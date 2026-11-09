"""Features: turn a table of homes into numbers a model can learn from.

The idea in one line: price every home in *today's dollars*, from what similar homes
nearby sold for, then let a gradient-boosted model learn the rest.

`FeatureBuilder.fit(train)` learns everything it needs from past sales only:

1. A **market index** per property type and month (the 2021-22 boom, the 2023-24
   correction). Every past sale is converted to "index-adjusted" log price, so a 2022
   sale and a 2026 sale can be compared.
2. **Neighbourhood price levels**: the adjusted log price per square foot of each
   community (and community x type), shrunk towards the type average when the
   community has few sales. Training rows get *out-of-fold* values, so a home never
   sees its own price.
3. **Comparable sales**: the 10 nearest past sales of the same type (by latitude and
   longitude), their adjusted price and price per square foot. Training rows skip
   themselves.

`transform(homes)` only uses the columns of a test file (never sold_price, date_sold
or the third-party estimate), so the same code runs in validation and on the test.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.neighbors import KDTree

PROPERTY_TYPES = ["Condo Apt", "Detached", "Semi-Detached", "Condo Townhouse",
                  "Freehold Townhouse", "Co-Op Apt"]
CATEGORICAL = ["house_type_name", "house_style", "municipality", "community"]

# Words in the listing remarks that move prices up or down.
KEYWORDS = {
    "kw_renovated": r"renovat|updated|remodel|brand new|new kitchen|upgraded|modern",
    "kw_needs_work": r"needs work|tlc|fixer|handyman|as[- ]is|investor|potential|bring your",
    "kw_luxury": r"luxur|custom|high[- ]end|designer|stunning|executive|chef",
    "kw_view": r"view|lake|ravine|skyline|waterfront|park",
    "kw_pool": r"pool",
    "kw_basement": r"basement|in-law|legal suite|separate entrance|income",
    "kw_transit": r"subway|ttc|transit|go station|lrt",
    "kw_corner": r"corner",
    "kw_penthouse": r"penthouse|ph\b",
}

K_COMPS = 10
SHRINK = 20          # how many sales a community needs before we trust its own average


def square_feet(house_area):
    """house_area is text, either one number ("622") or a range ("900-1099").
    Return a number: the number itself, or the middle of the range."""
    parts = house_area.astype(str).str.split("-", expand=True).reindex(columns=[0, 1])
    low = pd.to_numeric(parts[0], errors="coerce")
    high = pd.to_numeric(parts[1], errors="coerce").fillna(low)
    return (low + high) / 2


def month_number(dates):
    """Whole months since January 2021."""
    return (dates.dt.year - 2021) * 12 + dates.dt.month - 1


def xy(homes):
    """Coordinates in comparable units (a degree of longitude is 0.72 of a degree of latitude here)."""
    return np.column_stack([homes["latitude"].to_numpy(float), homes["longitude"].to_numpy(float) * 0.72])


class FeatureBuilder:
    def __init__(self, use_list_price=False):
        self.use_list_price = use_list_price

    # ------------------------------------------------------------------ fit
    def fit(self, train):
        train = train.reset_index(drop=True)
        log_price = np.log(train["sold_price"].to_numpy(float))
        sqft = square_feet(train["house_area"]).to_numpy(float)
        kind = train["house_type_name"].to_numpy()
        sold_month = month_number(train["date_sold"]).to_numpy()
        self.last_month = int(sold_month.max())

        # 1. Market index per type and month, from residuals after removing location.
        #    Two passes: location level -> monthly index -> location level again.
        index_value = np.zeros(len(train))
        for _ in range(2):
            adjusted = log_price - index_value
            group = train["community"].astype(str) + "|" + train["house_type_name"]
            level = pd.Series(adjusted).groupby(group.values).transform("mean").to_numpy()
            residual = log_price - level
            self.index = {}
            for t in np.unique(kind):
                mask = kind == t
                monthly = pd.Series(residual[mask]).groupby(sold_month[mask]).median()
                full = monthly.reindex(range(0, self.last_month + 1)).interpolate().bfill().ffill()
                smooth = full.rolling(3, min_periods=1, center=True).mean()
                # The last month only has past neighbours: use a trailing mean there.
                smooth.iloc[-1] = full.iloc[-2:].mean()
                self.index[t] = smooth - smooth.iloc[-1]     # 0 = price level of the last month
            index_value = self.index_at(kind, sold_month)

        adj = log_price - index_value                          # log price in last-month dollars
        adj_ppsf = adj - np.log(sqft)
        self.adj = adj

        # 2. Neighbourhood levels (out-of-fold for the training rows).
        self.type_mean = pd.Series(adj_ppsf).groupby(kind).median().to_dict()
        self.type_mean_price = pd.Series(adj).groupby(kind).median().to_dict()
        self.encoders = {}
        frame = pd.DataFrame({"community": train["community"].astype(str).values, "type": kind,
                              "adj_ppsf": adj_ppsf, "adj": adj})
        self.train_levels = pd.DataFrame(index=frame.index, columns=["comm_ppsf", "comm_type_ppsf", "comm_type_price"],
                                         dtype=float)
        for train_idx, hold_idx in KFold(5, shuffle=True, random_state=0).split(frame):
            tables = self._level_tables(frame.iloc[train_idx])
            self.train_levels.iloc[hold_idx] = self._apply_levels(frame.iloc[hold_idx], tables).values
        self.tables = self._level_tables(frame)

        # 3. Comparable sales: one KD-tree per property type.
        self.trees = {}
        for t in np.unique(kind):
            idx = np.where(kind == t)[0]
            self.trees[t] = (KDTree(xy(train.iloc[idx])), idx)
        self.comp_adj, self.comp_ppsf = adj, adj_ppsf
        self.train_comps = self._comps(train, exclude_self=True)

        # Category codes, learned from the training sales.
        # Columns with too many values for the tree model's categories are left out
        # (the neighbourhood levels above already carry their information).
        self.codes = {}
        for c in CATEGORICAL:
            values = sorted(train[c].astype(str).unique())
            if len(values) <= 250:
                self.codes[c] = {v: i for i, v in enumerate(values)}
        self._train_ref = train["ml_num"].to_numpy()
        return self

    def index_at(self, kind, month):
        """Market index (log scale, 0 = last training month) for each home."""
        month = np.clip(np.asarray(month), 0, self.last_month)
        out = np.zeros(len(month))
        for t, series in self.index.items():
            mask = np.asarray(kind) == t
            out[mask] = series.to_numpy()[month[mask]]
        return out

    def _level_tables(self, frame):
        comm = frame.groupby("community")["adj_ppsf"].agg(["mean", "count"])
        comm_type = frame.groupby(["community", "type"])[["adj_ppsf", "adj"]].agg(["mean", "count"])
        type_ppsf = frame.groupby("type")["adj_ppsf"].mean()
        type_adj = frame.groupby("type")["adj"].mean()
        return comm, comm_type, type_ppsf, type_adj

    def _apply_levels(self, frame, tables):
        comm, comm_type, type_ppsf, type_adj = tables
        prior_ppsf = frame["type"].map(type_ppsf).fillna(type_ppsf.mean()).to_numpy()
        prior_adj = frame["type"].map(type_adj).fillna(type_adj.mean()).to_numpy()
        c = comm.reindex(frame["community"].values)
        n = c["count"].fillna(0).to_numpy()
        comm_ppsf = (c["mean"].fillna(0).to_numpy() * n + prior_ppsf * SHRINK) / (n + SHRINK)
        key = pd.MultiIndex.from_arrays([frame["community"].values, frame["type"].values])
        ct = comm_type.reindex(key)
        n2 = ct[("adj_ppsf", "count")].fillna(0).to_numpy()
        ct_ppsf = (ct[("adj_ppsf", "mean")].fillna(0).to_numpy() * n2 + comm_ppsf * SHRINK) / (n2 + SHRINK)
        ct_price = (ct[("adj", "mean")].fillna(0).to_numpy() * n2 + prior_adj * SHRINK) / (n2 + SHRINK)
        return pd.DataFrame({"comm_ppsf": comm_ppsf, "comm_type_ppsf": ct_ppsf, "comm_type_price": ct_price},
                            index=frame.index)

    def _comps(self, homes, exclude_self=False):
        """Median adjusted price and price per sq ft of the K nearest past sales of the same type."""
        n = len(homes)
        out = {"comp_price": np.full(n, np.nan), "comp_ppsf": np.full(n, np.nan),
               "comp_price_near3": np.full(n, np.nan), "comp_distance": np.full(n, np.nan)}
        kind = homes["house_type_name"].to_numpy()
        points = xy(homes)
        for t in np.unique(kind):
            if t not in self.trees:
                continue
            tree, idx = self.trees[t]
            rows = np.where(kind == t)[0]
            k = min(K_COMPS + int(exclude_self), len(idx))
            dist, nb = tree.query(points[rows], k=k)
            if exclude_self:   # the nearest point of a training home is (usually) itself
                own = idx[nb] == rows[:, None]
                keep = ~own
                # Drop exactly one column per row: itself if found, else the farthest.
                keep[~own.any(axis=1), -1] = False
                dist = dist[keep].reshape(len(rows), k - 1)
                nb = nb[keep].reshape(len(rows), k - 1)
            sale = idx[nb]
            price = self.comp_adj[sale]
            ppsf = self.comp_ppsf[sale]
            out["comp_price"][rows] = np.median(price, axis=1)
            out["comp_ppsf"][rows] = np.nanmedian(np.where(np.isfinite(ppsf), ppsf, np.nan), axis=1)
            out["comp_price_near3"][rows] = np.median(price[:, :3], axis=1)
            out["comp_distance"][rows] = dist[:, -1]
        return pd.DataFrame(out, index=homes.index)

    # ------------------------------------------------------------ transform
    def transform(self, homes, training=False):
        """Features for `homes`. training=True only for the exact table given to fit()."""
        X = pd.DataFrame(index=homes.index)
        sqft = square_feet(homes["house_area"])
        X["square_feet"] = sqft
        X["log_sqft"] = np.log(sqft)
        X["area_is_range"] = homes["house_area"].astype(str).str.contains("-").astype(int)
        X["bedrooms"] = homes["bedrooms"]
        X["washrooms"] = homes["washrooms"]
        X["parking"] = homes["parking_total"]
        X["garage"] = homes["garage"]
        X["lot_front"] = homes["lot_front"]
        X["lot_depth"] = homes["lot_depth"]
        X["lot_area"] = homes["lot_front"] * homes["lot_depth"]
        X["latitude"] = homes["latitude"]
        X["longitude"] = homes["longitude"]
        X["listed_month_of_year"] = homes["date_listed"].dt.month
        for c in self.codes:
            X[c] = homes[c].astype(str).map(self.codes[c]).fillna(-1).astype(int)
            X.loc[X[c] < 0, c] = np.nan

        remarks = homes["remarks"].fillna("").str.lower()
        for name, pattern in KEYWORDS.items():
            X[name] = remarks.str.contains(pattern, regex=True).astype(int)
        X["remarks_length"] = remarks.str.len()

        if training:
            levels = self.train_levels.set_axis(homes.index)
            comps = self.train_comps.set_axis(homes.index)
        else:
            frame = pd.DataFrame({"community": homes["community"].astype(str).values,
                                  "type": homes["house_type_name"].values}, index=homes.index)
            levels = self._apply_levels(frame, self.tables)
            comps = self._comps(homes)
        X = pd.concat([X, levels, comps], axis=1)

        # Simple "comparable sales" valuations in today's dollars.
        X["value_from_comp_ppsf"] = X["comp_ppsf"] + X["log_sqft"]
        X["value_from_comm_ppsf"] = X["comm_type_ppsf"] + X["log_sqft"]

        if self.use_list_price:
            listed = month_number(homes["date_listed"]).to_numpy()
            index_listed = self.index_at(homes["house_type_name"].to_numpy(), listed)
            log_list = np.log(homes["list_price"].to_numpy(float))
            X["log_list_price"] = log_list
            X["list_in_today_dollars"] = log_list - index_listed
            X["list_vs_comps"] = X["list_in_today_dollars"] - X["comp_price"]
            X["list_vs_comm_value"] = X["list_in_today_dollars"] - X["value_from_comm_ppsf"]
            X["list_round_100k"] = (np.round(homes["list_price"]) % 100_000 == 0).astype(int)
            X["list_ends_900"] = (np.round(homes["list_price"]) % 10_000 >= 9_000).astype(int)
        return X

    def categorical_mask(self, X):
        return np.array([c in CATEGORICAL for c in X.columns])
