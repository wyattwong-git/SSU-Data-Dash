"""Chapter 2 of the story: what the market was doing, and what nearby homes sold for.

A tree model cannot extrapolate. If we simply gave it "months since 2021", every home
sold after the training data would get the same (last seen) time effect, and our test
runs up to four months past the last sale we can see. So we split the problem in two:

    log(price) = market level at the time of sale   <- MarketIndex, forecast explicitly
               + what this particular home is worth <- the tree model in models.py

This file provides the two things that split needs:

1. MarketIndex: a price level per property type per month, measured on *comparable*
   homes (a hedonic index), so that a month with many condos does not look like a crash.
2. Comps: what similar homes close by sold for.

THE ONE RULE OF THIS FILE: a comp only counts if it was sold BEFORE the home being valued
was listed. Every function that builds comps takes the query date and compares it with a
strict "<". tests/test_submission_model.py checks this by planting absurd future sales.
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingRegressor

import features

# How long after listing a home typically sells (the median days_on_market in the
# training data is 12). The test files have no sale date, so this is the best guess at
# when the sale will happen, and the only time information a real user would have too.
TYPICAL_DAYS_TO_SELL = 12

# Comps get less trustworthy as they age, so community averages come in two kinds:
# all history, and only this many days back.
RECENT_WINDOW_DAYS = 365

# An idea that did not earn its place. The hidden test's newest comps can be four months
# old, so we tried showing each training row comps as of 0 to 120 days before its listing.
# It made both tracks slightly worse on the development backtest (reports/experiments.md),
# so it is off. The switch stays so that experiments.py can reproduce that result.
TRAINING_COMP_STALENESS_DAYS = 0

# In Toronto a degree of longitude is 0.72 times as long as a degree of latitude.
LONGITUDE_SCALE = 0.72


def _days(dates):
    """Dates as whole days since 2000, so comparing them is plain number comparison."""
    return ((pd.to_datetime(dates) - pd.Timestamp("2000-01-01")).dt.days).to_numpy(dtype=np.int64)


class MarketIndex:
    """A log-price level for each property type and month, and a forecast beyond the data.

    level[type, month] is the median amount by which that month's sales beat the hedonic
    model, which knows the home but not the date. Because the hedonic model already
    accounts for size, type and location, this does not move when the mix of homes moves.
    Thin type-months (Co-Op Apt has few sales) are pulled towards the all-types level.
    """

    SHRINK = 50          # a type-month with 50 sales is half its own, half the all-types level
    MIN_SEASON_MONTHS = 24  # with less history than this we do not estimate a season
    # Share of the recent slope carried into the forecast. experiments.py tests this; see
    # reports/experiments.md for the result behind the value left here.
    TREND_SHARE = 0.0

    def fit(self, types, sold_months, residuals):
        df = pd.DataFrame({"type": np.asarray(types), "month": sold_months, "r": np.asarray(residuals)})
        months = pd.period_range(df["month"].min(), df["month"].max(), freq="M")
        overall = df.groupby("month")["r"].median().reindex(months).interpolate().bfill().ffill()

        self.last_month = months[-1]
        self.overall = overall
        self.levels = {}
        for kind, rows in df.groupby("type"):
            grouped = rows.groupby("month")["r"]
            weight = (grouped.size() / (grouped.size() + self.SHRINK)).reindex(months).fillna(0)
            own = grouped.median().reindex(months)
            self.levels[kind] = overall + weight * (own - overall).fillna(0)

        # The season: how far each calendar month usually sits from the 13-month average.
        # Measured on all types together, because one type alone is too noisy for it.
        self.season = pd.Series(0.0, index=range(1, 13))
        if len(months) >= self.MIN_SEASON_MONTHS:
            away_from_trend = overall - overall.rolling(13, center=True, min_periods=7).mean()
            by_calendar_month = away_from_trend.groupby(away_from_trend.index.month).median()
            self.season = (by_calendar_month - by_calendar_month.mean()).reindex(range(1, 13)).fillna(0.0)
        return self

    def value(self, types, months):
        """The level for each home: observed if the month is in the data, forecast if not.

        The forecast is deliberately humble. It takes the last three months with the
        season removed and puts back the season of the month asked for. There is no
        trend line: adding half the recent slope made off-market worse on the development
        backtest (June to September 2025), and a trend is the riskier bet anyway.
        """
        out = np.zeros(len(types))
        types = np.asarray(types)
        months = pd.PeriodIndex(months, freq="M")
        for kind in pd.unique(types):
            series = self.levels.get(kind, self.overall)
            without_season = series - self.season.reindex(series.index.month).to_numpy()
            current_level = without_season.iloc[-3:].mean()
            slope = (current_level - without_season.iloc[-6:-3].mean()) / 3 if len(series) >= 6 else 0.0
            where = np.flatnonzero(types == kind)
            for month in pd.unique(months[where]):
                if month in series.index:
                    level = series[month]
                elif month < series.index[0]:
                    level = series.iloc[0]
                else:
                    months_ahead = (month - self.last_month).n
                    level = current_level + self.TREND_SHARE * slope * months_ahead + self.season[month.month]
                out[where[months[where] == month]] = level
        return out


def _earlier_group_mean(sale_keys, sale_days, sale_values, query_keys, query_days, window=None):
    """For each query: the mean and count of values in the same group sold strictly earlier.

    Works by sorting all sales on (group, date) and keeping running totals, so each query
    only needs two binary searches. This is what makes rebuilding comps at every step of
    a walk-forward validation affordable.
    """
    codes, labels = pd.factorize(np.concatenate([sale_keys, query_keys]))
    sale_codes, query_codes = codes[: len(sale_keys)], codes[len(sale_keys):]

    usable = ~np.isnan(sale_values)
    base = np.int64(100_000)   # bigger than any day number we use
    combined = sale_codes[usable] * base + sale_days[usable]
    order = np.argsort(combined, kind="stable")
    combined = combined[order]
    running = np.concatenate([[0.0], np.cumsum(sale_values[usable][order])])

    end = np.searchsorted(combined, query_codes * base + query_days, side="left")   # strictly earlier
    start_days = query_days - window if window else np.zeros(len(query_days), dtype=np.int64)
    start = np.searchsorted(combined, query_codes * base + np.maximum(start_days, 0), side="left")
    count = end - start
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = (running[end] - running[start]) / count
    return mean, count


def _nearest_earlier(sales_xy, sale_days, sale_values, query_xy, query_days, sizes, pool=120):
    """Mean of each value column over the k nearest sales sold strictly earlier.

    A k-d tree finds the `pool` nearest sales of any date; we then throw away the ones
    not sold before the query date and average the closest k that remain. Returns a dict
    keyed by (value name, k), plus the distance to the k-th comp as a reliability signal.
    """
    results = {(name, k): np.full(len(query_xy), np.nan) for name in sale_values for k in sizes}
    distance_out = np.full(len(query_xy), np.nan)
    if len(sales_xy) == 0:
        return results, distance_out
    tree = cKDTree(sales_xy)
    pool = min(pool, len(sales_xy))

    for first in range(0, len(query_xy), 10_000):   # in chunks: the pool table is big
        chunk = np.arange(first, min(first + 10_000, len(query_xy)))
        distance, index = tree.query(query_xy[chunk], k=pool, workers=-1)
        if pool == 1:
            distance, index = distance[:, None], index[:, None]
        earlier = sale_days[index] < query_days[chunk, None]
        rank = np.cumsum(earlier, axis=1)
        for k in sizes:
            chosen = earlier & (rank <= k)
            taken = chosen.sum(axis=1)
            for name, values in sale_values.items():
                with np.errstate(invalid="ignore", divide="ignore"):
                    results[(name, k)][chunk] = (values[index] * chosen).sum(axis=1) / taken
        has_any = rank[:, -1] > 0
        reach = (earlier * distance).max(axis=1)
        distance_out[chunk[has_any]] = reach[has_any]
    return results, distance_out


class Market:
    """Everything learned from the past sales that both tracks share.

    Fitting it is the slow part of a model fit, and it is identical for the two tracks,
    so market_for() below keeps the most recent one instead of fitting it twice.
    """

    HEDONIC_ROUNDS = 150

    def fit(self, train):
        train = train.reset_index(drop=True)
        self.encoder = features.CategoryEncoder().fit(train)

        # Step 1: the hedonic model. It sees the home but not the date, so whatever it
        # cannot explain on average in a month is the market level of that month.
        X = features.static_features(train, self.encoder)
        log_price = np.log(train["sold_price"].to_numpy())
        hedonic = HistGradientBoostingRegressor(max_iter=self.HEDONIC_ROUNDS, learning_rate=0.1, max_leaf_nodes=63,
                                                early_stopping=False, random_state=0,
                                                categorical_features=features.CATEGORICAL)
        hedonic.fit(X, log_price)
        fitted = hedonic.predict(X)
        sold_months = train["date_sold"].dt.to_period("M")

        # Step 2: the index, then the "time-free" price of every sale.
        self.index = MarketIndex().fit(train["house_type_name"], sold_months, log_price - fitted)
        level = self.index.value(train["house_type_name"], sold_months)
        self.time_free_price = log_price - level

        # Step 3: the comp table. Each past sale leaves behind three facts: its price per
        # square foot, how far it beat the hedonic model (a local premium no size
        # difference can explain), and its sold-to-list ratio (list-aware track only).
        sqft = features.square_feet(train["house_area"]).to_numpy()
        self.sales = {
            "type": train["house_type_name"].to_numpy(dtype=str),
            "community": train["community"].to_numpy(dtype=str),
            "building": _building_key(train),
            "days": _days(train["date_sold"]),
            "xy": np.c_[train["longitude"].to_numpy() * LONGITUDE_SCALE, train["latitude"].to_numpy()],
            "ppsf": self.time_free_price - np.log(sqft),
            "premium": log_price - level - fitted,
            "sold_to_list": np.log(train["sold_price"].to_numpy() / train["list_price"].to_numpy()),
        }
        return self

    def level_when_sold(self, homes):
        """The market level at the sale date we expect: listing date plus a typical wait."""
        expected_sale = homes["date_listed"] + pd.Timedelta(days=TYPICAL_DAYS_TO_SELL)
        return self.index.value(homes["house_type_name"], expected_sale.dt.to_period("M"))

    def comps(self, homes, staleness_seed=None):
        """Comp features for each home, from sales strictly before its listing date.

        staleness_seed is only used while training, and only if the idea described at
        TRAINING_COMP_STALENESS_DAYS is switched on. Never pass it when predicting.
        """
        s = self.sales
        days = _days(homes["date_listed"])
        if staleness_seed is not None:
            rng = np.random.default_rng(staleness_seed)
            days = days - rng.integers(0, TRAINING_COMP_STALENESS_DAYS + 1, len(days))

        kinds = homes["house_type_name"].to_numpy(dtype=str)
        X = pd.DataFrame(index=homes.index)

        def add_group(prefix, sale_keys, query_keys, names, window=None):
            for name in names:
                mean, count = _earlier_group_mean(sale_keys, s["days"], s[name], query_keys, days, window)
                X[f"{prefix}_{name}"] = mean
            X[f"{prefix}_count"] = count

        everything = ["ppsf", "premium", "sold_to_list"]
        # Same community and type, all history and just the last year.
        community_key = np.char.add(s["community"], s["type"])
        query_community = np.char.add(homes["community"].to_numpy(dtype=str), kinds)
        add_group("community", community_key, query_community, everything)
        add_group("community_year", community_key, query_community, everything, RECENT_WINDOW_DAYS)
        # Same building: condos at one location share a level no size or area explains.
        add_group("building", s["building"], _building_key(homes), everything)

        # The nearest sales of the same type, however the area is named.
        values = {name: s[name] for name in everything}
        query_xy = np.c_[homes["longitude"].to_numpy() * LONGITUDE_SCALE, homes["latitude"].to_numpy()]
        X["comp_reach"] = np.nan
        for kind in pd.unique(kinds):
            rows = np.flatnonzero(kinds == kind)
            pool = np.flatnonzero(s["type"] == kind)
            found, reach = _nearest_earlier(s["xy"][pool], s["days"][pool], {n: v[pool] for n, v in values.items()},
                                            query_xy[rows], days[rows], sizes=(10, 40))
            for (name, k), result in found.items():
                column = f"nearest{k}_{name}"
                if column not in X:
                    X[column] = np.nan
                X.iloc[rows, X.columns.get_loc(column)] = result
            X.iloc[rows, X.columns.get_loc("comp_reach")] = reach
        return X


def _building_key(homes):
    """Homes at the exact same coordinates and of the same type: in practice, one building."""
    return (homes["latitude"].round(5).astype(str) + "," + homes["longitude"].round(5).astype(str) + ","
            + homes["house_type_name"].astype(str)).to_numpy(dtype=str)


_latest = {}


def market_for(train):
    """Fit a Market on these sales, or reuse the one just fitted on the very same sales."""
    key = (len(train), train["ml_num"].iloc[0], train["ml_num"].iloc[-1], train["date_sold"].max())
    if _latest.get("key") != key:
        _latest.update(key=key, market=Market().fit(train))
    return _latest["market"]
