"""The models. Every model has the same two steps:

    model.fit(train)        train = past sales, including sold_price and date_sold
    model.predict(homes)    homes = homes to value, with only the columns of a test file;
                            returns one predicted sold price per home

make_model() picks our model for each track (OffMarketModel, ListAwareModel, at the
bottom of this file). make_baseline() returns the two starter baselines, which we keep
to measure how much our ideas add.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import features


def make_model(track):
    """Return a fresh, untrained model for "off_market" or "list_aware"."""
    if track == "list_aware":
        return ListAwareModel()
    return OffMarketModel()


def make_baseline(track):
    """The two starter baselines, kept so `python run.py baselines` can compare against them."""
    if track == "list_aware":
        return ListPriceBaseline()
    return NearestSalesBaseline(k=10)


class ListPriceBaseline:
    """List-aware baseline: every home sells for exactly its list price."""

    def fit(self, train):
        return self   # nothing to learn

    def predict(self, homes):
        return homes["list_price"].to_numpy(dtype=float)


class NearestSalesBaseline:
    """Off-market baseline: the median sold price of the k nearest earlier sales of
    the same property type.

    "Nearest" means closest by latitude and longitude. "Earlier" means sold before the
    home was listed, so the model never looks into the future, even when the
    training table holds later sales. It makes no adjustment for size, condition or
    for prices moving since those sales: improving on that is your job.
    """

    def __init__(self, k=10):
        self.k = k

    def fit(self, train):
        columns = ["ml_num", "house_type_name", "date_sold", "latitude", "longitude", "sold_price"]
        # Newest sales first, so when several sales are equally near (a condo
        # building shares one location), the most recent ones are picked.
        self.sales = train[columns].sort_values(["date_sold", "ml_num"], ascending=[False, True])
        self.sales_by_type = dict(list(self.sales.groupby("house_type_name")))
        return self

    def predict(self, homes):
        return np.array([self.predict_one(home) for _, home in homes.iterrows()])

    def predict_one(self, home):
        sales = self.sales_by_type.get(home["house_type_name"], self.sales)
        earlier = sales[sales["date_sold"] < home["date_listed"]]
        if len(earlier) == 0:   # no earlier sale of this type: use every type
            earlier = self.sales[self.sales["date_sold"] < home["date_listed"]]

        # Distance in degrees. In Toronto a degree of longitude is only 0.72 times as
        # long as a degree of latitude (0.72 = cosine of 43.7 degrees north).
        dx = (earlier["longitude"].to_numpy() - home["longitude"]) * 0.72
        dy = earlier["latitude"].to_numpy() - home["latitude"]
        distance = np.sqrt(dx ** 2 + dy ** 2)

        nearest = np.argsort(distance, kind="stable")[: self.k]
        return float(np.median(earlier["sold_price"].to_numpy()[nearest]))


class FeatureModel:
    """A machine-learning model (gradient-boosted trees) on the features in features.py.

    It learns log(price) rather than price, so a 10% miss on a cheap condo counts
    as much as a 10% miss on an expensive house. Not used until you switch it on
    in make_model().
    """

    def __init__(self, make_features):
        self.make_features = make_features
        self.model = HistGradientBoostingRegressor(random_state=0)

    def fit(self, train):
        self.model.fit(self.make_features(train), np.log(train["sold_price"]))
        return self

    def predict(self, homes):
        return np.exp(self.model.predict(self.make_features(homes)))


# --------------------------------------------------------------------------------------
# Our models
# --------------------------------------------------------------------------------------

def gradient_boosting(loss="squared_error", quantile=None, max_iter=600, categorical=None):
    return HistGradientBoostingRegressor(
        loss=loss, quantile=quantile, learning_rate=0.05, max_iter=max_iter, max_leaf_nodes=63,
        min_samples_leaf=40, l2_regularization=1.0, categorical_features=categorical,
        early_stopping=False, random_state=0)


class _BoostedModel:
    """Shared machinery: a point model plus two quantile models for an 80% price range.

    Subclasses say what the target is (`target`) and how to turn a predicted target back
    into a price (`to_price`).
    """
    use_list_price = False
    LOW, HIGH = 0.10, 0.90

    def __init__(self, intervals=True):
        self.intervals = intervals

    def fit(self, train):
        train = train.reset_index(drop=True)
        self.builder = features.FeatureBuilder(self.use_list_price).fit(train)
        X = self.builder.transform(train, training=True)
        X = X.loc[:, X.notna().any()]          # a column with no values at all teaches nothing
        self.columns = list(X.columns)
        mask = self.builder.categorical_mask(X)
        y = self.target(train)
        self.point = gradient_boosting(categorical=mask).fit(X, y)
        if self.intervals:
            self.low = gradient_boosting("quantile", self.LOW, 250, mask).fit(X, y)
            self.high = gradient_boosting("quantile", self.HIGH, 250, mask).fit(X, y)
        return self

    def _X(self, homes):
        return self.builder.transform(homes)[self.columns]

    def predict(self, homes):
        return self.to_price(homes, self.point.predict(self._X(homes)))

    def predict_interval(self, homes):
        """80% range (10th and 90th percentile) of the sale price, one pair per home."""
        X = self._X(homes)
        low = self.to_price(homes, self.low.predict(X))
        high = self.to_price(homes, self.high.predict(X))
        point = self.to_price(homes, self.point.predict(X))
        return np.minimum(low, point), np.maximum(high, point)


class OffMarketModel(_BoostedModel):
    """Off-market: no list price.

    Target: log sold price in *last-month dollars* (sold price divided by the market index
    of its type in the month it sold). The test homes are then priced at the latest
    index level, which is all we know at prediction time.
    """

    def target(self, train):
        return self.builder.adj

    def to_price(self, homes, predicted):
        return np.exp(predicted)


class ListAwareModel(_BoostedModel):
    """List-aware: predict the sold-to-list ratio (in logs) and multiply by the list price.

    The list price carries most of the information; the model learns when homes sell
    over or under asking: under-priced listings (list below its comparable sales),
    neighbourhood, round-number list prices, the season.

    The typical sold-to-list ratio moves with the market (bidding wars in 2022, homes
    selling under asking later). A first version learned one ratio from all five years and
    predicted 2.4% too high in every validation month. So, like the price index, we keep a
    *ratio index*: the median log sold-to-list ratio per type and month. The model learns
    each sale's ratio relative to its own month, and the test homes get the ratio of the
    last three months of training sales.
    """
    use_list_price = True
    RECENT_MONTHS = 3

    def target(self, train):
        ratio = np.log(train["sold_price"].to_numpy(float) / train["list_price"].to_numpy(float))
        kind = train["house_type_name"].to_numpy()
        month = features.month_number(train["date_sold"]).to_numpy()
        frame = pd.DataFrame({"ratio": ratio, "kind": kind, "month": month})
        monthly = frame.groupby(["kind", "month"])["ratio"].transform("median").to_numpy()
        recent = frame[frame["month"] > month.max() - self.RECENT_MONTHS]
        overall = float(recent["ratio"].median())
        self.recent_ratio = recent.groupby("kind")["ratio"].median().to_dict()
        self.recent_overall = overall
        return ratio - monthly

    def to_price(self, homes, predicted):
        level = homes["house_type_name"].map(self.recent_ratio).fillna(self.recent_overall).to_numpy(float)
        return homes["list_price"].to_numpy(float) * np.exp(predicted + level)
