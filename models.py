"""Chapter 3 of the story: the models. Every model has the same two steps:

    model.fit(train)        train = past sales, including sold_price and date_sold
    model.predict(homes)    homes = homes to value, with only the columns of a test file;
                            returns one predicted sold price per home

make_model() picks the model for each track. ListPriceBaseline and NearestSalesBaseline
are the starter's baselines, kept so every result can be compared with them.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

import features
import market


def make_model(track):
    """Return a fresh, untrained model for "off_market" or "list_aware"."""
    if track == "list_aware":
        return ListAwareModel()
    return OffMarketModel()


def _boosted_trees():
    """The learner both tracks share.

    The loss is absolute error, because the competition ranks on the MEDIAN absolute
    error. Early stopping is off: it would hold out a random sample of rows, which is
    exactly the time-mixing split this project avoids.
    """
    return HistGradientBoostingRegressor(loss="absolute_error", max_iter=400, learning_rate=0.06,
                                         max_leaf_nodes=63, min_samples_leaf=30, l2_regularization=1.0,
                                         early_stopping=False, random_state=0,
                                         categorical_features=features.CATEGORICAL)


class OffMarketModel:
    """What is this home worth, with no list price?

    log(price) = market level (forecast by MarketIndex) + time-free value (these trees).
    The trees see the home and its earlier comps; they never see a date, so they cannot
    freeze at the last month they saw, and they never see a list price.
    """

    def fit(self, train):
        train = train.reset_index(drop=True)
        self.market = market.market_for(train)
        comps = self.market.comps(train, staleness_seed=0)
        comps = comps[[c for c in comps.columns if "sold_to_list" not in c]]
        X = features.static_features(train, self.market.encoder).join(comps)
        self.columns = list(X.columns)
        self.trees = _boosted_trees().fit(X, self.market.time_free_price)
        return self

    def predict(self, homes):
        X = features.off_market_features(homes, self.market)[self.columns]
        return np.exp(self.trees.predict(X) + self.market.level_when_sold(homes))


class ListAwareModel:
    """What will this listing close at, given its list price?

    The target is log(sold / list), not the price. The list price already carries most
    of what the market knows, so the trees only learn the correction: which listings
    are priced to draw bids and which will have to come down.
    """

    def fit(self, train):
        train = train.reset_index(drop=True)
        self.market = market.market_for(train)
        comps = self.market.comps(train, staleness_seed=0)
        X = features.list_aware_features(train, self.market, comps)
        self.columns = list(X.columns)
        self.trees = _boosted_trees().fit(X, np.log(train["sold_price"] / train["list_price"]))
        return self

    def predict(self, homes):
        X = features.list_aware_features(homes, self.market)[self.columns]
        return homes["list_price"].to_numpy(dtype=float) * np.exp(self.trees.predict(X))


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
