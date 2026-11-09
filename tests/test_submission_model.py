"""Checks on the rules that matter most: no peeking at the future, no list price off-market.

Run with `pytest`. Everything uses tiny made-up data, so nothing is downloaded.
"""
import numpy as np
import pandas as pd

import data
import features
import market
import models
from test_starter import make_listings


def test_comps_ignore_sales_after_the_listing_date():
    # Same market fit, same homes: the sold values of later sales are replaced by nonsense
    # directly in the comp table. Comps from before the listing date must not change.
    listings = make_listings(800)
    home = make_listings(5, seed=3, first_id=9000).assign(date_listed=pd.Timestamp("2025-06-01"))
    fitted = market.Market().fit(listings)
    before = fitted.comps(home)

    later = fitted.sales["days"] >= market._days(pd.Series([pd.Timestamp("2025-06-01")]))[0]
    assert later.any() and (~later).any()
    for name in ["ppsf", "premium", "sold_to_list"]:
        fitted.sales[name] = np.where(later, 1e6, fitted.sales[name])
    after = fitted.comps(home)

    pd.testing.assert_frame_equal(before, after)


def test_a_sale_on_the_listing_date_is_not_a_comp():
    days = np.array([100, 200, 300])
    values = np.array([1.0, 2.0, 3.0])
    keys = np.array(["a", "a", "a"])
    mean, count = market._earlier_group_mean(keys, days, values, np.array(["a"]), np.array([200]))
    assert count[0] == 1 and mean[0] == 1.0


def test_nearest_comps_skip_later_sales():
    xy = np.array([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]])
    days = np.array([500, 100, 100])    # the very nearest sale is in the future
    found, _ = market._nearest_earlier(xy, days, {"v": np.array([1e9, 2.0, 4.0])}, np.array([[0.0, 0.0]]),
                                       np.array([200]), sizes=(2,))
    assert found[("v", 2)][0] == 3.0


def test_off_market_features_contain_no_list_price():
    fitted = market.Market().fit(make_listings(800))
    homes = data.hide_answers(make_listings(20, seed=4, first_id=7000), "off_market")
    assert "list_price" not in homes.columns
    X = features.off_market_features(homes, fitted)     # would fail if it read list_price
    assert not [c for c in X.columns if "list" in c]


def test_off_market_prediction_does_not_move_with_a_list_price():
    model = models.OffMarketModel().fit(make_listings(800))
    homes = make_listings(20, seed=4, first_id=7000)
    first = model.predict(homes.assign(list_price=1.0))
    second = model.predict(homes.assign(list_price=1e9))
    assert (first == second).all()


def test_season_and_forecast_of_the_index():
    months = pd.period_range("2022-01", "2025-12", freq="M")
    # A flat market where June is always 10% up.
    frame = pd.DataFrame({"month": months.repeat(5)})
    frame["r"] = np.where(frame["month"].dt.month == 6, 0.10, 0.0)
    index = market.MarketIndex().fit(["Detached"] * len(frame), frame["month"], frame["r"])
    june, october = index.value(["Detached", "Detached"], pd.PeriodIndex(["2026-06", "2026-10"], freq="M"))
    assert june > october
    assert abs(october) < 0.02
