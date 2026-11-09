"""Chapter 1 of the story: describing a home in numbers.

Everything here comes from one row of a test file: only what is known before the sale.
Three columns of `listings` are deliberately never used:

- external_avm_estimate(_date): about 70% of these estimates are dated AFTER the sale, and
  the test files do not have the column. A model trained on it would learn from the future.
- days_on_market and date_sold: only known once the home has sold.
- list_price: never in the off-market features (rule of the case).

Comparable sales live in market.py, because those need the history of sales as well.
"""
import numpy as np
import pandas as pd

PROPERTY_TYPES = ["Condo Apt", "Detached", "Semi-Detached", "Condo Townhouse",
                  "Freehold Townhouse", "Co-Op Apt"]

# Columns the tree model should treat as labels rather than numbers.
CATEGORICAL = ["community", "house_type_name", "house_style", "municipality"]

# What the listing description says about condition. These groups were written by reading
# the remarks and checking which phrases go with a price above or below similar homes,
# on sales before June 2025 only. Phrases are grouped because no single phrase moved
# prices by more than about 2%, whereas the condition they describe does matter.
REMARK_FLAGS = {
    "renovated": r"fully (?:renovated|redone|updated)|renovated|turnkey|turn-key|gut|brand[- ]new|newly|top to bottom",
    "updated": r"updated|refreshed|refinished|new (?:kitchen|bathroom|flooring|furnace|roof|windows|central air)|newer",
    "original": r"original|dated|as[- ]is|personal touch|your vision|canvas|bring your|renovation ideas",
    "needs_work": r"needs? (?:work|attention|replacing|updating|tlc)|need replacing|fixer|handyman|worn|as[- ]is|invest in its potential",
    "high_end": r"quartz|stone counters|granite|frameless|spa-like|wide[- ]plank|smooth ceilings|built-in appliances|custom|stainless",
    "views": r"unobstructed|skyline|lake view|panoramic|floor-to-ceiling|ravine",
    "basement": r"finished basement|in-law|separate entrance|walk-?out",
    "outdoors": r"pool|deep lot|private backyard|south-facing|deck|patio|terrace|balcony",
    "amenities": r"concierge|gym|party room|locker|underground parking",
    "first_time_or_investor": r"first[- ]time|investor|starter|couple starting|start(?:ing)? out",
    "family_or_downsizer": r"growing family|downsizer|kids|room to grow",
    "commuter": r"commuter|transit|subway|highway",
}


def square_feet(house_area):
    """house_area is text, either one number ("622") or a range ("900-1099").
    Return a number: the number itself, or the middle of the range."""
    parts = house_area.astype(str).str.split("-", expand=True).reindex(columns=[0, 1])
    low = pd.to_numeric(parts[0], errors="coerce")
    high = pd.to_numeric(parts[1], errors="coerce").fillna(low)
    return (low + high) / 2


class CategoryEncoder:
    """Turn labels into integers using the training sales only. Labels first seen later
    become missing rather than an error (or a peek at the test file)."""

    def fit(self, train):
        self.codes = {column: {label: i for i, label in enumerate(sorted(train[column].astype(str).unique()))}
                      for column in CATEGORICAL}
        return self

    def transform(self, homes):
        return {column: homes[column].astype(str).map(self.codes[column]).astype(float)
                for column in CATEGORICAL}


def static_features(homes, encoder):
    """What a home is, ignoring when it is sold and what its neighbours sold for."""
    X = pd.DataFrame(index=homes.index)
    sqft = square_feet(homes["house_area"])
    X["log_sqft"] = np.log(sqft)
    X["bedrooms"] = homes["bedrooms"]
    X["washrooms"] = homes["washrooms"]
    X["parking"] = homes["parking_total"]
    X["garage"] = homes["garage"]
    X["sqft_per_room"] = sqft / (homes["bedrooms"] + homes["washrooms"]).clip(lower=1)
    # A lot is worth more than its area suggests when it is wide. Empty for condos,
    # which tree models handle without any filling in.
    X["lot_front"] = homes["lot_front"]
    X["lot_depth"] = homes["lot_depth"]
    X["log_lot_area"] = np.log(homes["lot_front"] * homes["lot_depth"])
    X["latitude"] = homes["latitude"]
    X["longitude"] = homes["longitude"]

    remarks = homes["remarks"].fillna("").str.lower()
    for name, pattern in REMARK_FLAGS.items():
        X["says_" + name] = remarks.str.contains(pattern, regex=True).astype(int)
    X["remark_words"] = remarks.str.split().str.len()

    for column, codes in encoder.transform(homes).items():
        X[column] = codes
    return X


def off_market_features(homes, market):
    """Off-market: the home plus its comps, with every trace of list price removed.

    That includes the sold-to-list ratios of past sales. They would arguably be allowed,
    but "the off-market model never touches a list price" is a simpler promise to check.
    """
    comps = market.comps(homes)
    comps = comps[[c for c in comps.columns if "sold_to_list" not in c]]
    return pd.concat([static_features(homes, market.encoder), comps], axis=1)


def list_aware_features(homes, market, comps=None):
    """List-aware: the question becomes "above or below the asking price, and by how much?"

    The useful signals are how the list price compares with what the comps suggest (an
    underpriced listing draws bids; an overpriced one sits) and how sales nearby have
    recently closed against their own list prices.
    """
    comps = market.comps(homes) if comps is None else comps
    X = pd.concat([static_features(homes, market.encoder), comps], axis=1)
    log_list = np.log(homes["list_price"].to_numpy())
    level = market.level_when_sold(homes)
    log_sqft = X["log_sqft"].to_numpy()
    X["log_list_price"] = log_list
    X["list_vs_nearest10"] = log_list - (X["nearest10_ppsf"].to_numpy() + level + log_sqft)
    X["list_vs_nearest40"] = log_list - (X["nearest40_ppsf"].to_numpy() + level + log_sqft)
    X["list_vs_community"] = log_list - (X["community_year_ppsf"].to_numpy() + level + log_sqft)
    X["list_vs_building"] = log_list - (X["building_ppsf"].to_numpy() + level + log_sqft)
    X["listed_on_round_number"] = (homes["list_price"].to_numpy() % 100_000 > 89_000).astype(int)
    # Spring bidding is hotter than autumn bidding, every year.
    X["calendar_month"] = homes["date_listed"].dt.month.to_numpy()
    return X
