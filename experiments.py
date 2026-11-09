"""What we tried, and what each idea was worth. Run once: python experiments.py

Every variant is scored on ONE development block: train on sales before June 2025,
predict June to September 2025. That is the hidden test's shape and season, one year
earlier. All choices were made here, so the official validation months (December 2025
to May 2026) stayed untouched: picking the best of many tries on the months you report
would flatter the score, which is one of the mistakes the case warns about.

The output, reports/experiments.md, is the "what didn't help" part of the report.
"""
import json
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import data
import features
import market
import models

ROOT = Path(__file__).parent
CUTOFF, END = "2025-06-01", "2025-10-01"


def score(predicted, homes):
    error = 100 * (predicted - homes["sold_price"].to_numpy()) / homes["sold_price"].to_numpy()
    return {"mdape": round(float(np.median(np.abs(error))), 2), "mape": round(float(np.mean(np.abs(error))), 2),
            "median_signed_error": round(float(np.median(error)), 2)}


def run_block(train, homes):
    market._latest.clear()   # a variant may change the Market, so never reuse a fitted one
    return {track: score(models.make_model(track).fit(train).predict(data.hide_answers(homes, track)), homes)
            for track in ["off_market", "list_aware"]}


def squared_loss_trees():
    return HistGradientBoostingRegressor(loss="squared_error", max_iter=400, learning_rate=0.06, max_leaf_nodes=63,
                                         min_samples_leaf=30, l2_regularization=1.0, early_stopping=False,
                                         random_state=0, categorical_features=features.CATEGORICAL)


def starter_style_block(train, homes):
    """Control: one tree model with the date as a feature, as FeatureModel did in the starter."""
    def design(rows, encoder):
        X = features.static_features(rows, encoder)
        X["months"] = (rows["date_listed"] - pd.Timestamp("2021-01-01")).dt.days / 30.44
        return X
    encoder = features.CategoryEncoder().fit(train)
    trees = models._boosted_trees().fit(design(train, encoder), np.log(train["sold_price"]))
    return score(np.exp(trees.predict(design(homes, encoder))), homes)


def main():
    listings = data.load("listings")
    train = listings[listings["date_sold"] < CUTOFF]
    homes = listings[(listings["date_sold"] >= CUTOFF) & (listings["date_sold"] < END)]
    results = {}

    def record(name, outcome):
        results[name] = outcome
        print(name, outcome, flush=True)

    # The variants below were measured against the draft, which aged the training comps.
    # Dropping that aging is the one change the experiments led to, giving the final design.
    aged = mock.patch.object(market, "TRAINING_COMP_STALENESS_DAYS", 120)
    with aged:
        record("Draft design (training comps aged 0-120 days)", run_block(train, homes))

    class NoSeason(market.MarketIndex):
        MIN_SEASON_MONTHS = 10_000
    with aged, mock.patch.object(market, "MarketIndex", NoSeason):
        record("No seasonal term in the forecast", run_block(train, homes))

    class HalfTrend(market.MarketIndex):
        TREND_SHARE = 0.5
    with aged, mock.patch.object(market, "MarketIndex", HalfTrend):
        record("Half of the recent trend added to the forecast", run_block(train, homes))

    record("FINAL design: draft with training comps not aged", run_block(train, homes))

    with aged, mock.patch.object(features, "REMARK_FLAGS", {}):
        record("No remark flags", run_block(train, homes))

    with aged, mock.patch.object(models, "_boosted_trees", squared_loss_trees):
        record("Squared-error loss", run_block(train, homes))

    record("Control: one model, date as a feature, no comps (off-market)",
           {"off_market": starter_style_block(train, homes)})

    # What a random split would have told us: the same control, tested on a random 20% of
    # the same months' worth of data. It sees sales from after the homes it predicts.
    recent = listings[listings["date_sold"] < END]
    held_out = recent.sample(frac=0.2, random_state=0)
    record("Warning: same control scored on a RANDOM 20% split (off-market)",
           {"off_market": starter_style_block(recent.drop(held_out.index), held_out)})

    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "experiments.json").write_text(json.dumps(results, indent=2))
    lines = ["# What we tried", "",
             f"Train on sales before {CUTOFF}, predict sales from then until {END} ({len(homes):,} homes).",
             "Errors in percent. Variants change one thing from the draft design.", "",
             "| Variant | Off-market MdAPE | Off-market MAPE | List-aware MdAPE | List-aware MAPE |", "|---|---|---|---|---|"]
    for name, outcome in results.items():
        off, aware = outcome.get("off_market", {}), outcome.get("list_aware", {})
        lines.append(f"| {name} | {off.get('mdape', '')} | {off.get('mape', '')} "
                     f"| {aware.get('mdape', '')} | {aware.get('mape', '')} |")
    (ROOT / "reports" / "experiments.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
