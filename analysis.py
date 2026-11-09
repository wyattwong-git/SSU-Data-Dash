"""Chapter 5 of the story: where the model fails, and how sure it is.

A single error number hides who pays for it. Everything here is measured on predictions
for sales the models had not seen (run.py validate and run.py backtest):

- error by property type, price, neighbourhood and how the sale actually played out;
- bias: whether a group of homes is consistently over- or under-valued;
- honest ranges: an 80% price range calibrated on one period and CHECKED on a later one;
- the simple "maximum error" benchmark from the handout, recomputed on the same homes.

Fields such as days_on_market appear below only to explain errors after the fact. They
are known only after a sale and no model ever receives them.
"""
import json

import numpy as np
import pandas as pd

import data
import features

COLOURS = {"off_market": "#2a78d6", "list_aware": "#eb6834"}
LABELS = {"off_market": "Off-market", "list_aware": "List-aware"}
TRACKS = ["off_market", "list_aware"]
# Calibrate ranges on the development block, the earliest period that resembles the test.
CALIBRATION_BLOCK = "Jun-Sep 2025 (development)"
RANGE_LEVEL = 0.80


def add_errors(rows):
    rows = rows.copy()
    rows["signed"] = 100 * (rows["predicted_price"] - rows["sold_price"]) / rows["sold_price"]
    rows["absolute"] = rows["signed"].abs()
    return rows


def summary(rows, by):
    """MdAPE, MAPE, bias and size of each group."""
    grouped = rows.groupby(by, observed=True)
    return pd.DataFrame({"homes": grouped.size(), "mdape": grouped["absolute"].median().round(2),
                         "mape": grouped["absolute"].mean().round(2),
                         "median_signed": grouped["signed"].median().round(2)})


def add_breakdown_columns(rows):
    rows = rows.copy()
    rows["price_band"] = pd.cut(rows["sold_price"], [0, 500e3, 750e3, 1e6, 1.5e6, 2.5e6, np.inf],
                                labels=["under $500k", "$500-750k", "$750k-1M", "$1-1.5M", "$1.5-2.5M", "over $2.5M"])
    ratio = rows["sold_price"] / rows["list_price"]
    rows["how_it_sold"] = pd.cut(ratio, [0, 0.93, 0.97, 1.03, 1.10, np.inf],
                                 labels=["far below list (<93%)", "below list (93-97%)", "near list (97-103%)",
                                         "above list (103-110%)", "bidding war (>110%)"])
    rows["time_on_market"] = pd.cut(rows["days_on_market"], [-1, 7, 30, 90, np.inf],
                                    labels=["within a week", "8-30 days", "31-90 days", "over 90 days"])
    return rows


def simple_benchmark(listings):
    """The handout's maximum error: median sale of the same type and neighbourhood over the
    previous 12 months. Recomputed on our validation homes as a check on our reading of it."""
    from run import VALIDATION_MONTHS
    scored = []
    for month in VALIDATION_MONTHS:
        past = listings[(listings["date_sold"] < month.start_time)
                        & (listings["date_sold"] >= month.start_time - pd.DateOffset(months=12))]
        homes = listings[listings["date_sold"].dt.to_period("M") == month]
        local = past.groupby(["house_type_name", "community"])["sold_price"].median().rename("local")
        by_type = past.groupby("house_type_name")["sold_price"].median().rename("by_type")
        merged = homes.join(local, on=["house_type_name", "community"]).join(by_type, on="house_type_name")
        predicted = merged["local"].fillna(merged["by_type"])
        scored.append(100 * (predicted - merged["sold_price"]).abs() / merged["sold_price"])
    return pd.concat(scored).dropna()


def calibrate_ranges(calibration):
    """Per type: how far sold prices fall from predictions, as factors covering the middle 80%.

    This is split-conformal in plain words: look at errors on homes the model did not train
    on, and widen each prediction by what those errors were.
    """
    low, high = (1 - RANGE_LEVEL) / 2, 1 - (1 - RANGE_LEVEL) / 2
    ratio = np.log(calibration["sold_price"] / calibration["predicted_price"])
    factors = ratio.groupby(calibration["house_type_name"]).quantile([low, high]).unstack()
    factors.columns = ["low", "high"]
    overall = [ratio.quantile(low), ratio.quantile(high)]
    return factors.reindex(features.PROPERTY_TYPES).fillna({"low": overall[0], "high": overall[1]})


def range_coverage(rows, factors):
    lower = rows["predicted_price"] * np.exp(rows["house_type_name"].map(factors["low"]))
    upper = rows["predicted_price"] * np.exp(rows["house_type_name"].map(factors["high"]))
    inside = (rows["sold_price"] >= lower) & (rows["sold_price"] <= upper)
    return float(inside.mean()), float(((upper - lower) / rows["predicted_price"]).median())


def _style(ax):
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def figure_by_group(validation, root):
    """Median error by property type: one figure, thin bars, labelled values."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 3.2))
    order = summary(validation[validation["track"] == "off_market"], "house_type_name")
    order = order.sort_values("mdape").index
    width = 0.34
    for shift, track in zip((-width / 2 - 0.01, width / 2 + 0.01), TRACKS):
        scores = summary(validation[validation["track"] == track], "house_type_name")["mdape"].reindex(order)
        bars = ax.bar(np.arange(len(order)) + shift, scores, width, color=COLOURS[track], label=LABELS[track])
        ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
    ax.set_xticks(range(len(order)), [o.replace(" ", "\n", 1) for o in order], fontsize=9)
    ax.set_ylabel("Median absolute % error")
    ax.set_title("Validation error by property type, Dec 2025 to May 2026", fontsize=11)
    ax.legend(frameon=False)
    _style(ax)
    fig.tight_layout()
    (root / "figures").mkdir(exist_ok=True)
    fig.savefig(root / "figures" / "error_by_type.png", dpi=170)
    plt.close(fig)


def figure_horizon(backtest, root):
    """The honesty figure: does error grow as the months since the last training sale grow?"""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(8, 2.9))
    styles = dict(zip(backtest["block"].unique(), ["-", "--", ":"]))
    for track, ax in zip(TRACKS, axes):
        rows = backtest[backtest["track"] == track].copy()
        months = pd.to_datetime(rows["date_sold"]).dt.to_period("M")
        first = rows["block"].map(months.groupby(rows["block"]).min())
        rows["ahead"] = [(m - f).n + 1 for m, f in zip(months, first)]
        for block, line in styles.items():
            scores = rows[rows["block"] == block].groupby("ahead")["absolute"].median()
            ax.plot(scores.index, scores.values, marker="o", linewidth=2, linestyle=line,
                    color=COLOURS[track], label=block.replace(" (development)", " (dev)"))
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xlabel("Months past the last training sale")
        ax.set_title(LABELS[track], fontsize=11)
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False, fontsize=7.5, loc="lower right")
        _style(ax)
    axes[0].set_ylabel("MdAPE (%)")
    fig.tight_layout()
    (root / "figures").mkdir(exist_ok=True)
    fig.savefig(root / "figures" / "error_by_horizon.png", dpi=170)
    plt.close(fig)


def write_all(root):
    validation = add_breakdown_columns(add_errors(pd.read_csv(root / "outputs" / "validation_predictions.csv")))
    backtest = add_breakdown_columns(add_errors(pd.read_csv(root / "outputs" / "backtest_predictions.csv")))
    listings = data.load("listings")
    out = {}

    # --- Overall, on both kinds of test ---
    for track in TRACKS:
        rows = validation[validation["track"] == track]
        blocks = summary(backtest[backtest["track"] == track], "block")
        out[track] = {"validation": summary(rows, "track").iloc[0].to_dict(),
                      "by_month": summary(rows, "month").to_dict("index"),
                      "backtest": blocks.to_dict("index"),
                      "by_type": summary(rows, "house_type_name").to_dict("index"),
                      "by_price": summary(rows, "price_band").to_dict("index"),
                      "by_sale_outcome": summary(rows, "how_it_sold").to_dict("index"),
                      "by_time_on_market": summary(rows, "time_on_market").to_dict("index")}
        community = summary(rows, "community")
        community = community[community["homes"] >= 40].sort_values("mdape")
        out[track]["best_communities"] = community.head(5).to_dict("index")
        out[track]["worst_communities"] = community.tail(5).to_dict("index")
        out[track]["community_mdape_spread"] = [float(community["mdape"].min()), float(community["mdape"].median()),
                                                float(community["mdape"].max())]

    benchmark = simple_benchmark(listings)
    out["simple_benchmark_on_validation"] = {"mdape": round(float(benchmark.median()), 2),
                                             "mape": round(float(benchmark.mean()), 2)}

    # --- Ranges: calibrate on the development block, check on later sales ---
    predictions_for_range = {}
    for track in TRACKS:
        calibration = backtest[(backtest["track"] == track) & (backtest["block"] == CALIBRATION_BLOCK)]
        factors = calibrate_ranges(calibration)
        checks = {"validation (1 month ahead)": validation[validation["track"] == track],
                  "Feb-May 2026 (4 months ahead)": backtest[(backtest["track"] == track)
                                                           & (backtest["block"] == "Feb-May 2026")]}
        out[track]["range"] = {"level": RANGE_LEVEL, "factors": factors.round(4).to_dict("index")}
        for name, rows in checks.items():
            if len(rows):
                covered, width = range_coverage(rows, factors)
                out[track]["range"][name] = {"coverage": round(covered, 3), "median_width": round(width, 3)}
        predictions_for_range[track] = factors

    # --- Ranges for the test homes, as a companion to the two submission files ---
    for track, factors in predictions_for_range.items():
        path = root / "submissions" / f"submission_{track}.csv"
        if not path.exists():
            continue
        homes = data.load(track)[["ml_num", "house_type_name"]]
        rows = pd.read_csv(path, dtype={"ml_num": str}).merge(homes, on="ml_num")
        rows["price_low_80"] = (rows["predicted_price"] * np.exp(rows["house_type_name"].map(factors["low"]))).round()
        rows["price_high_80"] = (rows["predicted_price"] * np.exp(rows["house_type_name"].map(factors["high"]))).round()
        (root / "reports").mkdir(exist_ok=True)
        rows[["ml_num", "predicted_price", "price_low_80", "price_high_80"]].to_csv(
            root / "reports" / f"price_ranges_{track}.csv", index=False)

    (root / "reports").mkdir(exist_ok=True)
    (root / "reports" / "error_analysis.json").write_text(json.dumps(out, indent=2, default=str))
    figure_by_group(validation, root)
    figure_horizon(backtest, root)
    print(json.dumps(out, indent=1, default=str))
