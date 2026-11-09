"""Run the whole pipeline. From this folder:

    python run.py validate   test the models on the last six months of sales
    python run.py report     score those predictions and draw a figure
    python run.py predict    predict the test homes and write the two submission files
    python run.py check      check the submission files before you hand them in
    python run.py baselines  the same walk-forward test for the two starter baselines
    python run.py analysis   where the model fails: errors by type, month, neighbourhood

On Windows, type py instead of python if python is not found.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import data
import models
from check_submission import check_all

ROOT = Path(__file__).parent
TRACKS = ["off_market", "list_aware"]
# Walk-forward validation: each of these months is predicted by models trained only
# on the sales before that month, the same way the real test works.
VALIDATION_MONTHS = pd.period_range("2025-12", "2026-05", freq="M")


def validate(make=models.make_model, file_name="validation_predictions.csv"):
    listings = data.load("listings")
    results = []
    for month in VALIDATION_MONTHS:
        train = listings[listings["date_sold"] < month.start_time]
        homes = listings[listings["date_sold"].dt.to_period("M") == month]
        for track in TRACKS:
            model = make(track).fit(train)
            visible = data.hide_answers(homes, track)
            frame = pd.DataFrame({
                "ml_num": homes["ml_num"], "track": track, "month": str(month),
                "house_type_name": homes["house_type_name"], "community": homes["community"],
                "sold_price": homes["sold_price"],
                "predicted_price": np.round(model.predict(visible))})
            if hasattr(model, "predict_interval"):
                low, high = model.predict_interval(visible)
                frame["low_80"], frame["high_80"] = np.round(low), np.round(high)
            results.append(frame)
        print(f"{month}: trained on {len(train):,} earlier sales, predicted {len(homes):,} homes")
    path = ROOT / "outputs" / file_name
    path.parent.mkdir(exist_ok=True)
    pd.concat(results).to_csv(path, index=False)
    print(f"Wrote {path.relative_to(ROOT)}")


def bootstrap_ci(values, statistic, repeats=1000):
    """95% interval: recompute the statistic on 1,000 resamples of the homes
    (drawn with replacement) and keep the middle 95% of the results."""
    rng = np.random.default_rng(0)
    results = [statistic(rng.choice(values, size=len(values))) for _ in range(repeats)]
    return [round(float(x), 2) for x in np.percentile(results, [2.5, 97.5])]


def metrics(predictions):
    """Scores for one track. Errors are in percent of the sold price."""
    signed = 100 * (predictions["predicted_price"] - predictions["sold_price"]) / predictions["sold_price"]
    absolute = signed.abs().to_numpy()
    extra = {}
    if "low_80" in predictions:
        inside = predictions["sold_price"].between(predictions["low_80"], predictions["high_80"])
        width = 100 * (predictions["high_80"] - predictions["low_80"]) / predictions["predicted_price"]
        extra = {"interval_80_coverage": round(float(100 * inside.mean()), 1),
                 "interval_80_median_width": round(float(width.median()), 1)}
    return extra | {
        "n": len(predictions),
        "mape": round(float(np.mean(absolute)), 2),          # mean absolute % error
        "mape_95ci": bootstrap_ci(absolute, np.mean),
        "mdape": round(float(np.median(absolute)), 2),       # median absolute % error
        "mdape_95ci": bootstrap_ci(absolute, np.median),
        "median_signed_error": round(float(np.median(signed)), 2),  # above 0: predicts too high
    }


def report():
    path = ROOT / "outputs" / "validation_predictions.csv"
    if not path.exists():
        sys.exit(f"{path} does not exist yet: run `python run.py validate` first")
    predictions = pd.read_csv(path)
    scores = {track: metrics(predictions[predictions["track"] == track]) for track in TRACKS}

    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "validation_report.json").write_text(json.dumps(scores, indent=2))
    lines = [
        "# Validation report", "",
        f"Walk-forward validation, {VALIDATION_MONTHS[0]} to {VALIDATION_MONTHS[-1]}: each month was "
        "predicted by models trained only on the sales before it.",
        "Errors are (predicted - sold) / sold. Brackets: bootstrap 95% interval.", "",
        "| Track | Homes | MAPE | MdAPE | Median signed error |", "|---|---|---|---|---|"]
    for track, s in scores.items():
        lines.append(f"| {track} | {s['n']:,} | {s['mape']:.2f}% ({s['mape_95ci'][0]:.2f} to {s['mape_95ci'][1]:.2f}) "
                     f"| {s['mdape']:.2f}% ({s['mdape_95ci'][0]:.2f} to {s['mdape_95ci'][1]:.2f}) "
                     f"| {s['median_signed_error']:+.2f}% |")
    if all("interval_80_coverage" in s for s in scores.values()):
        lines += ["", "80% price ranges (10th to 90th percentile models): a good range holds the sold price "
                  "80% of the time.", "", "| Track | Sold price inside the range | Median range width |",
                  "|---|---|---|"]
        lines += [f"| {t} | {s['interval_80_coverage']:.1f}% | {s['interval_80_median_width']:.1f}% of the price |"
                  for t, s in scores.items()]
    lines += [
        "", "MAPE: mean absolute percentage error. MdAPE: median absolute percentage error.",
        "A median signed error above 0 means the model tends to predict too high.",
        "The intervals treat homes as independent. Homes sold in the same month share the same",
        "market swings, so the real uncertainty is somewhat wider.", "",
        "![Error by price range](../figures/error_by_price_range.png)", ""]
    (ROOT / "reports" / "validation_report.md").write_text("\n".join(lines))
    write_latex_table(scores)
    error_by_price_range(predictions)
    print("\n".join(lines))
    print("Wrote reports/validation_report.md, reports/validation_report.json, reports/results_table.tex "
          "and figures/error_by_price_range.png")


def write_latex_table(scores):
    """The same scores as a LaTeX table, for report/report.tex to include."""
    rows = []
    for track, s in scores.items():
        name = track.replace("_", "-")
        rows.append(f"{name} & {s['n']:,} & {s['mdape']:.2f} ({s['mdape_95ci'][0]:.2f}--{s['mdape_95ci'][1]:.2f}) "
                    f"& {s['mape']:.2f} ({s['mape_95ci'][0]:.2f}--{s['mape_95ci'][1]:.2f}) "
                    f"& {s['median_signed_error']:+.2f} \\\\")
    table = "\n".join([
        "% Written by `python run.py report`. Do not edit by hand: rerun the command instead.",
        "\\begin{tabular}{lrccc}", "\\hline",
        "Track & Homes & MdAPE \\% (95\\% CI) & MAPE \\% (95\\% CI) & Median signed error \\% \\\\", "\\hline",
        *rows, "\\hline", "\\end{tabular}"])
    (ROOT / "reports" / "results_table.tex").write_text(table + "\n")


def error_by_price_range(predictions):
    """Figure: how big the errors are, from the cheapest tenth of homes to the dearest.

    Want more figures? Copy this function and group the errors by something else:
    house_type_name, month, or community (add it to the predictions file in validate()).
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colours = {"off_market": "#2a78d6", "list_aware": "#eb6834"}
    for track in TRACKS:
        rows = predictions[predictions["track"] == track]
        error = 100 * (rows["predicted_price"] - rows["sold_price"]).abs() / rows["sold_price"]
        decile = pd.qcut(rows["sold_price"], 10, labels=False) + 1   # 1 = cheapest tenth
        by_decile = error.groupby(decile)
        middle = by_decile.median()
        ax.plot(middle.index, middle.values, marker="o", linewidth=2, color=colours[track], label=track)
        ax.fill_between(middle.index, by_decile.quantile(0.25), by_decile.quantile(0.75),
                        color=colours[track], alpha=0.15, linewidth=0)
    prices = rows["sold_price"].groupby(decile).median()
    ax.set_xticks(prices.index, [f"{d}\n${p / 1e6:.2f}M" for d, p in prices.items()])
    ax.set_xlabel("Sold-price decile (1 = cheapest tenth of homes), with its median price")
    ax.set_ylabel("Absolute % error\n(line: median, band: middle 50%)")
    ax.set_title(f"Validation error by price range, {VALIDATION_MONTHS[0]} to {VALIDATION_MONTHS[-1]}")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#e5e5e5")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    (ROOT / "figures").mkdir(exist_ok=True)
    fig.savefig(ROOT / "figures" / "error_by_price_range.png", dpi=150)
    plt.close(fig)


def predict():
    listings = data.load("listings")
    (ROOT / "submissions").mkdir(exist_ok=True)
    for track in TRACKS:
        homes = data.load(track)
        model = models.make_model(track).fit(listings)
        predicted = model.predict(homes)
        path = ROOT / "submissions" / f"submission_{track}.csv"
        pd.DataFrame({"ml_num": homes["ml_num"], "predicted_price": np.round(predicted)}).to_csv(path, index=False)
        print(f"Wrote {path.relative_to(ROOT)} ({len(homes):,} homes)")
        if hasattr(model, "predict_interval"):   # the 80% ranges, kept apart from the submission file
            low, high = model.predict_interval(homes)
            (ROOT / "outputs").mkdir(exist_ok=True)
            pd.DataFrame({"ml_num": homes["ml_num"], "predicted_price": np.round(predicted),
                          "low_80": np.round(low), "high_80": np.round(high)}).to_csv(
                ROOT / "outputs" / f"test_ranges_{track}.csv", index=False)


def check():
    return check_all(ROOT / "submissions")


def baselines():
    """Walk-forward validation of the two starter baselines, then a side-by-side table."""
    validate(models.make_baseline, "validation_baselines.csv")
    compare()


def compare():
    rows = ["| Track | Model | MdAPE | MAPE | Median signed error |", "|---|---|---|---|---|"]
    for file_name, label in [("validation_baselines.csv", "Starter baseline"),
                             ("validation_predictions.csv", "Our model")]:
        path = ROOT / "outputs" / file_name
        if not path.exists():
            continue
        predictions = pd.read_csv(path)
        for track in TRACKS:
            s = metrics(predictions[predictions["track"] == track])
            rows.append(f"| {track} | {label} | {s['mdape']:.2f}% | {s['mape']:.2f}% | {s['median_signed_error']:+.2f}% |")
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "baseline_comparison.md").write_text("\n".join(rows) + "\n")
    print("\n".join(rows))


def analysis():
    """Where the model fails: error tables by type, month and neighbourhood, and two figures."""
    import matplotlib.pyplot as plt

    path = ROOT / "outputs" / "validation_predictions.csv"
    if not path.exists():
        sys.exit(f"{path} does not exist yet: run `python run.py validate` first")
    p = pd.read_csv(path)
    p["signed"] = 100 * (p["predicted_price"] - p["sold_price"]) / p["sold_price"]
    p["abs"] = p["signed"].abs()

    def table(by, min_n=1):
        g = p.groupby(["track", by]).agg(homes=("abs", "size"), mdape=("abs", "median"),
                                         mape=("abs", "mean"), median_signed=("signed", "median"))
        return g[g["homes"] >= min_n].round(2)

    by_type, by_month = table("house_type_name"), table("month")
    by_comm = table("community", min_n=30)
    worst = by_comm.reset_index().sort_values("mdape", ascending=False).groupby("track").head(10)
    best = by_comm.reset_index().sort_values("mdape").groupby("track").head(5)
    lines = ["# Where the model fails", "", "## By property type", "", by_type.to_markdown(), "",
             "## By month (each predicted only from earlier sales)", "", by_month.to_markdown(), "",
             "## Hardest neighbourhoods (at least 30 validation homes)", "", worst.to_markdown(index=False), "",
             "## Easiest neighbourhoods", "", best.to_markdown(index=False), ""]
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "error_analysis.md").write_text("\n".join(lines))

    (ROOT / "figures").mkdir(exist_ok=True)
    colours = {"off_market": "#2a78d6", "list_aware": "#eb6834"}
    fig, ax = plt.subplots(figsize=(8, 4.5))
    order = by_type.loc["off_market"].sort_values("mdape").index
    positions = np.arange(len(order))
    for i, track in enumerate(TRACKS):
        values = by_type.loc[track].reindex(order)["mdape"]
        ax.barh(positions + (i - 0.5) * 0.4, values, height=0.4, color=colours[track], label=track)
    ax.set_yticks(positions, order)
    ax.set_xlabel("Median absolute % error (MdAPE)")
    ax.set_title("Validation error by property type")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "error_by_type.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for track in TRACKS:
        m = by_month.loc[track]
        ax.plot(m.index, m["median_signed"], marker="o", color=colours[track], label=track)
    ax.axhline(0, color="#999", linewidth=1)
    ax.set_ylabel("Median signed error, %\n(above 0: we predicted too high)")
    ax.set_title("Bias by validation month")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "bias_by_month.png", dpi=150)
    plt.close(fig)
    print("\n".join(lines))
    print("Wrote reports/error_analysis.md, figures/error_by_type.png and figures/bias_by_month.png")


COMMANDS = {"validate": validate, "report": report, "predict": predict, "check": check,
            "baselines": baselines, "analysis": analysis}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in COMMANDS:
        sys.exit(__doc__)
    result = COMMANDS[sys.argv[1]]()
    sys.exit(1 if result is False else 0)
