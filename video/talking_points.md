# Video outline (5:00 maximum, judges stop at 5:00)

Say this in your own words, and don't read the report. Show the figures in `figures/` as slides or
screen-share this repository. Every number below comes from our own validation.

## 0:00 to 1:15: What we built
- Two questions: worth with no list price (off-market), and closing price of a listing (list-aware).
- A price is a moment in time, so we split it: market level when it sells + what this home is worth.
- Trees value the home using its features, remark phrases and earlier nearby sales.
- List-aware only predicts the correction to the asking price.

## 1:15 to 2:30: Why, and what didn't help
- Trees can't extrapolate, and the test is up to four months ahead. So we forecast the market ourselves.
- Season mattered: without it, off-market goes from about 7.3% to 8.4%.
- Remark phrases mattered: without them, about 8.2%.
- Didn't help: deliberately aging training comps (our own idea, dropped), and adding a trend.
- Refused: the third-party estimate, because 70% of its dates are after the sale.

## 2:30 to 3:45: How we validated
- All choices made on one block: train before June 2025, test June to September 2025.
- Then the required walk-forward, December 2025 to May 2026: **6.96% off-market, 3.96% list-aware**.
- That only looks one month ahead, so we also froze training for four months over three periods:
  **6.70 to 7.61% off-market, 4.17 to 4.46% list-aware**. Show `figures/error_by_horizon.png`.
- We expect about 7 to 8% and 4 to 4.5% on the hidden test.
- A comp only counts if sold before the listing date, and tests check this.
- A random split made the same simple model look 1.7 points better: 8.19% against 9.86%.

## 3:45 to 4:45: Where it fails
- Over $2.5M: 14.84% error off-market, undervalued by a median of 11.5%. Few comps, shrinks to typical.
  Show `figures/error_by_price_range.png`.
- Freehold (8.5 to 9%) against condos (5.9%): land and condition vary more. Show `figures/error_by_type.png`.
- Neighbourhoods from 3.39% to 13.01%: local trends our type-level index misses.
- Bidding wars: 12.18% in list-aware, nearly all undervaluation.
- 80% ranges cover 76 to 79%: slightly narrow, and we say so.

## 4:45 to 5:00: Close
- Don't rely on the off-market value above about $2.5M or for co-ops. Finish before 5:00.
