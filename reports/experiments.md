# What we tried

Train on sales before 2025-06-01, predict sales from then until 2025-10-01 (8,052 homes).
Errors in percent. Variants change one thing from the draft design.

| Variant | Off-market MdAPE | Off-market MAPE | List-aware MdAPE | List-aware MAPE |
|---|---|---|---|---|
| Draft design (training comps aged 0-120 days) | 7.35 | 10.0 | 4.24 | 5.28 |
| No seasonal term in the forecast | 8.43 | 11.39 | 4.48 | 5.58 |
| Half of the recent trend added to the forecast | 7.47 | 10.0 | 4.21 | 5.26 |
| FINAL design: draft with training comps not aged | 7.29 | 9.96 | 4.17 | 5.25 |
| No remark flags | 8.21 | 11.11 | 4.39 | 5.44 |
| Squared-error loss | 7.24 | 9.88 | 4.36 | 5.53 |
| Control: one model, date as a feature, no comps (off-market) | 9.86 | 12.9 |  |  |
| Warning: same control scored on a RANDOM 20% split (off-market) | 8.19 | 10.65 |  |  |
