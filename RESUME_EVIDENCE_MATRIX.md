# Resume Evidence Matrix — Retail Wave 1

| Claim | Evidence | Status | Resume boundary |
|---|---|---|---|
| RFM customer analysis | `segment_profiles.csv`, pipeline definitions | GREEN | Descriptive customer features |
| Clustering | k=2–6 silhouette; k=4 seed ARI ≥0.962 | GREEN | k=4 chosen for actionability, not natural truth |
| Campaign response | Group holdout AUC 0.826; top-decile lift 3.79× | GREEN | Observed response ranking, not uplift |
| Customer value profiling | Pre-campaign terciles; 7.78%/12.57%/16.65%; household bootstrap | GREEN | Observational, exposure-specific tiers |
| Segment/category differences | Segment sales-share index and penetration; minimum-support rule | GREEN | Retrospective product mix, not incremental sales |
| Basket analysis | support/confidence/lift; top lift 2.625 | GREEN | Association and test hypothesis only |
| Forecasting | 8-week chronological holdout; Seasonal Naive WAPE 9.304% | GREEN | One forecast origin, five commodities |
| Promotion uplift / causal effect | No randomized untreated counterfactual | RED | Do not claim |
| PCA / DID / survival | Not part of Wave 1 | YELLOW | Excluded from current resume |
| Price / inventory optimization | No true cost or on-hand inventory | RED | Do not claim |

Every technical term recommended in `RESUME_BULLETS_V3.md` is GREEN.
