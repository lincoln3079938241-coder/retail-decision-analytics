# Public Methodology Report

This case follows the sequence Business Problem → Data → EDA/quality → Household and Sales Analysis → Segmentation/Modeling → Insights → Recommendations.

RFM-style features are computed at household grain after excluding non-positive purchase lines. Skewed clustering fields are log-transformed and all fields standardized. k=2–6 inertia/silhouette diagnostics, seed stability, and a winsorization sensitivity run are exported. k=4 is retained for interpretability even though k=2 has higher silhouette.

Campaign features use only transactions before each campaign start and validation holds out whole households. The best model reached 0.826 ROC-AUC and 3.79× observed-response lift in the top score decile, but redemption is not incremental uplift. An additional pre-campaign value score combines rate-normalized sales, basket frequency, and inverse recency; its terciles show 7.78%, 12.57%, and 16.65% observed response. Confidence intervals use a 500-replicate household-cluster bootstrap.

Retrospective segment/category analysis reports within-segment sales share, basket penetration, and an index against all households. It is product-mix description, not a treatment effect. Department association rules use distinct department presence per basket; the leading pair has 1.63% support, asymmetric confidence of 8.43%/50.80%, and 2.62 lift. Forecasting uses a chronological eight-week holdout; Seasonal Naive wins at 9.30% WAPE.

Recommendations are hypotheses: prioritize higher-value households for efficient outreach tests, tailor category hypotheses to descriptive segment differences, evaluate cross-category placement with an A/B test, and require complex forecasts to beat Seasonal Naive across rolling origins. No promotion uplift is claimed.
