# Public Methodology Report

This case follows the sequence Business Problem → Data → EDA/quality → Household and Sales Analysis → Segmentation/Modeling → Insights → Recommendations.

RFM-style features are computed at household grain after excluding non-positive purchase lines. Skewed clustering fields are log-transformed and all fields standardized. k=2–6 inertia/silhouette diagnostics, seed stability, and a winsorization sensitivity run are exported. k=4 is retained for interpretability even though k=2 has higher silhouette.

Campaign features use only transactions before each campaign start and validation holds out whole households. The best model reached 0.826 ROC-AUC and 3.79× observed-response lift in the top score decile, but redemption is not incremental uplift. Department association rules use distinct department presence per basket; the leading pair has 1.63% support, asymmetric confidence of 8.43%/50.80%, and 2.62 lift. Forecasting uses a chronological eight-week holdout; Seasonal Naive wins at 9.30% WAPE.

Recommendations are hypotheses: protect high-value availability, test margin-aware offers for promotion-oriented households, evaluate cross-category placement with an A/B test, and require complex forecasts to beat Seasonal Naive across rolling origins.
