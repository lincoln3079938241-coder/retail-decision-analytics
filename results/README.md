# Result Dictionary

- `headline_metrics.json` — canonical rerun facts used by README and portfolio copy.
- `data_quality.json` — input scale and quality counters.
- `campaign_model_metrics.csv` — holdout metrics for compared classifiers.
- `campaign_targeting_deciles.csv` / `campaign_targeting_summary.json` — score-ranked targeting evidence.
- `campaign_response_by_type.csv` — aggregate response by campaign type.
- `pre_campaign_value_response.csv` / `pre_campaign_value_response_summary.json` — exposure-specific value tiers, observed response, and household-cluster bootstrap intervals.
- `pre_campaign_value_response_by_type.csv` — descriptive value-tier response by campaign description.
- `segment_profiles.csv` / `segment_silhouette_scores.csv` — aggregate cluster profiles and k diagnostics.
- `segment_category_profiles.csv` / `segment_category_top_overindex.csv` / `segment_category_summary.json` — retrospective segment/category mix, penetration, and index evidence.
- `department_association_rules.csv` — department-pair support, confidence, lift, and basket count.
- `forecast_model_summary.csv` / `forecast_metrics_by_commodity.csv` / `forecast_predictions.csv` — eight-week forecast benchmark.

Row-level household identifiers and raw transactions are intentionally excluded from the public result package.
