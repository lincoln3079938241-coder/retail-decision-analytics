# Retail Method Audit

| Check | Result |
|---|---|
| Transaction duplicates | 0 exact full-row and 0 duplicate commercial-line signatures |
| Household/basket grain | Distinct `household_key` for profiles; distinct `basket_id` for frequency and basket rules |
| RFM definitions | Recency from final valid day; frequency = distinct positive baskets; monetary = positive `sales_value` sum |
| Scaling/outliers | `log1p` on skewed fields, then StandardScaler; 1st/99th sensitivity ARI 0.943 |
| K selection | k=2 best silhouette 0.271; k=4 chosen for four descriptive profiles, silhouette 0.171 |
| Seed stability | Minimum ARI 0.962 across 10 alternative random seeds |
| Association rules | Distinct departments per basket; support/confidence/lift formulas independently checked |
| Campaign leakage | Only pre-campaign transactions; full-household group holdout |
| Forecast leakage | Final eight weeks held out chronologically; lag/rolling features use prior history only |

Release conclusion: technically explainable and publication-ready if the weak k=4 separation and non-causal response framing remain visible.
