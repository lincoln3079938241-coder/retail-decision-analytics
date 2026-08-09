# Interview Guide — Retail Decision Analytics

## 30-second version

I rebuilt two retail course streams into one reproducible decision system over 2.6M transaction lines. A household-grouped campaign holdout achieved 0.826 ROC-AUC and 3.79× top-decile observed-response lift. Pre-campaign value tiers showed 7.78% to 16.65% observed response, with household-cluster bootstrap intervals, and segment/category indices linked customer profiles to product hypotheses. I keep every result observational rather than causal.

## 2-minute version

The case follows customer → value/behavior → campaign/product response → decision. I audited transaction, basket, and household grains; built RFM-style profiles; held out households for campaign modeling; and created activity-normalized value tiers only from information available before each campaign. Higher-tier exposures responded at 16.65% versus 7.78% for the lower tier, but I call this descriptive because campaign assignment was not randomized. Retrospective category indices add product hypotheses, while the basket and forecast modules retain their original evidence boundaries.

## 18 questions

### 1. Why this business scope?

It connects three decisions—targeting, merchandising, and planning—using one consistent transaction evidence trail.

### 2. What is the transaction grain?

One product line within a basket; customer measures aggregate distinct baskets at household grain.

### 3. How are RFM fields defined?

Recency is dataset-end day minus last positive purchase day, frequency is distinct positive baskets, and monetary is positive sales value.

### 4. Why exclude non-positive lines?

Returns/adjustments should not count as positive purchase behavior; their counts remain in the quality audit.

### 5. Why K-means?

It offers an explainable distance-based vocabulary over continuous scaled behavior, not a claim of natural customer classes.

### 6. Why k=4 when k=2 scores better?

Four profiles support differentiated hypotheses, while I explicitly disclose 0.171 versus 0.271 silhouette.

### 7. Are clusters stable?

Across ten seeds the minimum ARI is 0.962; winsorization sensitivity produces ARI 0.943.

### 8. How did you prevent campaign leakage?

Every feature is computed before campaign start and the test split contains entirely unseen households.

### 9. What does 3.79× mean?

Top-decile observed response is 45.24% versus an 11.93% test baseline; it is ranking lift, not incremental uplift.

### 10. Why report PR-AUC?

It focuses on positive-class retrieval when response is less common than non-response.

### 11. How is pre-campaign value defined?

For each exposure I rate-normalize prior sales and baskets by elapsed days, combine their standardized logs with inverse recency, and divide the resulting score into terciles.

### 12. Why not use lifetime customer value?

Lifetime values would include behavior after the campaign and create horizon leakage. Every value-tier input is measured before that exposure starts.

### 13. What did the value tiers show?

Observed response was 7.78%, 12.57%, and 16.65% from lower to higher value; the higher-tier 95% household-bootstrap interval was 14.50%–18.97%.

### 14. Is that promotion uplift?

No. It is observed response profiling. Campaign mix and selection may differ, and there is no randomized no-contact control.

### 15. What does the category index measure?

Within-segment department sales share divided by the same department's share across all households; 1.0 means portfolio average.

### 16. Are category differences causal?

No. Segments use retrospective behavior, so the index supports assortment hypotheses, not incremental sales claims.

### 17. How are basket rules calculated?

Distinct departments per basket feed support, directional confidence, and lift relative to the consequent base rate.

### 18. Why not call the top rule a sales opportunity?

MEAT→seafood confidence is only 8.43%; association needs an experiment before business impact claims.

### Appendix: How is forecast leakage prevented?

The final eight weeks are never used to fit models; recursive features contain only prior values or prior predictions.

### Appendix: Why did Seasonal Naive win?

The short weekly series has repeat seasonality and the complex model did not consistently beat that structure.

### Appendix: What would production require?

Randomized incrementality tests, margin-aware thresholds, rolling forecasts, and ongoing calibration/drift monitoring.
