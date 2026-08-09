# Interview Guide — Retail Decision Analytics

## 30-second version

I rebuilt two retail course streams into one reproducible decision system over 2.6M transaction lines. A household-grouped campaign holdout achieved 0.826 ROC-AUC and 3.79× top-decile observed-response lift; a department pair reached 2.62 lift; and Seasonal Naive won the forecast benchmark at 9.30% WAPE. I keep cluster, association, and causal limitations explicit.

## 2-minute version

The case starts with three operating decisions: who to contact, what to test together, and what demand baseline to trust. I audited the transaction/basket/household grains, built RFM-style profiles, held out entire households for campaign modeling, calculated directional basket rules, and reserved the final eight weeks for forecasting. The analytical lesson is restraint: k=4 is more actionable but has weak silhouette; campaign ranking is not incremental uplift; MEAT→seafood confidence is low despite high lift; and a simple seasonal baseline beat Random Forest.

## 15 questions

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

### 11. How are basket rules calculated?

Distinct departments per basket feed support, directional confidence, and lift relative to the consequent base rate.

### 12. Why not call the top rule a sales opportunity?

MEAT→seafood confidence is only 8.43%; association needs an experiment before business impact claims.

### 13. How is forecast leakage prevented?

The final eight weeks are never used to fit models; recursive features contain only prior values or prior predictions.

### 14. Why did Seasonal Naive win?

The short weekly series has repeat seasonality and the complex model did not consistently beat that structure.

### 15. What would production require?

Randomized incrementality tests, margin-aware thresholds, rolling forecasts, and ongoing calibration/drift monitoring.
