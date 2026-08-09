# Retail Decision Analytics

A reproducible decision case over 2.6M transaction lines spanning customer profiles, campaign response ranking, basket affinities, and short-horizon forecasting. A household-grouped holdout produced 0.826 ROC-AUC and 3.79× top-decile observed-response lift. The strongest category pair had 2.62 lift but only 8.43% MEAT→seafood confidence, so it is framed as a test hypothesis. Seasonal Naive led the eight-week forecast benchmark at 9.30% WAPE. The four-profile solution is usable but weakly separated (silhouette 0.171 versus 0.271 for k=2).
