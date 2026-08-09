# Data Note

The project uses dunnhumby’s **The Complete Journey**, described by the provider as a representation of two years of household transactions for 2,500 frequent-shopper households plus direct-marketing history. Provider page: <https://www.dunnhumby.com/source-files/>. Review the provider’s current terms before use.

The repository does not include raw data. The local audit found 2,595,732 transaction lines, 276,484 baskets, 92,339 transacted products, 7,208 campaign exposures, 2,318 redemptions, and zero exact duplicate transaction rows.

Positive-purchase analytics exclude quantity ≤ 0 or sales ≤ 0 but retain counts of those rows in `results/data_quality.json`. Sales means `sales_value` from qualifying lines; it is not profit, margin, or net revenue. Household keys are anonymous grouping identifiers and are excluded from the public result package.

**Provenance:** original CA/SPMF work supplied the business context; the public pipeline, validation design, stability diagnostics, figures, and wording were independently rerun/refactored. Code is MIT-licensed. Data remain governed by the source provider’s terms.
