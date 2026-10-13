# Retrospective feature ablation

Already-reported public test; fixed hyperparameters; all variants reported. Positive delta means full minus removed.

| Removed group | Retrained NDCG@10 | Full-minus-removed | Query bootstrap 95% CI | Frozen-input sensitivity delta | Varying test queries |
|---|---:|---:|---|---:|---:|
| title_bm25 | 0.748146 | +0.001282 | [-0.001512, +0.004506] | +0.000349 | 96 |
| coverage | 0.750663 | -0.001235 | [-0.009575, +0.006523] | +0.002551 | 94 |
| char_trigram | 0.748043 | +0.001385 | [-0.014231, +0.016190] | +0.001568 | 100 |
| brand | 0.749976 | -0.000548 | [-0.006822, +0.004819] | -0.000358 | 21 |
| number | 0.747357 | +0.002071 | [-0.003079, +0.007710] | +0.003809 | 16 |
| phrase | 0.746529 | +0.002899 | [-0.001378, +0.007480] | +0.003444 | 81 |
| negation | 0.730169 | +0.019259 | [+0.007801, +0.032277] | +0.019542 | 26 |

Full reference NDCG@10: 0.749428.

These effects overlap and cannot be summed. CIs are exploratory and unadjusted across groups/metrics.
Removing a feature group is not removing shared query parsing or the entire feature engineering subsystem.
Full-catalog known-positive recall, activity denominators, reproduction checks and measured costs are in report.json.
The description column may be inactive in title/brand-only input; a zero effect then says nothing about richer descriptions.
