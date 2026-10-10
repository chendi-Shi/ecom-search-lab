# Public CPU benchmark

Catalog: 43,222; holdout queries: 100.

| Fixed fully judged candidates | NDCG@10 | Exact@1 | P@10 (E/S) |
|---|---:|---:|---:|
| title_bm25 | 0.713730 | 0.650000 | 0.819000 |
| learned_pairwise | 0.749428 | 0.710000 | 0.835000 |

| Full catalog | Known-positive Recall@10 | Known-positive Recall@100 |
|---|---:|---:|
| fts_bm25 | 0.193530 | 0.572474 |
| fts_pairwise | 0.190703 | 0.572474 |

Full-catalog judgments are incomplete. Recall is relative to known E/S labels; unjudged results are unknown.
Paired confidence intervals, denominators, diagnostic slices and measured serial latency are in `report.json`.
The model and data were frozen before test parsing. A second final run in this directory is rejected.
These are newly measured public-run results, not reproduction of the historical joint-encoder scores.
