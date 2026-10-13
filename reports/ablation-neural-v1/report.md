# Neural ranker ablation (retrospective)

Same 500 training / 200 validation / 200 previously evaluated test queries; all judged candidates per query.
Every learned variant is retrained with the same 7 leaves, 200 rounds and training labels. No new model is selected.

| Variant | Features | Test NDCG@10 | P@10 | Exact@1 |
|---|---:|---:|---:|---:|
| bm25 | 1 | 0.710818 | 0.827500 | 0.685000 |
| cross_encoder | 1 | 0.739240 | 0.842000 | 0.725000 |
| full15 | 15 | 0.774758 | 0.860000 | 0.805000 |
| without_bm25 | 14 | 0.778079 | 0.867000 | 0.775000 |
| without_coverage | 13 | 0.779203 | 0.867000 | 0.830000 |
| without_char_trigram | 14 | 0.778735 | 0.862500 | 0.760000 |
| without_brand | 13 | 0.777489 | 0.862500 | 0.790000 |
| without_number | 13 | 0.780379 | 0.867000 | 0.795000 |
| without_phrase_bigram | 13 | 0.775986 | 0.860500 | 0.790000 |
| without_negation | 13 | 0.749019 | 0.847000 | 0.745000 |
| without_ce_all | 12 | 0.755267 | 0.853500 | 0.765000 |
| without_pool_stats | 13 | 0.774748 | 0.860000 | 0.780000 |
| without_ce_raw | 14 | 0.775836 | 0.864500 | 0.780000 |
| without_percentile | 14 | 0.779615 | 0.868500 | 0.790000 |
| without_zscore | 14 | 0.774161 | 0.862500 | 0.815000 |

Delta is left minus right. Positive means the left variant performed better.

| Contrast | Delta NDCG@10 | Paired 95% CI | Family-corrected CI |
|---|---:|---|---|
| add_ce_to_lexical: without_pool_stats − without_ce_all | +0.019480 | [+0.007320, +0.031867] | [+0.001889, +0.037762] |
| drop_bm25: full15 − without_bm25 | -0.003321 | [-0.009946, +0.003112] | [-0.013238, +0.006296] |
| drop_coverage: full15 − without_coverage | -0.004445 | [-0.010711, +0.001522] | [-0.013747, +0.004333] |
| drop_char_trigram: full15 − without_char_trigram | -0.003977 | [-0.011940, +0.003754] | [-0.015802, +0.007252] |
| drop_brand: full15 − without_brand | -0.002731 | [-0.007918, +0.002363] | [-0.010556, +0.004734] |
| drop_number: full15 − without_number | -0.005621 | [-0.011256, -0.000179] | [-0.014089, +0.002302] |
| drop_phrase_bigram: full15 − without_phrase_bigram | -0.001228 | [-0.006112, +0.003558] | [-0.008412, +0.005731] |
| drop_negation: full15 − without_negation | +0.025739 | [+0.011935, +0.040474] | [+0.006436, +0.048128] |
| drop_ce_all: full15 − without_ce_all | +0.019491 | [+0.007250, +0.032224] | [+0.001644, +0.038575] |
| drop_pool_stats: full15 − without_pool_stats | +0.000010 | [-0.009952, +0.010229] | [-0.014586, +0.015258] |
| drop_ce_raw: full15 − without_ce_raw | -0.001078 | [-0.008052, +0.005936] | [-0.011344, +0.009210] |
| drop_percentile: full15 − without_percentile | -0.004857 | [-0.012378, +0.002538] | [-0.016024, +0.005863] |
| drop_zscore: full15 − without_zscore | +0.000597 | [-0.007408, +0.008513] | [-0.011127, +0.012299] |

## Interpretation limits

- The historical test labels/results were already observed. This is retrospective diagnosis, not a new untouched-test gain claim.
- Fixed hyperparameters estimate removal effects under this training recipe; they do not find the best individually tuned alternative.
- Feature interactions and correlated substitutes mean these removal effects are not additive percentages or unique causal attribution.
- without_ce_all is lexical12; without_pool_stats is lexical+raw-CE13. without_ce_raw still retains two CE-derived statistics and is NOT a no-neural baseline.
- NDCG contrasts include Bonferroni-corrected paired bootstrap intervals; secondary-metric 95% intervals are exploratory and unadjusted.
- Bootstrap resamples queries, not training runs. It does not quantify training-split uncertainty or upstream pretraining overlap.
- An inactive group cannot demonstrate benefit on these queries. Nonzero/within-pool-varying feature counts are included for every split.
- All candidates in each pool are judged. This result does not measure all-catalog retrieval, new images, personalization, online CTR or business lift.
- Frozen historical inputs and feature caches are local licensed assets, excluded from Git. Checksums verify identity; source checkout alone does not reconstruct the historical subset.
