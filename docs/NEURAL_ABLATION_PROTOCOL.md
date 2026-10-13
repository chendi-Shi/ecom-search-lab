# Frozen neural ranker: retrospective ablation

This experiment explains the historical 82,427-product neural experiment. It is
separate from the smaller downloadable public benchmark. Do not compare absolute
scores across those two datasets.

The historical 200-query test results have already been seen. This new experiment
is a **retrospective diagnosis**, not a fresh holdout or a new performance claim.
No variant selected by these results replaces the deployed model.

## Controlled comparison

All trained variants use exactly the historical SHA-selected 500 training queries
(18,996 judged pairs), 200 validation queries and 200 test queries. Every method
ranks the same complete judged candidate pool in the same original order. Training
uses LightGBM with 7 leaves, 200 rounds, learning rate 0.05, minimum leaf count 40,
L2 1 and seed 20261010. The complete parameters and input hashes are frozen in
`protocol.json` before execution. Hyperparameters are not retuned per variant.

The full model contains 12 lexical features, the raw fixed MiniLM CE logit, and
two statistics computed over each complete candidate pool: tied percentile and
population z-score. All CE scores use the original dynamic-int8, oneDNN, 32-pair
batch, 128-token configuration. Removing columns does not recompute or reorder
CE batches.

| Variant | Question answered |
|---|---|
| BM25, raw CE | Reference ranking scores; neither is refit in this experiment |
| `without_ce_all` (12 columns) | How does lexical-only LambdaMART perform? |
| `without_pool_stats` (13 columns) | What does adding the raw CE logit to lexical features change? |
| `full15` | Does exact retraining reproduce the historical model? |
| Full minus percentile / minus z-score | Is either pool statistic useful conditional on the other? |
| Full minus raw logit | Does the absolute logit add information beyond relative CE statistics? CE is still present. |
| Full minus BM25 / coverage / character trigram / brand / number / phrase-bigram / negation | Conditional effect of each lexical feature group under the same training recipe |

Every removal variant is retrained from scratch, instead of zeroing features at
inference on the already-trained model. Correlated features can substitute for
one another; a removal can improve a score. These effects are not additive
percentages and are not unique causal attribution.

## Integrity and inference

`freeze` checks the original selection/model/data/code hashes. `run` rejects any
change after freeze. Existing feature arrays are reusable only if their byte
checksum, ordered record hash, catalog hash, feature schema, source hashes and CE
runtime fingerprint all match. Relative features are recomputed from every pool's
raw logits and compared exactly. Otherwise pass no cache and recompute scores
with the verified local CE assets.

Test labels/features are opened only after all prescribed models are fitted and
their file hashes recorded. This ordering prevents new test-dependent model
choices; it does not undo the historical exposure of the test results. Full-model
validation/test predictions must match the historical exported tree predictions
within 1e-12 relative/absolute tolerance, and full-model aggregate test metrics
must match exactly. An additional final integrity check precedes report writing.

Primary metric is macro graded NDCG@10, gain `[0,1,3,7]`. P@10 and Recall@10 use
E/S relevance; Exact@1 uses E. Score ties use product ID ascending. Paired
bootstrap uses queries as units, 50,000 resamples, seed 20261010. All 13 planned
NDCG contrasts include both ordinary 95% and Bonferroni family-corrected intervals
(per-contrast confidence 99.6154%). Bootstrap coverage is approximate. Secondary
metric intervals are exploratory and unadjusted. Training-set uncertainty is not
included.

The report counts nonzero and within-query-varying features for each group. A
feature that does not vary across any candidates cannot affect ranking in that
sample. A zero-effect result with little activity is an evidence gap, not proof
that the underlying search requirement is solved or unimportant.

## Run with matching historical assets

The repository does not redistribute raw ESCI data, model weights or feature
caches. The exact historical subset is identified by the report hashes; preparing
a different public subset is not reproduction of these historical numbers.
Use an environment matching `requirements-accuracy.txt`, plus the exact historical
LightGBM version recorded in the model. Commands use portable placeholder paths:

```powershell
python scripts/ablate_neural.py freeze --data data/esci-v2 --reference outputs/accuracy-v1 --ce-model outputs/cross-encoder-assets/ms-marco-MiniLM-L6-v2 --feature-cache outputs/accuracy-v1 --output outputs/ablation-neural-v1
python scripts/ablate_neural.py run --data data/esci-v2 --reference outputs/accuracy-v1 --ce-model outputs/cross-encoder-assets/ms-marco-MiniLM-L6-v2 --feature-cache outputs/accuracy-v1 --output outputs/ablation-neural-v1
```

Only `protocol.json`, aggregate `report.json` and rendered `report.md` belong in
the public reports directory. `per_query_local.json`, boosters, score caches and
data stay under ignored `outputs/` and `data/`. The experiment measures fully
judged candidate ranking; it does not establish all-catalog recall, image
relevance, personalization, conversion or production latency.
