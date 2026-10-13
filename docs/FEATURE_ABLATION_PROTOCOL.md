# Feature ablation: what each component contributes

This protocol diagnoses the published **43,222-product public CPU model**. Its
100-query test set has already been reported. Results are **retrospective** and
cannot establish new unseen-test generalization. They are not the historical
82,427-product cross-encoder/LambdaMART experiment.

## Fixed experiment

Use the same 1,000 training queries, catalog, model initialization, pair sampling,
seed 42, 30 epochs, learning rate 0.05, L2 0.001, query normalization, and optimizer
as the already selected public model. No variant is selected or retuned using
these results. `PairwiseRanker.fit` itself is unchanged.

The result measures each group under this fixed training recipe. It does not
estimate the best achievable accuracy of every smaller model after separately
tuning its hyperparameters; that question needs training/validation selection
and a new independent test. Masking also retains the full feature-computation
path, so these timings cannot establish serving-speed savings from removal.

| Group | Original feature columns | Removal |
|---|---|---|
| title_bm25 | 0 | Title BM25 scalar |
| coverage | 1, 2 | Title and description token coverage |
| char_trigram | 3 | Character trigram similarity |
| brand | 4, 5 | Brand match and conflict |
| number | 6, 7 | Numeric value/unit match and conflict |
| phrase | 8, 9 | Bigram and phrase match |
| negation | 10, 11 | Literal negation conflict and satisfaction |

`phrase` here means ranking bigram/exact-positive-phrase features, not the query
planner's explicit quoted-phrase syntax.

Each leave-one-group-out variant is **retrained from the same initialization**.
Removed input columns are zero at both training and evaluation; the 12-column
schema and remaining features are identical. Shared preprocessing stays enabled:
for example, dropping negation columns does not remove negation-aware tokenization
inside BM25/coverage or the query planner. This is feature-column ablation, not
an ablation of every use of negation in the system.

The full 12-feature model is also retrained. Before any test scoring, its weights
and predictions on every cached training candidate must reproduce the original
model within absolute tolerance 1e-10. Evaluation also verifies full-model test
scores and every previously published fixed-pool/corpus macro metric.

A separate **inference sensitivity** experiment zeros each group in the original
frozen model without retraining. It answers whether that fitted model relies on
the group. Retraining asks how well other groups can compensate. These are
different questions, and their answers must not be combined into one gain.

## Identical candidates and uncertainty

All 15 conditions (full + 7 retrained removals + 7 sensitivity removals) use the
same fully judged candidates and one shared real retrieval pool per query. Each
real pool is retrieved once from the unchanged reference SQLite index with
`k=100`, then its IDs and fingerprint are saved. Ties use product ID ordering.
All conditions see identical candidate sets, including empty sets.

Report fixed-pool NDCG@10, Exact@1, E/S P@10 and known-positive Recall@10.
Full-catalog judgments are incomplete, so corpus results use known-positive
recall and judgment coverage; no full-catalog precision or NDCG is claimed.
Reranking cannot change Recall@100 when the 100-item candidate set is fixed.

Every comparison is **full minus removed**; positive means removing the group
hurts. Use paired query bootstrap with 2,000 resamples and seed 20261013. The
95% percentile intervals are exploratory and **not adjusted for the seven group
comparisons, two diagnostic modes and multiple metrics**. An interval crossing
zero does not establish absence of a useful effect. No model is promoted based
on this reused test. Correlated features and compensation mean deltas are
**non-additive**, not causal attribution or shares of the total BM25 improvement.

## Support and cost

For every feature and group, count candidate pairs with nonzero values, queries
with a nonzero value, and queries where at least one feature varies across
candidates. Constant-per-query features cannot change rankings directly.
Report an additional descriptive NDCG comparison on queries with varying group
values, retaining its query denominator. This is a diagnostic subset, not an
independent confirmatory test.

This catalog contains title and brand but empty descriptions. The description
coverage feature therefore has no observed opportunity to help; these data
cannot measure the value of real product descriptions. Literal negation and
brand heuristics can misread proper names, so activity alone is not correctness.

Training vectors are label-free and computed **once**, then reused across all
fits. A cache miss fails. The same original fitting method preserves pair order
and sampling. Record shared extraction seconds, separate per-variant fit seconds,
pair counts, total training cost, Python/platform information, and evaluation
time. Those serial CPU measurements are not online throughput claims.

## Reproduce

First prepare the real public dataset and complete the existing public benchmark
as documented in `PUBLIC_BENCHMARK.md`. Then from the repository root:

```powershell
python scripts/ablate_features.py prepare --data data/esci-public --reference outputs/public-cpu-v1 --output outputs/ablation-features-v1
python scripts/ablate_features.py train --data data/esci-public --reference outputs/public-cpu-v1 --output outputs/ablation-features-v1
python scripts/ablate_features.py evaluate --data data/esci-public --reference outputs/public-cpu-v1 --output outputs/ablation-features-v1
```

`run` executes the same three stages. Only the Python standard library is needed.
Use a new directory for a new declared experiment; existing stages are exclusive
and are not overwritten. `prepare` fingerprints config, source and all dataset
bytes without JSON-decoding test labels. `train` parses only catalog/train and
freezes models before `evaluate` opens test labels in this run. This ordering
does not erase prior exposure to the historical test set.

`protocol.json`, `models_frozen.json`, `training.json`, and aggregate
`report.json`/`report.md` are published evidence. Trained `models.json`, cache
vectors, candidate IDs and per-query results stay local under ignored `outputs/`.
The publication retains their hashes for provenance without distributing model
weights or underlying data. `attempts.json` records the completed run and any
interrupted attempt; the first interrupted training attempt never entered test
evaluation, and the second attempt reran every fit using the identical recipe.
