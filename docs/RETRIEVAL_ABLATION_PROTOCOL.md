# Retrieval ablation protocol

This retrospective study explains the previously published public CPU baseline.
The test set has already been seen; no new model or hyperparameter is selected.
Use a new output directory for each run. `prepare` fingerprints inputs, source,
the validation-selected ranker and a separately built title-only index. `evaluate`
checks fingerprints before loading test labels, takes an exclusive run lock, and
requires exact full-baseline ranking and metric reproduction.

```powershell
python scripts/ablate_retrieval.py prepare --data data/esci-public --reference outputs/public-cpu-v1 --output outputs/ablation-retrieval-v1
python scripts/ablate_retrieval.py evaluate --data data/esci-public --reference outputs/public-cpu-v1 --output outputs/ablation-retrieval-v1
```

Run the public data preparation and baseline first (see README). Python standard
library and SQLite FTS5 suffice; no neural dependency or new training is required.
Only aggregate `protocol.json` and `report.json` belong in a public report. Indexes,
data and reference per-query rankings remain local.

| Variant | Intervention | What stays fixed |
|---|---|---|
| full_k100 | Original title+brand FTS, planner, 100 candidates | Historical reference |
| no_brand_index_k100 | Blank brand field in a rebuilt FTS index | Restore original metadata before filtering/reranking |
| no_exclusion_filter_k100 | Disable exclusion filtering and its 3x oversampling | Positive terms, original index and ranker |
| raw_tokens_k100 | Remove stopword, negation and minus processing together | Lexical tokenizer, index and ranker |
| full_k50 | 50 candidates | Original planner/index/ranker |
| full_k200 | 200 candidates | Original planner/index/ranker |

Brand removal changes field TF/DF/document-length statistics as well as matches;
it is not merely setting a scoring coefficient to zero. Raw-token mode is a
bundled parser intervention. Changing candidates is intentionally distinct from
the identical-pool ranking ablation. The same trained pairwise weights and original
product metadata rank every returned pool.

Report known-positive Recall@10, Recall@100, whole-candidate-pool recall, judged
fraction@10, candidate count, and query-paired bootstrap intervals. Positive
`full_minus_variant` means the full baseline is better. Recall@100 for pool50 can
only use its 50 returned candidates; pool200's candidate recall uses all 200.
Unjudged pairs are unknown, not irrelevant. Also report the preplanned subset of
queries whose full parser produces exclusions, comparing full with no-exclusion.

Intervals use 2,000 resamples, seed 20261010, and are exploratory pointwise 95%
intervals with no multiple-comparison adjustment. They describe query sampling,
not training-seed robustness. Effects are conditional on other active components
and cannot be added into a total gain.

Timing excludes parser construction, and measures FTS SQL plus filtering, followed
by reranking. Model/catalog are resident; result caching is off and query feature
caching cleared per variant. Method order rotates by query. OS caches and other
machine load are uncontrolled: these timings do not establish speedups.
