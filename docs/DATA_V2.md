# Fresh ESCI v2 data

`data/esci-v2` freezes a new, independent public ESCI subset for the next experiment:
2,000 training queries, 200 validation queries, 200 final holdout queries, and
82,427 US products. Query IDs and NFKC/casefold/whitespace-normalized query texts
are disjoint from every query in the previous 250/50/100 dataset and from the
other v2 partitions. Existing data and reports are preserved.

The source is the official [amazon-science/esci-data release](https://github.com/amazon-science/esci-data/tree/7916cdf6ab75a462e77f20ab40428a10923998d5)
at commit `7916cdf6ab75a462e77f20ab40428a10923998d5`, under Apache-2.0.
Both complete local source files are verified against their official Git LFS
size and SHA256 before freezing:

| Source | Bytes | SHA256 |
| --- | ---: | --- |
| examples parquet | 51,286,808 | `4a735b693b4a424a6fc67f5be6e4c811495c488bbf66d02a602d308b2744263a` |
| products parquet | 1,108,857,465 | `25124442d064d64b26f74082d6fa09438d679efc0c183cf28d19064a2b65a265` |

The previous frozen local ESCI subset contains only the same 400 old queries.
That subset is not included in this public source distribution. New examples are selected from the complete examples parquet,
not from that prepared subset. Source location, old-file hashes, script hash,
selection hash, and every generated file hash are recorded in the v2 manifest.

## Selection and sealed holdout

`scripts/prepare_data_v2.py` first reads only `query_id`, `query`, `product_id`,
`product_locale`, `small_version`, and `split`. It does not project `esci_label`.
Only US `small_version=1` records enter the task. It excludes old query IDs and
old normalized texts, normalized texts crossing official partitions, and queries
with fewer than 20 distinct original judged product IDs. For duplicated normalized
queries, the representative is the first SHA256-hashed official query ID.

The fixed seed is `20261010`. Ranking hashes are
`SHA256(JSON([seed, tag, value]))`, with compact JSON separators. Eligible
queries are ordered by the hash of normalized query text. The first 2,000
official-training queries become train, the next 200 become val, and the first
200 unused official-test queries become final test. Source availability after
these exclusions is 3,703 official-training and 1,707 official-test queries.
No relevance grade or model score participates in any of these choices.

The script writes `selection.json` **before** opening the relevance-label column.
Only then does it automatically transcribe all selected official human judgments.
It checks the copied product IDs against the already frozen selection, and checks
that selection bytes and all previous dataset files remain unchanged. Conflicting
selected annotations cause failure; they never cause resampling. Identical
repeated query-product rows are deduplicated by product ID.
All output is immutable: an existing nonempty destination is refused.

`test.jsonl` and `corpus_test.jsonl` contain sealed final labels. Preparation
prints only label-blind counts and fingerprints. No holdout grade distribution,
positive-query percentage, label examples, or model score is produced. During
training or hyperparameter selection, read the train and val JSONL files directly.
**Do not call `load_dataset(data/esci-v2)` during training**, because that helper
also reads test. Release the final test only once after architecture, training,
candidate retrieval, and ranking configurations have been committed and frozen.

## Files and evaluation scope

| Split | Queries | Original judged query-product pairs |
| --- | ---: | ---: |
| train | 2,000 | 76,626 |
| val | 200 | 7,746 |
| final test | 200 | 7,646 |

These pair counts are independent of relevance grades. `train.jsonl`, `val.jsonl`,
and `test.jsonl` use **all** original human-judged products as `candidate_ids`,
ordered by a fixed label-blind hash; `qrels` has exactly these keys. Candidate
counts vary by query and are at least 20. `corpus_train.jsonl`, `corpus_val.jsonl`,
and `corpus_test.jsonl` preserve the same full judgments with `candidate_ids=[]`
for whole-pool retrieval evaluation. Integer grades remain E=3, S=2, C=1, I=0;
ranking gain is `2**grade - 1`, and known positives are grades >=2.

All rankers must be evaluated on the same complete candidate set for a fair
closed-pool comparison. If previous pairwise weights are reused with v2 lexical
catalog statistics, call that baseline `transferred_pairwise`: the BM25 statistics
have changed, so it is not an identical frozen predictor from v1. Corpus Recall
measures **known-positive Recall** over the finite v2 product union. Cross-query
unjudged pairs remain unknown; the benchmark is not Amazon full-catalog retrieval.

The product union is selected from judged IDs without observing relevance grades.
Every selected query's original judged products are present, so this within-query
product coverage is 100%. Product text remains official title and brand;
description is deliberately empty, and no images, video, prices, categories,
or user histories are invented.

## Coverage and resource limits

Keeping the old 15,171-product pool cannot yield this larger benchmark without
strong coverage bias. Among all fresh deduplicated US small-version queries,
only 39 official-training and 23 official-test queries have at least 20 original
judged products in that old pool. It covers roughly 2.7% of their original
query-product pairs. Therefore v2 uses an independent 82,427-product union.
Only 2,441 v2 products overlap the old pool; 79,986 require new embeddings.

The new pool still has selection bias: it is the union of judged candidates for
queries with at least 20 judgments, rather than a random sample of Amazon's
catalog or natural production traffic. The manifest also records old-pool coverage
on the actual eligible-query population. Coverage should be discussed alongside
accuracy, and v2 scores should not be presented as directly comparable to v1.

At 384 float32 dimensions, 82,427 product vectors occupy 126,607,872 bytes
(about 121 MiB) before ANN overhead. Pairwise or cross-encoder training can start
with a deterministic hash sample of 500 training queries; all 2,000 remain
available for a later larger run. The sample must come only from train and be
declared before validation comparisons. The data script does not train or score
any model.

## Reproduce

PyArrow is required for source preparation; serving the frozen JSONL files does
not require it. Pass a local directory containing the two verified official
parquet files. The default is the project-relative `data/raw/esci` directory.
The `--old` argument must reference the matching previous frozen subset; its
default is `data/esci`, which also must be prepared separately.

```powershell
python scripts/prepare_data_v2.py --raw <official-parquet-directory>
python scripts/prepare_data_v2.py --raw <official-parquet-directory> --freeze
```

The first command only inspects label-blind metadata. The second creates the
default `data/esci-v2` directory. To reproduce an already frozen subset, use a
different empty output directory and compare selection and data fingerprints;
the script will not overwrite the original.
