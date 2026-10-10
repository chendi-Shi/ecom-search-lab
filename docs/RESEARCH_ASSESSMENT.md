# What this project demonstrates, and what the evidence does not support

The public v0.4 path is a real-data CPU research baseline with an independently
repeatable preparation → training → validation selection → frozen evaluation →
serving workflow. It is not a production TikTok Shop search system.

## Findings from the newly executed protocol

On 100 held-out queries (3,741 human judgments), closed-candidate NDCG@10 rises
from 0.713730 to 0.749428. The paired query-bootstrap 95% interval for the
increment is [0.013043, 0.059581]. Exact@1 rises from 0.65 to 0.71, but its
increment interval [-0.02, 0.15] crosses zero.

On actual top100 retrieval candidates from the 43,222-product union, known-positive
Recall@10 changes from 0.193530 to 0.190703, interval [-0.015182, 0.009743]. This
does **not** support promoting the learned ranker as an end-to-end improvement.
Recall@100 is 0.572474 for both methods: reranking cannot recover an item absent
from the candidate pool. This is a finite judged-product union, not the full
Amazon catalog; cross-query unjudged pairs remain unknown.

The 38-query `long_ge5_tokens` diagnostic slice shows closed-pool NDCG improvement
but a corpus Recall@10 difference of -0.026368, interval [-0.047016, -0.006673].
Length is only an operational slice, not a human label of semantic complexity.
Slice intervals are exploratory and are not corrected for multiple comparisons.
The result is a warning about transfer from judged candidate pools to real
retrieval pools, not proof of a causal failure mechanism.

## Actionable research hypotheses

1. Investigate recall coverage before increasing reranker complexity. Candidate
   coverage limits what any reranker can achieve. On a fresh validation protocol,
   compare lexical, learned dense and fused candidates at fixed memory and latency.
2. Compare training on the judged pool with training on retrieval-generated
   candidates whose labels are actually known. Never assign negative labels to
   unjudged products just to fill the training set.
3. Check long queries, literal brands, numbers and negation separately. Current
   features do not establish unit conversion, attribute compliance, intent
   disambiguation or multilingual understanding. Inspect failure cases before
   designing new features; reserve new test data for the next final comparison.
4. Measure candidate-pool sizes and neural reranking costs on validation before
   freezing a quality/latency choice. The current p50/p95 is serial CPU timing,
   with model/catalog resident, no application result cache, and uncontrolled OS cache.

These are next experiments, not implemented gains. The current test results are
retained without retuning the model after evaluation.

## Mapping to the internship requirements

| JD area | Concrete evidence here | Remaining gap |
| --- | --- | --- |
| ML/DL and code | Supervised pairwise ranking, historical frozen CE + LambdaMART, tests and model/data binding | A new deep reranker is not trained by the stdlib baseline |
| Query understanding | Joint-encoder module, literal brand/number/negation features and failure slices | No proven robust complex/multilingual query understanding |
| Retrieval scale | Real finite catalogs, lexical retrieval, existing FAISS/HNSW/int8/mmap modules and resource reports | No billion-scale distributed indexing experiment |
| Multimodal and personalization | Separate real-data TinyCLIP and Beauty experiments, with documented failures | Not integrated and validated as a multimodal personalized ecommerce ranker |
| Research rigor | Real labels, reproducible sampling, train/val/test isolation, freeze hashes, paired intervals and negative results | No top-conference paper or online A/B evidence |

This can support an internship portfolio and an honest technical discussion.
It does not demonstrate all JD goals at production scale or establish online
conversion, satisfaction or safety benefits.
