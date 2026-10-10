# Published accuracy evidence

These JSON files are byte-for-byte copies of aggregate artifacts from the frozen
v0.3 CPU experiment. They describe that historical run, not newly generated
accuracy results for this source-only release.

| File | Contents |
| --- | --- |
| `report.json` | Final 200-query fully judged candidate ranking metrics and paired uncertainty |
| `end_to_end_report.json` | Fixed 50-query, common top100 retrieval-pool comparison |
| `validation.json` | Final validation summary |
| `protocol.json` | Frozen configuration and artifact fingerprints |
| `selection.json` | Model selection metadata |
| `validation_feature_probe.json` | Development feature comparisons on validation only |
| `verification.json` | Recorded 91-test suite result from the full local experiment environment |

Raw datasets, judgments, model weights, per-query predictions, feature caches,
indexes and local profile databases are not included. Historical paths in these
reports identify original artifacts; they do not imply those artifacts are part
of this public repository. Exact replay requires the matching inputs and runtime.

The public snapshot additionally contains a synthetic functional demo and portable
launch/data-preparation paths. Those packaging edits were made after the historical
experiment and are not new evidence of ranking quality. The frozen core algorithm
is retained without changes.

See [results](../../docs/ACCURACY_RESULTS.md),
[protocol](../../docs/ACCURACY_PROTOCOL.md), and
[source distribution instructions](../../README.md).
