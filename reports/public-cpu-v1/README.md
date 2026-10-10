# Independent public CPU experiment

This folder records the newly runnable public protocol, not the historical
CrossEncoder/LambdaMART experiment in `../accuracy/`.

Default preparation: 43,222 real ESCI products; 1,000/100/100 train/validation/test
queries; 38,239/3,985/3,741 human judgments. Official files are pinned and hashed.
Query identities and normalized texts are disjoint within this protocol;
overlap with earlier unrelated project experiments is not excluded.

`report.json` and `report.md` are the final aggregate results. `selection.json`
records the fixed validation search. `freeze.json` records source/data/artifact
fingerprints before the final labels were opened. `verification.json` records
local software tests, not ranking accuracy.

Raw data, weights, indexes and per-query diagnostic records stay in ignored
`data/` and `outputs/`. See [the complete runnable protocol](../../docs/PUBLIC_BENCHMARK.md).
