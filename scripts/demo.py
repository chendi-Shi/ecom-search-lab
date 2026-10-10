"""Run the real lexical search pipeline on small, hand-written synthetic fixtures.

Usage: python scripts/demo.py
Requires Python 3.10+ with SQLite FTS5; no packages or downloads are needed.
This smoke demo does not train or evaluate the ESCI accuracy model.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ecom_search.cli import SearchEngine
from ecom_search.index import ProductIndex, build_index


CATALOG = [
    {"id": "synthetic-001", "title": "DemoTrail lightweight running shoes", "brand": "DemoTrail", "description": "Hand-written synthetic demo product."},
    {"id": "synthetic-002", "title": "DemoPeak cushioned running shoes", "brand": "DemoPeak", "description": "Hand-written synthetic demo product."},
    {"id": "synthetic-003", "title": "DemoTrail waterproof hiking boots", "brand": "DemoTrail", "description": "Hand-written synthetic demo product."},
    {"id": "synthetic-004", "title": "DemoFlow insulated bottle 750 ml", "brand": "DemoFlow", "description": "Hand-written synthetic demo product."},
    {"id": "synthetic-005", "title": "DemoFlow insulated bottle 500 ml", "brand": "DemoFlow", "description": "Hand-written synthetic demo product."},
    {"id": "synthetic-006", "title": "DemoDesk adjustable laptop stand", "brand": "DemoDesk", "description": "Hand-written synthetic demo product."},
]
QUERIES = [
    "DemoTrail running shoes",
    "running shoes without demopeak",
    "insulated bottle 750 ml",
]
SCOPE = (
    "Synthetic functional smoke demo: six hand-written fictional products; "
    "no ESCI data, trained model, relevance judgments, or accuracy metrics."
)


def check(condition: bool, message: str) -> None:
    """Keep smoke checks active even when Python runs with -O."""
    if not condition:
        raise RuntimeError(f"Demo smoke check failed: {message}")


def run(output_dir: Path) -> dict:
    # Always allocate a new run directory; never overwrite a real catalog/index.
    directory = output_dir / uuid.uuid4().hex[:12]
    directory.mkdir(parents=True, exist_ok=False)
    index_path = directory / "synthetic-products.sqlite"
    build_index(CATALOG, index_path)
    engine = SearchEngine(index_path)
    try:
        results = []
        with ProductIndex(index_path) as index:
            for query in QUERIES:
                result = engine.search(query, k=len(CATALOG), pool=len(CATALOG))
                expected_ids = [p["id"] for p in index.search(query, k=len(CATALOG))]
                actual_ids = [p["id"] for p in result["products"]]
                check(actual_ids == expected_ids, "engine must preserve lexical index order")
                check(len(actual_ids) == len(set(actual_ids)), "result IDs must be unique")
                check(all(math.isfinite(p["score"]) for p in result["products"]), "scores must be finite")
                # Select fields explicitly: the ESCI-specific CLI data label does
                # not describe these fixtures and must not appear in this report.
                results.append({
                    "query": query,
                    "retrieval": result["retrieval"],
                    "ranking": result["ranking"],
                    "excluded_terms": list(result["plan"]["excluded_terms"]),
                    "products": [
                        {key: product[key] for key in ("id", "title", "brand", "score")}
                        for product in result["products"]
                    ],
                })
        check(results[0]["products"][0]["id"] == "synthetic-001", "exact brand/title fixture must rank first")
        check([p["id"] for p in results[1]["products"]] == ["synthetic-001"], "explicit brand exclusion must filter the matching fixture")
        check(results[2]["products"][0]["id"] == "synthetic-004", "numeric title fixture must rank first")
        repeated = engine.search(QUERIES[0], k=len(CATALOG), pool=len(CATALOG))
        check([p["id"] for p in repeated["products"]] == [p["id"] for p in results[0]["products"]], "cached search must retain order")
        check(engine.index.cache_hits > 0, "repeated search must use the cache")
    finally:
        engine.close()

    report = {"data_source": "synthetic", "scope": SCOPE, "products": len(CATALOG), "smoke_checks": "passed", "searches": results}
    report_path = directory / "synthetic-demo-report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(SCOPE)
    print("Smoke checks: PASSED (lexical order, exclusions, numeric match, finite scores, cache)")
    for result in results:
        print(f"\nQuery: {result['query']}")
        for rank, product in enumerate(result["products"], 1):
            print(f"  {rank}. {product['title']} [{product['id']}]")
    try:
        display_path = report_path.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        display_path = report_path
    print(f"\nSaved synthetic report: {display_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=SCOPE)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "demo", help="Parent folder for a new, uniquely named synthetic demo run")
    run(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
