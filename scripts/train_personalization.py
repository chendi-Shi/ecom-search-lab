"""Download a small public review subset and run time-safe CPU personalization."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import html
import json
from pathlib import Path
import re
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ecom_search.data import file_sha256, save_json, save_jsonl
from ecom_search.personalization import BPRModel, chronological_split, sampled_cases, evaluate_preferences, paired_user_bootstrap

SOURCE_PAGE = "https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/index.html"
SOURCE_BASE = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/"
SOURCES = {"Luxury_Beauty_5.json.gz": SOURCE_BASE + "categoryFilesSmall/Luxury_Beauty_5.json.gz",
           "meta_Luxury_Beauty.json.gz": SOURCE_BASE + "metaFiles2/meta_Luxury_Beauty.json.gz"}


def download(raw):
    raw.mkdir(parents=True, exist_ok=True)
    records, total = [], 0
    for name, url in SOURCES.items():
        path = raw / name
        if not path.exists():
            request = urllib.request.Request(url, headers={"User-Agent": "EcomSearchLab/0.1 research"})
            with urllib.request.urlopen(request, timeout=60) as response, path.with_suffix(path.suffix + ".part").open("wb") as handle:
                size = 0
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    size += len(block)
                    if size + total > 200_000_000:
                        raise ValueError("Public dataset download exceeds the fixed 200MB budget")
                    handle.write(block)
            path.with_suffix(path.suffix + ".part").replace(path)
        total += path.stat().st_size
        if total > 200_000_000:
            raise ValueError("Public dataset files exceed the fixed 200MB budget")
        records.append({"name": name, "url": url, "bytes": path.stat().st_size, "sha256": file_sha256(path),
                        "verification": "observed local content SHA256; publisher does not provide an immutable revision or checksum"})
    previous = raw / "source_manifest.json"
    if previous.exists():
        existing = json.loads(previous.read_text(encoding="utf-8"))
        if existing["files"] != records:
            raise ValueError("Downloaded source bytes differ from the first frozen local source manifest")
    manifest = {"dataset": "Amazon Reviews 2018 Luxury Beauty 5-core", "source_page": SOURCE_PAGE,
                "files": records, "license_status": "No explicit reusable data license confirmed; public research dataset, no rights relicensed",
                "citation": "Ni, Li and McAuley, EMNLP 2019, Justifying recommendations using distantly-labeled reviews and fined-grained aspects",
                "frozen_local_content": True, "raw_data_redistribution": False}
    save_json(previous, manifest)
    return manifest


def prepare(raw):
    events, identities, source_rows = [], set(), 0
    with gzip.open(raw / "Luxury_Beauty_5.json.gz", "rt", encoding="utf-8") as handle:
        for line in handle:
            source_rows += 1
            row = json.loads(line)
            user = "beauty-user-" + hashlib.sha256(row["reviewerID"].encode()).hexdigest()[:20]
            event = {"user_id": user, "product_id": row["asin"], "rating": float(row["overall"]),
                     "timestamp": int(row["unixReviewTime"]), "source": "amazon_2018_review_rating", "synthetic": False}
            key = (user, event["product_id"], event["timestamp"])
            if key in identities:
                continue
            identities.add(key)
            if not 1 <= event["rating"] <= 5:
                raise ValueError("Source rating outside 1..5")
            events.append(event)
    needed = {row["product_id"] for row in events}
    catalog = {}
    with gzip.open(raw / "meta_Luxury_Beauty.json.gz", "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["asin"] in needed:
                title = html.unescape(re.sub(r"<[^>]*>", " ", str(row.get("title", ""))))
                catalog[row["asin"]] = {"id": row["asin"], "title": " ".join(title.split()),
                                         "brand": str(row.get("brand", "")), "metadata_scope": "display-only crawl snapshot; not training feature"}
    for item in needed - set(catalog):
        catalog[item] = {"id": item, "title": "", "brand": "", "metadata_scope": "title unavailable"}
    return events, [catalog[item] for item in sorted(catalog)], {"original_review_rows": source_rows,
            "duplicate_user_product_timestamp_rows_removed": source_rows - len(events),
            "deduplication": "same hashed user, ASIN and review timestamp: retain first row in frozen source file"}


def run(raw, output, max_events=1000):
    start = time.perf_counter()
    if type(max_events) is not int or max_events < 1:
        raise ValueError("max_events must be a positive integer")
    if (output / "report.json").exists() or (output / "model.json").exists():
        raise FileExistsError("Existing experiment will not be overwritten; choose a new output directory")
    source_manifest = download(raw)
    events, catalog, source_counts = prepare(raw)
    splits, cutoffs = chronological_split(events)
    train_items = {row["product_id"] for row in splits["train"]}
    val_cases, val_counts = sampled_cases(splits["val"], events, train_items, limit=max_events)
    if not val_cases:
        raise ValueError("No eligible validation known-positive events")
    output.mkdir(parents=True, exist_ok=True)
    save_jsonl(output / "catalog.jsonl", catalog)
    for split, rows in splits.items():
        save_jsonl(output / f"{split}.jsonl", rows)
    fingerprints = {name: {"sha256": file_sha256(output / name), "bytes": (output / name).stat().st_size}
                    for name in ("catalog.jsonl", "train.jsonl", "val.jsonl", "test.jsonl")}
    choices, models = [], {}
    for epochs in (0, 5, 15, 30):
        model = BPRModel()
        info = model.fit(splits["train"], epochs=epochs)
        models[epochs] = model
        for weight in (0.0, 0.25, 0.5, 1.0):
            validation = evaluate_preferences(val_cases, model, weight)
            choices.append({"epochs": epochs, "personal_weight": weight, "validation_metrics": validation["metrics"], "training": info})
    selected = max(choices, key=lambda row: (row["validation_metrics"]["known_positive_ndcg@10"], -row["epochs"], -row["personal_weight"]))
    model = models[selected["epochs"]]
    frozen = {"model": model.to_dict(), "selection": selected, "data_fingerprints": fingerprints,
              "source": source_manifest, "cutoffs": cutoffs, "frozen_before_test": True,
              "selection_policy": "Fixed epochs [0,5,15,30] and rank fusion weights [0,.25,.5,1]; maximize validation known-positive NDCG@10; ties fewer epochs, then lower personal weight"}
    save_json(output / "model.json", frozen)
    # Test candidate sets are constructed only after the validation decision is frozen.
    test_cases, test_counts = sampled_cases(splits["test"], events, train_items, limit=max_events)
    for name, fingerprint in fingerprints.items():
        if file_sha256(output / name) != fingerprint["sha256"]:
            raise ValueError("Source split changed after validation freeze")
    baseline = evaluate_preferences(test_cases, model, 0.0)
    personal = evaluate_preferences(test_cases, model, selected["personal_weight"])
    report = {"dataset": source_manifest["dataset"], "source": source_manifest, "source_counts": source_counts,
              "review_events": len(events), "users": len({row["user_id"] for row in events}), "products": len(catalog),
              "train_known_products": len(train_items), "split_events": {split: len(rows) for split, rows in splits.items()},
              "cutoffs": cutoffs, "validation_choices": choices, "selected": selected,
              "validation_eligibility": val_counts, "test_eligibility": test_counts,
              "test": {"nonpersonalized_popularity": {key: value for key, value in baseline.items() if key != "per_event"},
                       "personalized_bpr": {key: value for key, value in personal.items() if key != "per_event"}},
              "data_fingerprints": fingerprints, "elapsed_seconds": time.perf_counter() - start,
              "paired_user_bootstrap": paired_user_bootstrap(baseline["per_event"], personal["per_event"]),
              "scope": "Global-time held-out review rating>=4; one known positive plus up to 99 uniform unjudged train-known items, excluding strictly earlier reviewed items; sampled ranking, not full-catalog recall or click/CTR",
              "history_policy": "Strict timestamp < event timestamp; no same-day review exchange; model trained only before train cutoff; cold users use popularity",
              "limitations": ["Original 5-core selection uses the complete dataset, so eligibility is not a prospective population sample",
                              "Review time is not purchase/exposure time; positive ratings are not click labels",
                              "Unreviewed sampled items are unjudged, can be false negatives, and are not known dislikes",
                              "Static metadata is display-only; no future average ratings, review count, or price used",
                              "Beauty catalog/users are separate from ESCI; this validates personalized recommendation, not personalized search relevance"],
              "created_at_utc": datetime.now(timezone.utc).isoformat()}
    save_jsonl(output / "per_event_test.jsonl", [{"baseline": left, "personalized": right}
              for left, right in zip(baseline["per_event"], personal["per_event"])])
    save_json(output / "report.json", report)
    serving_files = {**fingerprints, "model.json": {"sha256": file_sha256(output / "model.json"),
                                                    "bytes": (output / "model.json").stat().st_size}}
    save_json(output / "manifest.json", {"source": source_manifest, "files": serving_files, "cutoffs": cutoffs})
    print(json.dumps({"review_events": len(events), "users": report["users"], "products": len(catalog),
                      "selected": selected, "test": report["test"], "elapsed_seconds": report["elapsed_seconds"]}, ensure_ascii=False))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "outputs/personalization/raw")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/personalization/experiment")
    parser.add_argument("--max-events", type=int, default=1000)
    args = parser.parse_args()
    run(args.raw, args.output, args.max_events)


if __name__ == "__main__":
    main()
