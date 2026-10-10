"""Prepare the frozen public ESCI subset without modifying its source files."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ecom_search.data import file_sha256, load_dataset, load_jsonl, save_json, save_jsonl, validate_dataset

EXPECTED_SOURCE_MANIFEST = "b82fc3df59fd58690dbb898a8345abc38146a684a1518518d1186722d3e26c28"


def prepare(source, destination):
    source, destination = Path(source), Path(destination)
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    body = {key: value for key, value in manifest.items() if key != "manifest_content_sha256"}
    canonical = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    actual = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if actual != manifest.get("manifest_content_sha256") or actual != EXPECTED_SOURCE_MANIFEST:
        raise ValueError("Source manifest differs from the preselected frozen ESCI subset")
    if (manifest.get("products") != 15171 or manifest.get("candidates_per_query") != 20
            or manifest.get("product_metadata_scope") != "official_title_and_brand_only; description deliberately empty"):
        raise ValueError("Source subset scope differs")
    names = ["catalog.jsonl"] + [f"closed_candidates_{split}.jsonl" for split in ("train", "val", "test")]
    source_files = {}
    for name in names:
        expected = manifest["prepared_files"][name]
        path = source / name
        if file_sha256(path) != expected["sha256"] or path.stat().st_size != expected["bytes"]:
            raise ValueError(f"Frozen source fingerprint differs: {name}")
        source_files[name] = dict(expected)
    source_catalog = load_jsonl(source / "catalog.jsonl")
    catalog = []
    for product in source_catalog:
        provenance = product.get("provenance", {})
        if (product.get("synthetic") is not False or provenance.get("commit") != manifest["official_commit"]
                or provenance.get("dataset") != "amazon-science/esci-data"
                or provenance.get("metadata_scope") != "official_title_and_brand_only"):
            raise ValueError("Product provenance differs from the official nonsynthetic subset")
        catalog.append({"id": product["id"], "title": product["title"], "brand": product["brand"],
                        "description": "", "synthetic": False})
    splits = {}
    for split in ("train", "val", "test"):
        rows = load_jsonl(source / f"closed_candidates_{split}.jsonl")
        for row in rows:
            annotation = row.get("annotation_provenance", {})
            expected_official = "test" if split == "test" else "train"
            if (row.get("split") != split or row.get("synthetic") is not False
                    or row.get("annotation_source") != "human"
                    or annotation.get("official_split") != expected_official
                    or annotation.get("commit") != manifest["official_commit"]
                    or len(row.get("candidate_ids", [])) != 20):
                raise ValueError("Query split, judgments, or source provenance differs")
        if len(rows) != manifest["splits"][split]["queries"]:
            raise ValueError("Query count differs from frozen source")
        splits[split] = [{key: row[key] for key in (
            "query_id", "query", "candidate_ids", "qrels", "split", "synthetic",
            "annotation_source", "annotation_provenance", "candidate_sampling", "sampling_seed")}
                        for row in rows]
    validate_dataset(catalog, splits)
    if len(catalog) != 15171:
        raise ValueError("Product count differs from frozen source")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("Destination already contains files; choose an independent output directory")
    destination.mkdir(parents=True, exist_ok=True)
    save_jsonl(destination / "catalog.jsonl", catalog)
    for split, rows in splits.items():
        save_jsonl(destination / f"{split}.jsonl", rows)
    shutil.copyfile(manifest_path, destination / "source_manifest.json")
    shutil.copyfile(source / "ESCI_LICENSE", destination / "ESCI_LICENSE")
    output_names = ["catalog.jsonl", "train.jsonl", "val.jsonl", "test.jsonl", "source_manifest.json", "ESCI_LICENSE"]
    output_manifest = {
        "schema_version": 1, "dataset": "amazon-science/esci-data", "repository": manifest["repository"],
        "official_commit": manifest["official_commit"], "license": manifest["license"],
        "synthetic": False, "products": len(catalog), "queries": {s: len(r) for s, r in splits.items()},
        "metadata_scope": "official title and brand; description empty; no image/video or user histories",
        "evaluation_scope": "same fixed 20 fully human-judged candidates per query; not corpus recall",
        "candidates_per_query": 20, "selection": manifest["selection"], "sampling_seed": manifest["sampling_seed"],
        "split_policy": manifest["split_policy"], "ranking_gain": "2**grade - 1", "relevant_grade_min": 2,
        "source_subset_manifest_content_sha256": actual,
        "source_manifest_file_sha256": file_sha256(manifest_path), "source_files": source_files,
        "source_raw_verification_at_preparation": manifest["raw_source_verification_at_preparation"],
        "source_verification_scope": "verified frozen subset files; raw parquet is not reread by this preparation",
        "files": {name: {"sha256": file_sha256(destination / name), "bytes": (destination / name).stat().st_size}
                  for name in output_names},
    }
    save_json(destination / "manifest.json", output_manifest)
    return output_manifest


def add_corpus(source, destination):
    """Preserve all original judgments for observed-positive corpus evaluation."""
    source, destination = Path(source), Path(destination)
    catalog, train, val, test = load_dataset(destination)
    current = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    if current.get("source_subset_manifest_content_sha256") != EXPECTED_SOURCE_MANIFEST:
        raise ValueError("Destination does not use the preselected frozen source")
    source_manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    body = {key: value for key, value in source_manifest.items() if key != "manifest_content_sha256"}
    digest = hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True,
                                      separators=(",", ":")).encode("utf-8")).hexdigest()
    if digest != EXPECTED_SOURCE_MANIFEST:
        raise ValueError("Frozen source manifest changed")
    product_ids = {row["id"] for row in catalog}
    pending = []
    for split, records in zip(("train", "val", "test"), (train, val, test)):
        name = f"corpus_{split}.jsonl"
        expected = source_manifest["prepared_files"][name]
        path = source / name
        if file_sha256(path) != expected["sha256"] or path.stat().st_size != expected["bytes"]:
            raise ValueError(f"Corpus source fingerprint differs: {name}")
        corpus = load_jsonl(path)
        fixed_by_id = {row["query_id"]: row for row in records}
        if len(corpus) != len(records) or {row["query_id"] for row in corpus} != set(fixed_by_id):
            raise ValueError("Corpus queries differ from the frozen fixed-pool selection")
        for row in corpus:
            fixed = fixed_by_id[row["query_id"]]
            annotation = row.get("annotation_provenance", {})
            if (row.get("query") != fixed["query"] or row.get("split") != split
                    or row.get("synthetic") is not False or row.get("annotation_source") != "human"
                    or annotation.get("official_split") != ("test" if split == "test" else "train")
                    or annotation.get("commit") != source_manifest["official_commit"]):
                raise ValueError("Corpus query or split provenance differs")
            if any(pid not in product_ids or type(grade) is not int or not 0 <= grade <= 3
                   for pid, grade in row["qrels"].items()):
                raise ValueError("Corpus qrels have invalid product ids or grades")
            if any(row["qrels"].get(pid) != grade for pid, grade in fixed["qrels"].items()):
                raise ValueError("Fixed-pool and full-corpus judgments disagree")
        target = destination / name
        if target.exists() and file_sha256(target) != expected["sha256"]:
            raise FileExistsError("Existing corpus file differs; it will not be overwritten")
        pending.append((name, path, target, expected))
    for name, path, target, expected in pending:
        if not target.exists():
            shutil.copyfile(path, target)
        current["source_files"][name] = dict(expected)
        current["files"][name] = {"sha256": file_sha256(target), "bytes": target.stat().st_size}
    current["corpus_scope"] = ("15,171-product union of selected query judgments; corpus files preserve all original "
                               "per-query qrels; other query-product pairs remain unjudged; observed-positive recall "
                               "is not Amazon full-catalog recall")
    save_json(destination / "manifest.json", current)
    return current


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Directory containing the original frozen ESCI subset and manifest")
    parser.add_argument("--output", type=Path, default=ROOT / "data/esci")
    parser.add_argument("--add-corpus", action="store_true", help="Verify an existing prepared subset and add its corpus files")
    args = parser.parse_args()
    if not args.add_corpus:
        prepare(args.source, args.output)
    manifest = add_corpus(args.source, args.output)
    print(json.dumps({"products": manifest["products"], "queries": manifest["queries"],
                      "output": str(args.output.resolve()), "source_frozen_sha256": EXPECTED_SOURCE_MANIFEST}))


if __name__ == "__main__":
    main()
