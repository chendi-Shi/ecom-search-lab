"""Prepare a new ESCI cohort after excluding all registered prior query identities.

The portable registry contains hashes only. Selection reads identity columns, then
persists its decision before relevance labels are opened. Existing outputs fail.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ecom_search.data import normalized_query
from scripts import prepare_public_esci as base

PROTOCOL = "closure-esci-v1"
REGISTRY_PROTOCOL = "closure-esci-exclusions-v1"
SOURCE_FILES = ("scripts/prepare_closure_esci.py", "scripts/prepare_public_esci.py", "ecom_search/data.py")
FRESHNESS = ("Query ID and NFKC-casefold-whitespace query text are disjoint from every registered "
             "prior train/validation/test identity. Product overlap and semantic paraphrase overlap "
             "are not excluded. This does not establish that the pretrained encoder never saw ESCI.")
CAPACITY_DECISION = ("Default train count reduced from proposed 1000 to 800 before new labels were read: "
                     "4003 eligible official-train identities minus 2923 prior train/validation identities "
                     "leaves at most 1080, insufficient for 1000 train + 200 validation. Keep all exclusions, "
                     "minimum 20 candidates, 200 validation and 300 test; no label-based eligibility changes.")


def sha_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_id(value):
    """Official US numeric ID, independent of historical esci-us- prefix/zeros."""
    if isinstance(value, bool):
        raise ValueError("Invalid official query ID")
    text = str(value)
    if text.startswith("esci-us-"):
        text = text[len("esci-us-"):]
    if not re.fullmatch(r"[0-9]+", text):
        raise ValueError("Invalid official query ID")
    return str(int(text))


def identity_hashes(query_id, query):
    if not isinstance(query, str) or not normalized_query(query):
        raise ValueError("Invalid query text")
    return sha_text("esci-us-" + canonical_id(query_id)), sha_text(normalized_query(query))


def seal(value):
    value["content_sha256"] = base.canonical_sha(value)
    return value


def read_registry(path):
    registry = json.loads(Path(path).read_text(encoding="utf-8"))
    digest = registry.pop("content_sha256", None)
    if registry.get("protocol") != REGISTRY_PROTOCOL or digest != base.canonical_sha(registry):
        raise ValueError("Exclusion registry content hash or protocol mismatch")
    for name in ("query_id_sha256", "normalized_query_sha256"):
        values = registry.get(name)
        if not isinstance(values, list) or values != sorted(set(values)) or not values:
            raise ValueError("Exclusion registry must contain sorted nonempty unique hash sets")
        if any(not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value) for value in values):
            raise ValueError("Invalid exclusion hash")
    registry["content_sha256"] = digest
    return registry


def build_registry(sources, output):
    """Verify all historical identity files before opening any old records.

    sources is [(public_label, dataset_directory), ...]. All JSONL files except
    catalog are included, so corpus files and additional query files cannot be
    silently omitted. Every such file must have a prior manifest fingerprint.
    """
    output = Path(output)
    if output.exists():
        raise FileExistsError("Exclusion registry already exists; it is immutable")
    inputs, source_records, verified_directories, labels = [], [], [], set()
    for label, directory in sources:
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", label) or label in labels:
            raise ValueError("Source labels must be unique portable identifiers")
        labels.add(label)
        directory = Path(directory)
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("official_commit") != base.COMMIT:
            raise ValueError("Historical source has a different official commit")
        names = {path.name for path in directory.glob("*.jsonl") if path.name != "catalog.jsonl"}
        if not {"train.jsonl", "val.jsonl", "test.jsonl"}.issubset(names):
            raise ValueError("Historical source must include every train/val/test identity file")
        files = {"manifest.json": base.fingerprint(manifest_path)}
        for name in sorted(names):
            actual = base.fingerprint(directory / name)
            expected = manifest.get("files", {}).get(name)
            if actual != expected:
                raise ValueError(f"Historical identity file missing or changed: {label}/{name}")
            files[name] = actual
            inputs.append((label, directory / name, actual))
        source_records.append({"label": label, "files": files})
        verified_directories.append((directory, files))
    if not inputs:
        raise ValueError("At least one verified historical dataset is required")
    ids, texts, per_source = set(), set(), {}
    for label, path, expected in inputs:
        count = 0
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                raw_id = row["query_id"]
                provenance = row.get("annotation_provenance", {})
                if "official_query_id" in provenance and canonical_id(raw_id) != canonical_id(provenance["official_query_id"]):
                    raise ValueError("Historical query ID disagrees with provenance")
                qid_hash, text_hash = identity_hashes(raw_id, row["query"])
                ids.add(qid_hash)
                texts.add(text_hash)
                count += 1
        if base.fingerprint(path) != expected:
            raise ValueError("Historical identity file changed while extracting registry")
        per_source[f"{label}/{path.name}"] = count
    for directory, files in verified_directories:
        if any(base.fingerprint(directory / name) != expected for name, expected in files.items()):
            raise ValueError("Historical input changed before registry freeze")
    registry = seal({"schema_version": 1, "protocol": REGISTRY_PROTOCOL, "official_commit": base.COMMIT,
                     "query_id_hash_algorithm": "SHA256 UTF-8 of esci-us- + base-10 canonical official query ID",
                     "query_text_hash_algorithm": "SHA256 UTF-8 of NFKC-casefold-whitespace normalized query",
                     "scope": FRESHNESS, "sources": source_records, "source_identity_rows": per_source,
                     "query_id_sha256": sorted(ids), "normalized_query_sha256": sorted(texts),
                     "raw_queries_labels_or_local_paths_included": False})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(registry, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    return {"status": "registry_frozen", "unique_query_ids": len(ids), "unique_normalized_queries": len(texts),
            "source_files": len(inputs), "content_sha256": registry["content_sha256"],
            "registry_file": base.fingerprint(output)}


def select_fresh_metadata(rows, registry, **settings):
    """Exclude identities without inspecting esci_label or any grade-derived value."""
    old_ids, old_texts = set(registry["query_id_sha256"]), set(registry["normalized_query_sha256"])
    excluded_ids, excluded_text_ids, excluded_any_ids = set(), set(), set()
    scanned = 0

    def filtered():
        nonlocal scanned
        for row in rows:
            scanned += 1
            qid = canonical_id(row["query_id"])
            qid_hash, query_hash = identity_hashes(qid, row["query"])
            by_id, by_text = qid_hash in old_ids, query_hash in old_texts
            if by_id:
                excluded_ids.add(qid)
            if by_text:
                excluded_text_ids.add(qid)
            if by_id or by_text:
                excluded_any_ids.add(qid)
                continue
            yield row

    selected, chosen, products, selection = base.select_metadata(filtered(), **settings)
    for qid, item in selected.items():
        qid_hash, text_hash = identity_hashes(qid, item["query"])
        if qid_hash in old_ids or text_hash in old_texts:
            raise AssertionError("Freshness invariant violated")
    selection.pop("content_sha256")
    selection["protocol"] = PROTOCOL
    selection["default_sample_size_rationale"] = CAPACITY_DECISION
    selection["previous_experiment_exclusion"] = {
        "registry_content_sha256": registry["content_sha256"],
        "registered_unique_query_ids": len(old_ids), "registered_unique_normalized_queries": len(old_texts),
        "excluded_source_query_ids_by_id": len(excluded_ids),
        "excluded_source_query_ids_by_text": len(excluded_text_ids),
        "excluded_source_query_ids_union": len(excluded_any_ids),
        "selected_overlap_by_id": 0, "selected_overlap_by_normalized_query": 0,
        "freshness_scope": FRESHNESS}
    selection["summary"]["source_rows_scanned_without_labels"] = scanned
    selection["summary"]["exclusions"]["prior_query_identity"] = len(excluded_any_ids)
    selection["algorithm"] = "Exclude registered ID/text hashes, then " + selection["algorithm"]
    return selected, chosen, products, seal(selection)


def prepare(args):
    if args.output.exists():
        raise FileExistsError("Output already exists; choose a new immutable dataset directory")
    registry = read_registry(args.exclusions)
    registry_fingerprint = base.fingerprint(args.exclusions)
    source_hashes = {name: base.fingerprint(ROOT / name) for name in SOURCE_FILES}
    verified = base.verify_sources(args.raw, args.download)
    selected, chosen, products, selection = select_fresh_metadata(
        base.us_small_rows(args.raw / base.EXAMPLES,
                           ["query_id", "query", "split", "product_id", "product_locale", "small_version"]),
        registry, train_queries=args.train_queries, val_queries=args.val_queries, test_queries=args.test_queries,
        min_candidates=args.min_candidates, seed=args.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    base.write_json(args.output / "selection.json", selection)
    selection_file = base.fingerprint(args.output / "selection.json")
    # Grade-blind identities are now durable; only this subsequent stage copies labels.
    base.write_jsonl(args.output / "catalog.jsonl", base.collect_catalog(args.raw / base.PRODUCTS, products))
    labels = base.copy_labels(base.us_small_rows(args.raw / base.EXAMPLES,
        ["query_id", "product_id", "product_locale", "small_version", "esci_label"]), selected)
    for split, ids in chosen.items():
        records, corpus = [], []
        for qid in ids:
            item = selected[qid]
            common = {"query_id": "esci-us-" + qid, "query": item["query"], "split": split,
                      "synthetic": False, "annotation_source": "human", "sampling_seed": args.seed,
                      "annotation_provenance": {"source": base.REPOSITORY, "commit": base.COMMIT,
                          "official_query_id": qid, "official_split": item["official_split"]},
                      "qrels": {pid: labels[qid][pid] for pid in sorted(labels[qid])},
                      "candidate_sampling": "all original judged products; no grade-based filtering"}
            candidates = sorted(item["product_ids"],
                key=lambda pid: (base.token(args.seed, "candidate-order", qid + ":" + pid), pid))
            records.append({**common, "candidate_ids": candidates,
                            "evaluation_scope": "all_original_human_judged_candidates_for_query"})
            corpus.append({**common, "candidate_ids": [],
                           "evaluation_scope": "selected_query_product_union_with_incomplete_cross_query_judgments"})
        base.write_jsonl(args.output / f"{split}.jsonl", records)
        base.write_jsonl(args.output / f"corpus_{split}.jsonl", corpus)
    if base.fingerprint(args.output / "selection.json") != selection_file:
        raise ValueError("Frozen selection changed while copying labels")
    if base.fingerprint(args.exclusions) != registry_fingerprint:
        raise ValueError("Exclusion registry changed during preparation")
    if source_hashes != {name: base.fingerprint(ROOT / name) for name in SOURCE_FILES}:
        raise ValueError("Preparation source changed during run")
    manifest = {"schema_version": 1, "protocol": PROTOCOL, "dataset": "amazon-science/esci-data",
                "repository": base.REPOSITORY, "official_commit": base.COMMIT, "license": "Apache-2.0",
                "locale": "us", "synthetic": False, "task_version": "official small_version=1 retrieval task",
                **selection["summary"], "sampling_seed": args.seed, "minimum_judged_candidates": args.min_candidates,
                "default_sample_size_rationale": CAPACITY_DECISION,
                "selection_uses_esci_label": False, "selection_content_sha256": selection["content_sha256"],
                "previous_experiment_exclusion": selection["previous_experiment_exclusion"],
                "exclusion_registry_file": registry_fingerprint, "historical_identity_sources": registry["sources"],
                "split_policy": "train/val from official train; test from official test; ID and normalized text disjoint",
                "catalog_selection": "union of all judged products from selected queries, independent of relevance grade",
                "metadata_scope": "official title and brand; description empty; no image/video or user history",
                "evaluation_scope": "closed pools fully judged; corpus recall uses known positives in finite selected union; unjudged pairs unknown",
                "grade_mapping": base.GRADES, "ranking_gain": "2**grade - 1", "relevant_grade_min": 2,
                "holdout_status": "labels copied only; no grade summaries or model evaluation",
                "training_io_contract": "Read train and val only before model freeze; do not call load_dataset",
                "raw_files": verified, "source_files": source_hashes,
                "files": {path.name: base.fingerprint(path) for path in sorted(args.output.iterdir()) if path.is_file()}}
    manifest["manifest_content_sha256"] = base.canonical_sha(manifest)
    base.write_json(args.output / "manifest.json", manifest)
    return {"status": "prepared_fresh_query_protocol", **selection["summary"],
            "previous_experiment_exclusion": selection["previous_experiment_exclusion"],
            "selection_content_sha256": selection["content_sha256"],
            "manifest_content_sha256": manifest["manifest_content_sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    registry_parser = sub.add_parser("build-registry", help="One-time hash-only registry from verified historical datasets")
    registry_parser.add_argument("--source", action="append", required=True, metavar="LABEL=DATASET_DIR")
    registry_parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/closure-v1/exclusions.json")
    prepare_parser = sub.add_parser("prepare", help="Reproduce the new cohort using the published exclusion registry")
    prepare_parser.add_argument("--raw", type=Path, default=ROOT / "data/raw/esci")
    prepare_parser.add_argument("--download", action="store_true")
    prepare_parser.add_argument("--output", type=Path, default=ROOT / "data/esci-closure-v1")
    prepare_parser.add_argument("--exclusions", type=Path, default=ROOT / "benchmarks/closure-v1/exclusions.json")
    prepare_parser.add_argument("--train-queries", type=int, default=800)
    prepare_parser.add_argument("--val-queries", type=int, default=200)
    prepare_parser.add_argument("--test-queries", type=int, default=300)
    prepare_parser.add_argument("--min-candidates", type=int, default=20)
    prepare_parser.add_argument("--seed", type=int, default=20261013)
    args = parser.parse_args()
    if args.command == "build-registry":
        if any("=" not in value for value in args.source):
            parser.error("--source must be LABEL=DATASET_DIR")
        result = build_registry([value.split("=", 1) for value in args.source], args.output)
    else:
        result = prepare(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
