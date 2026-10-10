"""Freeze fresh, label-blind ESCI queries; the final test labels remain sealed.

The first pass projects *no esci_label column*. Query identity, official split,
and judged product IDs determine the selection. Only after writing selection.json
does a second pass copy the selected human judgments into the frozen files.
Nothing in the public summary depends on a relevance grade.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ecom_search.data import file_sha256, normalized_query, save_json, save_jsonl

COMMIT = "7916cdf6ab75a462e77f20ab40428a10923998d5"
REPOSITORY = "https://github.com/amazon-science/esci-data"
EXAMPLES = "shopping_queries_dataset_examples.parquet"
PRODUCTS = "shopping_queries_dataset_products.parquet"
GRADES = {"E": 3, "S": 2, "C": 1, "I": 0}
DEFAULT_RAW = ROOT / "data/raw/esci"


def token(seed, tag, value):
    payload = json.dumps([seed, tag, value], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def canonical_sha(value):
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fingerprint(path):
    path = Path(path)
    return {"bytes": path.stat().st_size, "sha256": file_sha256(path)}


def read_old_metadata(directory):
    """Project query identity from every old split without using old grades."""
    directory = Path(directory)
    ids, texts, files = set(), set(), {}
    for split in ("train", "val", "test"):
        for prefix in ("", "corpus_"):
            path = directory / f"{prefix}{split}.jsonl"
            if prefix and not path.exists():
                continue
            files[path.name] = fingerprint(path)
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    public_id = row["query_id"]
                    official_id = row.get("annotation_provenance", {}).get("official_query_id")
                    if official_id is None and public_id.startswith("esci-us-"):
                        official_id = public_id[len("esci-us-"):]
                    if official_id is None:
                        raise ValueError("Old query lacks an official query identity")
                    ids.add(str(official_id))
                    texts.add(normalized_query(row["query"]))
    for name in ("manifest.json", "source_manifest.json", "catalog.jsonl", "ESCI_LICENSE"):
        files[name] = fingerprint(directory / name)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest.get("files", {}).items():
        if Path(name).name != name:
            raise ValueError("Old manifest filename escapes its directory")
        if fingerprint(directory / name) != expected:
            raise ValueError(f"Old frozen file changed: {name}")
    with (directory / "catalog.jsonl").open(encoding="utf-8") as handle:
        product_ids = {json.loads(line)["id"] for line in handle}
    return ids, texts, product_ids, files


def raw_spec(old_directory):
    manifest = json.loads((Path(old_directory) / "source_manifest.json").read_text(encoding="utf-8"))
    if manifest["official_commit"] != COMMIT or manifest["locale"] != "us":
        raise ValueError("Old source is not the expected official US ESCI release")
    return {spec["name"]: spec for spec in manifest["raw_files"]}


def verify_raw(raw, specs, products=False):
    names = (EXAMPLES, PRODUCTS) if products else (EXAMPLES,)
    result = {}
    for name in names:
        actual = fingerprint(Path(raw) / name)
        expected = specs[name]
        if actual != {"bytes": expected["size"], "sha256": expected["sha256"]}:
            raise ValueError(f"Complete official source size/SHA differs: {name}")
        result[name] = {**actual, "source_url": expected["source_url"], "status": "complete_file_sha256_verified"}
    return result


def us_small_batches(path, columns, batch_size=65536):
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=columns, use_threads=False):
        mask = pc.and_(pc.equal(batch.column("product_locale"), "us"),
                       pc.equal(pc.cast(batch.column("small_version"), pa.int8()), 1))
        yield batch.filter(mask)


def select_queries(args, excluded_ids, excluded_texts, old_products):
    # IMPORTANT: esci_label is not requested in this pass.
    columns = ["query_id", "query", "product_id", "product_locale", "small_version", "split"]
    info, partitions = {}, defaultdict(set)
    rows_scanned = 0
    for batch in us_small_batches(args.raw / EXAMPLES, columns):
        rows_scanned += len(batch)
        for row in batch.to_pylist():
            qid, query, split = str(row["query_id"]), str(row["query"]), str(row["split"])
            normal = normalized_query(query)
            if not normal or split not in {"train", "test"}:
                raise ValueError("Invalid official query or split")
            partitions[normal].add(split)
            record = info.setdefault(qid, {"query": query, "normalized": normal, "official_split": split,
                                           "product_ids": set()})
            if (record["query"], record["official_split"]) != (query, split):
                raise ValueError("One official query ID has inconsistent text or split")
            record["product_ids"].add("us:" + str(row["product_id"]))
    cross_partition = {normal for normal, values in partitions.items() if len(values) > 1}
    representatives = {}
    exclusions = {"old_query_id_or_normalized_query": 0, "cross_official_partition": 0,
                  "below_minimum_judged_candidates": 0, "normalized_duplicate": 0}
    for qid in sorted(info, key=lambda value: token(args.seed, "representative", value)):
        row = info[qid]
        if qid in excluded_ids or row["normalized"] in excluded_texts:
            exclusions["old_query_id_or_normalized_query"] += 1
        elif row["normalized"] in cross_partition:
            exclusions["cross_official_partition"] += 1
        elif len(row["product_ids"]) < args.min_candidates:
            exclusions["below_minimum_judged_candidates"] += 1
        elif row["normalized"] in representatives:
            exclusions["normalized_duplicate"] += 1
        else:
            representatives[row["normalized"]] = qid
    def ordered(partition):
        return sorted((qid for qid in representatives.values() if info[qid]["official_split"] == partition),
                      key=lambda value: token(args.seed, "query-selection", info[value]["normalized"]))
    train_ids, test_ids = ordered("train"), ordered("test")
    if len(train_ids) < args.train_queries + args.val_queries or len(test_ids) < args.test_queries:
        raise ValueError("Not enough fresh label-blind eligible queries for the predeclared sizes")
    chosen = {"train": train_ids[:args.train_queries],
              "val": train_ids[args.train_queries:args.train_queries + args.val_queries],
              "test": test_ids[:args.test_queries]}
    selected = {qid: {**info[qid], "split": split} for split, ids in chosen.items() for qid in ids}
    for split in chosen:
        chosen[split].sort(key=lambda value: token(args.seed, "record-order", value))
    new_product_ids = set().union(*(row["product_ids"] for row in selected.values()))
    all_new_ids = set(selected)
    all_new_texts = {row["normalized"] for row in selected.values()}
    if all_new_ids & excluded_ids or all_new_texts & excluded_texts or len(all_new_texts) != len(selected):
        raise ValueError("Old or cross-split query leakage")
    old_coverage = {}
    for official in ("train", "test"):
        rows = [info[qid] for qid in representatives.values() if info[qid]["official_split"] == official]
        total = sum(len(row["product_ids"]) for row in rows)
        covered = sum(len(row["product_ids"] & old_products) for row in rows)
        old_coverage[official] = {"eligible_queries": len(rows), "source_judged_pairs": total,
                                  "old_catalog_pairs": covered, "pair_coverage": covered / total if total else 0.0,
                                  "queries_with_at_least_min_candidates_in_old_catalog":
                                  sum(len(row["product_ids"] & old_products) >= args.min_candidates for row in rows)}
    summary = {"queries": {split: len(ids) for split, ids in chosen.items()}, "products": len(new_product_ids),
               "judged_candidates": {split: sum(len(selected[qid]["product_ids"]) for qid in ids)
                                       for split, ids in chosen.items()},
               "source_rows_scanned_without_labels": rows_scanned, "excluded_old_unique_ids": len(excluded_ids),
               "excluded_old_normalized_queries": len(excluded_texts), "eligible_official_queries":
               {"train": len(train_ids), "test": len(test_ids)}, "exclusions": exclusions,
               "old_catalog_overlap": len(new_product_ids & old_products),
               "new_products_relative_to_old_catalog": len(new_product_ids - old_products),
               "old_catalog_coverage_for_eligible_queries": old_coverage}
    selection = {"schema_version": 2, "official_commit": COMMIT, "sampling_seed": args.seed,
                 "selection_uses_esci_label": False, "minimum_judged_candidates": args.min_candidates,
                 "algorithm": "SHA256(JSON([seed,tag,value])); normalized-query representative then query-selection ordering",
                 "summary": summary, "query_ids": {split: ["esci-us-" + qid for qid in ids] for split, ids in chosen.items()},
                 "query_identity_sha256": {"esci-us-" + qid: hashlib.sha256(row["normalized"].encode("utf-8")).hexdigest()
                                             for qid, row in selected.items()}}
    selection["content_sha256"] = canonical_sha(selection)
    return selected, chosen, new_product_ids, selection


def collect_catalog(raw, product_ids):
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    asin_ids = pa.array([pid.split(":", 1)[1] for pid in sorted(product_ids)])
    catalog = {}
    columns = ["product_id", "product_locale", "product_title", "product_brand"]
    for batch in pq.ParquetFile(Path(raw) / PRODUCTS).iter_batches(batch_size=32768, columns=columns, use_threads=False):
        mask = pc.and_(pc.equal(batch.column("product_locale"), "us"),
                       pc.is_in(batch.column("product_id"), value_set=asin_ids))
        for row in batch.filter(mask).to_pylist():
            pid = "us:" + str(row["product_id"])
            product = {"id": pid, "title": "" if row["product_title"] is None else str(row["product_title"]).strip(),
                       "brand": "" if row["product_brand"] is None else str(row["product_brand"]).strip(),
                       "description": "", "synthetic": False}
            if pid in catalog and catalog[pid] != product:
                raise ValueError("Conflicting official product metadata")
            catalog[pid] = product
    if set(catalog) != product_ids:
        raise ValueError("Selected judged product absent from official product metadata; selection will not be changed")
    return [catalog[pid] for pid in sorted(catalog)]


def copy_selected_judgments(raw, selected):
    """Automatically transcribe grades only after selection has been persisted.

    This function emits no grade-derived summaries, including for the holdout.
    Unselected queries are filtered before conversion to Python objects.
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    qids = pa.array([int(qid) for qid in selected], type=pa.int64())
    labels = {qid: {} for qid in selected}
    columns = ["query_id", "product_id", "product_locale", "small_version", "esci_label"]
    for batch in us_small_batches(Path(raw) / EXAMPLES, columns):
        mask = pc.is_in(pc.cast(batch.column("query_id"), pa.int64()), value_set=qids)
        for row in batch.filter(mask).to_pylist():
            qid, pid = str(row["query_id"]), "us:" + str(row["product_id"])
            if row["esci_label"] not in GRADES:
                raise ValueError("Selected annotation has an invalid official grade")
            grade = GRADES[row["esci_label"]]
            if pid in labels[qid] and labels[qid][pid] != grade:
                raise ValueError("Conflicting selected official annotations")
            labels[qid][pid] = grade
    if any(set(labels[qid]) != row["product_ids"] for qid, row in selected.items()):
        raise ValueError("Copied labels differ from frozen label-blind product IDs")
    return labels


def freeze(args):
    if args.train_queries <= 0 or args.val_queries <= 0 or args.test_queries <= 0 or args.min_candidates <= 0:
        raise ValueError("Query counts and minimum candidates must be positive")
    excluded_ids, excluded_texts, old_products, old_files = read_old_metadata(args.old)
    specs = raw_spec(args.old)
    verified = verify_raw(args.raw, specs, products=args.freeze)
    selected, chosen, product_ids, selection = select_queries(args, excluded_ids, excluded_texts, old_products)
    if not args.freeze:
        return {"status": "label_blind_inspection_only", **selection["summary"],
                "selection_content_sha256": selection["content_sha256"]}
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Output is not empty; frozen data will never be overwritten")
    args.output.mkdir(parents=True, exist_ok=True)
    # Persist the label-blind decisions before reading esci_label at all.
    save_json(args.output / "selection.json", selection)
    selection_file = fingerprint(args.output / "selection.json")
    catalog = collect_catalog(args.raw, product_ids)
    save_jsonl(args.output / "catalog.jsonl", catalog)
    labels = copy_selected_judgments(args.raw, selected)
    for split, qids in chosen.items():
        records, corpus = [], []
        for qid in qids:
            row = selected[qid]
            ordered_products = sorted(row["product_ids"], key=lambda pid: token(args.seed, "candidate-order", qid + ":" + pid))
            base = {"query_id": "esci-us-" + qid, "query": row["query"], "split": split,
                    "synthetic": False, "annotation_source": "human", "sampling_seed": args.seed,
                    "annotation_provenance": {"source": REPOSITORY, "commit": COMMIT,
                        "official_query_id": qid, "official_split": row["official_split"],
                        "query_origin": "official public ESCI query text"},
                    "qrels": {pid: labels[qid][pid] for pid in sorted(labels[qid])},
                    "candidate_sampling": "all original judged products; order label-blind SHA256; no grade-based filtering"}
            records.append({**base, "candidate_ids": ordered_products,
                            "evaluation_scope": "all_original_human_judged_candidates_for_query"})
            corpus.append({**base, "candidate_ids": [],
                           "evaluation_scope": f"{len(product_ids)}_product_selected_query_union_with_incomplete_cross_query_judgments"})
        save_jsonl(args.output / f"{split}.jsonl", records)
        save_jsonl(args.output / f"corpus_{split}.jsonl", corpus)
    # Verify selection is byte-for-byte stable through the annotation copy.
    if fingerprint(args.output / "selection.json") != selection_file:
        raise ValueError("Frozen selection changed while copying judgments")
    shutil.copyfile(args.old / "ESCI_LICENSE", args.output / "ESCI_LICENSE")
    # Preparation never updates the previous dataset or reports.
    if any(fingerprint(args.old / name) != expected for name, expected in old_files.items()):
        raise ValueError("An old frozen dataset file changed during preparation")
    files = {path.name: fingerprint(path) for path in sorted(args.output.iterdir()) if path.is_file()}
    manifest = {"schema_version": 2, "dataset": "amazon-science/esci-data", "repository": REPOSITORY,
                "official_commit": COMMIT, "license": "Apache-2.0", "locale": "us", "synthetic": False,
                "task_version": "official small_version=1 retrieval task", **selection["summary"],
                "sampling_seed": args.seed, "minimum_judged_candidates": args.min_candidates,
                "candidates_per_query": "all_original_judged_products_variable_count",
                "metadata_scope": "official title and brand; description empty; no image/video or user histories",
                "catalog_selection": "union of all judged product IDs of label-blind selected queries; no relevance-grade filtering",
                "catalog_qrel_coverage": 1.0,
                "catalog_bias": "query-judged union, selected for >=minimum candidate count; not Amazon full catalog",
                "split_policy": "train/val from official train; final test from unused official test; IDs and NFKC-casefold-whitespace queries disjoint from all old splits and each other",
                "selection_content_sha256": selection["content_sha256"], "selection_uses_esci_label": False,
                "old_dataset_files": old_files, "raw_files": verified,
                "raw_local_directory": str(args.raw.resolve()), "old_dataset_directory": str(args.old.resolve()),
                "preparation_script_sha256": file_sha256(Path(__file__)), "files": files,
                "ranking_gain": "2**grade - 1", "relevant_grade_min": 2,
                "grade_mapping": GRADES,
                "holdout_status": "sealed: labels copied automatically; no label summaries or model scores produced; release only once after model/configuration freeze",
                "holdout_files": ["test.jsonl", "corpus_test.jsonl"],
                "training_io_contract": "Read train.jsonl/corpus_train.jsonl and val.jsonl/corpus_val.jsonl directly. Do not call load_dataset (it loads test) during training or selection.",
                "evaluation_scope": "closed candidate scores use all original per-query human judgments; corpus Recall is known-positive Recall over this finite union; unjudged cross-query pairs are unknown"}
    manifest["manifest_content_sha256"] = canonical_sha(manifest)
    save_json(args.output / "manifest.json", manifest)
    return {"status": "frozen_holdout_sealed", **selection["summary"],
            "selection_content_sha256": selection["content_sha256"],
            "manifest_content_sha256": manifest["manifest_content_sha256"], "output": str(args.output.resolve())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--old", type=Path, default=ROOT / "data/esci")
    parser.add_argument("--output", type=Path, default=ROOT / "data/esci-v2")
    parser.add_argument("--train-queries", type=int, default=2000)
    parser.add_argument("--val-queries", type=int, default=200)
    parser.add_argument("--test-queries", type=int, default=200)
    parser.add_argument("--min-candidates", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--freeze", action="store_true", help="Persist selection then automatically copy labels; otherwise only inspect nonlabel metadata")
    args = parser.parse_args()
    print(json.dumps(freeze(args), ensure_ascii=False, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
