"""Prepare a reproducible new ESCI experiment directly from pinned official Parquet.

Query selection reads identity columns only. All grades are copied only after the
selection is saved. Existing output directories are never overwritten.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ecom_search.data import file_sha256, normalized_query

COMMIT = "7916cdf6ab75a462e77f20ab40428a10923998d5"
REPOSITORY = "https://github.com/amazon-science/esci-data"
EXAMPLES = "shopping_queries_dataset_examples.parquet"
PRODUCTS = "shopping_queries_dataset_products.parquet"
GRADES = {"E": 3, "S": 2, "C": 1, "I": 0}
# The official commit's Git LFS pointer objects contain these sizes and SHA256s.
SOURCES = {
    EXAMPLES: {"bytes": 51286808, "sha256": "4a735b693b4a424a6fc67f5be6e4c811495c488bbf66d02a602d308b2744263a"},
    PRODUCTS: {"bytes": 1108857465, "sha256": "25124442d064d64b26f74082d6fa09438d679efc0c183cf28d19064a2b65a265"},
}
for _name, _spec in SOURCES.items():
    _suffix = f"{COMMIT}/shopping_queries_dataset/{_name}"
    _spec["source_url"] = f"{REPOSITORY}/blob/{_suffix}"
    _spec["pointer_url"] = f"https://raw.githubusercontent.com/amazon-science/esci-data/{_suffix}"
    _spec["download_url"] = f"https://media.githubusercontent.com/media/amazon-science/esci-data/{_suffix}"


def canonical_sha(value):
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def token(seed, tag, value):
    return canonical_sha([seed, tag, value])


def fingerprint(path):
    return {"bytes": Path(path).stat().st_size, "sha256": file_sha256(path)}


def write_json(path, value):
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def verify_sources(raw, download=False):
    """Download only missing pinned files; reject mismatches rather than replace."""
    raw = Path(raw)
    verified = {}
    for name, spec in SOURCES.items():
        path = raw / name
        expected = {key: spec[key] for key in ("bytes", "sha256")}
        if not path.exists() and download:
            raw.mkdir(parents=True, exist_ok=True)
            partial = raw / (name + ".part-" + uuid.uuid4().hex)
            try:
                request = urllib.request.Request(spec["download_url"], headers={"User-Agent": "ecom-search-lab/0.3"})
                with urllib.request.urlopen(request, timeout=120) as response, partial.open("xb") as output:
                    copied = 0
                    while chunk := response.read(1024 * 1024):
                        copied += len(chunk)
                        if copied > spec["bytes"]:
                            raise ValueError(f"Download exceeds pinned size: {name}")
                        output.write(chunk)
                if fingerprint(partial) != expected:
                    raise ValueError(f"Downloaded file differs from pinned Git LFS object: {name}")
                # Exclusive creation prevents overwriting a concurrently created file.
                with path.open("xb") as output, partial.open("rb") as source:
                    while chunk := source.read(1024 * 1024):
                        output.write(chunk)
            finally:
                partial.unlink(missing_ok=True)
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name}. Supply --raw or explicitly add --download (about 1.16 GB).")
        if fingerprint(path) != expected:
            raise ValueError(f"Source differs from pinned official size/SHA256: {name}")
        verified[name] = {**expected, "source_url": spec["source_url"], "pointer_url": spec["pointer_url"]}
    return verified


def us_small_rows(path, columns):
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    for batch in pq.ParquetFile(path).iter_batches(batch_size=65536, columns=columns, use_threads=False):
        mask = pc.and_(pc.equal(batch.column("product_locale"), "us"),
                       pc.equal(pc.cast(batch.column("small_version"), pa.int8()), 1))
        yield from batch.filter(mask).to_pylist()


def select_metadata(rows, *, train_queries, val_queries, test_queries, min_candidates, seed):
    """Pure metadata selection, usable with tiny fixtures; esci_label is never read."""
    if min(train_queries, val_queries, test_queries, min_candidates) <= 0:
        raise ValueError("All query counts and min_candidates must be positive")
    info, partitions = {}, defaultdict(set)
    scanned = 0
    for row in rows:
        scanned += 1
        qid, query, split = str(row["query_id"]), row["query"], row["split"]
        if not isinstance(query, str) or not normalized_query(query) or split not in {"train", "test"}:
            raise ValueError("Invalid official query identity or partition")
        if row["product_id"] is None:
            raise ValueError("Missing official product ID")
        normal = normalized_query(query)
        partitions[normal].add(split)
        item = info.setdefault(qid, {"query": query, "normalized": normal,
                                    "official_split": split, "product_ids": set()})
        if (item["query"], item["official_split"]) != (query, split):
            raise ValueError("One query ID has conflicting text or official partition")
        item["product_ids"].add("us:" + str(row["product_id"]))
    cross_partition = {query for query, splits in partitions.items() if len(splits) > 1}
    representatives = {}
    excluded = {"cross_official_partition": 0, "below_minimum_candidates": 0, "normalized_duplicate": 0}
    for qid in sorted(info, key=lambda q: (token(seed, "representative", q), q)):
        item = info[qid]
        if item["normalized"] in cross_partition:
            excluded["cross_official_partition"] += 1
        elif len(item["product_ids"]) < min_candidates:
            excluded["below_minimum_candidates"] += 1
        elif item["normalized"] in representatives:
            excluded["normalized_duplicate"] += 1
        else:
            representatives[item["normalized"]] = qid
    def ordered(partition):
        return sorted((qid for qid in representatives.values() if info[qid]["official_split"] == partition),
                      key=lambda q: (token(seed, "query-selection", info[q]["normalized"]), q))
    train, test = ordered("train"), ordered("test")
    if len(train) < train_queries + val_queries or len(test) < test_queries:
        raise ValueError("Insufficient eligible queries; do not alter the protocol after seeing model results")
    chosen = {"train": train[:train_queries], "val": train[train_queries:train_queries + val_queries],
              "test": test[:test_queries]}
    for ids in chosen.values():
        ids.sort(key=lambda q: (token(seed, "record-order", q), q))
    selected = {qid: {**info[qid], "split": split} for split, ids in chosen.items() for qid in ids}
    products = set().union(*(row["product_ids"] for row in selected.values()))
    summary = {"queries": {split: len(ids) for split, ids in chosen.items()}, "products": len(products),
               "judged_candidates": {split: sum(len(selected[q]["product_ids"]) for q in ids)
                                      for split, ids in chosen.items()},
               "source_rows_scanned_without_labels": scanned,
               "eligible_official_queries": {"train": len(train), "test": len(test)}, "exclusions": excluded}
    selection = {"schema_version": 1, "protocol": "public-esci-v1", "official_commit": COMMIT,
                 "sampling_seed": seed, "minimum_judged_candidates": min_candidates,
                 "selection_uses_esci_label": False, "summary": summary,
                 "algorithm": "SHA256 canonical JSON [seed,tag,value]; one normalized-query representative; official train divided into train/val; official test retained",
                 "query_ids": {split: ["esci-us-" + qid for qid in ids] for split, ids in chosen.items()},
                 "query_identity_sha256": {"esci-us-" + qid: hashlib.sha256(row["normalized"].encode()).hexdigest()
                                           for qid, row in selected.items()}}
    selection["content_sha256"] = canonical_sha(selection)
    return selected, chosen, products, selection


def collect_catalog(path, product_ids):
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    asins = pa.array([pid.split(":", 1)[1] for pid in sorted(product_ids)])
    catalog = {}
    columns = ["product_id", "product_locale", "product_title", "product_brand"]
    for batch in pq.ParquetFile(path).iter_batches(batch_size=32768, columns=columns, use_threads=False):
        mask = pc.and_(pc.equal(batch.column("product_locale"), "us"),
                       pc.is_in(batch.column("product_id"), value_set=asins))
        for row in batch.filter(mask).to_pylist():
            pid = "us:" + str(row["product_id"])
            product = {"id": pid, "title": (row["product_title"] or "").strip(),
                       "brand": (row["product_brand"] or "").strip(), "description": "", "synthetic": False}
            if pid in catalog and catalog[pid] != product:
                raise ValueError("Conflicting official product metadata")
            catalog[pid] = product
    if set(catalog) != product_ids:
        raise ValueError("Selected products missing metadata; selection will not be silently changed")
    return [catalog[pid] for pid in sorted(catalog)]


def copy_labels(rows, selected):
    labels = {qid: {} for qid in selected}
    for row in rows:
        qid = str(row["query_id"])
        if qid not in selected:
            continue
        pid = "us:" + str(row["product_id"])
        if row["esci_label"] not in GRADES:
            raise ValueError("Unknown selected official ESCI label")
        grade = GRADES[row["esci_label"]]
        if pid in labels[qid] and labels[qid][pid] != grade:
            raise ValueError("Conflicting official judgments")
        labels[qid][pid] = grade
    if any(set(labels[q]) != row["product_ids"] for q, row in selected.items()):
        raise ValueError("Copied judgments do not match the frozen candidate identities")
    return labels


def prepare(args):
    if args.output.exists():
        raise FileExistsError("Output already exists; choose a new directory for an immutable dataset")
    verified = verify_sources(args.raw, args.download)
    identity_columns = ["query_id", "query", "split", "product_id", "product_locale", "small_version"]
    selected, chosen, products, selection = select_metadata(
        us_small_rows(args.raw / EXAMPLES, identity_columns), train_queries=args.train_queries,
        val_queries=args.val_queries, test_queries=args.test_queries,
        min_candidates=args.min_candidates, seed=args.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    # Persist the grade-blind decision before opening any esci_label column.
    write_json(args.output / "selection.json", selection)
    selection_file = fingerprint(args.output / "selection.json")
    write_jsonl(args.output / "catalog.jsonl", collect_catalog(args.raw / PRODUCTS, products))
    labels = copy_labels(us_small_rows(args.raw / EXAMPLES,
        ["query_id", "product_id", "product_locale", "small_version", "esci_label"]), selected)
    for split, ids in chosen.items():
        records, corpus = [], []
        for qid in ids:
            item = selected[qid]
            base = {"query_id": "esci-us-" + qid, "query": item["query"], "split": split,
                    "synthetic": False, "annotation_source": "human", "sampling_seed": args.seed,
                    "annotation_provenance": {"source": REPOSITORY, "commit": COMMIT,
                        "official_query_id": qid, "official_split": item["official_split"]},
                    "qrels": {pid: labels[qid][pid] for pid in sorted(labels[qid])},
                    "candidate_sampling": "all original judged products; no grade-based filtering"}
            candidates = sorted(item["product_ids"], key=lambda pid: (token(args.seed, "candidate-order", qid + ":" + pid), pid))
            records.append({**base, "candidate_ids": candidates,
                            "evaluation_scope": "all_original_human_judged_candidates_for_query"})
            corpus.append({**base, "candidate_ids": [],
                           "evaluation_scope": "selected_query_product_union_with_incomplete_cross_query_judgments"})
        write_jsonl(args.output / f"{split}.jsonl", records)
        write_jsonl(args.output / f"corpus_{split}.jsonl", corpus)
    if fingerprint(args.output / "selection.json") != selection_file:
        raise ValueError("Selection changed while copying judgments")
    files = {path.name: fingerprint(path) for path in sorted(args.output.iterdir()) if path.is_file()}
    manifest = {"schema_version": 1, "protocol": "public-esci-v1", "dataset": "amazon-science/esci-data",
                "repository": REPOSITORY, "official_commit": COMMIT, "license": "Apache-2.0", "locale": "us",
                "synthetic": False, "task_version": "official small_version=1 retrieval task",
                **selection["summary"], "sampling_seed": args.seed, "minimum_judged_candidates": args.min_candidates,
                "selection_uses_esci_label": False, "selection_content_sha256": selection["content_sha256"],
                "split_policy": "train/val from official train; test from official test; IDs and NFKC-casefold-whitespace query texts disjoint",
                "previous_experiment_exclusion": "none; this standalone protocol is not the historical accuracy-v1 holdout",
                "catalog_selection": "union of all judged products from selected queries, independent of relevance grade",
                "metadata_scope": "official title and brand; description empty; no image/video or user history",
                "evaluation_scope": "closed pools fully judged; corpus recall measures known positives in the finite selected union; unjudged pairs unknown",
                "grade_mapping": GRADES, "ranking_gain": "2**grade - 1", "relevant_grade_min": 2,
                "holdout_status": "copied without grade summaries or model evaluation; open only after train/validation selection",
                "raw_files": verified, "files": files, "preparation_script_sha256": file_sha256(Path(__file__))}
    manifest["manifest_content_sha256"] = canonical_sha(manifest)
    write_json(args.output / "manifest.json", manifest)
    return {"status": "prepared_new_public_protocol", **selection["summary"],
            "selection_content_sha256": selection["content_sha256"], "manifest_content_sha256": manifest["manifest_content_sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "data/raw/esci")
    parser.add_argument("--download", action="store_true", help="Download missing pinned official Parquet (about 1.16 GB)")
    parser.add_argument("--output", type=Path, default=ROOT / "data/esci-public")
    parser.add_argument("--train-queries", type=int, default=1000)
    parser.add_argument("--val-queries", type=int, default=100)
    parser.add_argument("--test-queries", type=int, default=100)
    parser.add_argument("--min-candidates", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261011)
    args = parser.parse_args()
    if min(args.train_queries, args.val_queries, args.test_queries, args.min_candidates) <= 0:
        parser.error("all query counts and min-candidates must be positive")
    print(json.dumps(prepare(args), ensure_ascii=False, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
