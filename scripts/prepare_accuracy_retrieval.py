"""CPU corpus retrieval assets and validation-only fusion selection.

Default execution reads catalog.jsonl, train.jsonl and val.jsonl only. It never
opens test/corpus_test. RetrievalPool and evaluate_frozen accept owner-supplied
records later, without changing the persisted strategy.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ecom_search.data import file_sha256, load_jsonl, normalized_query, save_json
from ecom_search.index import ProductIndex, build_index, catalog_digest
from ecom_search.disk_vectors import DiskVectorIndex, build_disk_store, _close_mapping
from ecom_search.query import plan_query
from ecom_search.ranking import FeatureExtractor
from ecom_search.vectors import _fingerprint, _load_encoder, _numpy, model_fingerprint, normalize_vectors


def text(product):
    return " ".join(str(product.get(field, "")) for field in ("title", "brand", "description")).strip()


def validate_records(rows, ids, expected_split=None):
    seen, queries = set(), set()
    for row in rows:
        qid, query = row.get("query_id"), row.get("query")
        if not qid or qid in seen or not isinstance(query, str) or not query.strip():
            raise ValueError("records require unique IDs and nonempty queries")
        seen.add(qid)
        queries.add(normalized_query(query))
        if expected_split and row.get("split") != expected_split:
            raise ValueError("record split differs from its permitted source")
        qrels = row.get("qrels")
        if not isinstance(qrels, dict) or not qrels or any(pid not in ids or type(g) is not int or not 0 <= g <= 3 for pid, g in qrels.items()):
            raise ValueError("invalid human qrels or product ID")
    return seen, queries


def reuse_vectors(directory, old_catalog, expected_model):
    """Reuse only identical complete encoder inputs with verified old bytes."""
    np = _numpy()
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if metadata["model"]["sha256"] != expected_model["sha256"]:
        raise ValueError("cached catalog uses a different text encoder")
    for name, key in (("vectors.npy", "vectors_sha256"), ("ids.json", "ids_sha256")):
        if file_sha256(directory / name) != metadata[key]:
            raise ValueError("old catalog cache checksum differs")
    ids = json.loads((directory / "ids.json").read_text(encoding="utf-8"))
    rows = load_jsonl(old_catalog)
    texts = [text(row) for row in rows]
    if ids != [row["id"] for row in rows] or metadata["input_fingerprint"] != _fingerprint({"ids": ids, "texts": texts, "model": expected_model["sha256"]}):
        raise ValueError("old catalog text and vector binding differ")
    mapping = {}
    for position, value in enumerate(texts):
        mapping.setdefault(value, position)
    matrix = np.load(directory / "vectors.npy", mmap_mode="r", allow_pickle=False)
    return matrix, mapping, {"vectors_sha256": metadata["vectors_sha256"], "catalog_sha256": file_sha256(old_catalog), "comparison": "identical title+brand+description text and identical model SHA; ID alone never permits reuse"}


def prepare_assets(catalog, data_fingerprints, output, model_path, reuse_dir, reuse_catalog, batch_size=64, threads=2):
    np = _numpy()
    model = model_fingerprint(model_path)
    ids, texts = [row["id"] for row in catalog], [text(row) for row in catalog]
    input_fingerprint = _fingerprint({"ids": ids, "texts": texts, "model": model["sha256"]})
    output.mkdir(parents=True, exist_ok=True)
    sqlite_path = output / "products.sqlite"
    if sqlite_path.is_file():
        with ProductIndex(sqlite_path) as index:
            if index.metadata["catalog_sha256"] != catalog_digest(catalog):
                raise ValueError("existing SQLite differs from the fixed catalog")
    else:
        save_json(output / "products.build.json", build_index(catalog, sqlite_path))
    folder = output / "float"
    metadata_path = folder / "metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["input_fingerprint"] != input_fingerprint or metadata["model"]["sha256"] != model["sha256"]:
            raise ValueError("existing vectors were encoded from different inputs")
        for name, info in metadata["files"].items():
            if file_sha256(folder / name) != info["sha256"]:
                raise ValueError("existing float vectors failed checksum")
        encoding = metadata["encoding"]
    else:
        folder.mkdir(parents=True, exist_ok=True)
        old, by_text, reused_source = reuse_vectors(reuse_dir, reuse_catalog, model)
        matrix = None
        try:
            encoder = _load_encoder(Path(model["path"]), threads)
            dimension = int(encoder.get_sentence_embedding_dimension())
            if old.shape[1] != dimension:
                raise ValueError("old vectors have a different dimension")
            checkpoint_path = folder / "encoding_progress.json"
            partial = folder / "vectors.partial.npy"
            completed = reused = new = 0
            encoder_seconds = elapsed_before = 0.0
            if checkpoint_path.is_file():
                checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                if checkpoint["input_fingerprint"] != input_fingerprint or not partial.is_file():
                    raise ValueError("encoding checkpoint belongs to different inputs")
                completed, reused, new = checkpoint["completed"], checkpoint["reused"], checkpoint["newly_encoded"]
                encoder_seconds, elapsed_before = checkpoint["encoder_seconds"], checkpoint["elapsed_seconds"]
                matrix = np.lib.format.open_memmap(partial, mode="r+")
                if matrix.shape != (len(ids), dimension):
                    raise ValueError("encoding checkpoint shape differs")
            else:
                matrix = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float32, shape=(len(ids), dimension))
            started = last_progress = time.perf_counter()
            for first in range(completed, len(ids), batch_size):
                stop = min(first + batch_size, len(ids))
                missing_positions = []
                for position in range(first, stop):
                    cached = by_text.get(texts[position])
                    if cached is None:
                        missing_positions.append(position)
                    else:
                        matrix[position] = old[cached]
                        reused += 1
                if missing_positions:
                    encoding_start = time.perf_counter()
                    encoded = encoder.encode([texts[position] for position in missing_positions], batch_size=batch_size,
                                             convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
                    matrix[missing_positions] = normalize_vectors(encoded)
                    encoder_seconds += time.perf_counter() - encoding_start
                    new += len(missing_positions)
                completed = stop
                if time.perf_counter() - last_progress >= 30 or completed == len(ids):
                    matrix.flush()
                    elapsed = elapsed_before + time.perf_counter() - started
                    remaining = sum(value not in by_text for value in texts[completed:])
                    progress = {"input_fingerprint": input_fingerprint, "completed": completed, "total": len(ids), "reused": reused,
                                "newly_encoded": new, "encoder_seconds": encoder_seconds, "elapsed_seconds": elapsed,
                                "estimated_remaining_seconds": remaining * encoder_seconds / new if new else None,
                                "threads": threads, "batch_size": batch_size}
                    save_json(checkpoint_path, progress)
                    print(json.dumps({"phase": "encoding_catalog", **progress}), flush=True)
                    last_progress = time.perf_counter()
            matrix.flush()
            _close_mapping(matrix)
            matrix = None
            partial.replace(folder / "vectors.npy")
            save_json(folder / "ids.json", ids)
            encoding = {"reused_rows": reused, "newly_encoded_rows": new, "encoder_seconds": encoder_seconds,
                        "elapsed_seconds": elapsed_before + time.perf_counter() - started, "threads": threads,
                        "batch_size": batch_size, "reuse_source": reused_source, "truncation": "unchanged local SentenceTransformer tokenizer/model configuration"}
            files = {name: {"sha256": file_sha256(folder / name), "bytes": (folder / name).stat().st_size} for name in ("vectors.npy", "ids.json")}
            metadata = {"schema_version": 1, "mode": "float32", "matrix": "vectors.npy", "count": len(ids), "dimension": dimension,
                        "model": model, "input_fingerprint": input_fingerprint, "files": files, "source_vectors_sha256": files["vectors.npy"]["sha256"],
                        "vectors_sha256": files["vectors.npy"]["sha256"], "ids_sha256": files["ids.json"]["sha256"],
                        "catalog_sha256": catalog_digest(catalog), "data_fingerprints": data_fingerprints, "encoding": encoding}
            save_json(metadata_path, metadata)
            checkpoint_path.unlink(missing_ok=True)
        finally:
            _close_mapping(matrix)
            _close_mapping(old)
    quantized = output / "int8"
    if not (quantized / "metadata.json").is_file():
        build_disk_store(folder, quantized, mode="int8", chunk=4096)
    with DiskVectorIndex(quantized, chunk=4096) as index:
        if index.metadata["input_fingerprint"] != input_fingerprint:
            raise ValueError("int8 input binding differs")
        workspace = index.workspace_estimate()
    return model, encoding, workspace


def fuse(lexical, dense, lexical_weight, depth, k=100):
    scores = {}
    for ranking, weight in ((lexical[:depth], lexical_weight), (dense[:depth], 1 - lexical_weight)):
        if weight:
            for rank, pid in enumerate(ranking, 1):
                scores[pid] = scores.get(pid, 0) + weight / (60 + rank)
    return [(pid, scores[pid]) for pid in sorted(scores, key=lambda pid: (-scores[pid], pid))[:k]]


class RetrievalPool:
    """Load the frozen strategy; default semantic query is the original text."""
    def __init__(self, directory, *, encoder=None, require_selection=True, strategy_path=None):
        self.directory = Path(directory).resolve()
        self.metadata = json.loads((self.directory / "metadata.json").read_text(encoding="utf-8"))
        for name, digest in self.metadata.get("artifacts", {}).items():
            if file_sha256(self.directory / name) != digest:
                raise ValueError("retrieval artifact checksum differs")
        self.index = ProductIndex(self.directory / "products.sqlite", cache_size=0)
        self.disk = None
        self.encoder = None
        self.model_path = None
        self.has_vectors = self.metadata.get("vectors_available", True)
        self._closed = False
        try:
            if self.index.metadata["catalog_sha256"] != self.metadata["catalog_sha256"]:
                raise ValueError("retrieval catalog binding differs")
            path = Path(strategy_path) if strategy_path else self.directory / "pool_strategy.json"
            if require_selection and self.metadata.get("pool_strategy_sha256") != file_sha256(path):
                raise ValueError("strategy checksum differs from the frozen binding")
            self.selection = json.loads(path.read_text(encoding="utf-8")) if require_selection else None
            if self.selection and (self.selection.get("frozen_before_holdout") is not True or self.selection["input_fingerprint"] != self.metadata["input_fingerprint"]):
                raise ValueError("strategy is not frozen or belongs to another catalog")
            if self.selection and (self.selection["selected"]["lexical_weight"] not in (0, .25, .5, .75, 1) or self.selection["selected"]["channel_depth"] not in (100, 200) or self.selection["rrf_constant"] != 60):
                raise ValueError("strategy is outside the preregistered grid")
            if not self.has_vectors:
                if self.metadata.get("mode") != "lexical_only" or not self.selection or self.selection.get("mode") != "lexical_only" or self.selection["selected"] != {"lexical_weight": 1, "channel_depth": 100}:
                    raise ValueError("vector-free serving requires the frozen lexical-only strategy")
            else:
                self.disk = DiskVectorIndex(self.directory / "int8", chunk=4096)
                if self.disk.metadata["input_fingerprint"] != self.metadata["input_fingerprint"] or self.disk.metadata["model"]["sha256"] != self.metadata["model"]["sha256"]:
                    raise ValueError("retrieval catalog/vector/encoder binding differs")
                model = self.metadata["model"]
                recorded = Path(model["path"])
                self.model_path = recorded if recorded.is_dir() else ROOT / "outputs/text-assets/minilm"
                if model_fingerprint(self.model_path)["sha256"] != model["sha256"]:
                    raise ValueError("encoder checksum differs")
                self.encoder = encoder or _load_encoder(self.model_path, 2)
        except Exception:
            self.close()
            raise

    def channels(self, query, depth=200, vector=None):
        plan = plan_query(query)
        lexical = [row["id"] for row in self.index.search(plan, depth)]
        if not self.has_vectors:
            return lexical, []
        if vector is None:
            vector = self.encoder.encode([query], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)[0]
        dense_k = min(1000, depth * 3) if plan.excluded_terms else depth
        dense = [pid for pid, score in self.disk.search_vector(vector, dense_k)]
        if plan.excluded_terms:
            forbidden = set(plan.excluded_terms)
            products = self.index.get_products(dense)
            dense = [pid for pid in dense if not forbidden & FeatureExtractor.affirmed_product_terms(products[pid])]
        return lexical, dense[:depth]

    def search(self, query, k=100, *, use_vector=True):
        if self._closed:
            raise ValueError("retrieval pool is closed")
        if self.selection is None:
            raise ValueError("search requires the validation-frozen strategy")
        if type(k) is not int or not 1 <= k <= 200 or type(use_vector) is not bool:
            raise ValueError("cutoff must be 1..200 and use_vector boolean; only top100 was selected")
        chosen = self.selection["selected"]
        if not self.has_vectors:
            # Keep the EXACT validation pool100 independent of requested k.
            # k=200 is accepted by UI, but this frozen strategy has <=100 items.
            return [(row["id"], row["retrieval_score"]) for row in self.index.search(query, chosen["channel_depth"])][:k]
        if not use_vector:
            return [(row["id"], row["retrieval_score"]) for row in self.index.search(query, k)]
        lexical, dense = self.channels(query, chosen["channel_depth"])
        return fuse(lexical, dense, chosen["lexical_weight"], chosen["channel_depth"], k)

    def close(self):
        if self._closed:
            return
        if self.disk is not None:
            self.disk.close()
        self.index.close()
        self._closed = True


def quality(ranked, qrels):
    positive, exact = {pid for pid, g in qrels.items() if g >= 2}, {pid for pid, g in qrels.items() if g == 3}
    return {**{f"known_positive_recall_at_{k}": len(set(ranked[:k]) & positive) / len(positive) if positive else None for k in (10, 100)},
            **{f"known_exact_recall_at_{k}": len(set(ranked[:k]) & exact) / len(exact) if exact else None for k in (10, 100)},
            **{f"judged_coverage_at_{k}": sum(pid in qrels for pid in ranked[:k]) / k for k in (10, 100)},
            "known_positive_count_at_10": len(set(ranked[:10]) & positive)}


def aggregate(rows):
    result = {"queries": len(rows)}
    for metric in ("known_positive_recall_at_10", "known_positive_recall_at_100", "known_exact_recall_at_10", "known_exact_recall_at_100", "judged_coverage_at_10", "judged_coverage_at_100", "known_positive_count_at_10"):
        values = [row[metric] for row in rows if row[metric] is not None]
        result[metric] = statistics.mean(values) if values else None
        result[metric + "_denominator"] = len(values)
    return result


def evaluate_frozen(directory, records, output=None):
    """Owner supplies permitted holdout rows after freezing; never selects again."""
    pool = RetrievalPool(directory)
    try:
        rows = [{"query_id": row["query_id"], **quality([pid for pid, score in pool.search(row["query"])], row["qrels"])} for row in records]
        result = {"metrics": aggregate(rows), "per_query": rows, "pool_strategy_sha256": file_sha256(pool.directory / "pool_strategy.json"),
                  "scope": "observed human E/S positives in fixed catalog; unjudged products remain unknown; frozen strategy unchanged"}
        if output:
            save_json(output, result)
        return result
    finally:
        pool.close()


def freeze_lexical(catalog, train, validation, fingerprints, output):
    """A fixed BM25 pool, not a claim that the unfinished dense build is ready."""
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    sqlite_path = output / "products.sqlite"
    digest = catalog_digest(catalog)
    if not sqlite_path.is_file():
        save_json(output / "products.build.json", build_index(catalog, sqlite_path))
    with ProductIndex(sqlite_path, cache_size=0) as index:
        if index.metadata["catalog_sha256"] != digest or catalog_digest(index.catalog()) != digest:
            raise ValueError("existing SQLite contents differ from the fixed catalog")
        rows = [{"query_id": row["query_id"], **quality([p["id"] for p in index.search(row["query"], 100)], row["qrels"])} for row in validation]
    input_fingerprint = _fingerprint({"catalog_sha256": digest, "mode": "lexical_only", "candidate_pool": 100,
                                      "lexical_backend": "SQLite FTS5 title weight3 brand weight1; current shared query parser"})
    selection = {"schema_version": 1, "mode": "lexical_only", "selected": {"lexical_weight": 1, "channel_depth": 100},
                 "rrf_constant": 60, "score_semantics": "raw SQLite title+brand BM25; no RRF/dense model in this mode",
                 "input_fingerprint": input_fingerprint, "catalog_sha256": digest, "catalog_file_sha256": fingerprints["catalog.jsonl"],
                 "validation_sha256": fingerprints["val.jsonl"], "validation_metrics": aggregate(rows),
                 "selection_policy": "fixed pool100 BM25 for same-candidate CE-vs-BM25 accuracy comparison; no fusion search or test tuning",
                 "frozen_before_holdout": True, "holdout_accessed": False, "vectors_available": False,
                 "selected_at_utc": datetime.now(timezone.utc).isoformat(), "relevance_scope": "observed human E/S positives only; unjudged products are unknown"}
    selection_path = output / "pool_strategy.json"
    save_json(selection_path, selection)
    save_json(output / "metadata.json", {"schema_version": 1, "mode": "lexical_only", "vectors_available": False,
              "catalog_sha256": digest, "catalog_file_sha256": fingerprints["catalog.jsonl"], "input_fingerprint": input_fingerprint,
              "data_fingerprints": fingerprints, "source": {"dataset": "Amazon ESCI", "annotation": "original human query-product relevance",
              "provenance": train[0].get("annotation_provenance", {})}, "serving_cutoff": 100,
              "pool_strategy_sha256": file_sha256(selection_path), "artifacts": {"products.sqlite": file_sha256(sqlite_path)},
              "dense_status": "not used or completed in this delivery; optional partial/checkpoint excluded from packages",
              "optional_resume_command": "python scripts/prepare_accuracy_retrieval.py --resume-dense --threads 2 --batch-size 64"})
    save_json(output / "validation_per_query.json", rows)
    report = {"mode": "lexical_only", "catalog_count": len(catalog), "train_queries_read_only_for_split_validation": len(train),
              "validation_queries": len(validation), "validation_metrics": selection["validation_metrics"], "data_fingerprints": fingerprints,
              "sqlite_bytes": sqlite_path.stat().st_size, "elapsed_seconds": time.perf_counter() - started,
              "vectors_available": False, "holdout_accessed": False, "same_candidate_pool": 100}
    save_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/esci-v2")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/accuracy-v1/retrieval")
    parser.add_argument("--model", type=Path, default=ROOT / "outputs/text-assets/minilm")
    parser.add_argument("--reuse", type=Path, default=ROOT / "outputs/ann/flat")
    parser.add_argument("--reuse-catalog", type=Path, default=ROOT / "data/esci/catalog.jsonl")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=2)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--lexical-only", action="store_true", help="Freeze standard-library SQLite pool100; never loads any encoder or vectors")
    modes.add_argument("--resume-dense", action="store_true", help="Resume OPTIONAL encoding/float/int8 build, preserving the frozen lexical strategy; no new selection")
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 64 or args.threads != 2:
        raise ValueError("this CPU run requires two threads and batch size <=64")
    if not args.resume_dense and (args.output / "pool_strategy.json").is_file():
        raise FileExistsError("strategy already frozen; use evaluate_frozen without rebuilding/reselection")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", CUDA_VISIBLE_DEVICES="", TOKENIZERS_PARALLELISM="false")
    started = time.perf_counter()
    paths = {name: args.data / name for name in ("catalog.jsonl", "train.jsonl", "val.jsonl")}
    fingerprints = {name: file_sha256(path) for name, path in paths.items()}
    catalog, train, validation = [load_jsonl(paths[name]) for name in paths]
    ids = {row["id"] for row in catalog}
    if not ids or len(ids) != len(catalog):
        raise ValueError("catalog requires unique nonempty product IDs")
    tr_ids, tr_queries = validate_records(train, ids, "train")
    va_ids, va_queries = validate_records(validation, ids, "val")
    if tr_ids & va_ids or tr_queries & va_queries:
        raise ValueError("train/validation query leakage")
    if args.lexical_only:
        report = freeze_lexical(catalog, train, validation, fingerprints, args.output)
        if any(file_sha256(paths[name]) != expected for name, expected in fingerprints.items()):
            raise ValueError("data changed during lexical freezing")
        print(json.dumps({"phase": "complete", **report}), flush=True)
        return
    grid = [{"lexical_weight": weight, "channel_depth": depth} for depth in (100, 200) for weight in (0, .25, .5, .75, 1)]
    model, encoding, workspace = prepare_assets(catalog, fingerprints, args.output, args.model, args.reuse, args.reuse_catalog, args.batch_size, args.threads)
    if args.resume_dense:
        report = {"optional_dense_complete": True, "default_strategy_unchanged": True, "encoding": encoding, "workspace": workspace,
                  "catalog_file_sha256": fingerprints["catalog.jsonl"], "model_sha256": model["sha256"], "data_fingerprints": fingerprints,
                  "holdout_accessed": False, "elapsed_seconds": time.perf_counter() - started}
        save_json(args.output / "dense_optional_report.json", report)
        print(json.dumps({"phase": "optional_dense_complete", **report}), flush=True)
        return
    input_fingerprint = _fingerprint({"ids": [row["id"] for row in catalog], "texts": [text(row) for row in catalog], "model": model["sha256"]})
    save_json(args.output / "metadata.json", {"schema_version": 1, "mode": "hybrid", "vectors_available": True, "catalog_sha256": catalog_digest(catalog), "input_fingerprint": input_fingerprint,
              "data_fingerprints": fingerprints, "model": model, "semantic_query": "original query, no stopword/negation stripping before embedding",
              "constraint_policy": "current shared lexical exclusions; catalog wording only", "threads": 2, "serving_cutoff": 100,
              "artifacts": {name: file_sha256(args.output / name) for name in ("products.sqlite", "float/metadata.json", "int8/metadata.json")}})
    selection_path = args.output / "pool_strategy.json"
    if selection_path.is_file():
        raise FileExistsError("strategy already frozen; evaluation must use evaluate_frozen without reselection")
    save_json(args.output / "selection_plan.json", {"grid": grid, "rrf_constant": 60, "cutoff": 100,
              "criterion": "highest validation macro known E/S Recall@100; ties exact Recall@100, smaller channel depth, single-channel simplicity, larger lexical weight",
              "validation_sha256": fingerprints["val.jsonl"], "holdout_accessed": False})
    pool = RetrievalPool(args.output, require_selection=False)
    try:
        np = _numpy()
        vectors = normalize_vectors(pool.encoder.encode([row["query"] for row in validation], batch_size=args.batch_size,
                                   convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False))
        np.save(args.output / "validation_query_vectors.npy", vectors, allow_pickle=False)
        scores = [[] for _ in grid]
        query_rows = []
        for number, (row, vector) in enumerate(zip(validation, vectors)):
            lexical, dense = pool.channels(row["query"], 200, vector)
            channels = {200: (lexical, dense)}
            # Oversampling depends on requested depth when exclusions apply.
            # Recompute that case so validation and serving have identical pools.
            channels[100] = pool.channels(row["query"], 100, vector) if plan_query(row["query"]).excluded_terms else (lexical[:100], dense[:100])
            record = {"query_id": row["query_id"], "strategies": []}
            for position, candidate in enumerate(grid):
                lexical, dense = channels[candidate["channel_depth"]]
                hits = fuse(lexical, dense, candidate["lexical_weight"], candidate["channel_depth"])
                metrics = quality([pid for pid, score in hits], row["qrels"])
                scores[position].append(metrics)
                record["strategies"].append({**candidate, **metrics})
            query_rows.append(record)
            if (number + 1) % 25 == 0:
                print(json.dumps({"phase": "validation_retrieval", "queries": number + 1, "total": len(validation)}), flush=True)
        results = [{**candidate, **aggregate(rows)} for candidate, rows in zip(grid, scores)]
        chosen = max(results, key=lambda row: (row["known_positive_recall_at_100"], row["known_exact_recall_at_100"] or 0,
                     -row["channel_depth"], int(row["lexical_weight"] in (0, 1)), row["lexical_weight"]))
        if any(file_sha256(paths[name]) != expected for name, expected in fingerprints.items()):
            raise ValueError("data changed during validation-only selection")
        selection = {"schema_version": 1, "selected": {key: chosen[key] for key in ("lexical_weight", "channel_depth")}, "results": results,
                     "input_fingerprint": input_fingerprint, "validation_sha256": fingerprints["val.jsonl"], "rrf_constant": 60,
                     "frozen_before_holdout": True, "holdout_accessed": False, "candidate_scope": "fixed 82k catalog; top100 after label-blind lexical/dense fusion",
                     "relevance_scope": "grade>=2 E/S known positives only; unjudged unknown", "selected_at_utc": datetime.now(timezone.utc).isoformat()}
        save_json(selection_path, selection)
        metadata_path = args.output / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["pool_strategy_sha256"] = file_sha256(selection_path)
        save_json(metadata_path, metadata)
        save_json(args.output / "validation_per_query.json", query_rows)
    finally:
        pool.close()
    report = {"catalog_count": len(catalog), "train_queries_read_only_for_split_validation": len(train), "validation_queries": len(validation),
              "encoding": encoding, "workspace": workspace, "selection": selection, "data_fingerprints": fingerprints,
              "elapsed_seconds": time.perf_counter() - started, "persistent_bytes": sum(p.stat().st_size for p in args.output.rglob("*") if p.is_file()),
              "validation_only": True, "holdout_accessed": False}
    save_json(args.output / "report.json", report)
    print(json.dumps({"phase": "complete", "catalog_count": len(catalog), "selected": selection["selected"], "validation_metrics": chosen,
                      "encoding": encoding, "elapsed_seconds": report["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
