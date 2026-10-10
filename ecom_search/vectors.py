"""Optional, local-only CPU embedding and FAISS retrieval benchmark.

The main application has no dependency on this module's optional libraries.
The same normalized embeddings are shared by FlatIP and HNSW. Validation alone
selects efSearch; test judgments are loaded only after that decision is saved.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import sys
import time
from typing import Any, Iterable


def _numpy():
    try:
        return importlib.import_module("numpy")
    except ImportError as exc:
        raise RuntimeError("Optional vector mode requires numpy in this Python environment") from exc


def _faiss(site: str | None = None):
    if site:
        dependency_path = Path(site).resolve()
        if not (dependency_path / "faiss" / "__init__.py").is_file():
            raise ValueError("--faiss-site must be a local site-packages directory containing faiss")
        # Append, never replace this environment's numpy/torch libraries.
        if str(dependency_path) not in sys.path:
            sys.path.append(str(dependency_path))
    try:
        return importlib.import_module("faiss")
    except ImportError as exc:
        raise RuntimeError("Optional ANN mode requires faiss-cpu; --faiss-site can reuse a local installation read-only") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{number}: expected a JSON object")
                rows.append(row)
    return rows


def normalize_vectors(vectors: Any):
    """Validate and return finite float32 unit vectors; zero vectors are invalid."""
    np = _numpy()
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] < 1:
        raise ValueError("vectors must have shape (count, positive_dimension)")
    if not np.isfinite(matrix).all():
        raise ValueError("vectors contain non-finite values")
    norms = np.linalg.norm(matrix.astype(np.float64), axis=1)
    if np.any(norms == 0):
        raise ValueError("zero vectors cannot be cosine normalized")
    return np.ascontiguousarray(matrix / norms[:, None], dtype=np.float32)


def _validate_ids(ids: Iterable[str], count: int) -> list[str]:
    raw_values = list(ids)
    if any(value is None for value in raw_values):
        raise ValueError("every vector must have one non-empty ID")
    values = [str(value) for value in raw_values]
    if len(values) != count or any(not value for value in values):
        raise ValueError("every vector must have one non-empty ID")
    if len(values) != len(set(values)):
        raise ValueError("duplicate product IDs")
    return values


def exact_cosine_search(ids: Iterable[str], vectors: Any, query_vector: Any, k: int):
    """Small NumPy reference implementation useful for independent correctness tests."""
    if k < 0:
        raise ValueError("k must be nonnegative")
    np = _numpy()
    matrix = normalize_vectors(vectors)
    values = _validate_ids(ids, len(matrix))
    query = normalize_vectors(np.asarray(query_vector, dtype=np.float32).reshape(1, -1))
    if query.shape[1] != matrix.shape[1]:
        raise ValueError("query dimension differs from index")
    scores = matrix @ query[0]
    positions = sorted(range(len(values)), key=lambda index: (-float(scores[index]), values[index]))[:k]
    return [(values[index], float(scores[index])) for index in positions]


def model_fingerprint(model_path: str | Path) -> dict:
    path = Path(model_path).resolve()
    if not path.is_dir() or not (path / "modules.json").is_file():
        raise ValueError("--model must be a local SentenceTransformer directory with modules.json; downloads are disabled")
    allowed = {".json", ".txt", ".bin", ".safetensors", ".model"}
    files = [{"path": file.relative_to(path).as_posix(), "sha256": _sha256(file), "bytes": file.stat().st_size}
             for file in sorted(path.rglob("*")) if file.is_file() and file.suffix.lower() in allowed
             and not any(part in {"onnx", "openvino", ".git"} for part in file.relative_to(path).parts)]
    if not any(row["path"].endswith((".safetensors", ".bin")) for row in files):
        raise ValueError("local model directory has no model weights")
    return {"path": str(path), "sha256": _fingerprint(files), "files": files}


def _load_encoder(model_path: Path, threads: int = 2):
    # A local directory plus local_files_only is required even when HF is logged in.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        torch = importlib.import_module("torch")
        module = importlib.import_module("sentence_transformers")
    except ImportError as exc:
        raise RuntimeError("Optional embedding mode requires CPU torch and sentence-transformers") from exc
    torch.set_num_threads(threads)
    return module.SentenceTransformer(str(model_path), device="cpu", local_files_only=True, trust_remote_code=False)


def _embeddings(texts: list[str], model: dict, cache_dir: Path, encoder: Any, batch_size: int):
    np = _numpy()
    key = _fingerprint({"texts": texts, "model": model["sha256"], "normalize": "l2-float32-v1"})
    array_path = cache_dir / f"{key}.npy"
    metadata_path = cache_dir / f"{key}.json"
    if array_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("sha256") != _sha256(array_path):
            raise ValueError("cached embedding checksum mismatch")
        vectors = np.load(array_path, allow_pickle=False)
        if len(vectors) != len(texts) or metadata.get("fingerprint") != key:
            raise ValueError("cached embedding metadata mismatch")
        return normalize_vectors(vectors), {"fingerprint": key, "cache_hit": True, "path": str(array_path), "sha256": metadata["sha256"]}
    if not texts:
        dimension = int(encoder.get_sentence_embedding_dimension())
        vectors = np.empty((0, dimension), dtype=np.float32)
    else:
        vectors = normalize_vectors(encoder.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                                                  normalize_embeddings=True, show_progress_bar=False))
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = array_path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, vectors, allow_pickle=False)
    temporary.replace(array_path)
    checksum = _sha256(array_path)
    _write_json(metadata_path, {"fingerprint": key, "sha256": checksum, "count": len(texts), "dimension": vectors.shape[1], "model_sha256": model["sha256"]})
    return vectors, {"fingerprint": key, "cache_hit": False, "path": str(array_path), "sha256": checksum}


def build(directory: str | Path, ids: Iterable[str], vectors: Any, *, model: dict | None = None,
          algorithm: str = "flat", ef_search: int = 64, hnsw_m: int = 16,
          faiss_site: str | None = None, input_fingerprint: str | None = None) -> dict:
    """Persist a cosine FAISS index and mapping; returns size/timing/checksums."""
    if algorithm not in {"flat", "hnsw"} or ef_search < 1 or hnsw_m < 2:
        raise ValueError("algorithm must be flat/hnsw and positive HNSW parameters")
    np = _numpy()
    faiss = _faiss(faiss_site)
    matrix = normalize_vectors(vectors)
    values = _validate_ids(ids, len(matrix))
    target = Path(directory).resolve()
    target.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    if algorithm == "flat":
        index = faiss.IndexFlatIP(matrix.shape[1])
    else:
        index = faiss.IndexHNSWFlat(matrix.shape[1], hnsw_m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 80
        index.hnsw.efSearch = ef_search
    index.add(matrix)
    build_seconds = time.perf_counter() - start
    faiss.write_index(index, str(target / "index.faiss"))
    np.save(target / "vectors.npy", matrix, allow_pickle=False)
    _write_json(target / "ids.json", values)
    report = {"schema_version": 1, "algorithm": algorithm, "metric": "normalized_cosine_inner_product",
              "count": len(values), "dimension": matrix.shape[1], "ef_search": ef_search if algorithm == "hnsw" else None,
              "hnsw_m": hnsw_m if algorithm == "hnsw" else None, "ef_construction": 80 if algorithm == "hnsw" else None,
              "build_seconds": build_seconds, "index_bytes": (target / "index.faiss").stat().st_size,
              "index_sha256": _sha256(target / "index.faiss"), "ids_sha256": _sha256(target / "ids.json"),
              "vectors_sha256": _sha256(target / "vectors.npy"), "input_fingerprint": input_fingerprint,
              "model": model, "faiss_version": getattr(faiss, "__version__", "unknown"),
              "faiss_module_path": str(Path(faiss.__file__).resolve()), "faiss_site": faiss_site}
    _write_json(target / "metadata.json", report)
    return report


class VectorIndex:
    """Load persisted index; text search uses precisely the recorded local model."""
    def __init__(self, directory: str | Path, *, faiss_site: str | None = None, encoder: Any = None):
        self.directory = Path(directory).resolve()
        self.metadata = json.loads((self.directory / "metadata.json").read_text(encoding="utf-8"))
        for name, field in [("index.faiss", "index_sha256"), ("ids.json", "ids_sha256"), ("vectors.npy", "vectors_sha256")]:
            if _sha256(self.directory / name) != self.metadata.get(field):
                raise ValueError(f"persisted {name} checksum mismatch")
        self.ids = json.loads((self.directory / "ids.json").read_text(encoding="utf-8"))
        _validate_ids(self.ids, self.metadata["count"])
        faiss = _faiss(faiss_site or self.metadata.get("faiss_site"))
        self.index = faiss.read_index(str(self.directory / "index.faiss"))
        if self.index.ntotal != len(self.ids) or self.index.d != self.metadata["dimension"]:
            raise ValueError("persisted FAISS shape differs from metadata")
        self.encoder = encoder

    def search_vector(self, query_vector: Any, k: int = 100) -> list[tuple[str, float]]:
        if not isinstance(k, int) or k < 0:
            raise ValueError("k must be a nonnegative integer")
        if k == 0 or not self.ids:
            return []
        np = _numpy()
        query = normalize_vectors(np.asarray(query_vector, dtype=np.float32).reshape(1, -1))
        if query.shape[1] != self.index.d:
            raise ValueError("query dimension differs from index")
        distances, positions = self.index.search(query, min(k, len(self.ids)))
        return [(self.ids[int(position)], float(score)) for position, score in zip(positions[0], distances[0]) if position >= 0]

    def search(self, query: str, k: int = 100) -> list[tuple[str, float]]:
        if not isinstance(k, int) or k < 0:
            raise ValueError("k must be a nonnegative integer")
        if k == 0 or not self.ids:
            return self.search_vector([], k)
        if self.encoder is None:
            model = self.metadata.get("model")
            if not model:
                raise ValueError("index has no local text encoder metadata; use search_vector")
            actual = model_fingerprint(model["path"])
            if actual["sha256"] != model["sha256"]:
                raise ValueError("local model changed since index creation")
            self.encoder = _load_encoder(Path(model["path"]))
        vectors = self.encoder.encode([str(query)], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        return self.search_vector(vectors[0], k)


def _query_rows(path: Path) -> list[dict]:
    rows = _jsonl(path)
    seen = set()
    for row in rows:
        query_id = str(row.get("query_id") or "")
        if not query_id or query_id in seen or not isinstance(row.get("query"), str):
            raise ValueError("queries require unique query_id and query text")
        seen.add(query_id)
        if row.get("candidate_ids"):
            raise ValueError("ANN evaluation requires observed corpus qrels (corpus_*.jsonl), not a fixed candidate pool")
        if not isinstance(row.get("qrels", {}), dict):
            raise ValueError("qrels must map product IDs to numeric relevance grades")
        for grade in row.get("qrels", {}).values():
            if not isinstance(grade, (int, float)) or not math.isfinite(grade):
                raise ValueError("non-finite/non-numeric relevance grade")
    return rows


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _evaluate(index: VectorIndex, queries: list[dict], vectors: Any, exact_neighbors: list[list[str]], *, repeats: int) -> dict:
    neighbor_recalls, positive_recalls, coverages, timings = [], [], [], []
    positive_query_count = 0
    # Warm-up is excluded; latency reports FAISS retrieval only, with cached query vectors.
    if len(vectors):
        index.search_vector(vectors[0], 100)
    for number, (row, vector) in enumerate(zip(queries, vectors)):
        result = index.search_vector(vector, 100)
        result_ids = [item[0] for item in result]
        exact_top = set(exact_neighbors[number][:10])
        if exact_top:
            neighbor_recalls.append(len(set(result_ids[:10]) & exact_top) / len(exact_top))
        qrels = {str(key): grade for key, grade in row.get("qrels", {}).items()}
        positives = {key for key, grade in qrels.items() if grade >= 2}
        if positives:
            positive_query_count += 1
            positive_recalls.append(len(set(result_ids) & positives) / len(positives))
        if result_ids:
            coverages.append(sum(product_id in qrels for product_id in result_ids) / len(result_ids))
        for _ in range(repeats):
            start = time.perf_counter()
            index.search_vector(vector, 100)
            timings.append((time.perf_counter() - start) * 1000)
    return {"queries": len(queries), "queries_with_known_positives": positive_query_count,
            "neighbor_recall_at_10": statistics.mean(neighbor_recalls) if neighbor_recalls else None,
            "known_positive_recall_at_100": statistics.mean(positive_recalls) if positive_recalls else None,
            "judged_coverage_at_100": statistics.mean(coverages) if coverages else None,
            "query_p50_ms": _percentile(timings, .50), "query_p95_ms": _percentile(timings, .95),
            "query_samples": len(timings), "latency_scope": "cached query vector normalization + FAISS search + ID mapping; no model inference",
            "recall_scope": "observed qrels only; grade >= 2 (E/S) positives; grade 1 complements excluded; unjudged products are unknown, not negatives",
            "neighbor_metric": "overlap of top 10 IDs from the serving k=100 result against FlatIP top 10",
            "averaging": "macro average across queries; positive recall excludes queries with no known E/S positives"}


def benchmark(catalog: str | Path, queries: str | Path, model_path: str | Path, output: str | Path, *,
              validation: str | Path, faiss_site: str | None = None, batch_size: int = 32,
              threads: int = 2, repeats: int = 3) -> dict:
    """Tune efSearch using validation, freeze it, then report untouched test metrics."""
    if threads < 1 or repeats < 1 or batch_size < 1:
        raise ValueError("threads, repeats and batch size must be positive")
    catalog_path, test_path, val_path = Path(catalog).resolve(), Path(queries).resolve(), Path(validation).resolve()
    if val_path == test_path or _sha256(val_path) == _sha256(test_path):
        raise ValueError("validation and test must be distinct files")
    target = Path(output).resolve()
    products = _jsonl(catalog_path)
    ids = _validate_ids([row.get("id", "") for row in products], len(products))
    if not products:
        raise ValueError("benchmark catalog must not be empty")
    texts = [" ".join(str(row.get(field, "")) for field in ("title", "brand", "description")).strip() for row in products]
    model = model_fingerprint(model_path)
    encoder = _load_encoder(Path(model["path"]), threads)
    faiss = _faiss(faiss_site)
    faiss.omp_set_num_threads(threads)
    catalog_vectors, catalog_cache = _embeddings(texts, model, target / "cache", encoder, batch_size)
    input_key = _fingerprint({"ids": ids, "texts": texts, "model": model["sha256"]})
    build_reports = {}
    indices = {}
    for algorithm in ("flat", "hnsw"):
        build_reports[algorithm] = build(target / algorithm, ids, catalog_vectors, model=model, algorithm=algorithm,
                                         faiss_site=faiss_site, input_fingerprint=input_key)
        indices[algorithm] = VectorIndex(target / algorithm, faiss_site=faiss_site, encoder=encoder)
    # Reading test text/labels is deliberately deferred until validation selection.
    val_rows = _query_rows(val_path)
    if not val_rows:
        raise ValueError("validation must contain at least one query")
    val_vectors, val_cache = _embeddings([row["query"] for row in val_rows], model, target / "cache", encoder, batch_size)
    val_exact = [[row[0] for row in indices["flat"].search_vector(vector, 100)] for vector in val_vectors]
    candidates = []
    for ef_search in (32, 64):
        indices["hnsw"].index.hnsw.efSearch = ef_search
        candidate = _evaluate(indices["hnsw"], val_rows, val_vectors, val_exact, repeats=repeats)
        candidates.append({"ef_search": ef_search, **candidate})
    # Predeclared policy: maximize geometric neighbor recall, smallest efSearch breaks ties.
    # Relevance labels and noisy wall-clock timing do not drive this selection.
    chosen = max(candidates, key=lambda row: (row["neighbor_recall_at_10"] or 0, -row["ef_search"]))
    selection = {"selected_ef_search": chosen["ef_search"], "policy": "maximize validation neighbor Recall@10, then smallest efSearch",
                 "validation_path": str(val_path), "validation_sha256": _sha256(val_path), "validation_results": candidates,
                 "test_labels_used_for_selection": False}
    _write_json(target / "selection.json", selection)
    selected_index = indices["hnsw"]
    selected_index.index.hnsw.efSearch = chosen["ef_search"]
    # Persist the frozen serving configuration before opening test judgments.
    selected_directory = target / "selected"
    selected_directory.mkdir(parents=True, exist_ok=True)
    for name in ("ids.json", "vectors.npy"):
        shutil.copyfile(target / "hnsw" / name, selected_directory / name)
    faiss.write_index(selected_index.index, str(selected_directory / "index.faiss"))
    selected_metadata = {**build_reports["hnsw"], "ef_search": chosen["ef_search"],
                         "index_sha256": _sha256(selected_directory / "index.faiss"), "selection_sha256": _sha256(target / "selection.json")}
    _write_json(selected_directory / "metadata.json", selected_metadata)
    test_rows = _query_rows(test_path)
    if {str(row["query_id"]) for row in val_rows} & {str(row["query_id"]) for row in test_rows}:
        raise ValueError("validation and test query IDs overlap")
    test_vectors, test_cache = _embeddings([row["query"] for row in test_rows], model, target / "cache", encoder, batch_size)
    test_exact = [[row[0] for row in indices["flat"].search_vector(vector, 100)] for vector in test_vectors]
    test_results = {}
    for algorithm in ("flat", "hnsw"):
        test_results[algorithm] = {**_evaluate(indices[algorithm], test_rows, test_vectors, test_exact, repeats=repeats),
                                   "build_seconds": build_reports[algorithm]["build_seconds"],
                                   "index_bytes": build_reports[algorithm]["index_bytes"]}
    report = {"schema_version": 1, "catalog_path": str(catalog_path), "catalog_sha256": _sha256(catalog_path),
              "test_path": str(test_path), "test_sha256": _sha256(test_path), "catalog_count": len(ids),
              "model": model, "threads": threads, "batch_size": batch_size, "embedding_cache": {"catalog": catalog_cache, "validation": val_cache, "test": test_cache},
              "selection": selection, "test": test_results, "build": build_reports,
              "serving_index": str(selected_directory), "limitations": ["CPU local-model benchmark; no production-scale claim", "known-positive recall is bounded by incomplete observed judgments", "latency excludes encoder inference"]}
    _write_json(target / "report.json", report)
    return report


def reevaluate_frozen(directory: str | Path, *, repeats: int = 3) -> dict:
    """Refresh metric instrumentation on frozen artifacts, without rebuilding/tuning.

    Useful when the shared metric contract is clarified while a long first-run
    encoding process is active. Every original data/model checksum must match.
    """
    if repeats < 1:
        raise ValueError("repeats must be positive")
    target = Path(directory).resolve()
    report = json.loads((target / "report.json").read_text(encoding="utf-8"))
    for path_field, sha_field in (("catalog_path", "catalog_sha256"), ("test_path", "test_sha256")):
        if _sha256(Path(report[path_field])) != report[sha_field]:
            raise ValueError(f"frozen {path_field} changed")
    selection = json.loads((target / "selection.json").read_text(encoding="utf-8"))
    if selection != report["selection"] or _sha256(Path(selection["validation_path"])) != selection["validation_sha256"]:
        raise ValueError("frozen validation or selection changed")
    model = model_fingerprint(report["model"]["path"])
    if model["sha256"] != report["model"]["sha256"]:
        raise ValueError("frozen local model changed")
    encoder = _load_encoder(Path(model["path"]), report["threads"])
    indices = {"flat": VectorIndex(target / "flat", encoder=encoder), "hnsw": VectorIndex(target / "selected", encoder=encoder)}
    if indices["hnsw"].metadata["ef_search"] != selection["selected_ef_search"]:
        raise ValueError("serving efSearch differs from frozen selection")
    faiss = _faiss(report["build"]["flat"].get("faiss_site"))
    faiss.omp_set_num_threads(report["threads"])
    rows = _query_rows(Path(report["test_path"]))
    vectors, cache = _embeddings([row["query"] for row in rows], model, target / "cache", encoder, report["batch_size"])
    exact = [[row[0] for row in indices["flat"].search_vector(vector, 100)] for vector in vectors]
    for algorithm, index in indices.items():
        report["test"][algorithm] = {**_evaluate(index, rows, vectors, exact, repeats=repeats),
                                      "build_seconds": report["build"][algorithm]["build_seconds"],
                                      "index_bytes": report["build"][algorithm]["index_bytes"]}
    report["embedding_cache"]["test"] = cache
    # Refresh validation instrumentation on exactly the same serialized graph.
    # Preserve selection.json as the immutable decision recorded before test.
    val_rows = _query_rows(Path(selection["validation_path"]))
    val_vectors, val_cache = _embeddings([row["query"] for row in val_rows], model, target / "cache", encoder, report["batch_size"])
    val_exact = [[row[0] for row in indices["flat"].search_vector(vector, 100)] for vector in val_vectors]
    validation_metrics = []
    for ef_search in (32, 64):
        indices["hnsw"].index.hnsw.efSearch = ef_search
        validation_metrics.append({"ef_search": ef_search, **_evaluate(indices["hnsw"], val_rows, val_vectors, val_exact, repeats=repeats)})
    indices["hnsw"].index.hnsw.efSearch = selection["selected_ef_search"]
    report["validation_metrics"] = validation_metrics
    report["embedding_cache"]["validation"] = val_cache
    report["metric_contract"] = {"positive_grade_minimum": 2, "positive_labels": ["E", "S"], "complement_label": "C (not counted for Recall)",
                                  "frozen_selection_reused": True, "index_rebuilt": False, "ef_search_reselected": False}
    report["selection_artifact_note"] = "selection.json is the original before-test decision; refreshed grade>=2 instrumentation is in validation_metrics/test"
    _write_json(target / "report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    command = subparsers.add_parser("benchmark", help="Compare normalized MiniLM FlatIP/HNSW on local data")
    command.add_argument("--catalog", required=True)
    command.add_argument("--queries", required=True, help="Untouched test queries with full available qrels")
    command.add_argument("--validation", help="Separate validation queries; defaults to corpus_val.jsonl or val.jsonl beside test")
    command.add_argument("--model", required=True, help="Local SentenceTransformer path; never downloads")
    command.add_argument("--output", required=True)
    command.add_argument("--faiss-site", help="Read-only local site-packages fallback for FAISS")
    command.add_argument("--batch-size", type=int, default=32)
    command.add_argument("--threads", type=int, default=2)
    command.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)
    query_path = Path(args.queries)
    validation = Path(args.validation) if args.validation else query_path.with_name("corpus_val.jsonl" if query_path.name.startswith("corpus_") else "val.jsonl")
    try:
        report = benchmark(args.catalog, args.queries, args.model, args.output, validation=validation,
                           faiss_site=args.faiss_site, batch_size=args.batch_size, threads=args.threads, repeats=args.repeats)
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps({"catalog_count": report["catalog_count"], "selection": report["selection"]["selected_ef_search"], "test": report["test"], "report": str(Path(args.output).resolve() / "report.json")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
