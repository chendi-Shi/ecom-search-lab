"""Optional frozen, local-only CPU CrossEncoder with content-bound score cache."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import numbers
import os
from pathlib import Path
import sqlite3
import threading
import time
import urllib.request
import uuid

MODEL_ID = "cross-encoder/ms-marco-MiniLM-L6-v2"
MODEL_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
MODEL_WEIGHT_SHA256 = "821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae"
MODEL_FILES = ("config.json", "model.safetensors", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt")
DOCUMENT_POLICY = "title. Brand: brand; no description; v1"


def _key(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".ce-" + uuid.uuid4().hex[:12] + ".tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_model(directory, *, max_bytes=100 * 1024 * 1024, progress=None):
    """Explicitly download only the fixed official model files; serving is offline."""
    if not isinstance(max_bytes, int) or max_bytes < 1:
        raise ValueError("max_bytes must be a positive integer")
    target = Path(directory).resolve()
    target.mkdir(parents=True, exist_ok=True)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    api_url = f"https://huggingface.co/api/models/{MODEL_ID}/revision/{MODEL_REVISION}?blobs=true"
    with opener.open(api_url, timeout=45) as response:
        metadata = json.loads(response.read(2 * 1024 * 1024))
    if metadata.get("sha") != MODEL_REVISION or metadata.get("cardData", {}).get("license") != "apache-2.0":
        raise ValueError("official fixed revision/license metadata differs")
    source = {row["rfilename"]: row for row in metadata["siblings"]}
    names = MODEL_FILES + ("README.md",)
    if any(name not in source for name in names):
        raise ValueError("fixed model revision is incomplete")
    total = sum(source[name]["size"] for name in names)
    if total > max_bytes:
        raise ValueError("model files exceed the explicit download budget")
    records = []
    downloaded = 0
    for name in names:
        expected = source[name]
        url = f"https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/{name}?download=true"
        path = target / name
        # A prior interrupted run may already contain complete validated files.
        valid = path.is_file() and path.stat().st_size == expected["size"]
        if valid:
            if "lfs" in expected:
                valid = _sha(path) == expected["lfs"]["sha256"]
            else:
                blob = b"blob " + str(expected["size"]).encode() + b"\0" + path.read_bytes()
                valid = hashlib.sha1(blob).hexdigest() == expected["blobId"]
        if not valid:
            part = path.with_suffix(path.suffix + ".part")
            count, next_update = 0, 8 * 1024 * 1024
            try:
                with opener.open(url, timeout=45) as response, part.open("wb") as stream:
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        count += len(block)
                        if count > expected["size"] or downloaded + count > max_bytes:
                            raise ValueError("model download exceeded declared size/budget")
                        stream.write(block)
                        if progress and count >= next_update:
                            progress({"file": name, "downloaded_bytes": count, "expected_bytes": expected["size"]})
                            next_update += 8 * 1024 * 1024
                if count != expected["size"]:
                    raise ValueError("incomplete model download")
                if "lfs" in expected:
                    actual = _sha(part)
                    if actual != expected["lfs"]["sha256"]:
                        raise ValueError("official model weight SHA256 mismatch")
                else:
                    blob = b"blob " + str(count).encode() + b"\0" + part.read_bytes()
                    if hashlib.sha1(blob).hexdigest() != expected["blobId"]:
                        raise ValueError("official model Git blob checksum mismatch")
                part.replace(path)
                downloaded += count
            finally:
                part.unlink(missing_ok=True)
        records.append({"path": name, "bytes": path.stat().st_size, "sha256": _sha(path), "source_url": url,
                        "official_blob_id": expected["blobId"], "official_lfs_sha256": expected.get("lfs", {}).get("sha256")})
        if progress:
            progress({"file": name, "complete": True, "bytes": path.stat().st_size})
    manifest = {"schema_version": 1, "model_id": MODEL_ID, "revision": MODEL_REVISION, "license": "apache-2.0",
                "source": f"https://huggingface.co/{MODEL_ID}/tree/{MODEL_REVISION}", "files": records,
                "downloaded_bytes_this_run": downloaded, "required_bytes": total, "frozen_pretrained": True}
    manifest["artifact_fingerprint"] = _key({"model_id": MODEL_ID, "revision": MODEL_REVISION,
                                             "files": [{key: row[key] for key in ("path", "bytes", "sha256")} for row in records]})
    _save(target / "model_manifest.json", manifest)
    return manifest


def model_fingerprint(directory):
    """Bind serving to the prepared fixed revision and every effective local file."""
    root = Path(directory).resolve()
    manifest = json.loads((root / "model_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("model_id") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError("CrossEncoder must use the fixed pretrained model revision")
    expected_names = set(MODEL_FILES) | {"README.md"}
    records = manifest.get("files", [])
    if len(records) != len(expected_names) or {row["path"] for row in records} != expected_names:
        raise ValueError("CrossEncoder model manifest is incomplete or has duplicate artifacts")
    for path in root.iterdir():
        if path.is_file() and path.suffix in {".json", ".txt", ".bin", ".safetensors", ".model"} and path.name not in expected_names | {"model_manifest.json"}:
            raise ValueError("untracked model/tokenizer file could change CrossEncoder inference")
    files = []
    for row in records:
        path = root / row["path"]
        if not path.is_file() or path.stat().st_size != row["bytes"] or _sha(path) != row["sha256"]:
            raise ValueError("CrossEncoder model/tokenizer checksum mismatch")
        files.append({key: row[key] for key in ("path", "bytes", "sha256")})
    if next(row["sha256"] for row in files if row["path"] == "model.safetensors") != MODEL_WEIGHT_SHA256:
        raise ValueError("CrossEncoder weights differ from the fixed official pretrained weights")
    fingerprint = _key({"model_id": MODEL_ID, "revision": MODEL_REVISION, "files": files})
    if manifest.get("artifact_fingerprint") != fingerprint:
        raise ValueError("CrossEncoder artifact fingerprint mismatch")
    return {"model_id": MODEL_ID, "revision": MODEL_REVISION, "license": manifest["license"],
            "path": str(root), "artifact_fingerprint": fingerprint, "files": files, "frozen_pretrained": True}


def product_text(product):
    """The shared experiment contract: title. Brand: brand, without description."""
    if not isinstance(product, dict):
        raise ValueError("product must be a dictionary")
    title, brand = product.get("title", ""), product.get("brand", "")
    if not isinstance(title, str) or not isinstance(brand, str):
        raise ValueError("product title and brand must be text")
    return f"{title.strip()}. Brand: {brand.strip()}"


def _versions():
    result = {}
    for name in ("torch", "transformers", "tokenizers"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "unavailable"
    return result


class LocalCrossEncoder:
    """Raw single-label logits; no probability calibration or fitting on qrels.

    predict/score_pairs accept exact (query, document) strings. Optional predictor
    injection is for dependency-free tests; normal inference always uses CPU,
    local_files_only=True, safetensors and trust_remote_code=False.
    """
    def __init__(self, model_path, *, cache_dir=None, threads=2, max_length=128, batch_size=32, precision="float32", predictor=None):
        for name, value, maximum in (("threads", threads, 32), ("max_length", max_length, 512), ("batch_size", batch_size, 1024)):
            if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= maximum:
                raise ValueError(f"{name} must be an integer in [1, {maximum}]")
        if precision not in ("float32", "dynamic_int8"):
            raise ValueError("precision must be float32 or dynamic_int8")
        self.model_info = model_fingerprint(model_path)
        self.threads, self.max_length, self.batch_size = threads, max_length, batch_size
        self.precision = precision
        self.config = {"schema_version": 2, "model_id": MODEL_ID, "revision": MODEL_REVISION,
                       "artifact_fingerprint": self.model_info["artifact_fingerprint"],
                       "adapter_source_sha256": _sha(Path(__file__)),
                       "max_length": max_length, "batch_size": batch_size, "threads": threads,
                       "device": "cpu", "dtype": "float32" if precision == "float32" else "qint8 Linear; float32 embeddings",
                       "precision": precision, "quantized_engine": "onednn" if precision == "dynamic_int8" else None,
                       "quantization": "torch.ao.quantization.quantize_dynamic(Linear,qint8)" if precision == "dynamic_int8" else None,
                       "cache_policy": "ordered_complete_call" if precision == "dynamic_int8" else "exact_pair",
                       "activation": "raw single-label logit",
                       "truncation": "longest_first", "document_policy": DOCUMENT_POLICY,
                       "runtime_versions": _versions(), "frozen_pretrained": True}
        self.config["backend"] = "transformers-local-cpu" if predictor is None else "injected-test-predictor"
        self.fingerprint = _key(self.config)
        # Keep ordinary Windows workspace paths below legacy MAX_PATH. The
        # complete fingerprint remains checked inside both metadata and SQLite.
        self.cache_directory = Path(cache_dir).resolve() / self.fingerprint[:24] if cache_dir is not None else None
        self._predictor = predictor
        self._torch = self._model = self._tokenizer = None
        self._lock = threading.Lock()
        self.last_report = None
        if self.cache_directory is not None:
            self.cache_directory.mkdir(parents=True, exist_ok=True)
            path = self.cache_directory / "metadata.json"
            metadata = {"fingerprint": self.fingerprint, "config": self.config, "model": self.model_info,
                        "cache_contract": "complete ordered query/document call + config SHA256; finite score list and full row checksum" if precision == "dynamic_int8" else "SHA256 of exact query/document strings; finite score and row checksum"}
            if path.is_file():
                previous = json.loads(path.read_text(encoding="utf-8"))
                if previous.get("fingerprint") != self.fingerprint or previous.get("config") != self.config:
                    raise ValueError("CrossEncoder cache metadata mismatch")
            else:
                _save(path, metadata)

    def _load(self):
        if self._model is not None or self._predictor is not None:
            return
        # Recheck before the first inference in case assets changed after init.
        if model_fingerprint(self.model_info["path"])["artifact_fingerprint"] != self.model_info["artifact_fingerprint"]:
            raise ValueError("CrossEncoder assets changed before model loading")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        try:
            torch = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
        except ImportError as exc:
            raise RuntimeError("optional CrossEncoder requires CPU torch and transformers") from exc
        torch.set_num_threads(self.threads)
        tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_info["path"], local_files_only=True, trust_remote_code=False, use_fast=True)
        model = transformers.AutoModelForSequenceClassification.from_pretrained(self.model_info["path"], local_files_only=True, trust_remote_code=False, use_safetensors=True).eval().to(device="cpu", dtype=torch.float32)
        if model.config.num_labels != 1:
            raise ValueError("CrossEncoder must expose a single relevance logit")
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        if self.precision == "dynamic_int8":
            if "onednn" not in torch.backends.quantized.supported_engines:
                raise RuntimeError("dynamic_int8 requires the frozen oneDNN quantization engine")
            torch.backends.quantized.engine = "onednn"
            model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8, inplace=False).eval()
        self._tokenizer, self._model, self._torch = tokenizer, model, torch

    def _predict(self, pairs):
        self._load()
        if self._predictor is not None:
            values = self._predictor(pairs, batch_size=self.batch_size)
            return list(values)
        if self.precision == "dynamic_int8":
            self._torch.backends.quantized.engine = "onednn"
        scores = []
        with self._torch.inference_mode():
            for start in range(0, len(pairs), self.batch_size):
                batch = pairs[start:start + self.batch_size]
                tokens = self._tokenizer([pair[0] for pair in batch], [pair[1] for pair in batch], padding=True,
                                         truncation="longest_first", max_length=self.max_length, return_tensors="pt")
                logits = self._model(**tokens).logits
                if tuple(logits.shape) != (len(batch), 1):
                    raise ValueError("CrossEncoder returned unexpected score dimensions")
                scores.extend(logits[:, 0].cpu().tolist())
        return scores

    def _cache(self):
        if self.cache_directory is None:
            return None
        connection = sqlite3.connect(str(self.cache_directory / "scores.sqlite3"), timeout=30)
        try:
            connection.execute("CREATE TABLE IF NOT EXISTS metadata (name TEXT PRIMARY KEY, value TEXT NOT NULL)")
            connection.execute("CREATE TABLE IF NOT EXISTS scores (pair_key TEXT PRIMARY KEY, score REAL NOT NULL, checksum TEXT NOT NULL)")
            connection.execute("CREATE TABLE IF NOT EXISTS complete_calls (call_key TEXT PRIMARY KEY, scores_json TEXT NOT NULL, checksum TEXT NOT NULL)")
            row = connection.execute("SELECT value FROM metadata WHERE name='fingerprint'").fetchone()
            if row is None:
                connection.execute("INSERT INTO metadata VALUES ('fingerprint', ?)", (self.fingerprint,))
            elif row[0] != self.fingerprint:
                raise ValueError("CrossEncoder score database fingerprint mismatch")
            connection.commit()
            return connection
        except Exception:
            connection.close()
            raise

    def _score_checksum(self, pair_key, score):
        return _key({"fingerprint": self.fingerprint, "pair_key": pair_key, "score": score})

    def _call_checksum(self, call_key, scores_json):
        return _key({"fingerprint": self.fingerprint, "call_key": call_key, "scores_json": scores_json})

    def _score_complete_call(self, pairs):
        """Dynamic activation scales require preserving every batch context.

        No deduplication or individual-row reuse: input order, duplicate rows,
        neighboring documents and configured batch boundaries remain identical.
        SQLite transactions atomically persist the complete checked score list.
        """
        started = time.perf_counter()
        call_key = _key({"fingerprint": self.fingerprint, "ordered_pairs": pairs})
        with self._lock:
            connection = self._cache()
            try:
                stored = connection.execute("SELECT scores_json, checksum FROM complete_calls WHERE call_key=?", (call_key,)).fetchone() if connection else None
                cache_hit = stored is not None
                if cache_hit:
                    if stored[1] != self._call_checksum(call_key, stored[0]):
                        raise ValueError("CrossEncoder complete-call cache checksum mismatch")
                    predicted = json.loads(stored[0])
                    if not isinstance(predicted, list):
                        raise ValueError("CrossEncoder complete-call cache requires a score list")
                    prediction_seconds = 0.
                else:
                    prediction_started = time.perf_counter()
                    predicted = self._predict(pairs) if pairs else []
                    prediction_seconds = time.perf_counter() - prediction_started if pairs else 0.
                if len(predicted) != len(pairs):
                    raise ValueError("CrossEncoder complete-call score count differs from input pair count")
                scores = []
                for value in predicted:
                    if isinstance(value, bool) or not isinstance(value, numbers.Real):
                        raise ValueError("CrossEncoder complete-call score must be finite numeric")
                    score = float(value)
                    if not math.isfinite(score):
                        raise ValueError("CrossEncoder produced a non-finite complete-call score")
                    scores.append(score)
                if connection and not cache_hit:
                    payload = json.dumps(scores, allow_nan=False, separators=(",", ":"))
                    connection.execute("INSERT INTO complete_calls VALUES (?, ?, ?)", (call_key, payload, self._call_checksum(call_key, payload)))
                    connection.commit()
                self.last_report = {"fingerprint": self.fingerprint, "call_sha256": call_key, "cache_policy": "ordered_complete_call",
                                    "pairs": len(pairs), "unique_pairs": len({_key(pair) for pair in pairs}),
                                    "newly_predicted_pairs": 0 if cache_hit else len(pairs), "reused_pairs": len(pairs) if cache_hit else 0,
                                    "complete_call_cache_hit": cache_hit, "prediction_seconds": prediction_seconds,
                                    "total_seconds": time.perf_counter() - started,
                                    "latency_scope": "complete ordered call; batch tokenization + CPU model, includes first load; no individual-pair reuse",
                                    "config": self.config}
                if self.cache_directory is not None:
                    _save(self.cache_directory / "last_score_report.json", self.last_report)
                return scores
            finally:
                if connection:
                    connection.close()

    def score_pairs(self, pairs):
        pairs = list(pairs)
        for pair in pairs:
            if not isinstance(pair, (tuple, list)) or len(pair) != 2 or any(not isinstance(text, str) or not text.strip() for text in pair):
                raise ValueError("each pair must contain non-empty query and document strings")
        pairs = [(pair[0], pair[1]) for pair in pairs]
        if self.precision == "dynamic_int8":
            return self._score_complete_call(pairs)
        keys = [_key(pair) for pair in pairs]
        started = time.perf_counter()
        with self._lock:
            connection = self._cache()
            resolved, missing = {}, {}
            try:
                for key, pair in zip(keys, pairs):
                    if key in resolved or key in missing:
                        continue
                    stored = connection.execute("SELECT score, checksum FROM scores WHERE pair_key=?", (key,)).fetchone() if connection else None
                    if stored is not None:
                        score = float(stored[0])
                        if not math.isfinite(score) or stored[1] != self._score_checksum(key, score):
                            raise ValueError("CrossEncoder cached score integrity mismatch")
                        resolved[key] = score
                    else:
                        missing[key] = pair
                prediction_started = time.perf_counter()
                predicted = self._predict(list(missing.values())) if missing else []
                prediction_seconds = time.perf_counter() - prediction_started if missing else 0.
                if len(predicted) != len(missing):
                    raise ValueError("CrossEncoder score count differs from input pair count")
                for key, value in zip(missing, predicted):
                    score = float(value)
                    if not math.isfinite(score):
                        raise ValueError("CrossEncoder produced a non-finite score")
                    resolved[key] = score
                if connection:
                    connection.executemany("INSERT INTO scores VALUES (?, ?, ?)",
                                           [(key, resolved[key], self._score_checksum(key, resolved[key])) for key in missing])
                    connection.commit()
                self.last_report = {"fingerprint": self.fingerprint, "pairs": len(pairs), "unique_pairs": len(set(keys)),
                                    "newly_predicted_pairs": len(missing), "reused_pairs": len(pairs) - len(missing),
                                    "prediction_seconds": prediction_seconds, "total_seconds": time.perf_counter() - started,
                                    "latency_scope": "batch tokenization + CPU model, includes first-load time when needed; cached rows skip inference",
                                    "config": self.config}
                if self.cache_directory is not None:
                    _save(self.cache_directory / "last_score_report.json", self.last_report)
                return [resolved[key] for key in keys]
            finally:
                if connection:
                    connection.close()

    def close(self):
        """Drop optional encoder references; SQLite is closed after each call.

        The adapter remains reusable and reloads local weights on a later miss.
        """
        with self._lock:
            self._model = self._tokenizer = self._torch = None

    def predict(self, pairs):
        return self.score_pairs(pairs)

    def rerank(self, query, documents, k=10):
        if not isinstance(k, int) or isinstance(k, bool) or k < 0:
            raise ValueError("k must be a nonnegative integer")
        documents = list(documents)
        ids = [document.get("id") if isinstance(document, dict) else None for document in documents]
        if any(not isinstance(item_id, str) or not item_id for item_id in ids) or len(set(ids)) != len(ids):
            raise ValueError("reranking needs unique non-empty product IDs")
        if k == 0 or not documents:
            return []
        scores = self.score_pairs([(query, product_text(document)) for document in documents])
        return sorted(zip(ids, scores), key=lambda row: (-row[1], row[0]))[:k]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--model", default="outputs/cross-encoder-assets/ms-marco-MiniLM-L6-v2")
    score = subparsers.add_parser("score")
    score.add_argument("--model", default="outputs/cross-encoder-assets/ms-marco-MiniLM-L6-v2")
    score.add_argument("--query", required=True)
    score.add_argument("--document", action="append", required=True)
    score.add_argument("--cache", default="outputs/cross-encoder-cache")
    score.add_argument("--threads", type=int, default=2)
    score.add_argument("--max-length", type=int, default=128)
    score.add_argument("--batch-size", type=int, default=32)
    score.add_argument("--precision", choices=("float32", "dynamic_int8"), default="float32")
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare_model(args.model, progress=lambda row: print(json.dumps(row), flush=True))
        else:
            adapter = LocalCrossEncoder(args.model, cache_dir=args.cache, threads=args.threads, max_length=args.max_length, batch_size=args.batch_size, precision=args.precision)
            scores = adapter.score_pairs([(args.query, document) for document in args.document])
            result = {"scores": scores, "report": adapter.last_report}
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
