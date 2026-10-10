"""Optional real-image CPU TinyCLIP product-pair retrieval and learned fusion.

This independent Fashion200K proxy task has no ESCI image joins and no human
search qrels. Only a small diagonal projection and modality gate are trained;
the pretrained vision/text encoders remain frozen and local-only.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import statistics
import threading
import time

ROOT = Path(__file__).resolve().parents[1]


def _numpy():
    try:
        return importlib.import_module("numpy")
    except ImportError as exc:
        raise RuntimeError("optional multimodal mode requires numpy") from exc


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _key(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _rows(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def normalize(vectors):
    np = _numpy()
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] < 1 or not np.isfinite(matrix).all():
        raise ValueError("embeddings must be a finite two-dimensional matrix")
    lengths = np.linalg.norm(matrix.astype(np.float64), axis=1)
    if np.any(lengths == 0):
        raise ValueError("zero embedding cannot be normalized")
    return np.ascontiguousarray(matrix / lengths[:, None], dtype=np.float32)


def fuse(text_vectors, image_vectors, *, gate=.5, text_projection=None, image_projection=None):
    """Shared diagonal projection followed by normalized weighted embedding sum."""
    np = _numpy()
    if not isinstance(gate, (int, float)) or not math.isfinite(gate) or not 0 <= gate <= 1:
        raise ValueError("gate must be finite in [0, 1]")
    if text_vectors is None and image_vectors is None:
        raise ValueError("text and/or image is required")
    text = normalize(text_vectors) if text_vectors is not None else None
    image = normalize(image_vectors) if image_vectors is not None else None
    if text is not None and image is not None and text.shape != image.shape:
        raise ValueError("text/image embedding shapes differ")
    def project(matrix, weights):
        if matrix is None or weights is None:
            return matrix
        weights = np.asarray(weights, dtype=np.float32)
        if weights.shape != (matrix.shape[1],) or not np.isfinite(weights).all():
            raise ValueError("projection must contain one finite weight per dimension")
        return normalize(matrix * np.exp(np.clip(weights, -2, 2)))
    text, image = project(text, text_projection), project(image, image_projection)
    if text is None:
        return image
    if image is None:
        return text
    return normalize(gate * text + (1 - gate) * image)


def _model_info(model_path):
    path = Path(model_path).resolve()
    names = ["config.json", "preprocessor_config.json", "tokenizer.json", "model.safetensors"]
    if not path.is_dir() or any(not (path / name).is_file() for name in names):
        raise ValueError("model must be a complete local TinyCLIP directory; no automatic downloads")
    files = [{"path": name, "sha256": _sha(path / name), "bytes": (path / name).stat().st_size} for name in names]
    return {"path": str(path), "sha256": _key(files), "files": files, "encoder_frozen": True}


class ClipEncoder:
    def __init__(self, model_path, *, threads=2):
        if threads < 1:
            raise ValueError("threads must be positive")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        self.info = _model_info(model_path)
        try:
            self.torch = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
            self.Image = importlib.import_module("PIL.Image")
        except ImportError as exc:
            raise RuntimeError("optional multimodal mode requires CPU torch, transformers and Pillow") from exc
        self.torch.set_num_threads(threads)
        path = Path(self.info["path"])
        self.model = transformers.CLIPModel.from_pretrained(str(path), local_files_only=True, use_safetensors=True).eval().to("cpu")
        self.tokenizer = transformers.CLIPTokenizerFast(tokenizer_file=str(path / "tokenizer.json"), model_max_length=77)
        self.image_processor = transformers.CLIPImageProcessor.from_pretrained(str(path), local_files_only=True)

    def encode(self, texts, images, *, batch_size=8):
        np = _numpy()
        if len(texts) != len(images) or batch_size < 1:
            raise ValueError("texts and images must pair; batch size must be positive")
        text_batches, image_batches = [], []
        text_seconds, image_seconds = 0., 0.
        with self.torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                batch_texts = texts[start:start + batch_size]
                begin = time.perf_counter()
                tokens = self.tokenizer(batch_texts, padding=True, truncation=True, max_length=77, return_tensors="pt")
                text = self.model.get_text_features(**tokens).cpu().numpy()
                text_seconds += time.perf_counter() - begin
                begin = time.perf_counter()
                pictures = []
                for image_path in images[start:start + batch_size]:
                    with self.Image.open(image_path) as image:
                        pictures.append(image.convert("RGB"))
                pixels = self.image_processor(images=pictures, return_tensors="pt")["pixel_values"]
                image = self.model.get_image_features(pixel_values=pixels).cpu().numpy()
                for picture in pictures:
                    picture.close()
                image_seconds += time.perf_counter() - begin
                text_batches.append(text)
                image_batches.append(image)
        dimension = self.model.config.projection_dim
        empty = np.empty((0, dimension), dtype=np.float32)
        return {"text": normalize(np.concatenate(text_batches)) if text_batches else empty,
                "image": normalize(np.concatenate(image_batches)) if image_batches else empty}, {
                    "count": len(texts), "text_encode_seconds": text_seconds, "image_encode_seconds": image_seconds,
                    "text_mean_ms": text_seconds * 1000 / len(texts) if texts else None,
                    "image_mean_ms": image_seconds * 1000 / len(texts) if texts else None,
                    "scope": "batched CPU encoding including tokenization/image decode+preprocess; excludes initial model load"}

    def query(self, text=None, image_path=None):
        if text is None and image_path is None:
            raise ValueError("text and/or image is required")
        vectors = {}
        with self.torch.inference_mode():
            if text is not None:
                tokens = self.tokenizer([str(text)], padding=True, truncation=True, max_length=77, return_tensors="pt")
                vectors["text"] = normalize(self.model.get_text_features(**tokens).cpu().numpy())
            if image_path is not None:
                with self.Image.open(image_path) as image:
                    picture = image.convert("RGB")
                try:
                    pixels = self.image_processor(images=[picture], return_tensors="pt")["pixel_values"]
                    vectors["image"] = normalize(self.model.get_image_features(pixel_values=pixels).cpu().numpy())
                finally:
                    picture.close()
        return vectors


def _cache_embeddings(rows, root, encoder, directory, *, text_field, batch_size):
    np = _numpy()
    payload = [{"text": str(row[text_field]), "image_sha256": row["image_sha256"]} for row in rows]
    key = _key({"inputs": payload, "model": encoder.info["sha256"], "normalization": "l2-float32-v1", "max_text_tokens": 77})
    directory.mkdir(parents=True, exist_ok=True)
    path, metadata_path = directory / (key + ".npz"), directory / (key + ".json")
    paths = []
    for row in rows:
        path_image = (root / row["image_path"]).resolve()
        if not path_image.is_relative_to(root) or _sha(path_image) != row["image_sha256"]:
            raise ValueError("image path/checksum mismatch")
        paths.append(path_image)
    begin = time.perf_counter()
    if path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("fingerprint") != key or metadata.get("sha256") != _sha(path):
            raise ValueError("embedding cache integrity mismatch")
        with np.load(path, allow_pickle=False) as stored:
            vectors = {name: normalize(stored[name]) for name in ("text", "image")}
        if len(vectors["text"]) != len(rows) or vectors["text"].shape != vectors["image"].shape:
            raise ValueError("cached embedding shape mismatch")
        return vectors, {**metadata["timing"], "cache_hit": True, "cache_load_seconds": time.perf_counter() - begin, "fingerprint": key}
    vectors, timing = encoder.encode([str(row[text_field]) for row in rows], paths, batch_size=batch_size)
    np.savez(path, text=vectors["text"], image=vectors["image"])
    metadata = {"fingerprint": key, "sha256": _sha(path), "timing": timing, "count": len(rows), "dimension": vectors["text"].shape[1]}
    _save(metadata_path, metadata)
    return vectors, {**timing, "cache_hit": False, "fingerprint": key}


def _features(vectors, mode, parameters=None):
    if mode not in {"text", "image", "fusion", "learned"}:
        raise ValueError("mode must be text/image/fusion/learned")
    if mode in {"text", "image"}:
        if mode not in vectors:
            raise ValueError(f"{mode} input is required for mode={mode}")
        return normalize(vectors[mode])
    parameters = parameters or {}
    return fuse(vectors.get("text"), vectors.get("image"), gate=parameters.get("gate", .5) if mode == "learned" else .5,
                text_projection=parameters.get("text_projection") if mode == "learned" else None,
                image_projection=parameters.get("image_projection") if mode == "learned" else None)


def evaluate_pairs(query_vectors, gallery_vectors, query_rows, catalog_ids, *, mode, parameters=None):
    """Identity-pair Recall@K; never interpreted as human relevance accuracy."""
    if len(set(catalog_ids)) != len(catalog_ids):
        raise ValueError("duplicate catalog IDs")
    queries = _features(query_vectors, mode, parameters)
    gallery = _features(gallery_vectors, mode, parameters)
    if len(queries) != len(query_rows):
        raise ValueError("query embedding/row count mismatch")
    np = _numpy()
    positions = {item_id: index for index, item_id in enumerate(catalog_ids)}
    recalls = {k: [] for k in (1, 5, 10)}
    ranks, latencies = [], []
    for vector, row in zip(queries, query_rows):
        if row["target_id"] not in positions:
            raise ValueError("paired target missing from catalog")
        begin = time.perf_counter()
        scores = gallery @ vector
        ordering = np.argsort(-scores, kind="stable")
        latencies.append((time.perf_counter() - begin) * 1000)
        rank = int(np.where(ordering == positions[row["target_id"]])[0][0]) + 1
        ranks.append(rank)
        for k in recalls:
            recalls[k].append(rank <= min(k, len(catalog_ids)))
    ordered = sorted(latencies)
    return {"queries": len(query_rows), "catalog_count": len(catalog_ids),
            **{f"pair_recall_at_{k}": statistics.mean(values) if values else None for k, values in recalls.items()},
            "median_target_rank": statistics.median(ranks) if ranks else None,
            "cached_search_p50_ms": statistics.median(latencies) if latencies else None,
            "cached_search_p95_ms": ordered[min(len(ordered) - 1, math.ceil(.95 * len(ordered)) - 1)] if ordered else None,
            "latency_scope": "cached embeddings; exact cosine scoring + stable full-gallery sort; excludes encoder",
            "ground_truth": "source same-product identity across distinct images; proxy, not human query relevance"}


def _validate_split(rows, catalog_ids, seen_products=None):
    seen_products = seen_products or set()
    query_ids, products = set(), set()
    for row in rows:
        if not row.get("query_id") or row["query_id"] in query_ids or row.get("target_id") not in catalog_ids:
            raise ValueError("queries need unique query_id and a real catalog target")
        if not row.get("product_id") or row["product_id"] in seen_products or row["product_id"] in products:
            raise ValueError("product leakage/duplicates across train, validation, test")
        query_ids.add(row["query_id"])
        products.add(row["product_id"])
    return products


def _train_snapshots(train_vectors, gallery_vectors, train_rows, ids, *, steps=(0, 25, 75, 150), threads=2):
    np = _numpy()
    torch = importlib.import_module("torch")
    torch.set_num_threads(threads)
    torch.manual_seed(20261009)
    if not train_rows:
        raise ValueError("training queries must not be empty")
    positions = {item_id: index for index, item_id in enumerate(ids)}
    positive_positions = [positions[row["target_id"]] for row in train_rows]
    # Negatives are restricted to TRAIN products; no val/test product embeddings
    # participate in the optimization, even though the retrieval gallery is public.
    query = {name: torch.from_numpy(np.asarray(train_vectors[name], dtype=np.float32).copy()) for name in ("text", "image")}
    gallery = {name: torch.from_numpy(np.asarray(gallery_vectors[name][positive_positions], dtype=np.float32).copy()) for name in ("text", "image")}
    dimension = query["text"].shape[1]
    text_weight = torch.nn.Parameter(torch.zeros(dimension))
    image_weight = torch.nn.Parameter(torch.zeros(dimension))
    gate_logit = torch.nn.Parameter(torch.tensor(0.))
    optimizer = torch.optim.Adam([text_weight, image_weight, gate_logit], lr=.02)
    labels = torch.arange(len(train_rows))
    def projected(vectors):
        text = torch.nn.functional.normalize(vectors["text"] * torch.exp(torch.clamp(text_weight, -2, 2)), dim=1)
        image = torch.nn.functional.normalize(vectors["image"] * torch.exp(torch.clamp(image_weight, -2, 2)), dim=1)
        gate = torch.sigmoid(gate_logit)
        return torch.nn.functional.normalize(gate * text + (1 - gate) * image, dim=1)
    snapshots, trace = {}, []
    begin = time.perf_counter()
    for step in range(max(steps) + 1):
        if step in steps:
            snapshots[step] = {"steps": step, "gate": float(torch.sigmoid(gate_logit).detach()), "text_projection": text_weight.detach().tolist(), "image_projection": image_weight.detach().tolist()}
        if step == max(steps):
            break
        optimizer.zero_grad()
        similarity = projected(query) @ projected(gallery).T / .07
        loss = (torch.nn.functional.cross_entropy(similarity, labels) + torch.nn.functional.cross_entropy(similarity.T, labels)) / 2
        loss = loss + .01 * (text_weight.square().mean() + image_weight.square().mean())
        loss.backward()
        optimizer.step()
        if step % 25 == 0:
            trace.append({"step": step + 1, "train_loss": float(loss.detach())})
    return snapshots, {"train_seconds": time.perf_counter() - begin, "train_queries": len(train_rows), "trainable_parameters": 2 * dimension + 1,
                       "encoder_frozen": True, "optimizer": "Adam lr=.02; symmetric InfoNCE temperature=.07; diagonal L2=.01", "seed": 20261009, "training_trace": trace, "negative_scope": "TRAIN products only"}


def benchmark(data, model, output, *, threads=2, batch_size=8):
    root, target = Path(data).resolve(), Path(output).resolve()
    if threads < 1 or batch_size < 1:
        raise ValueError("threads and batch size must be positive")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for name, record in manifest["data_files"].items():
        if _sha(root / name) != record["sha256"]:
            raise ValueError("data manifest mismatch")
    catalog = _rows(root / "catalog.jsonl")
    ids = [row["id"] for row in catalog]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("catalog must have unique non-empty IDs")
    train_rows, val_rows = _rows(root / "train.jsonl"), _rows(root / "val.jsonl")
    train_products = _validate_split(train_rows, set(ids))
    val_products = _validate_split(val_rows, set(ids), train_products)
    if not val_rows:
        raise ValueError("validation requires queries")
    for row in train_rows + val_rows:
        positive = catalog[ids.index(row["target_id"])]
        if positive["image_sha256"] == row["image_sha256"]:
            raise ValueError("same image would make identity retrieval trivial")
    encoder = ClipEncoder(model, threads=threads)
    gallery_vectors, gallery_timing = _cache_embeddings(catalog, root, encoder, target / "cache", text_field="text", batch_size=batch_size)
    train_vectors, train_timing = _cache_embeddings(train_rows, root, encoder, target / "cache", text_field="query", batch_size=batch_size)
    val_vectors, val_timing = _cache_embeddings(val_rows, root, encoder, target / "cache", text_field="query", batch_size=batch_size)
    snapshots, training = _train_snapshots(train_vectors, gallery_vectors, train_rows, ids, threads=threads)
    val_results = []
    for step, parameters in snapshots.items():
        metrics = evaluate_pairs(val_vectors, gallery_vectors, val_rows, ids, mode="learned", parameters=parameters)
        val_results.append({"steps": step, "gate": parameters["gate"], **metrics})
    chosen = max(val_results, key=lambda row: (row["pair_recall_at_1"], row["pair_recall_at_5"], -row["steps"]))
    parameters = snapshots[chosen["steps"]]
    selected = target / "selected"
    selected.mkdir(parents=True, exist_ok=True)
    selection = {"selected_steps": chosen["steps"], "selected_gate": chosen["gate"], "policy": "maximize validation paired Recall@1, then Recall@5, then fewer optimization steps", "candidate_steps": [0, 25, 75, 150], "validation_results": val_results, "test_labels_used_for_selection": False, "validation_sha256": _sha(root / "val.jsonl")}
    # Decision and serving artifacts are persisted before parsing test queries.
    _save(target / "selection.json", selection)
    _save(selected / "model.json", parameters)
    _save(selected / "catalog.json", [{**row, "image_relative_path": row["image_path"], "image_path": str((root / row["image_path"]).resolve())} for row in catalog])
    np = _numpy()
    np.savez(selected / "gallery.npz", text=gallery_vectors["text"], image=gallery_vectors["image"])
    serving = {"schema_version": 1, "model": encoder.info, "data_root": str(root), "data_manifest_sha256": _sha(root / "manifest.json"), "selection_sha256": _sha(target / "selection.json"), "catalog_count": len(ids), "threads": threads,
               "artifacts": {name: _sha(selected / name) for name in ("model.json", "catalog.json", "gallery.npz")}, "task": manifest["task"]}
    _save(selected / "metadata.json", serving)
    test_rows = _rows(root / "test.jsonl")
    _validate_split(test_rows, set(ids), train_products | val_products)
    for row in test_rows:
        if catalog[ids.index(row["target_id"])]["image_sha256"] == row["image_sha256"]:
            raise ValueError("same query/catalog image would make identity retrieval trivial")
    test_vectors, test_timing = _cache_embeddings(test_rows, root, encoder, target / "cache", text_field="query", batch_size=batch_size)
    test_metrics = {mode: evaluate_pairs(test_vectors, gallery_vectors, test_rows, ids, mode=mode, parameters=parameters) for mode in ("text", "image", "fusion", "learned")}
    report = {"schema_version": 1, "task": manifest["task"], "data_manifest_sha256": _sha(root / "manifest.json"), "dataset_revision": manifest["revision"], "model": encoder.info, "catalog_count": len(ids), "source_images": len(manifest["images"]), "split_counts": manifest["split_counts"],
              "threads": threads, "batch_size": batch_size, "encoding": {"gallery": gallery_timing, "train": train_timing, "val": val_timing, "test": test_timing}, "training": training, "selection": selection, "test": test_metrics, "serving_index": str(selected),
              "limitations": ["paired same-product identity proxy, not human e-commerce search relevance", "small subset of early Fashion200K shard; category-biased, not a representative commerce corpus", "catalog text is published source text; its annotation method is unspecified in the dataset card; source short title/category3 is the proxy query", "no statistical significance or production performance claim", "encoder-pretraining overlap with these old web images is unknown"]}
    _save(target / "report.json", report)
    return report


class MultimodalIndex:
    """Reusable local UI API with arbitrary text, image, or joint queries."""
    def __init__(self, directory, *, encoder=None, data_root=None, model_path=None):
        self.directory = Path(directory).resolve()
        self.metadata = json.loads((self.directory / "metadata.json").read_text(encoding="utf-8"))
        for name, digest in self.metadata["artifacts"].items():
            if _sha(self.directory / name) != digest:
                raise ValueError("serving artifact checksum mismatch")
        self.parameters = json.loads((self.directory / "model.json").read_text(encoding="utf-8"))
        rows = json.loads((self.directory / "catalog.json").read_text(encoding="utf-8"))
        self.ids = [row["id"] for row in rows]
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("duplicate catalog IDs")
        self.data_root = None
        original_data_root = self.metadata.get("data_root")
        self._model_path_override = Path(model_path).resolve() if model_path is not None else None
        if original_data_root or data_root is not None:
            original_root = Path(original_data_root).resolve() if original_data_root else None
            local_root = ROOT / "data" / "multimodal"
            # A complete copied project owns its local assets even when the
            # recorded original directory remains accessible on this machine.
            # Never recover from a local checksum failure by reading old assets.
            if data_root is not None:
                self.data_root = Path(data_root).resolve()
            elif local_root.is_dir():
                self.data_root = local_root.resolve()
            else:
                self.data_root = original_root
            expected_manifest = self.metadata.get("data_manifest_sha256")
            if not self.data_root.is_dir() or (expected_manifest and _sha(self.data_root / "manifest.json") != expected_manifest):
                raise ValueError("relocated data manifest differs from the frozen source")
            for row in rows:
                if "image_path" not in row:
                    continue
                original_path = Path(row["image_path"])
                relative = row.get("image_relative_path")
                if relative is None:
                    if not original_path.is_absolute():
                        relative = original_path
                    else:
                        try:
                            relative = original_path.relative_to(original_root or self.data_root)
                        except ValueError:
                            relative = Path("images") / original_path.name
                path = (self.data_root / relative).resolve()
                if not path.is_relative_to(self.data_root):
                    raise ValueError("relocated image path escaped the frozen data directory")
                if not path.is_file() or (row.get("image_sha256") and _sha(path) != row["image_sha256"]):
                    raise ValueError("relocated catalog image checksum mismatch")
                row["image_path"] = str(path.resolve())
        self.catalog = {row["id"]: row for row in rows}
        np = _numpy()
        with np.load(self.directory / "gallery.npz", allow_pickle=False) as stored:
            self.vectors = {name: normalize(stored[name]) for name in ("text", "image")}
        if len(self.vectors["text"]) != len(self.ids) or self.vectors["image"].shape != self.vectors["text"].shape:
            raise ValueError("serving vector/catalog shape mismatch")
        self.encoder = encoder
        self._encoder_lock = threading.Lock()
        self._gallery_features = {mode: _features(self.vectors, mode, self.parameters) for mode in ("text", "image", "fusion", "learned")}

    def _ensure_encoder(self):
        if self.encoder is None:
            with self._encoder_lock:
                if self.encoder is None:
                    original = Path(self.metadata["model"]["path"])
                    local_path = ROOT / "outputs" / "multimodal-assets" / "tinyclip"
                    if self._model_path_override is not None:
                        model_path = self._model_path_override
                    else:
                        model_path = local_path if local_path.is_dir() else original
                    actual = _model_info(model_path)
                    if actual["sha256"] != self.metadata["model"]["sha256"]:
                        raise ValueError("local pretrained encoder changed")
                    self.resolved_model_path = str(model_path.resolve())
                    self.encoder = ClipEncoder(actual["path"], threads=self.metadata.get("threads", 2))

    def search_vectors(self, vectors, *, k=10, mode="learned", candidate_ids=None):
        if not isinstance(k, int) or k < 0:
            raise ValueError("k must be a nonnegative integer")
        mode = mode.lower()
        if mode not in self._gallery_features:
            raise ValueError("mode must be text/image/fusion/learned")
        if k == 0 or not self.ids:
            return []
        query = _features(vectors, mode, self.parameters)
        if query.shape[0] != 1 or query.shape[1] != self.vectors["text"].shape[1]:
            raise ValueError("search requires one query with matching dimension")
        np = _numpy()
        positions = range(len(self.ids))
        if candidate_ids is not None:
            candidates = list(candidate_ids)
            if len(candidates) != len(set(candidates)) or any(item_id not in self.catalog for item_id in candidates):
                raise ValueError("rerank IDs must be unique catalog IDs")
            allowed = set(candidates)
            positions = [index for index, item_id in enumerate(self.ids) if item_id in allowed]
        scores = self._gallery_features[mode] @ query[0]
        positions = sorted(positions, key=lambda index: (-float(scores[index]), self.ids[index]))[:k]
        return [(self.ids[index], float(scores[index])) for index in positions]

    def search(self, query="", image_path=None, k=10, mode="learned"):
        if not isinstance(k, int) or k < 0:
            raise ValueError("k must be a nonnegative integer")
        if k == 0 or not self.ids:
            return []
        text = str(query).strip() or None
        if text is None and image_path is None:
            raise ValueError("text and/or image is required")
        self._ensure_encoder()
        return self.search_vectors(self.encoder.query(text, image_path), k=k, mode=mode)

    def rerank(self, candidate_ids, query="", image_path=None, k=10, mode="learned"):
        if not isinstance(k, int) or k < 0:
            raise ValueError("k must be a nonnegative integer")
        if k == 0 or not self.ids:
            return []
        text = str(query).strip() or None
        if text is None and image_path is None:
            raise ValueError("text and/or image is required")
        self._ensure_encoder()
        return self.search_vectors(self.encoder.query(text, image_path), k=k, mode=mode, candidate_ids=candidate_ids)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    bench = subparsers.add_parser("benchmark")
    bench.add_argument("--data", default="data/multimodal")
    bench.add_argument("--model", default="outputs/multimodal-assets/tinyclip")
    bench.add_argument("--output", default="outputs/multimodal")
    bench.add_argument("--threads", type=int, default=2)
    bench.add_argument("--batch-size", type=int, default=8)
    search = subparsers.add_parser("search")
    search.add_argument("--index", default="outputs/multimodal/selected")
    search.add_argument("--query", default="")
    search.add_argument("--image")
    search.add_argument("--mode", choices=["text", "image", "fusion", "learned"], default="learned")
    search.add_argument("--k", type=int, default=10)
    args = parser.parse_args(argv)
    try:
        if args.command == "benchmark":
            result = benchmark(args.data, args.model, args.output, threads=args.threads, batch_size=args.batch_size)
            print(json.dumps({"test": result["test"], "training": result["training"], "selection": {"steps": result["selection"]["selected_steps"], "gate": result["selection"]["selected_gate"]}, "report": str(Path(args.output).resolve() / "report.json")}, ensure_ascii=False, indent=2))
        else:
            index = MultimodalIndex(args.index)
            results = index.search(args.query, args.image, args.k, args.mode)
            print(json.dumps([{"id": item_id, "score": score, **index.catalog[item_id]} for item_id, score in results], ensure_ascii=False, indent=2))
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
