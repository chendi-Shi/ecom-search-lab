"""Optional CPU Query -> Semantic ID generation, with no nearest-neighbor fallback.

The RQ codebooks fit only train-relevant product vectors. A small conditional
GRU learns SID sequences from train query/product labels. Constrained beam search
generates catalog SID paths. Exact cosine is used only inside a generated SID's
collision bucket; generation probability orders the buckets.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

import numpy as np
import torch
from torch import nn

from .data import file_sha256, load_jsonl, normalized_query, save_json, save_jsonl
from .index import ProductIndex, catalog_digest
from .vectors import _embeddings, _load_encoder, model_fingerprint, normalize_vectors

SCHEMA_VERSION = 1


def _fingerprint(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def _kmeans(matrix, count=8, seed=42, iterations=30):
    """Seeded train-only Lloyd kmeans; small codebooks, float64 distances."""
    x = np.asarray(matrix, dtype=np.float64)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all() or not 1 <= count <= len(x):
        raise ValueError("invalid training vectors/codebook size")
    rng = np.random.default_rng(seed)
    centers = [x[int(rng.integers(len(x)))].copy()]
    nearest = np.sum((x - centers[0]) ** 2, axis=1)
    for _ in range(1, count):
        total = float(nearest.sum())
        index = int(rng.choice(len(x), p=nearest / total)) if total > 0 else int(rng.integers(len(x)))
        centers.append(x[index].copy())
        nearest = np.minimum(nearest, np.sum((x - centers[-1]) ** 2, axis=1))
    centers = np.asarray(centers)
    for _ in range(iterations):
        distance = np.maximum(0, np.sum(x * x, axis=1)[:, None]
                              + np.sum(centers * centers, axis=1)[None, :] - 2 * x @ centers.T)
        assignment = distance.argmin(axis=1)
        updated = centers.copy()
        for code in range(count):
            members = x[assignment == code]
            if len(members):
                updated[code] = members.mean(axis=0)
            else:
                updated[code] = x[int(np.argmax(distance.min(axis=1)))]
        if np.max(np.abs(updated - centers)) < 1e-6:
            centers = updated
            break
        centers = updated
    return centers.astype(np.float32)


def fit_residual_codebooks(train_vectors, levels=4, codebook_size=8, seed=42):
    residual = np.asarray(train_vectors, dtype=np.float32).copy()
    books = []
    for level in range(levels):
        centers = _kmeans(residual, codebook_size, seed + level)
        assignment = _nearest_codes(residual, centers)
        books.append(centers)
        residual -= centers[assignment]
    return np.asarray(books, dtype=np.float32)


def _nearest_codes(vectors, centers):
    distance = (np.sum(vectors * vectors, axis=1)[:, None]
                + np.sum(centers * centers, axis=1)[None, :] - 2 * vectors @ centers.T)
    return np.argmin(distance, axis=1)


def assign_semantic_ids(vectors, codebooks):
    residual = np.asarray(vectors, dtype=np.float32).copy()
    books = np.asarray(codebooks, dtype=np.float32)
    if residual.ndim != 2 or books.ndim != 3 or residual.shape[1] != books.shape[2]:
        raise ValueError("vector/codebook shape mismatch")
    if not np.isfinite(residual).all() or not np.isfinite(books).all():
        raise ValueError("non-finite vectors/codebooks")
    codes = []
    for centers in books:
        assignment = _nearest_codes(residual, centers)
        codes.append(assignment)
        residual -= centers[assignment]
    return np.stack(codes, axis=1).astype(np.int32)


class QuerySIDDecoder(nn.Module):
    """Frozen dense query representation conditions a learned autoregressive GRU."""
    def __init__(self, dimension=384, levels=4, codebook_size=8, hidden=128, token_dim=32):
        super().__init__()
        self.dimension, self.levels, self.codebook_size = dimension, levels, codebook_size
        self.hidden_size, self.token_dim = hidden, token_dim
        self.context = nn.Sequential(nn.Linear(dimension, hidden), nn.Tanh())
        self.token_embedding = nn.Embedding(levels * codebook_size + 1, token_dim)
        self.cell = nn.GRUCell(token_dim + hidden, hidden)
        self.output = nn.Linear(hidden, codebook_size)
        self.dropout = nn.Dropout(0.1)
        self.bos = levels * codebook_size

    def step(self, context, hidden, previous):
        hidden = self.cell(torch.cat((self.token_embedding(previous), context), dim=1), hidden)
        return self.output(self.dropout(hidden)), hidden

    def forward(self, query_vectors, target_codes):
        context = self.context(query_vectors)
        hidden = context
        previous = torch.full((len(query_vectors),), self.bos, dtype=torch.long, device=query_vectors.device)
        logits = []
        for level in range(self.levels):
            output, hidden = self.step(context, hidden, previous)
            logits.append(output)
            previous = target_codes[:, level] + level * self.codebook_size
        return torch.stack(logits, dim=1)

    def config(self):
        return {"dimension": self.dimension, "levels": self.levels,
                "codebook_size": self.codebook_size, "hidden": self.hidden_size, "token_dim": self.token_dim}


class SIDRetriever:
    """In-memory constrained decoder; inject vectors to avoid encoder timing."""
    def __init__(self, model, ids, vectors, codes, beam_width=30):
        self.model = model.eval()
        self.ids = list(ids)
        self.vectors = normalize_vectors(vectors)
        self.codes = np.asarray(codes, dtype=np.int32)
        if len(self.ids) != len(self.vectors) or self.codes.shape != (len(self.ids), model.levels):
            raise ValueError("IDs/vectors/codes shape mismatch")
        if len(set(self.ids)) != len(self.ids) or np.any(self.codes < 0) or np.any(self.codes >= model.codebook_size):
            raise ValueError("duplicate IDs or invalid Semantic ID tokens")
        self.beam_width = beam_width
        self.children: dict[tuple, set] = {}
        self.buckets: dict[tuple, list[int]] = {}
        for position, row in enumerate(self.codes):
            sid = tuple(map(int, row))
            self.buckets.setdefault(sid, []).append(position)
            for level, code in enumerate(sid):
                self.children.setdefault(sid[:level], set()).add(code)

    @torch.inference_mode()
    def search_vector(self, query_vector, k=100, beam_width=None):
        if type(k) is not int or not 1 <= k <= 1000:
            raise ValueError("k must be an integer in [1,1000]")
        beam_width = self.beam_width if beam_width is None else beam_width
        if type(beam_width) is not int or not 1 <= beam_width <= 100:
            raise ValueError("beam_width must be an integer in [1,100]")
        start = time.perf_counter()
        q = normalize_vectors(np.asarray(query_vector, dtype=np.float32).reshape(1, -1))
        if q.shape[1] != self.model.dimension:
            raise ValueError("query dimension mismatch")
        context = self.model.context(torch.from_numpy(q))
        # Each entry holds a generated prefix, unmasked log probability, and state.
        beam = [((), 0.0, context[0].clone())]
        raw_prefix, raw_hidden = (), context.clone()
        for level in range(self.model.levels):
            previous = torch.tensor([self.model.bos if not prefix else (level - 1) * self.model.codebook_size + prefix[-1]
                                     for prefix, _, _ in beam], dtype=torch.long)
            logits, states = self.model.step(context.expand(len(beam), -1),
                                            torch.stack([state for _, _, state in beam]), previous)
            log_probs = torch.log_softmax(logits, dim=-1)
            expanded = []
            for row, (prefix, score, _) in enumerate(beam):
                for code in sorted(self.children.get(prefix, ())):
                    expanded.append((prefix + (code,), score + float(log_probs[row, code]), states[row].clone()))
            beam = sorted(expanded, key=lambda item: (-item[1], item[0]))[:beam_width]
            if not beam:
                raise ValueError("catalog trie has no legal path")
            raw_previous = torch.tensor([self.model.bos if not raw_prefix else (level - 1) * self.model.codebook_size + raw_prefix[-1]])
            raw_logits, raw_hidden = self.model.step(context, raw_hidden, raw_previous)
            raw_prefix += (int(raw_logits.argmax(dim=-1)[0]),)
        raw_invalid = raw_prefix not in self.buckets
        result, terminal_rows, dot_count = [], [], 0
        for sid_rank, (sid, log_probability, _) in enumerate(beam):
            positions = self.buckets[sid]
            if len(positions) > 1:
                scores = self.vectors[positions] @ q[0]
                dot_count += len(positions)
                ordered = sorted(zip(positions, scores), key=lambda row: (-float(row[1]), self.ids[row[0]]))
            else:
                ordered = [(positions[0], None)]
            # Ordinal API score preserves probability-first bucket ordering and
            # cosine-only within-bucket ordering. It is not a calibrated relevance probability.
            for position, cosine in ordered:
                result.append((self.ids[position], -float(len(result))))
            terminal_rows.append({"sid": list(sid), "log_probability": log_probability, "bucket_size": len(positions)})
        diagnostics = {"generated_sids": len(beam), "expanded_product_candidates": len(result),
                       "collision_dot_products": dot_count, "unconstrained_greedy_invalid_id": raw_invalid,
                       "constrained_invalid_id_count": sum(sid not in self.buckets for sid, _, _ in beam),
                       "retrieval_ms": 1000 * (time.perf_counter() - start), "terminals": terminal_rows,
                       "score_semantics": "ordinal rank; SID buckets ordered by generated log probability, cosine only within collision buckets"}
        return result[:k], diagnostics


class GenerativeIndex:
    def __init__(self, directory, *, encoder=None, model_path=None, threads=2):
        start = time.perf_counter()
        self.directory = Path(directory).resolve()
        self.metadata = json.loads((self.directory / "metadata.json").read_text(encoding="utf-8"))
        if self.metadata.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported generative artifact schema")
        for name, expected in self.metadata["files"].items():
            if Path(name).name != name or file_sha256(self.directory / name) != expected:
                raise ValueError(f"generative artifact checksum mismatch: {name}")
        torch.set_num_threads(threads)
        from threadpoolctl import threadpool_limits
        self._thread_limiter = threadpool_limits(limits=threads)
        self.model = QuerySIDDecoder(**self.metadata["decoder_config"])
        self.model.load_state_dict(torch.load(self.directory / "decoder.pt", map_location="cpu", weights_only=True))
        ids = json.loads((self.directory / "ids.json").read_text(encoding="utf-8"))
        vectors = np.load(self.directory / "vectors.npy", allow_pickle=False)
        codebooks = np.load(self.directory / "codebooks.npy", allow_pickle=False)
        codes = np.load(self.directory / "codes.npy", allow_pickle=False)
        if not np.array_equal(assign_semantic_ids(vectors, codebooks), codes):
            raise ValueError("saved codes do not match product vectors/codebooks")
        self.retriever = SIDRetriever(self.model, ids, vectors, codes, self.metadata["selected_beam_width"])
        self.encoder = encoder
        self.model_path = Path(model_path or self.metadata["embedding_model"]["path"])
        if model_path is None and not self.model_path.is_dir():
            self.model_path = Path(__file__).resolve().parents[1] / "outputs" / "text-assets" / "minilm"
        self.load_seconds = time.perf_counter() - start
        self.threads = threads

    def _encode(self, query):
        if not isinstance(query, str) or not query.strip() or len(query) > 2000:
            raise ValueError("query must be nonempty text of at most 2000 characters")
        if self.encoder is None:
            actual = model_fingerprint(self.model_path)
            if actual["sha256"] != self.metadata["embedding_model"]["sha256"]:
                raise ValueError("embedding model differs from the frozen generative model")
            self.encoder = _load_encoder(self.model_path, self.threads)
        return self.encoder.encode([query], convert_to_numpy=True, normalize_embeddings=True,
                                   show_progress_bar=False)[0]

    def search(self, query, k=100):
        return self.search_with_diagnostics(query, k)[0]

    def search_with_diagnostics(self, query, k=100):
        start = time.perf_counter()
        vector = self._encode(query)
        encode_ms = 1000 * (time.perf_counter() - start)
        result, diagnostics = self.retriever.search_vector(vector, k)
        diagnostics.update({"encoder_ms": encode_ms, "end_to_end_ms": 1000 * (time.perf_counter() - start)})
        return result, diagnostics


def _validate_records(rows, valid_ids, split):
    seen, texts = set(), set()
    for row in rows:
        if row.get("split", split) != split or not isinstance(row.get("query"), str):
            raise ValueError(f"invalid {split} record")
        qid = row.get("query_id")
        text = normalized_query(row["query"])
        if not qid or qid in seen or not text or text in texts:
            raise ValueError(f"duplicate/empty {split} query")
        seen.add(qid)
        texts.add(text)
        qrels = row.get("qrels")
        if not isinstance(qrels, dict) or any(pid not in valid_ids or type(grade) is not int or grade not in range(4)
                                               for pid, grade in qrels.items()):
            raise ValueError("qrels must be catalog product IDs with grades 0..3")
    return seen, texts


def _positive_recall(ranked, qrels, k):
    positives = {pid for pid, grade in qrels.items() if grade >= 2}
    return len(set(ranked[:k]) & positives) / len(positives) if positives else None


def evaluate_generated(retriever, rows, vectors, beam, *, include_rows=False):
    results, recall_values = [], {10: [], 100: []}
    diagnostics = []
    for row, vector in zip(rows, vectors):
        hits, info = retriever.search_vector(vector, 100, beam)
        ids = [pid for pid, _ in hits]
        metrics = {}
        for k in recall_values:
            value = _positive_recall(ids, row["qrels"], k)
            metrics[f"known_positive_recall_at_{k}"] = value
            if value is not None:
                recall_values[k].append(value)
        diagnostics.append(info)
        results.append({"query_id": row["query_id"], "query": row["query"], "ranking": ids,
                        "metrics": metrics, "diagnostics": info})
    report = {"queries": len(rows), "queries_with_known_positives": len(recall_values[10]),
              **{f"known_positive_recall_at_{k}": statistics.mean(values) if values else None
                 for k, values in recall_values.items()},
              "mean_generated_sids": statistics.mean(d["generated_sids"] for d in diagnostics),
              "mean_product_candidates": statistics.mean(d["expanded_product_candidates"] for d in diagnostics),
              "mean_collision_dot_products": statistics.mean(d["collision_dot_products"] for d in diagnostics),
              "unconstrained_greedy_invalid_id_rate": statistics.mean(d["unconstrained_greedy_invalid_id"] for d in diagnostics),
              "constrained_invalid_id_rate": sum(d["constrained_invalid_id_count"] for d in diagnostics)
                / max(1, sum(d["generated_sids"] for d in diagnostics)),
              "retrieval_p50_ms": float(np.percentile([d["retrieval_ms"] for d in diagnostics], 50)),
              "retrieval_p95_ms": float(np.percentile([d["retrieval_ms"] for d in diagnostics], 95)),
              "latency_scope": "cached query embedding + GRU constrained beam + collision disambiguation; excludes MiniLM encoding"}
    return (report, results) if include_rows else report


def _evaluate_baseline(rows, rank_fn):
    recalls = {10: [], 100: []}
    times = []
    for row in rows:
        start = time.perf_counter()
        ranked = rank_fn(row)
        times.append(1000 * (time.perf_counter() - start))
        for k in recalls:
            value = _positive_recall(ranked, row["qrels"], k)
            if value is not None:
                recalls[k].append(value)
    return {"queries": len(rows), "queries_with_known_positives": len(recalls[10]),
            **{f"known_positive_recall_at_{k}": statistics.mean(values) if values else None for k, values in recalls.items()},
            "retrieval_p50_ms": float(np.percentile(times, 50)), "retrieval_p95_ms": float(np.percentile(times, 95))}


def train_experiment(data_dir, vector_dir, bm25_index, output_dir, *, epochs=(30, 60, 100),
                     beams=(10, 30), seed=42, threads=2, levels=4, codebook_size=8):
    """Train/val select, freeze artifacts, then open final test labels once."""
    started = time.perf_counter()
    data_dir, vector_dir, output_dir = map(Path, (data_dir, vector_dir, output_dir))
    if output_dir.exists() and any(path.name != "cache" for path in output_dir.iterdir()):
        raise FileExistsError("choose an empty generative output directory; existing artifacts are never overwritten")
    if not epochs or any(type(e) is not int or e < 1 for e in epochs) or tuple(sorted(set(epochs))) != tuple(epochs):
        raise ValueError("epochs must be an increasing unique positive grid")
    if not beams or any(type(b) is not int or not 1 <= b <= 100 for b in beams):
        raise ValueError("beam grid must contain integers 1..100")
    torch.set_num_threads(threads)
    from threadpoolctl import threadpool_limits
    thread_limiter = threadpool_limits(limits=threads)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    np.random.seed(seed)
    vector_metadata = json.loads((vector_dir / "metadata.json").read_text(encoding="utf-8"))
    for name, field in (("vectors.npy", "vectors_sha256"), ("ids.json", "ids_sha256")):
        if file_sha256(vector_dir / name) != vector_metadata[field]:
            raise ValueError("input Flat vector checksum mismatch")
    ids = json.loads((vector_dir / "ids.json").read_text(encoding="utf-8"))
    vectors = normalize_vectors(np.load(vector_dir / "vectors.npy", allow_pickle=False))
    if len(ids) != len(vectors) or len(set(ids)) != len(ids):
        raise ValueError("input vector mapping mismatch")
    catalog = load_jsonl(data_dir / "catalog.jsonl")
    if ids != [row["id"] for row in catalog]:
        raise ValueError("input Flat vector IDs/order differ from catalog")
    actual_input = _fingerprint({"ids": ids, "texts": [" ".join(str(row.get(field, "")) for field in ("title", "brand", "description")).strip()
                                                       for row in catalog], "model": vector_metadata["model"]["sha256"]})
    if actual_input != vector_metadata["input_fingerprint"]:
        raise ValueError("Flat vectors were encoded from different catalog text")
    model_info = model_fingerprint(vector_metadata["model"]["path"])
    if model_info["sha256"] != vector_metadata["model"]["sha256"]:
        raise ValueError("local embedding model differs from Flat index")
    train_path, val_path, test_path = [data_dir / f"corpus_{split}.jsonl" for split in ("train", "val", "test")]
    fingerprints = {path.name: file_sha256(path) for path in (train_path, val_path, test_path, data_dir / "catalog.jsonl")}
    train, val = load_jsonl(train_path), load_jsonl(val_path)
    train_seen, train_texts = _validate_records(train, set(ids), "train")
    val_seen, val_texts = _validate_records(val, set(ids), "val")
    if train_seen & val_seen or train_texts & val_texts or not train or not val:
        raise ValueError("train and validation must be nonempty/disjoint")
    id_position = {pid: position for position, pid in enumerate(ids)}
    fit_ids = sorted({pid for row in train for pid, grade in row["qrels"].items() if grade >= 2})
    if len(fit_ids) < codebook_size:
        raise ValueError("too few train-relevant products for codebook fitting")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"phase": "encoder_and_queries", "pid": os.getpid(), "threads": threads}), flush=True)
    encoder = _load_encoder(Path(model_info["path"]), threads)
    train_queries, train_cache = _embeddings([row["query"] for row in train], model_info, output_dir / "cache", encoder, 32)
    val_queries, val_cache = _embeddings([row["query"] for row in val], model_info, output_dir / "cache", encoder, 32)
    quantizer_start = time.perf_counter()
    print(json.dumps({"phase": "fit_train_only_codebooks", "fit_products": len(fit_ids)}), flush=True)
    codebooks = fit_residual_codebooks(vectors[[id_position[pid] for pid in fit_ids]], levels, codebook_size, seed)
    codes = assign_semantic_ids(vectors, codebooks)
    quantizer_seconds = time.perf_counter() - quantizer_start
    query_positions, target_codes, target_weights = [], [], []
    for number, row in enumerate(train):
        for pid, grade in sorted(row["qrels"].items()):
            if grade >= 2:
                query_positions.append(number)
                target_codes.append(codes[id_position[pid]])
                target_weights.append(1.0 if grade == 3 else 0.5)
    x = torch.from_numpy(train_queries[np.asarray(query_positions)].copy())
    y = torch.from_numpy(np.asarray(target_codes, dtype=np.int64))
    weights = torch.tensor(target_weights, dtype=torch.float32)
    model = QuerySIDDecoder(vectors.shape[1], levels, codebook_size)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count > 10_000_000:
        raise ValueError("decoder exceeds the CPU experiment parameter limit")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.0001)
    criterion = nn.CrossEntropyLoss(reduction="none")
    shuffle_generator = torch.Generator().manual_seed(seed)
    train_history, selection, best = [], [], None
    training_start = time.perf_counter()
    print(json.dumps({"phase": "training", "targets": len(x), "parameters": parameter_count,
                      "codebook_seconds": quantizer_seconds}), flush=True)
    for epoch in range(1, max(epochs) + 1):
        model.train()
        permutation = torch.randperm(len(x), generator=shuffle_generator)
        loss_sum = 0.0
        for batch in permutation.split(128):
            optimizer.zero_grad(set_to_none=True)
            logits = model(x[batch], y[batch])
            per_target = criterion(logits.reshape(-1, codebook_size), y[batch].reshape(-1)).reshape(-1, levels).mean(dim=1)
            loss = (per_target * weights[batch]).sum() / weights[batch].sum()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(batch)
        train_history.append({"epoch": epoch, "weighted_teacher_forcing_cross_entropy": loss_sum / len(x)})
        if epoch % 10 == 0:
            print(json.dumps({"phase": "training_epoch", **train_history[-1]}), flush=True)
        if epoch in epochs:
            retriever = SIDRetriever(model, ids, vectors, codes)
            for beam in beams:
                result = evaluate_generated(retriever, val, val_queries, beam)
                candidate = {"epochs": epoch, "beam_width": beam, "validation": result}
                selection.append(candidate)
                key = (result["known_positive_recall_at_10"] or 0.0, -epoch, -beam)
                if best is None or key > best["key"]:
                    best = {"key": key, "epochs": epoch, "beam_width": beam,
                            "state": copy.deepcopy(model.state_dict())}
            print(json.dumps({"phase": "validation", "epoch": epoch, "loss": train_history[-1],
                              "selected_epochs_so_far": best["epochs"], "selected_beam_so_far": best["beam_width"]}), flush=True)
    training_seconds = time.perf_counter() - training_start
    model.load_state_dict(best["state"])
    model.eval()
    collision_retriever = SIDRetriever(model, ids, vectors, codes, best["beam_width"])
    bucket_sizes = [len(members) for members in collision_retriever.buckets.values()]
    collision_info = {"catalog_items": len(ids), "unique_semantic_ids": len(bucket_sizes),
                      "items_in_collision_buckets": sum(size for size in bucket_sizes if size > 1),
                      "item_collision_rate": sum(size for size in bucket_sizes if size > 1) / len(ids),
                      "excess_item_collision_rate": (len(ids) - len(bucket_sizes)) / len(ids),
                      "max_bucket_size": max(bucket_sizes), "mean_bucket_size": statistics.mean(bucket_sizes)}
    save_json(output_dir / "selection.json", {"grid": {"epochs": list(epochs), "beam_width": list(beams)},
              "criterion": "highest validation known-positive Recall@10; ties fewer epochs then smaller beam",
              "selected_epochs": best["epochs"], "selected_beam_width": best["beam_width"],
              "results": selection, "test_labels_used": False})
    torch.save(model.state_dict(), output_dir / "decoder.pt")
    np.save(output_dir / "codebooks.npy", codebooks, allow_pickle=False)
    np.save(output_dir / "codes.npy", codes, allow_pickle=False)
    np.save(output_dir / "vectors.npy", vectors, allow_pickle=False)
    save_json(output_dir / "ids.json", ids)
    save_json(output_dir / "fit_product_ids.json", fit_ids)
    files = {name: file_sha256(output_dir / name) for name in (
        "decoder.pt", "codebooks.npy", "codes.npy", "vectors.npy", "ids.json", "fit_product_ids.json", "selection.json")}
    metadata = {"schema_version": SCHEMA_VERSION, "decoder_config": model.config(), "parameter_count": parameter_count,
                "selected_epochs": best["epochs"], "selected_beam_width": best["beam_width"], "files": files,
                "embedding_model": model_info, "catalog_sha256": catalog_digest(catalog), "data_fingerprints": fingerprints,
                "seed": seed, "threads": threads, "fit_product_count": len(fit_ids),
                "source_code_sha256": file_sha256(Path(__file__)),
                "quantizer_fit_scope": "unique E/S judged products from corpus_train.jsonl only; val/test labels never fit codebooks",
                "training_scope": "query embedding -> autoregressive SID targets from train E/S products; Exact weight=1, Substitute=.5",
                "frozen_before_test": True, "collision": collision_info,
                "generation_score": "beam sum of unmasked per-code log probabilities; bucket-local cosine disambiguation; no dense fallback"}
    save_json(output_dir / "metadata.json", metadata)
    # All selection and persisted model/codebooks exist before final test labels open.
    for name, expected in fingerprints.items():
        if file_sha256(data_dir / name) != expected:
            raise ValueError("data changed after model selection")
    test = load_jsonl(test_path)
    test_seen, test_texts = _validate_records(test, set(ids), "test")
    if test_seen & (train_seen | val_seen) or test_texts & (train_texts | val_texts) or not test:
        raise ValueError("test must be nonempty and disjoint")
    test_queries, test_cache = _embeddings([row["query"] for row in test], model_info, output_dir / "cache", encoder, 32)
    loaded = GenerativeIndex(output_dir, encoder=encoder, threads=threads)
    generated, test_rows = evaluate_generated(loaded.retriever, test, test_queries, best["beam_width"], include_rows=True)
    with ProductIndex(bm25_index, cache_size=0) as lexical:
        if catalog_digest(lexical.catalog()) != metadata["catalog_sha256"]:
            raise ValueError("BM25 baseline uses a different catalog")
        bm25 = _evaluate_baseline(test, lambda row: [product["id"] for product in lexical.search(row["query"], 100)])
    query_by_id = {row["query_id"]: vector for row, vector in zip(test, test_queries)}
    def flat_rank(row):
        scores = vectors @ query_by_id[row["query_id"]]
        positions = np.argpartition(-scores, min(99, len(scores) - 1))[:100]
        positions = sorted(positions, key=lambda position: (-float(scores[position]), ids[position]))
        return [ids[position] for position in positions]
    flat = _evaluate_baseline(test, flat_rank)
    artifacts_bytes = sum((output_dir / name).stat().st_size for name in files)
    import psutil
    memory = psutil.Process().memory_info()
    report = {"schema_version": SCHEMA_VERSION, "experiment": "true query-conditioned GRU SID generation, CPU",
              "split_queries": {"train": len(train), "val": len(val), "test": len(test)},
              "train_targets": len(x), "train_relevant_products": len(fit_ids), "parameter_count": parameter_count,
              "selected_epochs": best["epochs"], "selected_beam_width": best["beam_width"],
              "test": {"generative_gru_sid": generated, "bm25_title_brand": bm25, "flat_cosine_same_minilm": flat},
              "collision": collision_info, "training_history": train_history,
              "resources": {"threads": threads, "device": "cpu", "quantizer_fit_seconds": quantizer_seconds,
                            "training_and_validation_seconds": training_seconds, "total_seconds": time.perf_counter() - started,
                            "decoder_load_seconds": loaded.load_seconds, "persistent_artifacts_bytes": artifacts_bytes,
                            "decoder_weights_bytes": (output_dir / "decoder.pt").stat().st_size,
                            "collision_vectors_bytes": (output_dir / "vectors.npy").stat().st_size,
                            "rss_bytes_at_end": memory.rss, "peak_working_set_bytes": getattr(memory, "peak_wset", None),
                            "rss_scope": "whole training process including frozen encoder and Python; disk bytes reported separately"},
              "embedding_cache": {"train": train_cache, "validation": val_cache, "test": test_cache},
              "data_fingerprints": fingerprints, "model_metadata_sha256": file_sha256(output_dir / "metadata.json"),
              "limits": ["250 training queries are insufficient for industrial generative search; low recall is reported honestly",
                         "Known-positive E/S recall uses incomplete observed judgments; unjudged items are unknown",
                         "Catalog is the same frozen 15,171-product public subset for every baseline",
                         "MiniLM remains a frozen dense input encoder; the trained GRU genuinely generates code tokens",
                         "Full product vectors remain stored for collision disambiguation; no index-free memory-saving claim",
                         "Codebooks fit train-relevant items only; all catalog items enter a label-free inference prefix trie",
                         "Runtime excludes MiniLM inference in retrieval-only comparisons; no online-impact or billion-scale claim"]}
    save_json(output_dir / "report.json", report)
    save_jsonl(output_dir / "per_query_test.jsonl", test_rows)
    lines = ["# CPU 生成式商品检索", "", "冻结 MiniLM Query 向量 → 学习到的 GRU → 4 层 Semantic ID；合法 trie 解码；碰撞 bucket 内精确 cosine 消歧。", "",
             "| 同商品池模型 | Known-positive Recall@10 | Known-positive Recall@100 |", "|---|---:|---:|"]
    for name, metrics in report["test"].items():
        lines.append(f"| {name} | {metrics['known_positive_recall_at_10']:.6f} | {metrics['known_positive_recall_at_100']:.6f} |")
    lines.extend(["", f"验证选择 epochs={best['epochs']}，beam={best['beam_width']}；解码器参数 {parameter_count:,}。",
                  f"训练加验证 {training_seconds:.2f}s；码本拟合 {quantizer_seconds:.2f}s；碰撞向量存储 {(output_dir / 'vectors.npy').stat().st_size:,} bytes。", "",
                  "仅 train E/S 商品拟合 RQ 码本和训练 query→SID；val 选择参数；选定模型/码本/指纹先保存，随后一次 final test。",
                  "未标商品不是负样本。小数据结果不能代表千亿候选、线上转化收益或工业级生成式检索。"])
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
