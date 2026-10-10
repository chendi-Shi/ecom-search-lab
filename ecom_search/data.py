"""Small, dependency-free data contracts and content fingerprints."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unicodedata


def load_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{number}: expected a JSON object")
            rows.append(value)
    return rows


def save_json(path, obj):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True,
                                 allow_nan=False) + "\n", encoding="utf-8")


def save_jsonl(path, rows):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                    allow_nan=False) + "\n")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_query(query):
    return " ".join(unicodedata.normalize("NFKC", query).casefold().split())


def validate_dataset(catalog, splits):
    products = set()
    for product in catalog:
        product_id = product.get("id")
        if not isinstance(product_id, str) or not product_id or product_id in products:
            raise ValueError("Product ids must be unique nonempty strings")
        if not isinstance(product.get("title"), str) or not isinstance(product.get("brand", ""), str):
            raise ValueError("Product title and brand must be strings")
        products.add(product_id)
    if not products:
        raise ValueError("Catalog is empty")
    query_ids, normalized_splits = set(), {}
    for split, records in splits.items():
        for record in records:
            query_id, query = record.get("query_id"), record.get("query")
            if not isinstance(query_id, str) or not query_id or query_id in query_ids:
                raise ValueError("Query ids must be unique across all splits")
            query_ids.add(query_id)
            if not isinstance(query, str) or not normalized_query(query):
                raise ValueError("Query must be a nonempty string")
            normalized = normalized_query(query)
            if normalized in normalized_splits and normalized_splits[normalized] != split:
                raise ValueError("Normalized query crosses dataset splits")
            normalized_splits[normalized] = split
            if record.get("split", split) != split:
                raise ValueError("Record split differs from its file")
            candidates, qrels = record.get("candidate_ids"), record.get("qrels")
            if not isinstance(candidates, list) or not candidates or any(
                    not isinstance(pid, str) or pid not in products for pid in candidates):
                raise ValueError("Candidates must be nonempty catalog product ids")
            if len(candidates) != len(set(candidates)):
                raise ValueError("Candidate ids must be unique")
            if not isinstance(qrels, dict) or set(qrels) != set(candidates):
                raise ValueError("Fixed candidate pools require exactly one judgment per candidate")
            if any(pid not in products or type(grade) is not int or not 0 <= grade <= 3
                   for pid, grade in qrels.items()):
                raise ValueError("Qrels must use catalog product ids and integer grades 0..3")
            full_qrels = record.get("full_qrels", {})
            if not isinstance(full_qrels, dict) or any(
                    pid not in products or type(grade) is not int or not 0 <= grade <= 3
                    for pid, grade in full_qrels.items()):
                raise ValueError("Full qrels must use catalog product ids and integer grades 0..3")


def load_dataset(directory):
    root = Path(directory)
    catalog = load_jsonl(root / "catalog.jsonl")
    splits = {split: load_jsonl(root / f"{split}.jsonl") for split in ("train", "val", "test")}
    validate_dataset(catalog, splits)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for name, fingerprint in manifest.get("files", {}).items():
            # Manifest filenames must remain inside the dataset directory.
            if Path(name).name != name or name in {".", ".."}:
                raise ValueError("Dataset manifest contains an unsafe filename")
            target = root / name
            if (not target.is_file() or file_sha256(target) != fingerprint["sha256"]
                    or target.stat().st_size != fingerprint["bytes"]):
                raise ValueError(f"Dataset fingerprint differs: {name}")
    return catalog, splits["train"], splits["val"], splits["test"]
