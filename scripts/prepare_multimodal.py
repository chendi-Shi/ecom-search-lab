"""Download a bounded, immutable real Fashion200K multi-view proxy dataset.

No credentials, generated images, ESCI joins, or remote Python execution.
Parquet row groups are fetched with verified HTTP byte ranges instead of the
whole 368 MB shard. Every transferred range and extracted image is inventoried.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import ssl
import threading
from urllib.parse import quote
from urllib.request import Request, urlopen

MODEL_REPO = "wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M"
MODEL_REVISION = "a2a8c6eaa2549ad66eb7c31b85022bf58273a26c"
DATA_REPO = "Marqo/fashion200k"
DATA_REVISION = "b9d0e6c9c14337620e442c93f40f7906942fe893"
SHARD = "data/data-00000-of-00009.parquet"
SHARD_BYTES = 368297137
SHARD_SHA256 = "f015251237b9a321a4b7e000e497c7168bf9aebc6e9cb600f414487c5c37ffff"
MODEL_FILES = ("config.json", "preprocessor_config.json", "tokenizer.json", "model.safetensors", "README.md")
LICENSE_SOURCES = {
    "TinyCLIP-MIT.txt": "https://raw.githubusercontent.com/microsoft/Cream/f3a5fd8aef84b6c9bc35a9179311aa15928773e5/TinyCLIP/LICENSE",
    "Fashion200K-Apache-2.0.txt": "https://raw.githubusercontent.com/xthan/fashion-200k/7a817ad0318dbb7771cec0a8e75d9f74987094f9/LICENSE",
}


def sha256(value):
    if isinstance(value, bytes):
        return hashlib.sha256(value).hexdigest()
    digest = hashlib.sha256()
    with Path(value).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class PublicDownloader:
    def __init__(self, maximum_bytes):
        self.maximum_bytes = maximum_bytes
        self.transferred = 0
        self.lock = threading.Lock()
        self.context = ssl.create_default_context()

    def fetch(self, url, *, start=None, length=None):
        headers = {"User-Agent": "ecom-search-lab-public-multimodal/1.0"}
        if start is not None:
            headers["Range"] = f"bytes={start}-{start + length - 1}"
        with urlopen(Request(url, headers=headers), context=self.context, timeout=45) as response:
            if start is not None:
                expected = f"bytes {start}-{start + length - 1}/"
                if response.status != 206 or not response.headers.get("Content-Range", "").startswith(expected):
                    raise ValueError("public server did not honor exact HTTP Range; full download refused")
            chunks, total = [], 0
            while chunk := response.read(1024 * 1024):
                with self.lock:
                    self.transferred += len(chunk)
                    if self.transferred > self.maximum_bytes:
                        raise ValueError("download byte budget exceeded")
                chunks.append(chunk)
                total += len(chunk)
                if length is not None and total > length:
                    raise ValueError("HTTP Range exceeded requested size")
            data = b"".join(chunks)
        if length is not None and len(data) != length:
            raise ValueError("HTTP Range byte count mismatch")
        return data


def download_licenses(downloader, output):
    output = Path(output).resolve()
    directory = output / "licenses"
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for name, url in LICENSE_SOURCES.items():
        data = downloader.fetch(url)
        path = directory / name
        path.write_bytes(data)
        records.append({"path": "licenses/" + name, "url": url, "bytes": len(data), "sha256": sha256(data)})
    write_json(output / "license_manifest.json", {"files": records, "note": "source notices retained verbatim; source product photographs retain third-party rights"})
    return records


def download_model(downloader, assets):
    info_url = f"https://huggingface.co/api/models/{MODEL_REPO}/revision/{MODEL_REVISION}?blobs=true"
    info = json.loads(downloader.fetch(info_url))
    if info["sha"] != MODEL_REVISION:
        raise ValueError("model revision mismatch")
    entries = {entry["rfilename"]: entry for entry in info["siblings"]}
    model_dir = assets / "tinyclip"
    records = []
    for name in MODEL_FILES:
        entry = entries[name]
        target = model_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/{quote(name, safe='/')}"
        expected_sha = entry.get("lfs", {}).get("sha256")
        if target.is_file() and target.stat().st_size == entry["size"] and (not expected_sha or sha256(target) == expected_sha):
            data = target.read_bytes()
        elif entry["size"] > 4 * 1024 * 1024:
            segments_dir = assets / "model-ranges"
            segments_dir.mkdir(parents=True, exist_ok=True)
            chunk_size = 4 * 1024 * 1024
            ranges = [(start, min(chunk_size, entry["size"] - start)) for start in range(0, entry["size"], chunk_size)]
            def segment(pair):
                start, length = pair
                path = segments_dir / f"{start:012d}-{length}.bin"
                if not path.is_file() or path.stat().st_size != length:
                    path.write_bytes(downloader.fetch(url + f"?download=true&segment={start}", start=start, length=length))
                print(json.dumps({"event": "model_segment", "start": start, "bytes": length}), flush=True)
                return path
            with ThreadPoolExecutor(max_workers=2) as pool:
                paths = list(pool.map(segment, ranges))
            with target.open("wb") as stream:
                for path in paths:
                    stream.write(path.read_bytes())
            data = target.read_bytes()
        else:
            data = downloader.fetch(url)
            target.write_bytes(data)
        if len(data) != entry["size"] or (expected_sha and sha256(data) != expected_sha):
            raise ValueError("model artifact SHA256/size mismatch")
        git_blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        if not expected_sha and git_blob != entry["blobId"]:
            raise ValueError("model small-file Git blob checksum mismatch")
        records.append({"path": name, "url": url, "bytes": len(data), "sha256": sha256(data), "publisher_sha256": expected_sha})
    write_json(model_dir / "source_manifest.json", {"repo": MODEL_REPO, "revision": MODEL_REVISION, "license": "MIT", "files": records})
    return model_dir, records


class RangeParquet(io.RawIOBase):
    def __init__(self, downloader, cache_dir):
        super().__init__()
        self.downloader, self.cache_dir = downloader, cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.position, self.records = 0, {}
        self.url = f"https://huggingface.co/datasets/{DATA_REPO}/resolve/{DATA_REVISION}/{SHARD}"

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position
    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else SHARD_BYTES + offset
        if self.position < 0 or self.position > SHARD_BYTES:
            raise ValueError("Parquet seek outside source")
        return self.position

    def read(self, size=-1):
        if size < 0:
            raise ValueError("unbounded Parquet reads disabled")
        size = min(size, SHARD_BYTES - self.position)
        if size == 0:
            return b""
        start = self.position
        chunks = []
        while size:
            length = min(size, 4 * 1024 * 1024)
            path = self.cache_dir / f"{start:012d}-{length}.bin"
            if path.is_file() and path.stat().st_size == length:
                data = path.read_bytes()
            else:
                data = self.downloader.fetch(self.url + f"?download=true&segment={start}&length={length}", start=start, length=length)
                path.write_bytes(data)
            self.records[(start, length)] = {"start": start, "bytes": length, "sha256": sha256(data), "file": str(path.resolve())}
            chunks.append(data)
            start += length
            size -= length
        data = b"".join(chunks)
        self.position += len(data)
        return data


def prepare(output, assets, products=200, max_transfer_mb=190):
    if not 30 <= products <= 250 or not 100 <= max_transfer_mb <= 500:
        raise ValueError("products must be 30..250 (images <=500), transfer budget 100..500 MB")
    from PIL import Image
    import pyarrow.parquet as pq
    output, assets = Path(output).resolve(), Path(assets).resolve()
    output.mkdir(parents=True, exist_ok=True)
    assets.mkdir(parents=True, exist_ok=True)
    downloader = PublicDownloader(max_transfer_mb * 1024 * 1024)
    model_dir, model_records = download_model(downloader, assets)
    licenses = download_licenses(downloader, output)
    # Retain the source card and original author license as local evidence.
    card_url = f"https://huggingface.co/datasets/{DATA_REPO}/resolve/{DATA_REVISION}/README.md"
    card = downloader.fetch(card_url)
    (output / "SOURCE_CARD.md").write_bytes(card)
    source = RangeParquet(downloader, assets / "fashion-ranges")
    parquet = pq.ParquetFile(source)
    groups = {}
    scanned_row_groups = []
    for group_number in range(parquet.num_row_groups):
        print(json.dumps({"event": "parquet_row_group", "row_group": group_number}), flush=True)
        rows = parquet.read_row_group(group_number, columns=["image", "category1", "category2", "category3", "text", "item_ID"]).to_pylist()
        scanned_row_groups.append(group_number)
        for row in rows:
            view_id = str(row["item_ID"])
            product_id = view_id.rsplit("_", 1)[0]
            image = row["image"].get("bytes")
            if not image:
                continue
            digest = sha256(image)
            members = groups.setdefault(product_id, [])
            if digest in {member["image_sha256"] for member in members} or len(members) >= 2:
                continue
            try:
                with Image.open(io.BytesIO(image)) as decoded:
                    if min(decoded.size) < 16:
                        continue
                    decoded.verify()
            except Exception:
                continue
            members.append({**row, "image_sha256": digest, "source_row_group": group_number})
        eligible = [key for key, members in groups.items() if len(members) >= 2]
        if len(eligible) >= products:
            break
    eligible = [key for key, members in groups.items() if len(members) >= 2]
    if len(eligible) < products:
        raise ValueError(f"only {len(eligible)} multi-view products within budget")
    # Sampling/splits use IDs only; never visual features or retrieval results.
    eligible.sort(key=lambda key: hashlib.sha256(("20261009:" + key).encode()).hexdigest())
    selected = eligible[:products]
    counts = {"train": int(products * .6), "val": int(products * .2)}
    counts["test"] = products - counts["train"] - counts["val"]
    images = output / "images"
    images.mkdir(parents=True, exist_ok=True)
    catalog, splits = [], {name: [] for name in counts}
    image_records = []
    for index, product_id in enumerate(selected):
        split = "train" if index < counts["train"] else "val" if index < counts["train"] + counts["val"] else "test"
        first, second = sorted(groups[product_id], key=lambda row: row["item_ID"])[:2]
        paths = []
        for row in (first, second):
            relative = f"images/{row['item_ID']}.jpg"
            (output / relative).write_bytes(row["image"]["bytes"])
            paths.append(relative)
            image_records.append({"path": relative, "item_ID": row["item_ID"], "product_id": product_id, "sha256": row["image_sha256"], "bytes": len(row["image"]["bytes"]), "source_row_group": row["source_row_group"]})
        item_id = "fashion200k:" + product_id
        # The source publishes original product titles and additional captions.
        caption = str(first["text"]).split("|", 1)[-1].strip()
        catalog.append({"id": item_id, "title": first["category3"], "text": caption, "source_text": first["text"], "image_path": paths[0], "image_sha256": first["image_sha256"], "category": first["category1"], "subcategory": first["category2"], "product_id": product_id, "synthetic": False, "source": DATA_REPO})
        splits[split].append({"query_id": "fashion-pair:" + product_id, "query": first["category3"], "image_path": paths[1], "image_sha256": second["image_sha256"], "target_id": item_id, "product_id": product_id, "split": split, "label_source": "same source product identity across distinct views; proxy, not human query relevance", "synthetic": False})
    for filename, rows in [("catalog.jsonl", catalog), *[(name + ".jsonl", rows) for name, rows in splits.items()]]:
        (output / filename).write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    manifest = {"schema_version": 1, "dataset": DATA_REPO, "revision": DATA_REVISION, "source_url": f"https://huggingface.co/datasets/{DATA_REPO}", "original_source": "https://github.com/xthan/fashion-200k", "license": "Apache-2.0 declared by original repository and HF dataset card; source product photographs retain third-party rights",
                "task": "multi-view paired-product identity retrieval proxy; no human search query qrels", "sampling": "first sufficient source row groups, then SHA256(seed:product_id), two distinct source images per product", "sampling_seed": 20261009,
                "catalog_count": len(catalog), "images": image_records, "split_counts": counts, "data_files": {name: {"sha256": sha256(output / name), "bytes": (output / name).stat().st_size} for name in ("catalog.jsonl", "train.jsonl", "val.jsonl", "test.jsonl")},
                "source_card_sha256": sha256(card), "shard": {"path": SHARD, "bytes": SHARD_BYTES, "publisher_sha256": SHARD_SHA256, "full_file_sha256_verified": False, "read_row_groups": scanned_row_groups, "downloaded_ranges": list(source.records.values())},
                "model": {"repo": MODEL_REPO, "revision": MODEL_REVISION, "license": "MIT", "local_path": str(model_dir), "files": model_records}, "license_files": licenses, "network_bytes_this_run": downloader.transferred, "data_range_bytes": sum(row["bytes"] for row in source.records.values()), "text_scope": "original source product title used for query; published extra image caption used for catalog text; neither is a human search query annotation"}
    write_json(output / "manifest.json", manifest)
    print(json.dumps({"event": "prepared", "catalog": len(catalog), "images": len(image_records), "network_bytes": downloader.transferred, "model": str(model_dir), "manifest": str(output / "manifest.json")}), flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/multimodal")
    parser.add_argument("--assets-root", default="outputs/multimodal-assets")
    parser.add_argument("--products", type=int, default=200)
    parser.add_argument("--max-transfer-mb", type=int, default=190)
    parser.add_argument("--licenses-only", action="store_true", help="Add fixed source license notices without altering the frozen dataset/checkpoint")
    arguments = parser.parse_args()
    if arguments.licenses_only:
        records = download_licenses(PublicDownloader(1024 * 1024), arguments.output)
        print(json.dumps({"event": "source_licenses_saved", "files": records}), flush=True)
    else:
        prepare(arguments.output, arguments.assets_root, arguments.products, arguments.max_transfer_mb)
