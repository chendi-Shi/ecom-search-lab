"""Verify the accuracy ZIP and run its relocated CPU search without holdout reads.

The parent verifies every archive byte against CRC and DELIVERY_MANIFEST SHA256.
The isolated child runs real inference from the retained extracted copy, denying
network, original-project files, evaluation labels/results and feature caches.
Only the CPU interpreter and its installed dependencies may live elsewhere.
"""
from __future__ import annotations

import argparse
import binascii
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import time
import traceback
from urllib.parse import urlsplit
from urllib.request import url2pathname
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "ecom-search-lab/"
MARKER = "ACCURACY_DELIVERY_RUNTIME="


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def relative_name(value):
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError("ZIP paths must be relative POSIX paths")
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or any(part in {".", ".."} for part in parsed.parts) or parsed.as_posix() != value:
        raise ValueError("unsafe ZIP path")
    return parsed


def extract_verified(archive_path, destination):
    """Hash held-out files as opaque bytes only; never deserialize their labels."""
    manifest_name = PREFIX + "DELIVERY_MANIFEST.json"
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(set(names)) != len(names) or manifest_name not in names:
            raise ValueError("ZIP requires unique names and a delivery manifest")
        for info in infos:
            name = info.filename.rstrip("/") if info.is_dir() else info.filename
            relative_name(name)
            if name != PREFIX.rstrip("/") and not name.startswith(PREFIX):
                raise ValueError("ZIP entry escaped the project prefix")
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("ZIP symlinks are unsupported")
        manifest_bytes = archive.read(manifest_name)
        manifest = json.loads(manifest_bytes)
        if manifest.get("full") is not True or manifest.get("accuracy") is not True:
            raise ValueError("the full accuracy delivery ZIP is required")
        rows = manifest.get("files")
        if not isinstance(rows, list) or not rows:
            raise ValueError("empty delivery manifest")
        records = {}
        for row in rows:
            name = row["path"]
            relative_name(name)
            if name in records or type(row.get("bytes")) is not int or row["bytes"] < 0:
                raise ValueError("invalid/duplicate manifest entry")
            if "local-profiles" in name.lower() or name.startswith("outputs/uploads/"):
                raise ValueError("personal profiles/uploads must not be delivered")
            records[name] = row
        actual = {info.filename for info in infos if not info.is_dir()}
        if {PREFIX + name for name in records} | {manifest_name} != actual:
            raise ValueError("ZIP files differ from its manifest")
        required = {
            "ecom_search/accuracy.py", "ecom_search/cross_encoder.py", "ecom_search/ranking.py",
            "ecom_search/cli.py", "scripts/prepare_accuracy_retrieval.py", "scripts/verify_accuracy_delivery.py",
            "outputs/accuracy-v1/model.json", "outputs/accuracy-v1/retrieval/products.sqlite",
            "outputs/accuracy-v1/retrieval/metadata.json", "outputs/accuracy-v1/retrieval/pool_strategy.json",
            "outputs/cross-encoder-assets/ms-marco-MiniLM-L6-v2/model_manifest.json",
            "outputs/cross-encoder-assets/ms-marco-MiniLM-L6-v2/model.safetensors",
            "run-accuracy.ps1",
        }
        if required - records.keys():
            raise ValueError("missing accuracy serving artifacts: " + ", ".join(sorted(required - records.keys())))
        total = 0
        for name, row in records.items():
            info = archive.getinfo(PREFIX + name)
            if info.file_size != row["bytes"]:
                raise ValueError("manifest byte count mismatch: " + name)
            output = destination / name
            if not output.resolve().is_relative_to(destination.resolve()):
                raise ValueError("extraction escaped its destination")
            output.parent.mkdir(parents=True, exist_ok=True)
            digest, crc, size = hashlib.sha256(), 0, 0
            with archive.open(info) as source, output.open("xb") as target:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(block)
                    crc = binascii.crc32(block, crc)
                    size += len(block)
                    target.write(block)
            if size != row["bytes"] or digest.hexdigest() != row["sha256"] or crc & 0xFFFFFFFF != info.CRC:
                raise ValueError("delivery SHA256/CRC mismatch: " + name)
            if sha256(output) != row["sha256"]:
                raise ValueError("extracted file mismatch: " + name)
            total += size
        (destination / "DELIVERY_MANIFEST.json").write_bytes(manifest_bytes)
        return manifest, {"files": len(records), "uncompressed_bytes": total, "all_sha256_verified": True,
                          "all_crc_verified": True, "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest()}


def prohibited_runtime_file(relative):
    """No serving result may come from stored evaluation answers or labels."""
    parts = [part.lower() for part in relative.parts]
    name = parts[-1] if parts else ""
    if parts[:2] == ["data", "esci-v2"] and name in {"test.jsonl", "corpus_test.jsonl"}:
        return True
    if parts[:2] == ["outputs", "accuracy-v1"]:
        if any("cache" in part or part == "features" for part in parts[2:]):
            return True
        if (name.startswith(("test", "corpus-test")) or "holdout" in name
                or name in {"report.json", "end_to_end_report.json", "final-lock.json"}):
            return True
    return False


def runtime_check(project, original):
    project, original = Path(project).resolve(), Path(original).resolve()
    sys.path.insert(0, str(project))
    counts = {"relocated_file_opens": 0, "blocked_original_reads": 0,
              "blocked_evaluation_reads": 0, "blocked_network_attempts": 0,
              "blocked_loopback_capability_probes": 0}
    local_probes = []

    def audit(event, args):
        if event == "socket.bind" and len(args) > 1 and args[1] == ("::1", 0):
            caller = sys._getframe(1)
            if (caller.f_globals.get("__name__") == "urllib3.util.connection"
                    and caller.f_code.co_name == "_has_ipv6"):
                # urllib3 probes local IPv6 support during import. Keep the
                # socket blocked: this is not an outbound connection and does
                # not need to count as an attempted remote model download.
                counts["blocked_loopback_capability_probes"] += 1
                local_probes.append({"event": event, "address": ["::1", 0], "blocked": True,
                                     "caller": "urllib3.util.connection._has_ipv6",
                                     "source_line": caller.f_lineno})
                raise OSError("local IPv6 capability probing is disabled during delivery verification")
        if event in {"socket.connect", "socket.getaddrinfo", "socket.sendto", "socket.bind"}:
            counts["blocked_network_attempts"] += 1
            raise RuntimeError("network is disabled during delivery verification")
        if event not in {"open", "sqlite3.connect"} or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        value = os.fsdecode(args[0])
        if event == "sqlite3.connect" and value == ":memory:":
            return
        if event == "sqlite3.connect" and value.startswith("file:"):
            uri = urlsplit(value)
            value = url2pathname(("//" + uri.netloc if uri.netloc else "") + uri.path)
        path = Path(value).resolve()
        if path.is_relative_to(project):
            if prohibited_runtime_file(path.relative_to(project)):
                counts["blocked_evaluation_reads"] += 1
                raise PermissionError("serving must not read holdout labels/results or feature caches")
            counts["relocated_file_opens"] += 1
        elif path.is_relative_to(original):
            counts["blocked_original_reads"] += 1
            raise PermissionError("original-project access is forbidden after relocation")

    sys.addaudithook(audit)
    result = {"passed": False, "offline": True, "original_project_reads_forbidden": True,
              "evaluation_reads_forbidden": True, "query": "coleman tumbler 20 oz", "pool": 100, "k": 5}
    engine = None
    try:
        import ecom_search
        from ecom_search import accuracy
        from ecom_search.index import catalog_digest

        class NoCacheAccuracyRanker(accuracy.AccuracyRanker):
            # Only storage is disabled; model bytes, batching and numeric score()
            # are inherited unchanged. This avoids Windows MAX_PATH cache paths.
            def __init__(self, path, extractor, **kwargs):
                kwargs["cache_dir"] = None
                super().__init__(path, extractor, **kwargs)

        original_score = accuracy.AccuracyRanker.score
        accuracy.AccuracyRanker = NoCacheAccuracyRanker
        if NoCacheAccuracyRanker.score is not original_score:
            raise ValueError("delivery probe altered numeric scoring")
        from ecom_search.cli import SearchEngine
        if not Path(ecom_search.__file__).resolve().is_relative_to(project):
            raise ValueError("Python imported the original project")
        directory = project / "outputs/accuracy-v1"
        engine = SearchEngine(directory / "retrieval/products.sqlite", accuracy_path=directory / "model.json",
                              strategy_path=directory / "retrieval/pool_strategy.json")
        sqlite_path = Path(engine.index.connection.execute("PRAGMA database_list").fetchone()[2]).resolve()
        if sqlite_path != directory / "retrieval/products.sqlite":
            raise ValueError("SQLite was not relocated")
        catalog = engine.index.catalog()
        if len(catalog) != 82427 or len({p["id"] for p in catalog}) != len(catalog):
            raise ValueError("expected the 82,427-item catalog with unique IDs")
        catalog_sha = catalog_digest(catalog)
        if catalog_sha != engine.accuracy.metadata["catalog_sha256"]:
            raise ValueError("accuracy/catalog fingerprint differs")
        encoder = engine.accuracy.encoder
        if not Path(encoder.model_info["path"]).resolve().is_relative_to(project) or encoder.cache_directory is not None:
            raise ValueError("CrossEncoder escaped the relocated project or used a cache")
        candidates = engine.retrieval_pool.search(result["query"], k=100)
        candidate_ids = [pid for pid, _ in candidates]
        if len(candidate_ids) != len(set(candidate_ids)) or not 5 <= len(candidate_ids) <= 100:
            raise ValueError("invalid frozen candidate pool")
        accurate = engine.search(result["query"], k=5, pool=100)
        fast = engine.search(result["query"], k=5, pool=100, use_accuracy=False)
        expected_ranking = "cpu_" + engine.accuracy.metadata["kind"]
        if accurate["ranking"] != expected_ranking or not accurate["accuracy"]["applied"]:
            raise ValueError("accuracy mode did not run")
        if fast["ranking"] != "retrieval_order" or fast["accuracy"]["applied"]:
            raise ValueError("fast mode did not preserve the retrieval order")
        if accurate["retrieval"] != fast["retrieval"]:
            raise ValueError("accuracy/fast retrieval modes differ")
        for response in (accurate, fast):
            products = response["products"]
            ids = [p["id"] for p in products]
            if response["candidate_count"] != len(candidate_ids) or len(products) != 5 or len(set(ids)) != 5:
                raise ValueError("search returned invalid candidate/result counts")
            if not set(ids) <= set(candidate_ids) or any(not math.isfinite(p["score"]) for p in products):
                raise ValueError("invalid or nonfinite search results")
        if [p["id"] for p in fast["products"]] != candidate_ids[:5]:
            raise ValueError("fast mode changed the frozen candidate order")
        score_report = encoder.last_report
        if score_report["newly_predicted_pairs"] != len(candidate_ids) or score_report["reused_pairs"]:
            raise ValueError("CPU inference reused cached answers")
        if any(counts[key] for key in ("blocked_original_reads", "blocked_evaluation_reads", "blocked_network_attempts")):
            raise ValueError("serving attempted a prohibited access, even if a dependency caught the audit error")
        result.update(passed=True, imported_package=str(Path(ecom_search.__file__).resolve()),
                      catalog_count=len(catalog), catalog_sha256=catalog_sha, sqlite_path=str(sqlite_path),
                      candidate_count=len(candidate_ids), candidates_identical=True,
                      model_path=encoder.model_info["path"], cross_encoder_fingerprint=encoder.fingerprint,
                      model_kind=engine.accuracy.metadata["kind"], fresh_cpu_pairs=score_report["newly_predicted_pairs"],
                      cache_disabled=True, numeric_score_method_unchanged=True,
                      accuracy={"ranking": accurate["ranking"], "ids": [p["id"] for p in accurate["products"]]},
                      fast={"ranking": fast["ranking"], "ids": [p["id"] for p in fast["products"]]})
    except Exception as exc:
        result.update(error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    finally:
        if engine is not None:
            engine.close()
        result["access_audit"] = counts
        result["blocked_local_capability_probes"] = local_probes
    print(MARKER + json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
    return result["passed"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=ROOT.parent / "ecom-search-lab-accuracy-cpu.zip")
    parser.add_argument("--python", type=Path, default=Path(sys.executable), help="CPU Python with torch/transformers/numpy installed")
    parser.add_argument("--timeout", type=int, default=600, help="isolated runtime timeout in seconds")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--project", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--original", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        return 0 if runtime_check(args.project, args.original) else 1
    started = time.perf_counter()
    output = ROOT / "outputs/accuracy-v1/delivery_verification.json"
    archive_path = args.zip.resolve()
    report = {"passed": False, "created_at_utc": datetime.now(timezone.utc).isoformat(), "zip": str(archive_path)}
    save(output, report)
    try:
        report["zip_sha256"] = sha256(archive_path)
        run_root = ROOT / "outputs/accuracy-delivery-check" / uuid.uuid4().hex[:8]
        project = run_root / "ecom-search-lab"
        project.mkdir(parents=True, exist_ok=False)
        report["extracted_project"] = str(project)
        manifest, report["integrity"] = extract_verified(archive_path, project)
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                   CUDA_VISIBLE_DEVICES="", TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                   OPENBLAS_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1")
        command = [str(args.python.resolve()), "-I", "-B", str(project / "scripts/verify_accuracy_delivery.py"),
                   "--child", "--project", str(project), "--original", str(ROOT)]
        completed = subprocess.run(command, cwd=project, env=env, capture_output=True, text=True,
                                   encoding="utf-8", errors="replace", timeout=args.timeout)
        payloads = [line[len(MARKER):] for line in completed.stdout.splitlines() if line.startswith(MARKER)]
        report["runtime"] = json.loads(payloads[-1]) if len(payloads) == 1 else {"passed": False, "error": "missing/duplicate child result"}
        report["child_exit_code"] = completed.returncode
        if completed.returncode or not report["runtime"].get("passed"):
            report["child_stderr_tail"] = completed.stderr[-3000:]
            raise RuntimeError("offline relocated CPU search failed")
        if any(sha256(project / row["path"]) != row["sha256"] for row in manifest["files"]):
            raise ValueError("runtime changed a delivered file")
        if sha256(archive_path) != report["zip_sha256"]:
            raise ValueError("ZIP changed during verification")
        report.update(passed=True, immutable_delivered_files_rechecked=True, retained_test_copy=True,
                      scope="Delivery integrity and real offline inference only; no new ranking-quality or latency claim")
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    report["elapsed_seconds"] = time.perf_counter() - started
    save(output, report)
    print(json.dumps({key: report.get(key) for key in ("passed", "zip", "zip_sha256", "extracted_project", "elapsed_seconds", "error")}, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
