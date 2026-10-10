"""Verify a full CPU ZIP, then run its relocated serving artifacts offline.

Only the extracted copy's metadata paths are invalidated. Every other delivered
byte remains checked against DELIVERY_MANIFEST. The extracted copy is retained.
The orchestrator uses the standard library; --python supplies optional CPU deps.
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
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "ecom-search-lab/"
MANIFEST = PREFIX + "DELIVERY_MANIFEST.json"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def relative_name(value):
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError("archive paths must be relative POSIX paths")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {".", ".."} for part in path.parts) or path.as_posix() != value:
        raise ValueError(f"unsafe archive path: {value}")
    return path


def extract_verified(archive_path, destination):
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)) or MANIFEST not in names:
            raise ValueError("ZIP requires unique names and DELIVERY_MANIFEST.json")
        for info in infos:
            name = info.filename[:-1] if info.is_dir() else info.filename
            relative_name(name)
            if not name.startswith(PREFIX.rstrip("/")) or (not info.is_dir() and not name.startswith(PREFIX)):
                raise ValueError("ZIP contains an entry outside ecom-search-lab")
        manifest = json.loads(archive.read(MANIFEST))
        if manifest.get("full") is not True:
            raise ValueError("relocation test requires the full CPU delivery")
        files = manifest.get("files")
        if not isinstance(files, list) or not files:
            raise ValueError("empty delivery manifest")
        records = {}
        for row in files:
            relative_name(row["path"])
            if row["path"] in records:
                raise ValueError("duplicate delivery manifest entry")
            if "local-profiles" in row["path"].lower() or row["path"].startswith("outputs/uploads/"):
                raise ValueError("delivery contains local profiles or uploads")
            records[row["path"]] = row
        if {PREFIX + name for name in records} | {MANIFEST} != {info.filename for info in infos if not info.is_dir()}:
            raise ValueError("ZIP files and delivery manifest differ")
        count = total = 0
        for name, row in records.items():
            info = archive.getinfo(PREFIX + name)
            if info.file_size != row["bytes"]:
                raise ValueError(f"ZIP byte count mismatch: {name}")
            output = destination / name
            if not output.resolve().is_relative_to(destination.resolve()):
                raise ValueError("extraction escaped the destination")
            output.parent.mkdir(parents=True, exist_ok=True)
            digest, crc, size = hashlib.sha256(), 0, 0
            with archive.open(info) as source, output.open("xb") as target:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(block)
                    crc = binascii.crc32(block, crc)
                    size += len(block)
                    target.write(block)
            if size != row["bytes"] or digest.hexdigest() != row["sha256"] or crc & 0xFFFFFFFF != info.CRC:
                raise ValueError(f"delivery SHA256/CRC mismatch: {name}")
            if sha256(output) != row["sha256"]:
                raise ValueError(f"extracted file differs: {name}")
            count += 1
            total += size
        manifest_bytes = archive.read(MANIFEST)  # Reading to EOF also checks its ZIP CRC.
        (destination / "DELIVERY_MANIFEST.json").write_bytes(manifest_bytes)
        return manifest, {"files": count, "uncompressed_bytes": total, "sha256_verified": True, "crc_verified": True,
                          "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(), "local_profiles_excluded": True}


def invalidate_old_paths(project):
    changes = []
    edits = {
        "outputs/disk-vectors-v1/int8/metadata.json": [("model", "path")],
        "outputs/generative-v1/metadata.json": [("embedding_model", "path")],
        "outputs/multimodal/selected/metadata.json": [("model", "path"), ("data_root",)],
    }
    for name, fields in edits.items():
        path = project / name
        obj = json.loads(path.read_text(encoding="utf-8"))
        before = sha256(path)
        for field in fields:
            parent = obj
            for component in field[:-1]:
                parent = parent[component]
            missing = project.parent / "__missing_original_paths__" / name.replace("/", "_") / "_".join(field)
            if missing.exists():
                raise ValueError("missing-path fixture unexpectedly exists")
            parent[field[-1]] = str(missing)
        save(path, obj)
        changes.append({"path": name, "fields": [".".join(field) for field in fields],
                        "original_sha256": before, "test_copy_sha256": sha256(path)})
    return changes


def runtime_check(project, original, output):
    project, original, output = Path(project).resolve(), Path(original).resolve(), Path(output).resolve()
    sys.path.insert(0, str(project))
    read_counts = {"relocated_file_opens": 0, "blocked_original_reads": 0, "network_attempts": 0}

    def audit(event, args):
        if event == "socket.connect":
            read_counts["network_attempts"] += 1
            raise RuntimeError("network is disabled during delivery verification")
        if event not in {"open", "sqlite3.connect"} or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if path.is_relative_to(project):
            read_counts["relocated_file_opens"] += 1
        elif path.is_relative_to(original) and path != output:
            read_counts["blocked_original_reads"] += 1
            raise PermissionError(f"original project access is forbidden: {path}")

    sys.addaudithook(audit)
    result = {"passed": False, "cwd": str(Path.cwd()), "offline": True, "original_project_reads_forbidden": True}
    engine = None
    try:
        import ecom_search
        from ecom_search import vectors
        from ecom_search.index import catalog_digest
        real_load = vectors._load_encoder
        loaded_text_models = []

        def local_encoder(path, *args, **kwargs):
            path = Path(path).resolve()
            if not path.is_relative_to(project):
                raise ValueError("text encoder escaped the relocated project")
            loaded_text_models.append(str(path))
            return real_load(path, *args, **kwargs)

        vectors._load_encoder = local_encoder  # Observe paths, then run the real CPU loader.
        from ecom_search.cli import SearchEngine
        from ecom_search.generative import GenerativeIndex
        from ecom_search.multimodal import MultimodalIndex, _model_info
        from ecom_search.personalization import BeautyRecommender
        if not Path(ecom_search.__file__).resolve().is_relative_to(project):
            raise ValueError("Python imported the original project")
        engine = SearchEngine(project / "outputs/products.sqlite", project / "outputs/ranking/model.json",
                              disk_path=project / "outputs/disk-vectors-v1/int8")
        sqlite_path = Path(engine.index.connection.execute("PRAGMA database_list").fetchone()[2]).resolve()
        if sqlite_path != project / "outputs/products.sqlite":
            raise ValueError("SQLite was opened outside the relocated project")
        catalog_hash = catalog_digest(engine.index.catalog())
        disk_result = engine.search("running shoes without Nike", 5, 100)
        if not disk_result["products"] or "disk_int8" not in disk_result["retrieval"]:
            raise ValueError("disk-backed hybrid search did not run")
        if any("nike" in (row["title"] + " " + row["brand"]).lower() for row in disk_result["products"]):
            raise ValueError("negative query constraint failed after relocation")
        text_path = project / "outputs/text-assets/minilm"
        text_sha = vectors.model_fingerprint(text_path)["sha256"]
        if text_sha != engine.disk.metadata["model"]["sha256"]:
            raise ValueError("relocated text encoder SHA differs")
        result["disk_search"] = {"passed": True, "products": len(disk_result["products"]), "ids": [row["id"] for row in disk_result["products"]],
                                 "sqlite_path": str(sqlite_path), "catalog_sha256": catalog_hash,
                                 "model_path": str(text_path), "model_sha256": text_sha}
        generator = GenerativeIndex(project / "outputs/generative-v1")
        generated = generator.search("wireless headphones", 5)
        if not generated or generator.model_path.resolve() != text_path or generator.metadata["embedding_model"]["sha256"] != text_sha or generator.metadata["catalog_sha256"] != catalog_hash:
            raise ValueError("generative relocation/model binding failed")
        result["generative_search"] = {"passed": True, "products": len(generated), "ids": [pid for pid, score in generated],
                                       "model_path": str(generator.model_path.resolve()), "model_sha256": text_sha}
        multimodal = MultimodalIndex(project / "outputs/multimodal/selected")
        data_root = project / "data/multimodal"
        image_paths = [Path(row["image_path"]).resolve() for row in multimodal.catalog.values()]
        if multimodal.data_root.resolve() != data_root or any(not path.is_relative_to(data_root) for path in image_paths):
            raise ValueError("multimodal images still reference the original project")
        text_hits = multimodal.search(query="black dress", k=5, mode="text")
        image_hits = multimodal.search(image_path=image_paths[0], k=5, mode="image")
        joint_hits = multimodal.search(query="black dress", image_path=image_paths[0], k=5, mode="learned")
        mm_path = project / "outputs/multimodal-assets/tinyclip"
        mm_sha = _model_info(mm_path)["sha256"]
        if any(len(hits) != 5 or any(not math.isfinite(score) for pid, score in hits) for hits in [text_hits, image_hits, joint_hits]) or Path(multimodal.resolved_model_path) != mm_path or mm_sha != multimodal.metadata["model"]["sha256"]:
            raise ValueError("multimodal text/image/joint relocation failed")
        result["multimodal_search"] = {"passed": True, "text_products": len(text_hits), "image_products": len(image_hits), "joint_products": len(joint_hits),
                                      "model_path": str(mm_path), "model_sha256": mm_sha, "data_root": str(data_root),
                                      "catalog_images_under_relocated_root": len(image_paths), "sample_image": str(image_paths[0])}
        beauty = BeautyRecommender(project / "outputs/personalization/experiment")
        if not beauty.users:
            raise ValueError("Beauty contains no training users")
        user = beauty.users[0]
        recommendations = beauty.recommend(user, 5)
        if len(recommendations) != 5 or any(row["id"] in beauty.reviewed[user] or row["cold_user"] or row["personal_weight"] != beauty.personal_weight for row in recommendations):
            raise ValueError("frozen Beauty recommendation failed")
        result["beauty_recommend"] = {"passed": True, "products": len(recommendations), "ids": [row["id"] for row in recommendations],
                                     "personal_weight": beauty.personal_weight, "selected_epochs": beauty.selection["epochs"],
                                     "history_scope": beauty.history_scope, "serving_directory": str(project / "outputs/personalization/experiment")}
        result.update(passed=True, imported_package=str(Path(ecom_search.__file__).resolve()), loaded_text_model_paths=loaded_text_models)
    except Exception as exc:
        result.update(error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    finally:
        if engine is not None:
            engine.close()
        result["access_audit"] = read_counts
        save(output, result)
    return result["passed"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=ROOT.parent / "ecom-search-lab-full-cpu.zip")
    parser.add_argument("--python", type=Path, default=Path(sys.executable), help="CPU interpreter with numpy/torch/transformers/sentence-transformers/Pillow")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/delivery_verification.json")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--project", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--original", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--child-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        return 0 if runtime_check(args.project, args.original, args.child_output) else 1
    started = time.perf_counter()
    archive_path, output = args.zip.resolve(), args.output.resolve()
    report = {"passed": False, "created_at_utc": datetime.now(timezone.utc).isoformat(), "zip": str(archive_path)}
    save(output, report)  # A failed rerun cannot leave an old passed=true report.
    try:
        report["zip_sha256"] = sha256(archive_path)
        run_root = ROOT / "outputs/delivery-test" / uuid.uuid4().hex
        project = run_root / "ecom-search-lab"
        project.mkdir(parents=True)
        report["extracted_project"] = str(project)
        manifest, report["integrity"] = extract_verified(archive_path, project)
        report["test_copy_path_changes"] = invalidate_old_paths(project)
        env = os.environ.copy()
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                   CUDA_VISIBLE_DEVICES="", TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
        child_output = run_root / "runtime.json"
        command = [str(args.python.resolve()), "-I", str(Path(__file__).resolve()), "--child", "--project", str(project),
                   "--original", str(ROOT), "--child-output", str(child_output)]
        completed = subprocess.run(command, cwd=project, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
        report["runtime"] = json.loads(child_output.read_text(encoding="utf-8")) if child_output.is_file() else {"passed": False, "error": "child report missing"}
        report["child_exit_code"] = completed.returncode
        if completed.returncode or not report["runtime"].get("passed"):
            report["child_stderr_tail"] = completed.stderr[-2000:]
            raise RuntimeError("offline relocated runtime checks failed")
        changed = {row["path"] for row in report["test_copy_path_changes"]}
        if any(sha256(project / row["path"]) != row["sha256"] for row in manifest["files"] if row["path"] not in changed):
            raise ValueError("runtime changed a delivered data/model/selection/source artifact")
        if sha256(archive_path) != report["zip_sha256"]:
            raise ValueError("ZIP changed during delivery verification")
        report.update(passed=True, immutable_delivered_files_rechecked=True, retained_test_copy=True,
                      scope="Offline relocation of frozen serving artifacts; verifies delivery and real inference, does not rerun experiments or establish new quality/latency results")
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    report["elapsed_seconds"] = time.perf_counter() - started
    save(output, report)
    print(json.dumps({key: report.get(key) for key in ("passed", "zip", "zip_sha256", "extracted_project", "elapsed_seconds", "error")}, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
