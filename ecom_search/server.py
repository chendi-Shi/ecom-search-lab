from __future__ import annotations

import base64
import binascii
from http.server import BaseHTTPRequestHandler, HTTPServer
import io
import json
from pathlib import Path
import time
from urllib.parse import urlsplit, parse_qs, unquote
import uuid

from .cli import SearchEngine

ROOT = Path(__file__).resolve().parents[1]


def experiment_cards():
    path = ROOT / "outputs/experiment-summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"cards": []}


def health_payload(engine, multimodal=None, beauty=None):
    """Capabilities come from loaded, fingerprint-checked serving artifacts."""
    accuracy = engine.accuracy is not None
    mode = (getattr(engine.retrieval_pool, "selection", {}).get("mode", "validation_selected")
            if engine.retrieval_pool is not None else
            "hybrid_rrf" if engine.vector is not None or engine.disk is not None else "lexical_only")
    return {"status": "ok", "products": int(engine.index.metadata["products"]),
            "trained_ranker": engine.model is not None, "generative": engine.generative is not None and engine.retrieval_pool is None,
            "multimodal": multimodal is not None, "beauty": beauty is not None,
            "disk": engine.disk is not None or bool(engine.retrieval_pool is not None and
                engine.retrieval_pool.metadata.get("vectors_available", False)),
            "profiles": engine.profile_store is not None,
            "accuracy": accuracy, "accuracy_default": accuracy,
            "accuracy_kind": engine.accuracy.metadata["kind"] if accuracy else None,
            "accuracy_pool_limit": 200, "accuracy_evaluated_pool": 100 if accuracy else None,
            "retrieval_mode": mode, "candidate_source": "词法召回（BM25）" if mode == "lexical_only" else
            "多路召回融合" if mode == "hybrid_rrf" else "验证集选定召回策略",
            "retrieval_strategy": engine.retrieval_pool is not None}


def search_request(engine, obj):
    # JSON null/strings/numbers must not silently become a boolean switch.
    if "use_accuracy" in obj and not isinstance(obj["use_accuracy"], bool):
        raise ValueError("use_accuracy must be a boolean")
    return engine.search(obj.get("query"), obj.get("k", 5), obj.get("pool", 100),
        user_id=obj.get("user_id"), personalization_weight=obj.get("personalization_weight", .25),
        use_generative=obj.get("use_generative", False), use_accuracy=obj.get("use_accuracy"))


def serve_local(index_path: Path, model_path: Path | None, port: int,
                vector_path: Path | None = None, *, generative_path=None,
                disk_path=None, profile_path=None, multimodal_path=None, beauty_path=None,
                accuracy_path=None, strategy_path=None):
    engine = SearchEngine(index_path, model_path, vector_path, generative_path=generative_path,
                          disk_path=disk_path, profile_path=profile_path,
                          accuracy_path=accuracy_path, strategy_path=strategy_path)
    multimodal = beauty = None
    try:
        if multimodal_path:
            from .multimodal import MultimodalIndex
            multimodal = MultimodalIndex(Path(multimodal_path))
        if beauty_path:
            from .personalization import BeautyRecommender
            beauty = BeautyRecommender(Path(beauty_path))
    except Exception:
        engine.close()
        raise

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(30)

        def response(self, status, payload, content_type="application/json; charset=utf-8"):
            body = payload if isinstance(payload, bytes) else (payload.encode("utf-8") if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False).encode("utf-8"))
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def product_card(self, pid, score=None):
            p = multimodal.catalog[pid]
            card = {"id": pid, "title": p.get("title", p.get("text", "")), "source": "Fashion200K real paired photographs"}
            if score is not None: card["score"] = float(score)
            return card

        def do_GET(self):
            url = urlsplit(self.path)
            try:
                if url.path == "/":
                    self.response(200, (ROOT / "web/index.html").read_text(encoding="utf-8"), "text/html; charset=utf-8")
                elif url.path == "/health":
                    self.response(200, health_payload(engine, multimodal, beauty))
                elif url.path == "/experiments":
                    self.response(200, experiment_cards())
                elif url.path == "/multimodal/catalog":
                    if multimodal is None: raise ValueError("multimodal index is not configured")
                    self.response(200, {"products": [self.product_card(pid) for pid in list(multimodal.catalog)[:16]]})
                elif url.path.startswith("/image/"):
                    pid = unquote(url.path[len("/image/"):])
                    if multimodal is None or pid not in multimodal.catalog: raise ValueError("unknown image product ID")
                    # Open only paths from the verified catalog, never a client-supplied path.
                    path = Path(multimodal.catalog[pid]["image_path"])
                    types = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
                    self.response(200, path.read_bytes(), types.get(path.suffix.lower(), "application/octet-stream"))
                elif url.path == "/recommend/users":
                    if beauty is None: raise ValueError("Beauty experiment is not configured")
                    self.response(200, {"users": list(beauty.users)[:50]})
                elif url.path == "/recommend":
                    if beauty is None: raise ValueError("Beauty experiment is not configured")
                    user = parse_qs(url.query).get("user", [""])[0]
                    self.response(200, {"products": beauty.recommend(user, k=8), "scope": "冻结 BPR + 热门融合；只读取训练期历史；独立 Beauty 数据集"})
                else:
                    self.response(404, {"error": "not found"})
            except (ValueError, TypeError, KeyError) as exc:
                self.response(400, {"error": str(exc)})

        def read_object(self, max_bytes):
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= max_bytes: raise ValueError(f"request body must be 1..{max_bytes} bytes")
            obj = json.loads(self.rfile.read(length))
            if not isinstance(obj, dict): raise ValueError("request body must be a JSON object")
            return obj

        def image_search(self, obj):
            if multimodal is None: raise ValueError("multimodal index is not configured")
            query = obj.get("query", "")
            if not isinstance(query, str) or len(query) > 2000: raise ValueError("query must be text of at most 2000 characters")
            k = obj.get("k", 8)
            if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= 100: raise ValueError("k must be in [1, 100]")
            mode = obj.get("mode", "learned")
            if mode not in {"learned", "image", "text", "fusion"}: raise ValueError("unknown multimodal mode")
            reference, encoded = obj.get("reference_id"), obj.get("image_base64")
            if "image_path" in obj: raise ValueError("client image paths are not accepted")
            if mode == "text": reference, encoded = None, None
            if reference and encoded: raise ValueError("choose one reference image or upload")
            image_path = temporary = None
            started = time.perf_counter()
            try:
                if reference:
                    if not isinstance(reference, str) or reference not in multimodal.catalog: raise ValueError("unknown image product ID")
                    image_path = Path(multimodal.catalog[reference]["image_path"])
                elif encoded:
                    if not isinstance(encoded, str): raise ValueError("image_base64 must be text")
                    if encoded.startswith("data:"):
                        header, encoded = encoded.split(",", 1)
                        if header not in {"data:image/jpeg;base64", "data:image/png;base64", "data:image/webp;base64"}: raise ValueError("unsupported image type")
                    raw = base64.b64decode(encoded, validate=True)
                    if not raw or len(raw) > 2 * 1024 * 1024: raise ValueError("image must be 1 byte to 2 MiB")
                    from PIL import Image
                    try:
                        with Image.open(io.BytesIO(raw)) as im:
                            if im.format not in {"PNG", "JPEG", "WEBP"} or im.width * im.height > 4_000_000: raise ValueError("image must be PNG/JPEG/WebP, at most 4 million pixels")
                            im.verify()
                    except Image.DecompressionBombError as exc:
                        raise ValueError("image dimensions exceed the supported limit") from exc
                    upload_dir = ROOT / "outputs/uploads"
                    upload_dir.mkdir(parents=True, exist_ok=True)
                    temporary = upload_dir / (uuid.uuid4().hex + ".img")
                    temporary.write_bytes(raw)
                    image_path = temporary
                rows = multimodal.search(query=query, image_path=image_path, k=k, mode=mode)
                return {"products": [self.product_card(pid, score) for pid, score in rows], "mode": mode, "latency_ms": 1000 * (time.perf_counter() - started), "scope": "200 real Fashion200K products, frozen TinyCLIP + trainable projection; separate from ESCI"}
            finally:
                if temporary is not None and temporary.resolve().is_relative_to((ROOT / "outputs/uploads").resolve()): temporary.unlink(missing_ok=True)

        def do_POST(self):
            try:
                if self.path == "/search":
                    obj = self.read_object(16384)
                    self.response(200, search_request(engine, obj))
                elif self.path == "/profile":
                    obj = self.read_object(16384)
                    if engine.profile_store is None: raise ValueError("local profiles are not configured")
                    user = obj.get("user_id")
                    if obj.get("action") == "clear":
                        engine.profile_store.clear(user)
                        result = {"cleared": True}
                    else: result = engine.record_preference(user, obj.get("product_id"), obj.get("action"))
                    self.response(200, result)
                elif self.path == "/multimodal/search":
                    self.response(200, self.image_search(self.read_object(3 * 1024 * 1024)))
                else:
                    self.response(404, {"error": "not found"})
            except (ValueError, TypeError, KeyError, binascii.Error, OSError) as exc:
                self.response(400, {"error": str(exc)})
            except (RuntimeError, ImportError) as exc:
                self.response(500, {"error": "model execution failed: " + str(exc)})

        def log_message(self, *args):
            pass

    httpd = None
    try:
        httpd = HTTPServer(("127.0.0.1", port), Handler)
        print(f"Local search: http://127.0.0.1:{port}", flush=True)
        httpd.serve_forever()
    finally:
        if httpd is not None: httpd.server_close()
        engine.close()
