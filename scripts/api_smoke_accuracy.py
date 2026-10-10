"""Exercise the real v0.3 HTTP service; no evaluation-label reads."""
import argparse
import atexit
import base64
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import urllib.error
import urllib.request
from urllib.parse import quote
import uuid

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--port", type=int, default=18866)
args = parser.parse_args()
ROOT = Path(__file__).resolve().parents[1]
target = ROOT / "outputs/accuracy-v1/api_smoke.json"
base = f"http://127.0.0.1:{args.port}"
checks, completed = [], False


def save(status):
    result = {"status": status, "passed": status == "completed", "checks": checks,
              "at": datetime.now(timezone.utc).isoformat(), "base_url": base,
              "scope": "real local single-worker HTTP; cold/warm latency is diagnostic, not a load test"}
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def request(path, obj=None, expected=200, binary=False):
    req = urllib.request.Request(base + path, data=None if obj is None else json.dumps(obj).encode(),
        headers={} if obj is None else {"Content-Type": "application/json"})
    try:
        response = urllib.request.urlopen(req, timeout=300)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        data = response.read()
        assert response.status == expected, (path, response.status, data[:1000])
        return data if binary else json.loads(data)


def passed(name, **details):
    checks.append({"check": name, "passed": True, **details})


target.parent.mkdir(parents=True, exist_ok=True)
save("running")
atexit.register(lambda: None if completed else save("failed"))
health = request("/health")
assert health["status"] == "ok" and health["products"] == 82427
assert health["accuracy"] and health["accuracy_default"] and health["retrieval_mode"] == "lexical_only"
assert health["profiles"] and health["multimodal"] and health["beauty"] and not health["generative"]
passed("frozen_accuracy_and_independent_modules", kind=health["accuracy_kind"], products=health["products"])

query = "coleman tumbler 20 oz"
accurate = request("/search", {"query": query, "k": 100, "pool": 100})
fast = request("/search", {"query": query, "k": 100, "pool": 100, "use_accuracy": False})
assert accurate["accuracy"]["applied"] and not fast["accuracy"]["applied"]
assert accurate["ranking"] == "cpu_" + health["accuracy_kind"] and fast["ranking"] == "retrieval_order"
assert accurate["products"] and {p["id"] for p in accurate["products"]} == {p["id"] for p in fast["products"]}
assert accurate["candidate_count"] == fast["candidate_count"]
assert len(accurate["products"]) == len({p["id"] for p in accurate["products"]})
assert all(math.isfinite(p["score"]) for p in accurate["products"])
assert accurate["products"] == sorted(accurate["products"], key=lambda p: (-p["score"], p["id"]))
passed("real_reranking_of_identical_retrieval_candidates", candidates=accurate["candidate_count"],
       accurate_ms=accurate["latency_ms"], fast_ms=fast["latency_ms"], top5=accurate["products"][:5])
repeat = request("/search", {"query": query, "k": 100, "pool": 100})
assert [(p["id"], p["score"]) for p in accurate["products"]] == [(p["id"], p["score"]) for p in repeat["products"]]
passed("whole_call_cache_repeats_identically", latency_ms=repeat["latency_ms"])

negative = request("/search", {"query": "running shoes without Nike", "k": 8, "pool": 100})
assert negative["products"] and all("nike" not in (p["title"] + " " + p["brand"]).casefold() for p in negative["products"])
passed("negative_brand_constraint_with_accuracy", returned=len(negative["products"]))
user = "accuracy-smoke-" + uuid.uuid4().hex
pid = accurate["products"][0]["id"]
try:
    request("/profile", {"user_id": user, "product_id": pid, "action": "like"})
    personal = request("/search", {"query": query, "k": 100, "user_id": user})
    assert personal["personalization"]["applied"] and any(p["id"] == pid and p["preference_score"] > 0 for p in personal["products"])
    passed("explicit_profile_after_new_relevance_scores")
    request("/profile", {"user_id": user, "product_id": pid, "action": "hide"})
    hidden = request("/search", {"query": query, "k": 100, "user_id": user})
    assert all(p["id"] != pid for p in hidden["products"])
    passed("hidden_product_removed")
finally:
    request("/profile", {"user_id": user, "action": "clear"})

catalog = request("/multimodal/catalog")["products"]
image_id = catalog[0]["id"]
raw = request("/image/" + quote(image_id, safe=""), binary=True)
image = request("/multimodal/search", {"reference_id": image_id, "query": "black dress", "mode": "learned", "k": 5})
assert len(image["products"]) == 5
upload = request("/multimodal/search", {"image_base64": base64.b64encode(raw).decode(), "mode": "image", "k": 5})
assert len(upload["products"]) == 5
passed("real_image_and_upload_modules_retained")
users = request("/recommend/users")["users"]
assert request("/recommend?user=" + quote(users[0], safe=""))["products"]
passed("independent_history_recommendation_retained")
cards = request("/experiments")["cards"]
assert any(card["task"].startswith("ESCI v2") for card in cards)
passed("fresh_accuracy_evidence_in_page")
for label, obj in [
    ("boolean_mode", {"query": "shoes", "use_accuracy": "true"}),
    ("bounded_cpu_pool", {"query": "shoes", "pool": 201}),
    ("generative_catalog_mismatch", {"query": "shoes", "use_generative": True}),
]:
    assert request("/search", obj, expected=400)["error"]
    passed("reject_" + label)
completed = True
print(json.dumps(save("completed"), ensure_ascii=False))
