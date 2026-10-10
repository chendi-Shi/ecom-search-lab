"""Local explicit preferences and small CPU Bayesian personalized ranking.

Client likes/hides are explicit user inputs. Review ratings are a separate
research source; neither source is fabricated from ESCI relevance judgments.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sqlite3
import time


def _identifier(value, label):
    if not isinstance(value, str) or not value.strip() or len(value) > 200:
        raise ValueError(f"{label} must be a nonempty string of at most 200 characters")
    return value


class PersonalizationStore:
    """One local SQLite database, without authentication or remote synchronization."""
    def __init__(self, database):
        path = Path(database)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("CREATE TABLE IF NOT EXISTS preferences(id INTEGER PRIMARY KEY,user_id TEXT NOT NULL,product_id TEXT NOT NULL,action TEXT NOT NULL,timestamp REAL NOT NULL)")
        self.connection.execute("CREATE INDEX IF NOT EXISTS user_time ON preferences(user_id,timestamp,id)")
        self.connection.commit()

    def record(self, user_id, product_id, action, timestamp=None):
        _identifier(user_id, "user_id")
        _identifier(product_id, "product_id")
        if action not in {"like", "hide", "history"}:
            raise ValueError("action must be like, hide, or history")
        timestamp = time.time() if timestamp is None else timestamp
        if isinstance(timestamp, bool) or not isinstance(timestamp, (float, int)) or not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("timestamp must be finite nonnegative Unix seconds")
        cursor = self.connection.execute("INSERT INTO preferences(user_id,product_id,action,timestamp) VALUES(?,?,?,?)",
                                         (user_id, product_id, action, timestamp))
        self.connection.commit()
        return {"id": cursor.lastrowid, "user_id": user_id, "product_id": product_id,
                "action": action, "timestamp": timestamp, "source": "explicit_local_client"}

    def events(self, user_id, before_timestamp=None):
        _identifier(user_id, "user_id")
        if before_timestamp is None:
            rows = self.connection.execute("SELECT * FROM preferences WHERE user_id=? ORDER BY timestamp,id", (user_id,))
        else:
            if isinstance(before_timestamp, bool) or not isinstance(before_timestamp, (float, int)) or not math.isfinite(before_timestamp):
                raise ValueError("before_timestamp must be finite Unix seconds")
            rows = self.connection.execute("SELECT * FROM preferences WHERE user_id=? AND timestamp<? ORDER BY timestamp,id", (user_id, before_timestamp))
        return [dict(row) for row in rows]

    history = events

    def hidden(self, user_id, before_timestamp=None):
        latest = {event["product_id"]: event["action"] for event in self.events(user_id, before_timestamp)
                  if event["action"] in {"like", "hide"}}
        return {pid for pid, action in latest.items() if action == "hide"}

    def profile(self, user_id):
        events = self.events(user_id)
        latest = {event["product_id"]: event["action"] for event in events if event["action"] in {"like", "hide"}}
        return {"user_id": user_id, "events": len(events), "liked_ids": sorted(pid for pid, action in latest.items() if action == "like"),
                "hidden_ids": sorted(self.hidden(user_id)), "source": "explicit_local_client", "local_only": True}

    def clear(self, user_id):
        _identifier(user_id, "user_id")
        self.connection.execute("DELETE FROM preferences WHERE user_id=?", (user_id,))
        self.connection.commit()

    def close(self):
        self.connection.close()


def _hash_vector(product, dimensions=128):
    text = " ".join(str(product.get(field, "")) for field in ("title", "brand", "description"))
    vector = [0.0] * dimensions
    for term in re.findall(r"\w+", text.casefold()):
        digest = hashlib.sha256(term.encode()).digest()
        vector[int.from_bytes(digest[:4], "little") % dimensions] += 1 if digest[4] & 1 else -1
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


class ProfileRanker:
    """Recency-weighted interest vector from explicit local preferences.

The default token hash encoder is a transparent untrained lexical baseline.
Callers may supply a mapping of actual cached normalized product embeddings.
Weights below are a UI heuristic, independent from the review BPR experiment.
"""
    def __init__(self, store, catalog, vectors=None, weights=None):
        self.store = store
        self.catalog = {row["id"]: row for row in catalog}
        self.weights = {"like": 1.0, "hide": -1.0, "history": 0.15, "half_life_days": 30.0, "confidence_events": 3.0}
        self.weights.update(weights or {})
        if any(not isinstance(value, (float, int)) or isinstance(value, bool) or not math.isfinite(value)
               for value in self.weights.values()) or self.weights["half_life_days"] <= 0 or self.weights["confidence_events"] <= 0:
            raise ValueError("Profile weights must be finite with positive decay and confidence scales")
        self.vectors = {pid: list(vector) for pid, vector in vectors.items()} if vectors is not None else {
            pid: _hash_vector(row) for pid, row in self.catalog.items()}
        dimensions = {len(vector) for vector in self.vectors.values()}
        if len(dimensions) > 1 or any(not math.isfinite(value) for vector in self.vectors.values() for value in vector):
            raise ValueError("Cached vectors need a common dimension and finite values")
        for pid, vector in self.vectors.items():
            norm = math.sqrt(sum(value * value for value in vector)) or 1.0
            self.vectors[pid] = [value / norm for value in vector]
        self.backend = "cached-product-embeddings" if vectors is not None else "untrained-token-hash-profile"

    def hidden(self, user_id, before_timestamp=None):
        return self.store.hidden(user_id, before_timestamp)

    def score(self, query, products, user_id, before_timestamp=None):
        cutoff = time.time() if before_timestamp is None else before_timestamp
        events = self.store.events(user_id, cutoff)
        # Only the latest state per item contributes, so repeated clicks cannot
        # multiply a preference or override a later hide via old likes.
        latest = {}
        for event in events:
            previous = latest.get(event["product_id"])
            if event["action"] == "history" and previous and previous["action"] in {"like", "hide"}:
                continue
            latest[event["product_id"]] = event
        weighted = [(self.vectors[event["product_id"]], self.weights[event["action"]] *
                     2 ** (-max(0.0, cutoff - event["timestamp"]) / (86400 * self.weights["half_life_days"])) )
                    for event in latest.values() if event["product_id"] in self.vectors]
        if not weighted:
            return {product["id"]: 0.0 for product in products}
        interest = [sum(vector[index] * weight for vector, weight in weighted) for index in range(len(weighted[0][0]))]
        norm = math.sqrt(sum(value * value for value in interest))
        confidence = min(1.0, sum(abs(weight) for _, weight in weighted) / self.weights["confidence_events"])
        if not norm:
            return {product["id"]: 0.0 for product in products}
        return {product["id"]: confidence * sum(a * b for a, b in zip(interest, self.vectors.get(product["id"], []))) / norm
                for product in products}


class BPRModel:
    """Seeded 16-factor BPR trained on rating>=4 events and unjudged samples."""
    def __init__(self, dimensions=16, seed=20261009):
        self.dimensions, self.seed = dimensions, seed
        self.users, self.items, self.popularity = {}, {}, {}

    def fit(self, events, epochs=15, learning_rate=0.04, regularization=0.01):
        if any(event.get("split") != "train" for event in events):
            raise ValueError("BPR training accepts only explicit train events")
        if type(epochs) is not int or epochs < 0:
            raise ValueError("epochs must be a nonnegative integer")
        rng = random.Random(self.seed)
        item_ids = sorted({event["product_id"] for event in events})
        user_ids = sorted({event["user_id"] for event in events})
        self.users = {user: [rng.gauss(0, 0.08) for _ in range(self.dimensions)] for user in user_ids}
        self.items = {item: [rng.gauss(0, 0.08) for _ in range(self.dimensions)] for item in item_ids}
        positives = [event for event in events if event["rating"] >= 4]
        counts = Counter(event["product_id"] for event in positives)
        self.popularity = {item: counts[item] for item in item_ids}
        seen = defaultdict(set)
        for event in events:
            seen[event["user_id"]].add(event["product_id"])
        updates, last_loss = 0, None
        for _ in range(epochs):
            shuffled = list(positives)
            rng.shuffle(shuffled)
            loss, count = 0.0, 0
            for event in shuffled:
                user, positive = event["user_id"], event["product_id"]
                negative = None
                for attempt in range(30):
                    candidate = item_ids[rng.randrange(len(item_ids))]
                    if candidate not in seen[user]:
                        negative = candidate
                        break
                if negative is None:
                    continue
                u, p, n = self.users[user], self.items[positive], self.items[negative]
                difference = sum(a * (b - c) for a, b, c in zip(u, p, n))
                gradient = 1 / (1 + math.exp(max(-40.0, min(40.0, difference))))
                loss += math.log1p(math.exp(max(-40.0, min(40.0, -difference))))
                count += 1
                for index in range(self.dimensions):
                    old_u, old_p, old_n = u[index], p[index], n[index]
                    u[index] += learning_rate * (gradient * (old_p - old_n) - regularization * old_u)
                    p[index] += learning_rate * (gradient * old_u - regularization * old_p)
                    n[index] += learning_rate * (-gradient * old_u - regularization * old_n)
                updates += 1
            last_loss = loss / count if count else None
        return {"epochs": epochs, "dimensions": self.dimensions, "positive_events": len(positives), "updates": updates,
                "last_sampled_pair_loss": last_loss, "negative_scope": "uniform unreviewed train-known items; not verified dislikes"}

    def score(self, user_id, product_id):
        if user_id not in self.users or product_id not in self.items:
            return 0.0
        return sum(a * b for a, b in zip(self.users[user_id], self.items[product_id]))

    def to_dict(self):
        return {"version": 1, "dimensions": self.dimensions, "seed": self.seed,
                "users": self.users, "items": self.items, "popularity": self.popularity}

    @classmethod
    def from_dict(cls, data):
        if data.get("version") != 1:
            raise ValueError("Unsupported BPR model version")
        model = cls(data["dimensions"], data["seed"])
        model.users, model.items, model.popularity = data["users"], data["items"], data["popularity"]
        if any(len(vector) != model.dimensions or any(not math.isfinite(value) for value in vector)
               for table in (model.users, model.items) for vector in table.values()):
            raise ValueError("Invalid BPR vectors")
        return model


def chronological_split(events):
    ordered = sorted(events, key=lambda event: (event["timestamp"], event["user_id"], event["product_id"]))
    if len(ordered) < 10:
        raise ValueError("Need at least ten real events for chronological splitting")
    train_cutoff = ordered[int(len(ordered) * 0.8)]["timestamp"]
    test_cutoff = ordered[int(len(ordered) * 0.9)]["timestamp"]
    if train_cutoff >= test_cutoff:
        raise ValueError("Distinct chronological boundaries are required")
    splits = {"train": [], "val": [], "test": []}
    for event in ordered:
        split = "train" if event["timestamp"] < train_cutoff else "val" if event["timestamp"] < test_cutoff else "test"
        splits[split].append(dict(event, split=split))
    return splits, {"train_end_unix_seconds_exclusive": train_cutoff, "val_end_unix_seconds_exclusive": test_cutoff,
                    "policy": "global chronological 80/10/10 timestamp quantiles; all equal timestamps remain in one split"}


def sampled_cases(events, all_events, train_items, candidate_count=100, limit=1000, seed=20261009):
    """Known positive + uniform unjudged decoys; only strictly earlier history."""
    history = defaultdict(list)
    for event in all_events:
        history[event["user_id"]].append(event)
    for rows in history.values():
        rows.sort(key=lambda event: event["timestamp"])
    pool = sorted(train_items)
    eligible, counts = [], Counter()
    for event in events:
        if event["rating"] < 4:
            counts["not_positive"] += 1
            continue
        if event["product_id"] not in train_items:
            counts["target_outside_train_catalog"] += 1
            continue
        prior = [row for row in history[event["user_id"]] if row["timestamp"] < event["timestamp"]]
        seen = {row["product_id"] for row in prior}
        if event["product_id"] in seen:
            counts["repeat_review_excluded"] += 1
            continue
        available = [item for item in pool if item not in seen and item != event["product_id"]]
        event_key = f"{seed}:{event['user_id']}:{event['product_id']}:{event['timestamp']}"
        rng = random.Random(int.from_bytes(hashlib.sha256(event_key.encode()).digest()[:8], "big"))
        candidates = [event["product_id"]] + rng.sample(available, min(candidate_count - 1, len(available)))
        rng.shuffle(candidates)
        eligible.append({"user_id": event["user_id"], "target_id": event["product_id"], "timestamp": event["timestamp"],
                         "candidate_ids": candidates, "history_events": len(prior),
                         "history_max_timestamp": max((row["timestamp"] for row in prior), default=None),
                         "selection_hash": hashlib.sha256(event_key.encode()).hexdigest()})
    counts["eligible_events"] = len(eligible)
    eligible.sort(key=lambda case: case["selection_hash"])
    return eligible[:limit], dict(counts)


def evaluate_preferences(cases, model, personal_weight, k=10):
    if not 0 <= personal_weight <= 1:
        raise ValueError("Personal fusion weight must be in [0,1]")
    rows = []
    for case in cases:
        ids, user = case["candidate_ids"], case["user_id"]
        pop_order = sorted(ids, key=lambda pid: (-model.popularity.get(pid, 0), pid))
        pop_ranks = {pid: rank for rank, pid in enumerate(pop_order)}
        cold = user not in model.users
        effective = 0.0 if cold else personal_weight
        personal_order = sorted(ids, key=lambda pid: (-model.score(user, pid), pid))
        personal_ranks = {pid: rank for rank, pid in enumerate(personal_order)}
        fused = sorted(ids, key=lambda pid: ((1 - effective) * pop_ranks[pid] + effective * personal_ranks[pid], pid))
        target_rank = fused.index(case["target_id"]) + 1
        hit = target_rank <= k
        rows.append({**case, "cold_user": cold, "target_rank": target_rank, "known_positive_recall@10": float(hit),
                     "known_positive_ndcg@10": 1 / math.log2(target_rank + 1) if hit else 0.0})
    def aggregate(values):
        return {"events": len(values), **{key: sum(row[key] for row in values) / len(values) if values else None
                for key in ("known_positive_recall@10", "known_positive_ndcg@10")}}
    return {"metrics": aggregate(rows), "cold_users": aggregate([row for row in rows if row["cold_user"]]),
            "warm_users": aggregate([row for row in rows if not row["cold_user"]]), "per_event": rows}


def paired_user_bootstrap(baseline_rows, personalized_rows, repetitions=2000, seed=20261009):
    """Report user-cluster uncertainty without changing any selected parameters."""
    if len(baseline_rows) != len(personalized_rows) or not baseline_rows:
        raise ValueError("Paired bootstrap needs aligned nonempty predictions")
    groups = defaultdict(list)
    for baseline, personal in zip(baseline_rows, personalized_rows):
        for field in ("user_id", "target_id", "timestamp", "candidate_ids"):
            if baseline[field] != personal[field]:
                raise ValueError("Paired predictions use different user events or candidates")
        groups[baseline["user_id"]].append((personal["known_positive_ndcg@10"] - baseline["known_positive_ndcg@10"],
                                             personal["known_positive_recall@10"] - baseline["known_positive_recall@10"]))
    summaries = [(sum(pair[0] for pair in values), sum(pair[1] for pair in values), len(values))
                 for values in groups.values()]
    rng = random.Random(seed)
    draws = [[], []]
    for _ in range(repetitions):
        total_ndcg, total_recall, count = 0.0, 0.0, 0
        for _ in summaries:
            ndcg, recall, events = summaries[rng.randrange(len(summaries))]
            total_ndcg += ndcg
            total_recall += recall
            count += events
        draws[0].append(total_ndcg / count)
        draws[1].append(total_recall / count)
    result = {"unit": "users resampled with replacement; all sampled test events retained per drawn user",
              "users": len(groups), "events": len(baseline_rows), "repetitions": repetitions, "seed": seed}
    for index, name in enumerate(("known_positive_ndcg@10", "known_positive_recall@10")):
        ordered = sorted(draws[index])
        result[name] = {"delta_personalized_minus_popularity": sum(summary[index] for summary in summaries) / len(baseline_rows),
                        "percentile_95_interval": [ordered[int(repetitions * 0.025)], ordered[min(repetitions - 1, int(repetitions * 0.975))]]}
    return result


class BeautyRecommender:
    """Serve the frozen Beauty model using only train-period reviewed history."""
    def __init__(self, directory):
        from .data import file_sha256, load_jsonl
        root = Path(directory)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        required = ("catalog.jsonl", "train.jsonl", "model.json")
        for name in required:
            fingerprint = manifest.get("files", {}).get(name)
            if not fingerprint or file_sha256(root / name) != fingerprint["sha256"] or (root / name).stat().st_size != fingerprint["bytes"]:
                raise ValueError(f"Beauty serving artifact fingerprint differs or is absent: {name}")
        frozen = json.loads((root / "model.json").read_text(encoding="utf-8"))
        if frozen.get("frozen_before_test") is not True:
            raise ValueError("Beauty serving needs a validation-frozen model")
        for name in ("catalog.jsonl", "train.jsonl"):
            if frozen["data_fingerprints"].get(name) != manifest["files"][name]:
                raise ValueError("Beauty model and catalog/train binding disagree")
        self.model = BPRModel.from_dict(frozen["model"])
        self.personal_weight = frozen["selection"]["personal_weight"]
        if not isinstance(self.personal_weight, (int, float)) or not 0 <= self.personal_weight <= 1:
            raise ValueError("Invalid frozen Beauty fusion weight")
        self.selection = frozen["selection"]
        self.source = frozen["source"]
        self.catalog = {row["id"]: row for row in load_jsonl(root / "catalog.jsonl")}
        self.users = sorted(self.model.users)
        self.reviewed = defaultdict(set)
        train = load_jsonl(root / "train.jsonl")
        cutoff = frozen["cutoffs"]["train_end_unix_seconds_exclusive"]
        for event in train:
            if event.get("split") != "train" or event["timestamp"] >= cutoff:
                raise ValueError("Beauty serving history contains a post-training event")
            self.reviewed[event["user_id"]].add(event["product_id"])
        if {event["product_id"] for event in train} != set(self.model.items) or not set(self.model.items) <= set(self.catalog):
            raise ValueError("Beauty trained items and serving catalog differ")
        self.history_scope = "Only frozen training-period review history; validation/test interactions are not read for recommendation"

    def recommend(self, user_id, k=5):
        _identifier(user_id, "user_id")
        if type(k) is not int or not 1 <= k <= 100:
            raise ValueError("Recommendation k must be an integer in [1,100]")
        ids = sorted(set(self.model.items) - self.reviewed.get(user_id, set()))
        cold = user_id not in self.model.users
        weight = 0.0 if cold else self.personal_weight
        pop_order = sorted(ids, key=lambda pid: (-self.model.popularity.get(pid, 0), pid))
        pop_rank = {pid: rank for rank, pid in enumerate(pop_order)}
        scores = {pid: self.model.score(user_id, pid) for pid in ids}
        personal_order = sorted(ids, key=lambda pid: (-scores[pid], pid))
        personal_rank = {pid: rank for rank, pid in enumerate(personal_order)}
        fused = sorted(ids, key=lambda pid: ((1 - weight) * pop_rank[pid] + weight * personal_rank[pid], pid))
        return [{**self.catalog[pid], "product_id": pid, "personal_score": scores[pid],
                 "train_positive_reviews": self.model.popularity.get(pid, 0), "cold_user": cold,
                 "personal_weight": weight, "selected_epochs": self.selection["epochs"],
                 "source": "Amazon Luxury Beauty review ratings; separate from ESCI search", "history_scope": self.history_scope}
                for pid in fused[:k]]
