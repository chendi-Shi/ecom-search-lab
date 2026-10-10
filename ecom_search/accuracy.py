"""CPU relevance reranking: frozen joint text encoder plus exported LambdaMART.

Training uses LightGBM; serving the numeric trees needs only Python. Labels and
held-out queries are never loaded by the serving path.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from functools import lru_cache
from pathlib import Path

from .cross_encoder import LocalCrossEncoder, product_text
from .data import file_sha256
from .index import catalog_digest
from .ranking import FEATURE_NAMES, FeatureExtractor, _tokens, _positive_tokens

RAW_CE_COLUMN = 12
BASE_ACCURACY_FEATURES = [*FEATURE_NAMES, "cross_encoder_logit"]
ACCURACY_FEATURES = [*BASE_ACCURACY_FEATURES, "ce_query_percentile", "ce_query_zscore"]
ACCURACY_SCHEMA_VERSION = 2


def complete_pool_features(base_rows):
    """Extend a complete query pool with tied CE percentile and population z-score.

    Input/order is preserved. Every caller must pass the whole candidate pool,
    never individual chunks: these two features depend on all pool logits.
    NumPy matches the validation probe's float64 mean/std arithmetic exactly.
    """
    if len(base_rows) == 0:
        return []
    import numpy as np
    rows = np.asarray(base_rows, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != len(BASE_ACCURACY_FEATURES) or not np.isfinite(rows).all():
        raise ValueError("base feature vector shape/finiteness mismatch")
    logits = rows[:, RAW_CE_COLUMN]
    count = len(rows)
    order = np.argsort(logits, kind="stable")
    percentile = np.zeros(count, dtype=np.float64)
    begin = 0
    while begin < count:
        end = begin + 1
        while end < count and logits[order[end]] == logits[order[begin]]:
            end += 1
        # Equivalent to (scipy.stats.rankdata(method='average') - 1)/(n-1).
        percentile[order[begin:end]] = ((begin + end - 1) / 2) / max(1, count - 1)
        begin = end
    deviation = float(logits.std())
    zscore = (logits - logits.mean()) / deviation if deviation > 1e-12 else np.zeros(count)
    result = np.column_stack((rows, percentile, zscore))
    if not np.isfinite(result).all():
        raise ValueError("nonfinite query-relative features")
    return result.tolist()


def build_pool_features(query, products, extractor, logits):
    """Shared training, corpus evaluation and serving complete-pool builder."""
    if len(products) != len(logits):
        raise ValueError("candidate/logit count mismatch")
    if len({p["id"] for p in products}) != len(products):
        raise ValueError("duplicate reranking candidates")
    return complete_pool_features([
        extractor.features(query, product) + [float(score)]
        for product, score in zip(products, logits)
    ])


class CompactFeatureExtractor(FeatureExtractor):
    """Identical lexical features without retaining every catalog trigram set.

    Document frequency/average length/known brands use the complete catalog.
    Expensive per-product statistics are computed on demand with a 512-item LRU.
    This keeps an 82k-item CPU demo from consuming gigabytes of Python sets.
    """
    def __init__(self, catalog):
        self.products, self._df = {}, Counter()
        brands, total_length = set(), 0
        for product in catalog:
            pid = product.get("id")
            if not isinstance(pid, str) or not pid or pid in self.products:
                raise ValueError("catalog IDs must be unique nonempty strings")
            self.products[pid] = dict(product)
            positive = _positive_tokens(_tokens(product.get("title", "")))
            self._df.update(set(positive))
            total_length += len(positive)
            brand = _tokens(product.get("brand", ""))
            if brand:
                brands.add(brand)
        self._size = len(self.products)
        self._avg_title_length = (total_length / self._size if self._size else 1.) or 1.
        self._brands = frozenset(brands)

    @lru_cache(maxsize=512)
    def _stats_by_id(self, pid):
        return self._make_stats(self.products[pid])

    def _product_stats(self, product):
        pid = str(product.get("id", ""))
        if pid not in self.products:
            raise ValueError("product is absent from catalog")
        return self._stats_by_id(pid)


def export_trees(booster, iterations):
    dump = booster.dump_model(num_iteration=iterations)
    if dump.get("num_class") != 1 or dump.get("average_output"):
        raise ValueError("only additive single-output regression trees are supported")
    if dump["feature_names"] != ACCURACY_FEATURES:
        raise ValueError("LightGBM feature schema mismatch")

    def convert(node):
        if "leaf_value" in node:
            return {"value": float(node["leaf_value"])}
        if node["decision_type"] != "<=":
            raise ValueError("categorical trees are unsupported")
        return {"feature": node["split_feature"], "threshold": float(node["threshold"]),
                "left": convert(node["left_child"]), "right": convert(node["right_child"])}
    return [convert(tree["tree_structure"]) for tree in dump["tree_info"]]


class TreeEnsemble:
    def __init__(self, trees):
        if not isinstance(trees, list) or not 1 <= len(trees) <= 1000:
            raise ValueError("invalid tree count")
        self.trees = trees
        count = 0

        def validate(node, depth=0):
            nonlocal count
            count += 1
            if depth > 32 or count > 100000 or not isinstance(node, dict):
                raise ValueError("invalid tree structure")
            if set(node) == {"value"}:
                values = [node["value"]]
            elif set(node) == {"feature", "threshold", "left", "right"}:
                if type(node["feature"]) is not int or not 0 <= node["feature"] < len(ACCURACY_FEATURES):
                    raise ValueError("invalid split feature")
                values = [node["threshold"]]
                validate(node["left"], depth + 1)
                validate(node["right"], depth + 1)
            else:
                raise ValueError("invalid tree node")
            if any(type(x) not in (int, float) or not math.isfinite(x) for x in values):
                raise ValueError("tree values must be finite")
        for tree in trees:
            validate(tree)

    def predict(self, rows):
        result = []
        for row in rows:
            if len(row) != len(ACCURACY_FEATURES) or any(not math.isfinite(float(x)) for x in row):
                raise ValueError("feature vector shape/finiteness mismatch")
            score = 0.0
            for tree in self.trees:
                node = tree
                while "value" not in node:
                    node = node["left"] if row[node["feature"]] <= node["threshold"] else node["right"]
                score += node["value"]
            if not math.isfinite(score):
                raise ValueError("nonfinite ensemble output")
            result.append(score)
        return result


def model_scores(model, features):
    if model.get("schema_version") != ACCURACY_SCHEMA_VERSION or model.get("feature_names") != ACCURACY_FEATURES:
        raise ValueError("unsupported accuracy model schema")
    if model["kind"] == "cross_encoder":
        if any(len(row) != len(ACCURACY_FEATURES) or any(not math.isfinite(float(value)) for value in row) for row in features):
            raise ValueError("feature vector shape/finiteness mismatch")
        return [float(row[RAW_CE_COLUMN]) for row in features]
    if model["kind"] == "lambdamart":
        return TreeEnsemble(model["trees"]).predict(features)
    raise ValueError("unsupported accuracy model kind")


class AccuracyRanker:
    def __init__(self, path, extractor, *, ce_model_path=None, cache_dir=None, encoder=None):
        self.path = Path(path)
        self.metadata = json.loads(self.path.read_text(encoding="utf-8"))
        model_scores(self.metadata, [])
        if self.metadata["catalog_sha256"] != catalog_digest(list(extractor.products.values())):
            raise ValueError("accuracy ranker catalog mismatch")
        self.extractor = extractor
        root = Path(__file__).resolve().parents[1]
        if self.metadata.get("ranking_source_sha256") != file_sha256(root / "ecom_search/ranking.py"):
            raise ValueError("accuracy feature implementation changed")
        if self.metadata.get("accuracy_source_sha256") != file_sha256(Path(__file__)):
            raise ValueError("accuracy implementation changed")
        ce_path = Path(ce_model_path) if ce_model_path else root / "outputs/cross-encoder-assets/ms-marco-MiniLM-L6-v2"
        self.encoder = encoder or LocalCrossEncoder(ce_path, cache_dir=cache_dir,
            threads=2, max_length=self.metadata["cross_encoder"]["max_length"], batch_size=32,
            precision=self.metadata["cross_encoder"]["config"]["precision"])
        if self.encoder.fingerprint != self.metadata["cross_encoder"]["fingerprint"]:
            raise ValueError("accuracy cross-encoder fingerprint mismatch")
        self.ensemble = TreeEnsemble(self.metadata["trees"]) if self.metadata["kind"] == "lambdamart" else None

    def score(self, query, products):
        if not products:
            return {}
        if len({p["id"] for p in products}) != len(products):
            raise ValueError("duplicate reranking candidates")
        logits = self.encoder.score_pairs([(query, product_text(p)) for p in products])
        rows = build_pool_features(query, products, self.extractor, logits)
        scores = self.ensemble.predict(rows) if self.ensemble else logits
        return {p["id"]: float(s) for p, s in zip(products, scores)}

    def close(self):
        close = getattr(self.encoder, "close", None)
        if close:
            close()
