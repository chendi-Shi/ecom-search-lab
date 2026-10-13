"""Shared, deterministic retrieval and ranking for the independent closure study.

The soft policy changes only natural-language hard exclusion. It keeps the
legacy positive query terms and ranking features, and still honors explicit
``-word`` exclusions using the same literal catalog heuristic as ProductIndex.
Both offline evaluation and the closure service call this module.
"""
from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
import re
from typing import Mapping, Sequence
import unicodedata

from .data import load_jsonl, normalized_query, validate_dataset
from .index import ProductIndex, catalog_digest
from .query import QueryPlan, plan_query
from .ranking import FeatureExtractor, PairwiseRanker


POLICIES = ("legacy", "soft")
RANKERS = ("bm25", "judged", "retrieved")
# Match the feature tokenizer's decimal form. A hyphen inside a token is not
# an exclusion operator; the minus must start at the beginning or whitespace.
_EXPLICIT_MINUS = re.compile(r"(?<!\S)-([^\W_]+(?:\.[0-9]+)?)", re.UNICODE)


def _validate_query(query: str) -> None:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be non-empty text")
    if len(query) > 2000:
        raise ValueError("query must be <= 2000 characters")


def _bounded_integer(value: int, name: str, maximum: int) -> None:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [1, {maximum}]")


def ranking_query(query: str) -> str:
    """Use the historical normalization, including its no-positive fallback."""
    _validate_query(query)
    try:
        return plan_query(query).ranking_query
    except ValueError:
        # Input type, emptiness, and length were checked above. The only
        # remaining legacy error is a query with no searchable positive terms.
        return normalized_query(query)


def query_plan(query: str, policy: str = "legacy") -> QueryPlan:
    """Return an empty plan for valid text with no searchable positive terms.

    Such queries remain in evaluation with zero retrieved candidates, exactly
    as in benchmark_public. Invalid input is still rejected, not swallowed.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown retrieval policy: {policy!r}")
    _validate_query(query)
    try:
        plan = plan_query(query)
    except ValueError:
        return QueryPlan(query, (), (), (), (),
                         ("Query has no searchable positive terms",),
                         normalized_query(query))
    if policy == "legacy":
        return plan
    normalized = unicodedata.normalize("NFKC", query).casefold()
    explicit = tuple(dict.fromkeys(_EXPLICIT_MINUS.findall(normalized)))
    warnings = tuple(message for message in plan.warnings
                     if not message.startswith("Literal exclusions use catalog wording"))
    if plan.excluded_terms or explicit:
        warnings += ("Natural-language exclusions are soft ranking cues; explicit -word "
                     "exclusions filter literal affirmed catalog terms",)
    return replace(plan, excluded_terms=explicit, warnings=warnings)


class ClosureEngine:
    """Catalog-bound search shared by fresh offline and online evaluation.

    ``models`` maps ``judged`` and/or ``retrieved`` to 12 finite weights. BM25
    always means SQLite's title+brand retrieval score, not a learned feature.
    Artifact provenance and release authorization are verified by the caller.
    """

    def __init__(self, catalog_path: str | Path, index_path: str | Path,
                 models: Mapping[str, Sequence[float]] | None = None):
        self.models: dict[str, PairwiseRanker] = {}
        if models is not None:
            if not isinstance(models, Mapping):
                raise ValueError("models must map ranker names to weight sequences")
            for name, weights in models.items():
                if name not in RANKERS[1:]:
                    raise ValueError(f"unknown learned ranker: {name!r}")
                if isinstance(weights, (str, bytes)):
                    raise ValueError("model weights must be a numeric sequence")
                self.models[name] = PairwiseRanker(list(weights))
        self.catalog = load_jsonl(catalog_path)
        validate_dataset(self.catalog, {})
        # No result-cache warmup advantage in the offline latency measurements.
        self.index = ProductIndex(index_path, cache_size=0)
        try:
            if self.index.metadata.get("catalog_sha256") != catalog_digest(self.catalog):
                raise ValueError("index catalog fingerprint differs from supplied catalog")
            self.extractor = FeatureExtractor(self.catalog)
        except BaseException:
            self.index.close()
            raise

    def close(self) -> None:
        self.index.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def retrieve(self, query: str, policy: str = "legacy", pool: int = 100) -> list[dict]:
        """Return ProductIndex hit dictionaries in deterministic retrieval order."""
        _bounded_integer(pool, "pool", 1000)
        return self.index.search(query_plan(query, policy), k=pool)

    def rank(self, query: str, hits: Sequence[Mapping], ranker: str = "bm25") -> list[dict]:
        """Return the complete pool as ``{id, score}``, descending score then ID."""
        query_text = ranking_query(query)
        if ranker not in RANKERS:
            raise ValueError(f"unknown ranker: {ranker!r}")
        if ranker != "bm25" and ranker not in self.models:
            raise ValueError(f"ranker weights were not loaded: {ranker}")
        seen: set[str] = set()
        ranked = []
        for hit in hits:
            product_id = hit.get("id")
            if product_id not in self.extractor.products or product_id in seen:
                raise ValueError("ranking hits must contain unique catalog product ids")
            seen.add(product_id)
            if ranker == "bm25":
                raw_score = hit.get("retrieval_score")
                if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
                    raise ValueError("retrieval_score must be numeric and finite")
                score = float(raw_score)
                if not math.isfinite(score):
                    raise ValueError("retrieval_score must be numeric and finite")
            else:
                product = self.extractor.products[product_id]
                score = self.models[ranker].score(self.extractor.features(query_text, product))
            ranked.append({"id": product_id, "score": score})
        return sorted(ranked, key=lambda item: (-item["score"], item["id"]))

    def search(self, query: str, policy: str = "legacy", ranker: str = "bm25",
               pool: int = 100, k: int = 10) -> dict:
        """Return the full ranking plus the top-k catalog result snapshots.

        ``candidate_ids`` preserves retrieval order. ``ranking`` contains every
        candidate exactly once, so evaluators need no independent ranking path.
        """
        _bounded_integer(pool, "pool", 1000)
        _bounded_integer(k, "k", pool)
        hits = self.retrieve(query, policy, pool)
        ranking = self.rank(query, hits, ranker)
        hit_scores = {hit["id"]: hit["retrieval_score"] for hit in hits}
        results = [dict(self.extractor.products[row["id"]], score=row["score"],
                        retrieval_score=hit_scores[row["id"]]) for row in ranking[:k]]
        return {"query": query, "policy": policy, "ranker": ranker,
                "pool": pool, "k": k, "plan": query_plan(query, policy).to_dict(),
                "candidate_ids": [hit["id"] for hit in hits],
                "ranking": ranking, "results": results}
