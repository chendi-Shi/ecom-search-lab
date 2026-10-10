"""Small, inspectable CPU learning-to-rank implementation using only Python.

Features depend on the query and catalog fields, never relevance labels. All
features have a fixed [0, 1] scale; no statistics are fitted on held-out labels.
BM25 is a title-only baseline, while the learned model also sees brand and
description. The linguistic features are deliberately conservative heuristics:
English stopwords/negation, literal catalog-brand mentions, and numeric values
within the same explicit unit (or bare-number slot). They are not a multilingual
semantic parser, unit converter, price filter, or proof of attribute compliance.
Missing attributes do not count as conflicts. Brand homonyms and multiple bare
model numbers can cause errors, which should be reviewed in query slices.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import math
import random
import re
import unicodedata
from typing import Any, Mapping


FEATURE_VERSION = 1
MODEL_VERSION = 1
FEATURE_NAMES = [
    "title_bm25", "title_token_coverage", "description_token_coverage",
    "char_trigram_similarity", "brand_match", "brand_conflict",
    "number_match", "number_conflict", "bigram_match", "phrase_match",
    "negation_conflict", "negation_satisfied",
]
_STOPWORDS = frozenset("a an the for of to in on with and or is are be by any please me i want looking find".split())
_NEGATORS = frozenset(("not", "no", "without", "excluding", "exclude", "non"))
_TOKEN = re.compile(r"[^\W_]+(?:\.[0-9]+)?", re.UNICODE)
_NUMBER = re.compile(
    r"(?<![\w.])(\d+(?:\.\d+)?)\s*"
    r"(inches|inch|mah|pack|pcs|gb|tb|mb|cm|mm|kg|ml|oz|hz|w|v|g|l|in)?(?![\w.])",
    re.IGNORECASE,
)
_UNIT_ALIASES = {"inches": "in", "inch": "in", "pcs": "pack"}


def _normalize(value: Any) -> str:
    return unicodedata.normalize("NFKC", str(value or "")).casefold()


def _tokens(value: Any) -> tuple[str, ...]:
    return tuple(_TOKEN.findall(_normalize(value)))


def _negated(tokens: tuple[str, ...]) -> frozenset[str]:
    """Only the next content token is scoped; '<attribute> free' is supported."""
    result: set[str] = set()
    for index, token in enumerate(tokens):
        if token in _NEGATORS:
            for following in tokens[index + 1:index + 4]:
                if following in _NEGATORS or following in ("and", "or"):
                    break
                if following not in _STOPWORDS:
                    result.add(following)
                    break
        elif token == "free" and index and tokens[index - 1] not in _STOPWORDS:
            result.add(tokens[index - 1])
    return frozenset(result)


def _positive_tokens(tokens: tuple[str, ...]) -> tuple[str, ...]:
    forbidden = _negated(tokens)
    return tuple(t for t in tokens if t not in _STOPWORDS and t not in _NEGATORS
                 and t != "free" and t not in forbidden)


def analyze_lexical_query(query: str) -> dict:
    """Expose the exact existing ranking token/negation rules to retrieval.

    This does not alter feature extraction or a saved model's interpretation.
    Exclusions are lexical soft constraints, not guarantees of product semantics.
    Explicit '-term' syntax must be normalized to 'without term' by the caller
    before both this function and FeatureExtractor receive the query.
    """
    if not isinstance(query, str):
        raise ValueError("query must be text")
    tokens = _tokens(query)
    forbidden = _negated(tokens)
    return {
        "positive_terms": tuple(dict.fromkeys(_positive_tokens(tokens))),
        "excluded_terms": tuple(dict.fromkeys(token for token in tokens if token in forbidden)),
    }


def _trigrams(tokens: tuple[str, ...]) -> frozenset[str]:
    # Per-word padding gives a typo feature without creating cross-word grams.
    return frozenset(padded[i:i + 3] for token in tokens for padded in ("^^" + token + "$$",)
                     for i in range(len(padded) - 2))


def _numbers(text: str) -> frozenset[tuple[str, str]]:
    result = set()
    for match in _NUMBER.finditer(_normalize(text)):
        whole, _, fractional = match.group(1).partition(".")
        whole = whole.lstrip("0") or "0"
        fractional = fractional.rstrip("0")
        number = whole + ("." + fractional if fractional else "")
        unit = match.group(2) or ""
        result.add((number, _UNIT_ALIASES.get(unit, unit)))
    return frozenset(result)


def _affirmed_numbers(tokens: tuple[str, ...]) -> frozenset[tuple[str, str]]:
    # Keep unit tokens that are also stopwords (notably inches abbreviated 'in').
    forbidden = _negated(tokens)
    return _numbers(" ".join(t for t in tokens if t not in forbidden
                             and t not in _NEGATORS and t != "free"))


def _contains_phrase(haystack: tuple[str, ...], needle: tuple[str, ...]) -> bool:
    return bool(needle) and any(haystack[i:i + len(needle)] == needle
                               for i in range(len(haystack) - len(needle) + 1))


def _finite_vector(values: Any, length: int, label: str) -> list[float]:
    if not isinstance(values, (list, tuple)) or len(values) != length:
        raise ValueError(f"{label} must contain exactly {length} values")
    result = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label} must contain numeric, finite values")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"{label} must contain numeric, finite values")
        result.append(value)
    return result


@dataclass(frozen=True)
class _ProductStats:
    title_tokens: tuple[str, ...]
    title_positive: tuple[str, ...]
    title_counts: Counter
    description_positive: frozenset[str]
    trigrams: frozenset[str]
    brand: tuple[str, ...]
    numbers: frozenset[tuple[str, str]]
    affirmed: frozenset[str]
    negated: frozenset[str]


class FeatureExtractor:
    """Catalog-only token/DF cache. ``products`` maps string IDs to snapshots."""

    names = list(FEATURE_NAMES)

    @staticmethod
    def affirmed_product_terms(product: Mapping) -> frozenset[str]:
        """Shared literal constraint check; negated title terms are not affirmed.

        This is a heuristic over supplied catalog text, not a verified product
        attribute extractor. Missing attributes cannot establish compliance.
        """
        # The constraint pass needs no Counter, trigrams or numeric parsing.
        return (frozenset(_positive_tokens(_tokens(product.get("title",""))))
                | frozenset(_positive_tokens(_tokens(product.get("description",""))))
                | frozenset(_tokens(product.get("brand",""))))

    def __init__(self, catalog: list[dict]):
        self.products: dict[str, dict] = {}
        self._stats: dict[str, _ProductStats] = {}
        self._df: Counter = Counter()
        for product in catalog:
            if not isinstance(product, Mapping) or product.get("id") is None:
                raise ValueError("each product must have a nonempty id")
            product_id = str(product["id"])
            if not product_id or product_id in self.products:
                raise ValueError(f"duplicate or empty product id: {product_id!r}")
            snapshot = dict(product, id=product_id)
            self.products[product_id] = snapshot
            stats = self._make_stats(snapshot)
            self._stats[product_id] = stats
            self._df.update(set(stats.title_positive))
        self._size = len(self.products)
        self._avg_title_length = (
            sum(len(s.title_positive) for s in self._stats.values()) / self._size
            if self._size else 1.0
        ) or 1.0
        self._brands = frozenset(s.brand for s in self._stats.values() if s.brand)

    @staticmethod
    def _make_stats(product: Mapping) -> _ProductStats:
        title = _tokens(product.get("title", ""))
        description = _tokens(product.get("description", ""))
        brand = _tokens(product.get("brand", ""))
        # Keep negation scope within each field; a title ending in 'no' must not
        # accidentally negate the first word of an unrelated description.
        negated = _negated(title) | _negated(description)
        positive = _positive_tokens(title)
        affirmed = frozenset(positive) | frozenset(_positive_tokens(description)) | frozenset(brand)
        numeric_values = _affirmed_numbers(title) | _affirmed_numbers(description)
        return _ProductStats(title, positive, Counter(positive),
                             frozenset(_positive_tokens(description)), _trigrams(positive),
                             brand, numeric_values, affirmed, negated)

    def _product_stats(self, product: dict) -> _ProductStats:
        product_id = str(product.get("id", ""))
        if product_id not in self._stats:
            raise ValueError(f"product is absent from catalog: {product_id!r}")
        return self._stats[product_id]

    @lru_cache(maxsize=2048)
    def _query_stats(self, query: str) -> tuple:
        tokens = _tokens(query)
        positive = _positive_tokens(tokens)
        brands = frozenset(brand for brand in self._brands if _contains_phrase(tokens, brand))
        return tokens, positive, _trigrams(positive), brands, _affirmed_numbers(tokens), _negated(tokens)

    def bm25(self, query: str, product: dict) -> float:
        """Nonnegative title-only BM25 with k1=1.2, b=0.75, catalog DF."""
        stats = self._product_stats(product)
        query_terms = set(self._query_stats(query)[1])
        k1, b = 1.2, 0.75
        normalizer = k1 * (1 - b + b * len(stats.title_positive) / self._avg_title_length)
        result = 0.0
        for token in query_terms:
            tf = stats.title_counts.get(token, 0)
            if tf:
                idf = math.log1p((self._size - self._df[token] + 0.5) / (self._df[token] + 0.5))
                result += idf * tf * (k1 + 1) / (tf + normalizer)
        return result

    def features(self, query: str, product: dict) -> list[float]:
        stats = self._product_stats(product)
        _, positive, grams, brands, numbers, forbidden = self._query_stats(query)
        terms = set(positive)
        title_terms = set(stats.title_positive)
        bm25 = self.bm25(query, product)
        matched_numbers = numbers & stats.numbers
        product_units = {unit for _, unit in stats.numbers}
        conflicting_numbers = {pair for pair in numbers - stats.numbers if pair[1] in product_units}
        query_bigrams = set(zip(positive, positive[1:]))
        title_bigrams = set(zip(stats.title_positive, stats.title_positive[1:]))
        values = [
            bm25 / (1 + bm25),
            len(terms & title_terms) / len(terms) if terms else 0.0,
            len(terms & stats.description_positive) / len(terms) if terms else 0.0,
            2 * len(grams & stats.trigrams) / (len(grams) + len(stats.trigrams))
            if grams or stats.trigrams else 0.0,
            float(bool(brands) and stats.brand in brands),
            float(bool(brands) and bool(stats.brand) and stats.brand not in brands),
            len(matched_numbers) / len(numbers) if numbers else 0.0,
            len(conflicting_numbers) / len(numbers) if numbers else 0.0,
            len(query_bigrams & title_bigrams) / len(query_bigrams) if query_bigrams else 0.0,
            float(_contains_phrase(stats.title_positive, positive)),
            len(forbidden & stats.affirmed) / len(forbidden) if forbidden else 0.0,
            len(forbidden & stats.negated) / len(forbidden) if forbidden else 0.0,
        ]
        return [float(value) for value in values]


def _record_field(record: Any, name: str, default: Any = None) -> Any:
    return record.get(name, default) if isinstance(record, Mapping) else getattr(record, name, default)


class PairwiseRanker:
    """Linear model trained by seeded pairwise logistic SGD on judged candidates.

    There is no bias: it cancels within a query. At most 100 unequal-grade pairs
    per query are sampled uniformly within each record without replacement. Unjudged candidates
    never become negatives. The model stores weights only, not labels/records.
    """

    def __init__(self, weights: list | None = None):
        self.weights = _finite_vector(weights if weights is not None else [1.0] + [0.0] * (len(FEATURE_NAMES) - 1),
                                      len(FEATURE_NAMES), "weights")

    def score(self, features: list[float]) -> float:
        values = _finite_vector(features, len(FEATURE_NAMES), "features")
        result = sum(weight * value for weight, value in zip(self.weights, values))
        if not math.isfinite(result):
            raise ValueError("non-finite model score")
        return result

    @staticmethod
    def _loss(weights: list[float], pairs: list[list[float]], l2: float) -> float:
        if not pairs:
            return 0.0
        total = 0.0
        for difference in pairs:
            margin = sum(w * x for w, x in zip(weights, difference))
            total += max(-margin, 0.0) + math.log1p(math.exp(-abs(margin)))
        return total / len(pairs) + 0.5 * l2 * sum(w * w for w in weights)

    def fit(self, train_records: list, extractor: FeatureExtractor, epochs: int = 30,
            learning_rate: float = 0.05, seed: int = 42, l2: float = 0.001) -> dict:
        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 0:
            raise ValueError("epochs must be a nonnegative integer")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        for value, name in ((learning_rate, "learning_rate"), (l2, "l2")):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if learning_rate <= 0 or l2 < 0:
            raise ValueError("learning_rate must be positive and l2 nonnegative")
        if extractor.names != FEATURE_NAMES:
            raise ValueError("feature schema mismatch")
        self.weights = _finite_vector(self.weights, len(FEATURE_NAMES), "weights")
        rng = random.Random(seed)
        pairs: list[list[float]] = []
        sampled_per_query: dict[str, int] = {}
        for record in train_records:
            split = _record_field(record, "split")
            if split is not None and str(split).casefold() not in ("train", "training"):
                raise ValueError("fit accepts training records only; held-out labels are forbidden")
            query = _record_field(record, "query")
            qrels = _record_field(record, "qrels")
            candidate_ids = _record_field(record, "candidate_ids")
            if not isinstance(query, str) or not isinstance(qrels, Mapping) or not isinstance(candidate_ids, (list, tuple)):
                raise ValueError("records require query, qrels mapping and candidate_ids list")
            normalized_qrels = {}
            for product_id, grade in qrels.items():
                product_id = str(product_id)
                if product_id in normalized_qrels:
                    raise ValueError("duplicate relevance product id after normalization")
                if isinstance(grade, bool) or not isinstance(grade, (int, float)) or not math.isfinite(grade) or grade not in (0, 1, 2, 3):
                    raise ValueError("relevance grades must be integers from 0 through 3")
                normalized_qrels[product_id] = int(grade)
            groups: dict[int, list[str]] = {}
            for product_id in sorted(set(map(str, candidate_ids)) & normalized_qrels.keys()):
                if product_id not in extractor.products:
                    raise ValueError(f"judged candidate is absent from catalog: {product_id!r}")
                groups.setdefault(normalized_qrels[product_id], []).append(product_id)
            blocks = []
            total_pairs = 0
            for high in sorted(groups, reverse=True):
                for low in sorted(groups, reverse=True):
                    if high > low:
                        size = len(groups[high]) * len(groups[low])
                        blocks.append((total_pairs, size, groups[high], groups[low]))
                        total_pairs += size
            if not total_pairs:
                continue
            remaining = 100 - sampled_per_query.get(query, 0)
            sample_count = min(remaining, total_pairs)
            if not sample_count:
                continue
            sampled_per_query[query] = sampled_per_query.get(query, 0) + sample_count
            vector_cache = {}
            for index in rng.sample(range(total_pairs), sample_count):
                for offset, size, highs, lows in blocks:
                    if offset <= index < offset + size:
                        local = index - offset
                        high_id, low_id = highs[local // len(lows)], lows[local % len(lows)]
                        for product_id in (high_id, low_id):
                            if product_id not in vector_cache:
                                vector_cache[product_id] = extractor.features(query, extractor.products[product_id])
                        pairs.append([a - b for a, b in zip(vector_cache[high_id], vector_cache[low_id])])
                        break
        history = [self._loss(self.weights, pairs, l2)]
        if not math.isfinite(history[0]):
            raise ValueError("initial training loss is non-finite; use smaller weights/l2")
        for _ in range(epochs):
            rng.shuffle(pairs)
            for difference in pairs:
                margin = sum(w * x for w, x in zip(self.weights, difference))
                # Stable sigmoid(-margin), the negative logistic loss gradient.
                if margin >= 0:
                    exp_negative = math.exp(-margin)
                    gradient_scale = exp_negative / (1 + exp_negative)
                else:
                    gradient_scale = 1 / (1 + math.exp(margin))
                self.weights = [w + learning_rate * (gradient_scale * x - l2 * w)
                                for w, x in zip(self.weights, difference)]
                if not all(math.isfinite(w) for w in self.weights):
                    raise ValueError("training diverged; reduce learning_rate")
            history.append(self._loss(self.weights, pairs, l2))
            if not math.isfinite(history[-1]):
                raise ValueError("training loss diverged; reduce learning_rate")
        return {
            "epochs": epochs, "seed": seed, "learning_rate": learning_rate, "l2": l2,
            "training_records": len(train_records), "queries_with_pairs": len(sampled_per_query),
            "pairs": len(pairs), "pair_count": len(pairs), "max_pairs_per_query": 100,
            "initial_loss": history[0], "final_loss": history[-1], "loss_history": history,
            "feature_version": FEATURE_VERSION, "feature_names": list(FEATURE_NAMES),
            "weights": dict(zip(FEATURE_NAMES, self.weights)),
        }

    def to_dict(self) -> dict:
        return {"model": "pairwise_logistic_linear", "model_version": MODEL_VERSION,
                "feature_version": FEATURE_VERSION, "feature_names": list(FEATURE_NAMES),
                "weights": _finite_vector(self.weights, len(FEATURE_NAMES), "weights")}

    @classmethod
    def from_dict(cls, obj: dict) -> PairwiseRanker:
        if not isinstance(obj, Mapping) or obj.get("model") != "pairwise_logistic_linear":
            raise ValueError("unsupported model format")
        if type(obj.get("model_version")) is not int or obj["model_version"] != MODEL_VERSION:
            raise ValueError("unsupported model version")
        if type(obj.get("feature_version")) is not int or obj["feature_version"] != FEATURE_VERSION:
            raise ValueError("unsupported feature version")
        if obj.get("feature_names") != FEATURE_NAMES:
            raise ValueError("feature names/order mismatch")
        if "weights" not in obj or obj["weights"] is None:
            raise ValueError("serialized model has no weights")
        return cls(obj["weights"])
