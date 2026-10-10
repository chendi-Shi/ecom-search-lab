"""Conservative English lexical parsing; no learned intent labels are claimed."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import re
import unicodedata

STOP_WORDS = frozenset("a an the i me my find show want need looking look for please with and or of to is are that which some can you buy under budget only".split())


def tokens(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).casefold()
    return re.findall(r"[^\W_]+(?:[.-][^\W_]+)*", text, flags=re.UNICODE)


@dataclass(frozen=True)
class QueryPlan:
    original: str
    positive_terms: tuple[str, ...]
    excluded_terms: tuple[str, ...]
    phrases: tuple[str, ...]
    numeric_terms: tuple[str, ...]
    warnings: tuple[str, ...]
    ranking_query: str

    def to_dict(self) -> dict:
        return asdict(self)


def plan_query(query: str) -> QueryPlan:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be non-empty text")
    if len(query) > 2000:
        raise ValueError("query must be <= 2000 characters")
    normalized = unicodedata.normalize("NFKC", query).casefold()
    # Share the feature extractor's exact heuristic. Leading '-word' is made
    # explicit before BOTH retrieval and ranking see the normalized query.
    ranking_query = re.sub(r'(?<!\S)-([^\W_]+)',r'without \1',normalized)
    from .ranking import analyze_lexical_query
    attributes = analyze_lexical_query(ranking_query)
    phrases = tuple(" ".join(tokens(p)) for p in re.findall(r'"([^"]+)"', ranking_query))
    terms = tuple(attributes["positive_terms"])
    negatives = tuple(attributes["excluded_terms"])
    if not terms:
        raise ValueError("query has no searchable positive terms")
    numeric = tuple(t for t in terms if any(c.isdigit() for c in t))
    warnings = []
    if negatives:
        warnings.append("Literal exclusions use catalog wording; missing or ambiguous attributes still require verification")
    if re.search(r"[$€¥]|\b(?:under|budget|cheaper|price)\b", normalized):
        warnings.append("ESCI has no price data; no budget filter was applied")
    if re.search(r"[^\x00-\x7f]", normalized):
        warnings.append("This frozen benchmark is US English; other languages are not validated")
    return QueryPlan(query, terms, negatives, phrases, numeric, tuple(warnings),ranking_query)
