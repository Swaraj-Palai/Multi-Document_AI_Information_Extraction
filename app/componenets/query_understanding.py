"""
Query Understanding Module (Advanced RAG - Pre-Retrieval Stage)

Analyzes the user's question BEFORE document retrieval and produces
structured retrieval instructions. It does NOT generate the final answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from langchain_core.prompts import PromptTemplate

from app.common.logger import get_logger
from app.common.custom_exceptions import CustomException
from app.config.config import (
    QUERY_UNDERSTANDING_ENABLED,
    QUERY_UNDERSTANDING_USE_LLM,
    QUERY_UNDERSTANDING_DEFAULT_TOP_K,
    QUERY_UNDERSTANDING_COMPLEX_BOOST_TOP_K,
    TOP_K,
)

logger = get_logger(__name__)


@dataclass
class QueryUnderstanding:
    """Structured understanding of a user question before retrieval."""

    original_query: str
    cleaned_query: str
    intent: str = "unknown"
    confidence: float = 0.0
    entities: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    complexity: str = "simple"
    requires_documents: bool = True
    suggested_top_k: int = TOP_K
    search_strategy: str = "semantic"
    needs_query_rewrite: bool = True
    language: Optional[str] = None
    instructions: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_query": self.original_query,
            "cleaned_query": self.cleaned_query,
            "intent": self.intent,
            "confidence": self.confidence,
            "entities": self.entities,
            "keywords": self.keywords,
            "constraints": self.constraints,
            "complexity": self.complexity,
            "requires_documents": self.requires_documents,
            "suggested_top_k": self.suggested_top_k,
            "search_strategy": self.search_strategy,
            "needs_query_rewrite": self.needs_query_rewrite,
            "language": self.language,
            "instructions": self.instructions,
        }


# ---------------------------------------------------------------------------
# Rule-based patterns (no LLM required)
# ---------------------------------------------------------------------------

_INTENT_PATTERNS = [
    (r"\b(what is|what are|define|definition of|meaning of|stands for)\b", "definition", 0.85),
    (r"\b(explain|how does|how do|why does|why do|what causes|what is the reason)\b", "explanation", 0.85),
    (r"\b(compare|comparison|difference between|versus|vs\.|better|worse|similarities|contrast)\b", "comparison", 0.85),
    (r"\b(list|enumerate|what are the|name the|all the|examples of|types of|kinds of)\b", "listing", 0.80),
    (r"\b(how to|steps|procedure|process|how can i|method to|way to|guide|tutorial)\b", "procedure", 0.85),
    (r"\b(calculate|compute|how much|how many|what is the total|sum|average|percentage|ratio|formula)\b", "calculation", 0.80),
    (r"\b(who|when|where|which|what year|what date|in what year)\b", "factual", 0.75),
    (r"\b(hello|hi|hey|greetings|good morning|good afternoon|good evening)\b", "greeting", 0.90),
    (r"\b(help|what can you do|how does this work|instructions|about this app|capabilities)\b", "meta", 0.85),
]

_COMPLEXITY_SIGNALS = [
    r"\b(and|or|but|however|although|whereas|meanwhile|additionally|furthermore|moreover)\b",
    r"\b(before|after|during|since|until|between|from|to|through)\b",
    r"\b(not|without|except|excluding|rather than|instead of)\b",
    r"\b(more than|less than|greater than|fewer than|above|below|over|under)\b",
    r"\b(compare|contrast|versus|vs\.|difference|similar|relationship|correlation)\b",
    r"\b(if|then|unless|provided that|assuming|given that)\b",
    r"\?.*\?",
]

_CONSTRAINT_PATTERNS = [
    (r"\b(not|without|except|excluding|exclude|other than|besides)\b", "exclusion"),
    (r"\b(more than|greater than|over|above)\s*(\d+)", "minimum"),
    (r"\b(less than|fewer than|under|below)\s*(\d+)", "maximum"),
    (r"\b(between)\s*(\d+)\s*(and|to|-)\s*(\d+)", "range"),
    (r"\b(\d{4})\b", "year"),
    (r"\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b", "date"),
    (r"\b(\d+(?:\.\d+)?)\s*(%|percent)\b", "percentage"),
    (r"\b(in|on|at|from|since|before|after|during|by)\s+(\w+)\s+(\d{4})\b", "temporal"),
]

_ENTITY_PATTERNS = [
    (r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", "proper_noun"),
    (r"\b(?:AI|ML|API|HTTP|URL|PDF|DOCX|OCR|LLM|RAG|FAISS|SQL|CPU|GPU|USB|HTML|CSS|JSON|XML|AWS|GCP|Azure)\b", "acronym"),
]

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "must", "shall", "to", "of", "in",
    "for", "on", "with", "at", "by", "from", "as", "into", "through",
    "during", "before", "after", "above", "below", "between", "under",
    "again", "further", "then", "once", "here", "there", "when", "where",
    "why", "how", "all", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than",
    "too", "very", "just", "don", "now", "i", "me", "my", "myself", "we",
    "our", "ours", "ourselves", "you", "your", "yours", "yourself",
    "yourselves", "he", "him", "his", "himself", "she", "her", "hers",
    "herself", "it", "its", "itself", "they", "them", "their", "theirs",
    "themselves", "what", "which", "who", "whom", "this", "that", "these",
    "those", "am", "is", "are", "was", "were", "be", "been", "being",
}


def _simple_tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9]+(?:\.[A-Za-z0-9]+)*", text.lower())
    return [t for t in tokens if t and t not in _STOPWORDS and len(t) > 1]


# ---------------------------------------------------------------------------
# LLM-powered refinement (optional)
# ---------------------------------------------------------------------------

_QUERY_UNDERSTANDING_PROMPT = """
You are a Query Understanding Engine for a Retrieval-Augmented Generation (RAG) system.

Your ONLY job is to analyze the user's question and produce a structured JSON object
describing how the retrieval system should process it. You do NOT answer the question.

Analyze:
1. intent: one of [definition, explanation, comparison, listing, procedure, calculation, factual, greeting, meta, unknown]
2. entities: important people, places, organizations, technologies, documents mentioned
3. keywords: the most important retrieval terms (lowercase, no stopwords)
4. constraints: exclusions, numeric thresholds, dates, ranges, conditions
5. complexity: simple | moderate | complex
6. requires_documents: true if the answer likely depends on uploaded documents, false if general knowledge / greeting
7. suggested_top_k: number of document chunks to retrieve (4 default, up to 12)
8. search_strategy: semantic | keyword | hybrid

Respond with ONLY valid JSON, no markdown, no explanation, no extra text.

USER QUESTION:
{question}

JSON OUTPUT:
"""


def _llm_refine(query: str, understanding: QueryUnderstanding, llm) -> QueryUnderstanding:
    try:
        prompt = PromptTemplate(
            template=_QUERY_UNDERSTANDING_PROMPT,
            input_variables=["question"],
        ).format(question=query.strip())

        response = llm.invoke(prompt)
        raw = response.content if hasattr(response, "content") else str(response)
        raw = raw.strip()

        if raw.startswith("```"):
            raw = raw.strip("`")
            first_nl = raw.find("\n")
            if first_nl != -1:
                raw = raw[first_nl + 1:]
        if raw.endswith("```"):
            raw = raw[:-3]

        import json
        data = json.loads(raw)

        if isinstance(data, dict):
            if data.get("intent"):
                understanding.intent = str(data["intent"])
            if data.get("confidence") is not None:
                try:
                    understanding.confidence = float(data["confidence"])
                except (TypeError, ValueError):
                    pass
            if isinstance(data.get("entities"), list):
                understanding.entities = [str(e) for e in data["entities"]]
            if isinstance(data.get("keywords"), list):
                understanding.keywords = [str(k) for k in data["keywords"]]
            if isinstance(data.get("constraints"), dict):
                understanding.constraints = data["constraints"]
            if data.get("complexity") in ("simple", "moderate", "complex"):
                understanding.complexity = str(data["complexity"])
            if isinstance(data.get("requires_documents"), bool):
                understanding.requires_documents = bool(data["requires_documents"])
            if data.get("suggested_top_k") is not None:
                try:
                    understanding.suggested_top_k = int(data["suggested_top_k"])
                except (TypeError, ValueError):
                    pass
            if data.get("search_strategy") in ("semantic", "keyword", "hybrid"):
                understanding.search_strategy = str(data["search_strategy"])
            if isinstance(data.get("needs_query_rewrite"), bool):
                understanding.needs_query_rewrite = bool(data["needs_query_rewrite"])

        logger.info("LLM query understanding refinement applied.")
    except Exception as e:
        logger.warning(f"LLM query understanding refinement failed, keeping rule-based result: {e}")

    return understanding


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def _classify_intent(query_lower: str) -> tuple:
    best_intent = "unknown"
    best_conf = 0.30

    for pattern, intent, conf in _INTENT_PATTERNS:
        if re.search(pattern, query_lower, re.IGNORECASE):
            if conf > best_conf:
                best_intent = intent
                best_conf = conf

    return best_intent, best_conf


def _extract_entities_and_keywords(query: str) -> tuple:
    entities: List[str] = []
    seen_ent: set = set()

    for pattern, _label in _ENTITY_PATTERNS:
        for match in re.finditer(pattern, query):
            ent = match.group(0)
            if ent not in seen_ent:
                seen_ent.add(ent)
                entities.append(ent)

    tokens = _simple_tokenize(query)
    entity_lower = {e.lower() for e in entities}
    keywords = [t for t in tokens if t not in entity_lower]

    seen_kw: set = set()
    deduped_keywords: List[str] = []
    for kw in keywords:
        if kw not in seen_kw:
            seen_kw.add(kw)
            deduped_keywords.append(kw)

    return entities, deduped_keywords


def _extract_constraints(query_lower: str) -> Dict[str, Any]:
    constraints: Dict[str, Any] = {}

    exclusions: List[str] = []
    for match in re.finditer(r"\b(not|without|except|excluding|exclude|other than|besides)\b\s+(\w+)", query_lower):
        exclusions.append(match.group(2))
    if exclusions:
        constraints["exclusions"] = exclusions

    minimums: List[str] = []
    for match in re.finditer(r"\b(more than|greater than|over|above)\s*(\d+)", query_lower):
        minimums.append(match.group(2))
    if minimums:
        constraints["minimum_values"] = minimums

    maximums: List[str] = []
    for match in re.finditer(r"\b(less than|fewer than|under|below)\s*(\d+)", query_lower):
        maximums.append(match.group(2))
    if maximums:
        constraints["maximum_values"] = maximums

    ranges: List[str] = []
    for match in re.finditer(r"\bbetween\s*(\d+)\s*(and|to|-)\s*(\d+)", query_lower):
        ranges.append(f"{match.group(1)} to {match.group(3)}")
    if ranges:
        constraints["numeric_ranges"] = ranges

    years: List[str] = []
    for match in re.finditer(r"\b(\d{4})\b", query_lower):
        years.append(match.group(1))
    if years:
        constraints["years"] = years

    percentages: List[str] = []
    for match in re.finditer(r"\b(\d+(?:\.\d+)?)\s*(%|percent)\b", query_lower):
        percentages.append(match.group(1) + "%")
    if percentages:
        constraints["percentages"] = percentages

    return constraints


def _assess_complexity(query, query_lower, intent, entities, keywords, constraints) -> str:
    score = 0
    word_count = len(query.split())
    if word_count > 25:
        score += 2
    elif word_count > 12:
        score += 1

    if query.count("?") > 1:
        score += 2

    signal_hits = sum(1 for pat in _COMPLEXITY_SIGNALS if re.search(pat, query_lower))
    score += min(signal_hits, 3)
    score += len(constraints)

    if len(entities) + len(keywords) > 8:
        score += 1

    if intent in ("comparison", "procedure", "calculation"):
        score += 1

    if score >= 5:
        return "complex"
    if score >= 2:
        return "moderate"
    return "simple"


def _build_instructions(understanding: QueryUnderstanding) -> str:
    parts: List[str] = []

    parts.append(f"Intent: {understanding.intent} (confidence {understanding.confidence:.2f})")
    parts.append(f"Complexity: {understanding.complexity}")
    parts.append(f"Requires documents: {understanding.requires_documents}")
    parts.append(f"Suggested top_k: {understanding.suggested_top_k}")
    parts.append(f"Search strategy: {understanding.search_strategy}")
    parts.append(f"Query rewrite needed: {understanding.needs_query_rewrite}")

    if understanding.entities:
        parts.append(f"Entities: {', '.join(understanding.entities)}")
    if understanding.keywords:
        parts.append(f"Keywords: {', '.join(understanding.keywords[:10])}")
    if understanding.constraints:
        parts.append(f"Constraints: {understanding.constraints}")

    if understanding.intent == "comparison":
        parts.append("Retrieval guidance: retrieve passages mentioning BOTH compared entities; look for comparison tables or 'vs' sections.")
    elif understanding.intent == "procedure":
        parts.append("Retrieval guidance: prioritize numbered steps, headings, and 'how-to' sections.")
    elif understanding.intent == "calculation":
        parts.append("Retrieval guidance: prioritize passages with numeric values, formulas, percentages, and tables.")
    elif understanding.intent == "listing":
        parts.append("Retrieval guidance: prioritize bulleted/numbered lists and enumerated sections.")
    elif understanding.intent == "definition":
        parts.append("Retrieval guidance: prioritize glossary-style definitions and 'is/are' statements.")

    if understanding.constraints.get("exclusions"):
        parts.append("Filter guidance: exclude passages primarily about the excluded entities.")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def understand_query(
    query: str,
    llm=None,
    language: Optional[str] = None,
) -> QueryUnderstanding:
    """
    Analyze the user's question BEFORE document retrieval.

    Returns a QueryUnderstanding object with intent, entities, keywords,
    constraints, complexity, and retrieval instructions.

    Does NOT generate the final answer.
    """
    if not query or not query.strip():
        return QueryUnderstanding(
            original_query=query,
            cleaned_query="",
            intent="unknown",
            confidence=0.0,
            requires_documents=True,
            suggested_top_k=TOP_K,
            instructions="Empty query; no retrieval needed.",
        )

    cleaned = " ".join(query.strip().split())
    query_lower = cleaned.lower()

    intent, conf = _classify_intent(query_lower)
    entities, keywords = _extract_entities_and_keywords(cleaned)
    constraints = _extract_constraints(query_lower)
    complexity = _assess_complexity(
        cleaned, query_lower, intent, entities, keywords, constraints
    )

    suggested_top_k = QUERY_UNDERSTANDING_DEFAULT_TOP_K
    requires_documents = True
    needs_rewrite = True
    search_strategy = "semantic"

    if intent == "greeting":
        requires_documents = False
        suggested_top_k = 0
        needs_rewrite = False
        search_strategy = "semantic"
    elif intent == "meta":
        requires_documents = False
        suggested_top_k = 0
        needs_rewrite = False
        search_strategy = "semantic"
    elif intent == "calculation":
        suggested_top_k = min(QUERY_UNDERSTANDING_COMPLEX_BOOST_TOP_K, 12)
        search_strategy = "keyword"
    elif intent == "comparison":
        suggested_top_k = min(QUERY_UNDERSTANDING_COMPLEX_BOOST_TOP_K, 12)
        search_strategy = "hybrid"
    elif complexity == "complex":
        suggested_top_k = min(QUERY_UNDERSTANDING_COMPLEX_BOOST_TOP_K, 12)
        search_strategy = "hybrid"
    elif complexity == "moderate":
        suggested_top_k = min(QUERY_UNDERSTANDING_DEFAULT_TOP_K + 2, 10)
        search_strategy = "semantic"
    # BUG FIX: "unknown" or low-confidence intents still need to retrieve
    # documents. Previously these fell through with no override, defaulting to
    # top_k but with a poor keyword set. We now extract and keep keywords,
    # ensuring the query still retrieves relevant document chunks.
    elif intent == "unknown":
        # Extract any meaningful keywords from the query and boost top_k
        # so the retrieval has enough candidates to find relevant chunks.
        if keywords:
            suggested_top_k = min(max(kw.count for kw in keywords if kw), 8) or QUERY_UNDERSTANDING_DEFAULT_TOP_K
        else:
            suggested_top_k = min(QUERY_UNDERSTANDING_DEFAULT_TOP_K + 2, 10)
        search_strategy = "hybrid"

    understanding = QueryUnderstanding(
        original_query=query,
        cleaned_query=cleaned,
        intent=intent,
        confidence=conf,
        entities=entities,
        keywords=keywords,
        constraints=constraints,
        complexity=complexity,
        requires_documents=requires_documents,
        suggested_top_k=suggested_top_k,
        search_strategy=search_strategy,
        needs_query_rewrite=needs_rewrite,
        language=language,
    )

    understanding.instructions = _build_instructions(understanding)

    if QUERY_UNDERSTANDING_USE_LLM and llm is not None:
        understanding = _llm_refine(cleaned, understanding, llm)
        understanding.instructions = _build_instructions(understanding)

    logger.info(
        f"Query understood: intent={understanding.intent}, "
        f"complexity={understanding.complexity}, "
        f"top_k={understanding.suggested_top_k}, "
        f"requires_docs={understanding.requires_documents}"
    )

    return understanding


def get_query_understanding_summary(understanding: QueryUnderstanding) -> str:
    return (
        f"[{understanding.intent}] [{understanding.complexity}] "
        f"top_k={understanding.suggested_top_k} "
        f"docs_required={understanding.requires_documents} "
        f":: {understanding.cleaned_query[:80]}"
    )

