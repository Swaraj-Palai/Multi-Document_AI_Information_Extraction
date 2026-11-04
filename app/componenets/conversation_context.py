"""
Conversation-aware RAG Module

Analyzes the current user question using relevant information from previous
messages in the same conversation. It does NOT generate the final answer.

Responsibilities:
  1. Fetch recent conversation history from SQLite (bounded count/tokens).
  2. Resolve pronouns ("it", "this", "that", "they") and references.
  3. Understand follow-ups ("explain more", "give an example", "compare it").
  4. Detect topic shifts (user started a completely new topic).
  5. Produce a contextualized retrieval query (for document search).
  6. Preserve the original question for final answer generation.

Previous messages help understand the QUESTION; uploaded documents provide
the EVIDENCE for answers. Previous assistant answers are never treated as
unquestionable facts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from app.common.logger import get_logger
from app.common.custom_exceptions import CustomException
from app.config.config import (
    CONVERSATION_AWARE_ENABLED,
    CONVERSATION_AWARE_MAX_MESSAGES,
    CONVERSATION_AWARE_MAX_TOKENS,
    CONVERSATION_AWARE_USE_LLM,
    CONVERSATION_AWARE_NEW_TOPIC_THRESHOLD,
)

logger = get_logger(__name__)


@dataclass
class ConversationContext:
    """Structured result of conversation-aware analysis."""

    chat_id: Optional[str]
    original_question: str
    contextualized_question: str
    references_resolved: List[str] = field(default_factory=list)
    is_follow_up: bool = False
    topic_shift_detected: bool = False
    context_used: bool = False
    relevant_previous_messages: List[Dict[str, Any]] = field(default_factory=list)
    resolution_method: str = "none"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chat_id": self.chat_id,
            "original_question": self.original_question,
            "contextualized_question": self.contextualized_question,
            "references_resolved": self.references_resolved,
            "is_follow_up": self.is_follow_up,
            "topic_shift_detected": self.topic_shift_detected,
            "context_used": self.context_used,
            "relevant_previous_messages": self.relevant_previous_messages,
            "resolution_method": self.resolution_method,
        }


# ---------------------------------------------------------------------------
# Pronoun / reference resolution patterns (rule-based, no LLM required)
# ---------------------------------------------------------------------------

_FOLLOW_UP_PATTERNS = [
    (r"\b(explain more|elaborate|tell me more|go on|continue)\b", "explain_more"),
    (r"\b(give an example|an example|for example|example)\b", "give_example"),
    (r"\b(compare it|compare this|compare that|comparison)\b", "compare_it"),
    (r"\b(what about|how about)\b", "what_about"),
    (r"\b(in more detail|detailed|details)\b", "more_detail"),
    (r"\b(summarize|summary|recap)\b", "summarize"),
    (r"\b(the first|the second|the third|the last)\b", "reference_item"),
]

_ENTITY_EXTRACT_PATTERNS = [
    (r"\b(?:BGE[- ]M3|BGE M3)\b", "BGE-M3"),
    (r"\b(?:MiniLM|miniLM|Mini LM)\b", "MiniLM"),
    (r"\b(?:FAISS|faiss)\b", "FAISS"),
    (r"\b(?:LLM|llm)\b", "LLM"),
    (r"\b(?:RAG|rag)\b", "RAG"),
    (r"\b(?:PDF|pdf)\b", "PDF"),
    (r"\b(?:DOCX|docx)\b", "DOCX"),
    (r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", "proper_noun"),
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
    "those", "am",
}


def _simple_tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9]+(?:\.[A-Za-z0-9]+)*", text.lower())
    return [t for t in tokens if t and t not in _STOPWORDS and len(t) > 1]


def _extract_topics(text: str) -> List[str]:
    topics: List[str] = []
    seen: set = set()
    for pattern, label in _ENTITY_EXTRACT_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            term = match.group(0)
            key = term.lower()
            if key not in seen:
                seen.add(key)
                topics.append(term)
    for tok in _simple_tokenize(text):
        if tok not in seen and len(tok) > 3:
            seen.add(tok)
            topics.append(tok)
    return topics


# ---------------------------------------------------------------------------
# History retrieval
# ---------------------------------------------------------------------------

def get_recent_messages(
    chat_id: Optional[str],
    max_messages: int = CONVERSATION_AWARE_MAX_MESSAGES,
    max_tokens: int = CONVERSATION_AWARE_MAX_TOKENS,
) -> List[Dict[str, Any]]:
    if not chat_id:
        return []

    try:
        from app.componenets.history import get_messages
        all_msgs = get_messages(chat_id) or []
    except Exception as e:
        logger.warning(f"Could not fetch messages for chat '{chat_id}': {e}")
        return []

    if not all_msgs:
        return []

    recent = all_msgs[-max_messages:]

    selected: List[Dict[str, Any]] = []
    total_chars = 0
    for msg in reversed(recent):
        content = msg.get("message", "") or msg.get("content", "")
        approx_tokens = len(content) // 4 + 1
        if total_chars + approx_tokens * 4 > max_tokens * 4 and selected:
            break
        selected.insert(0, {
            "role": msg.get("role", "user"),
            "content": content,
            "timestamp": msg.get("timestamp", msg.get("time", "")),
        })
        total_chars += len(content)

    return selected


# ---------------------------------------------------------------------------
# Rule-based reference resolution
# ---------------------------------------------------------------------------

def _find_last_entity(messages: List[Dict[str, Any]]) -> Optional[str]:
    for msg in reversed(messages):
        content = msg.get("content", "")
        topics = _extract_topics(content)
        if topics:
            return topics[0]
    return None


def _find_last_assistant_topic(messages: List[Dict[str, Any]]) -> Optional[str]:
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            topics = _extract_topics(msg.get("content", ""))
            if topics:
                return topics[0]
    return None


def _resolve_rule_based(
    current_question: str,
    messages: List[Dict[str, Any]],
) -> tuple:
    resolved = current_question.strip()
    references: List[str] = []
    is_follow_up = False
    method = "rule-based"

    last_entity = _find_last_entity(messages)
    last_assistant_topic = _find_last_assistant_topic(messages)
    fallback_topic = last_assistant_topic or last_entity

    q_lower = current_question.lower().strip()

    for pattern, follow_type in _FOLLOW_UP_PATTERNS:
        if re.search(pattern, q_lower):
            is_follow_up = True
            break

    # Resolve pronouns (it, this, that, they, its, etc.) -> last known topic
    # Whenever the question contains a pronoun and we have a fallback topic,
    # attempt resolution. This handles both explicit follow-ups ("explain more")
    # and implicit ones ("What are its advantages?").
    if fallback_topic:
        pronoun_pattern = r"\b(it|this|that|they|them|their|its)\b"
        matches = list(re.finditer(pronoun_pattern, resolved, re.IGNORECASE))
        if matches:
            # Mark as follow-up when pronouns are present (it's a reference)
            is_follow_up = True
            parts = []
            last_idx = 0
            for m in matches:
                pronoun = m.group(0)
                parts.append(resolved[last_idx:m.start()])
                parts.append(fallback_topic)
                references.append(f"{pronoun} -> {fallback_topic}")
                last_idx = m.end()
            parts.append(resolved[last_idx:])
            resolved = "".join(parts)
            method = "rule-based"

    if re.search(r"\bcompare\s+(it|this|that)\b", q_lower, re.IGNORECASE) and fallback_topic:
        resolved = re.sub(
            r"\bcompare\s+(it|this|that)\b",
            f"compare {fallback_topic} with",
            resolved,
            flags=re.IGNORECASE,
        )
        references.append(f"compare it -> compare {fallback_topic} with")

    if re.search(r"\b(the (first|second|third|last))\b", q_lower, re.IGNORECASE):
        references.append("references previous list item")

    # Handle follow-up patterns without pronouns: append topic context
    if is_follow_up and fallback_topic and fallback_topic.lower() not in resolved.lower():
        follow_up_bare = re.match(
            r"^(explain more|elaborate|tell me more|go on|continue|give me an example|give an example|an example|for example|example|in more detail|detailed|details|summarize|summary|recap)\s*\.?$",
            resolved,
            re.IGNORECASE,
        )
        if follow_up_bare:
            resolved = f"{resolved} of {fallback_topic}"
            references.append(f"appended topic: {fallback_topic}")
        elif re.search(r"\b(the (first|second|third|last))\b", q_lower, re.IGNORECASE):
            # "Explain the second advantage" -> "Explain the second advantage of BGE-M3"
            resolved = f"{resolved} of {fallback_topic}"
            references.append(f"appended topic: {fallback_topic}")

    resolved = re.sub(r"\s+", " ", resolved).strip()

    return resolved, references, is_follow_up, method


# ---------------------------------------------------------------------------
# LLM-based reference resolution (optional)
# ---------------------------------------------------------------------------

_RESOLUTION_PROMPT = """
You are a Conversation Context Resolution Engine for a RAG system.

Your ONLY job is to rewrite the user's current question by resolving pronouns
and references using the conversation history. You do NOT answer the question.

CONVERSATION HISTORY (most recent last):
{history}

CURRENT USER QUESTION:
{question}

TASK:
- Replace pronouns (it, this, that, they, them, its) with what they refer to
  from the conversation history.
- Expand follow-ups like "explain more", "give an example", "compare it" into
  complete, self-contained questions.
- If the question is already self-contained, return it unchanged.
- Do NOT treat previous assistant answers as facts.
- Do NOT answer the question.
- Return ONLY the rewritten question, no explanation, no markdown, no quotes.

REWRITTEN QUESTION:
"""


def _resolve_llm_based(
    current_question: str,
    messages: List[Dict[str, Any]],
    llm,
) -> tuple:
    try:
        history_text = ""
        for m in messages[-6:]:
            role = m.get("role", "user")
            content = m.get("content", "")
            history_text += f"{role.capitalize()}: {content}\n"

        from langchain_core.prompts import PromptTemplate
        prompt = PromptTemplate(
            template=_RESOLUTION_PROMPT,
            input_variables=["history", "question"],
        ).format(history=history_text, question=current_question.strip())

        response = llm.invoke(prompt)
        raw = response.content if hasattr(response, "content") else str(response)
        raw = raw.strip()

        if raw.startswith('"') and raw.endswith('"'):
            raw = raw[1:-1].strip()
        if raw.startswith("'") and raw.endswith("'"):
            raw = raw[1:-1].strip()
        if raw.startswith("Rewritten Question:"):
            raw = raw.split(":", 1)[-1].strip()

        if not raw or len(raw) < 2:
            return current_question, [], False, "llm-failed"

        references = ["llm-resolved references"]
        is_follow_up = raw.lower().strip() != current_question.lower().strip()
        return raw, references, is_follow_up, "llm"
    except Exception as e:
        logger.warning(f"LLM reference resolution failed: {e}")
        return current_question, [], False, "llm-failed"


# ---------------------------------------------------------------------------
# Topic shift detection
# ---------------------------------------------------------------------------

def _detect_topic_shift(
    current_question: str,
    messages: List[Dict[str, Any]],
) -> bool:
    if not messages:
        return False

    current_topics = set(t.lower() for t in _extract_topics(current_question))
    if not current_topics:
        return False

    # If the question contains pronouns (it, this, that, they, etc.), it's
    # likely a reference to a previous topic, not a new topic.
    pronoun_pattern = r"\b(it|this|that|they|them|their|its|these|those)\b"
    if re.search(pronoun_pattern, current_question, re.IGNORECASE):
        return False

    prior_topics: set = set()
    for m in messages[-6:]:
        for t in _extract_topics(m.get("content", "")):
            prior_topics.add(t.lower())

    if not prior_topics:
        return False

    overlap = current_topics & prior_topics
    if len(overlap) == 0 and len(current_topics) > 0:
        return True

    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_conversation_context(
    chat_id: Optional[str],
    current_question: str,
    llm=None,
) -> ConversationContext:
    if not CONVERSATION_AWARE_ENABLED or not chat_id:
        return ConversationContext(
            chat_id=chat_id,
            original_question=current_question,
            contextualized_question=current_question,
            context_used=False,
            resolution_method="disabled",
        )

    messages = get_recent_messages(chat_id)

    history_for_resolution = [
        m for m in messages
        if m.get("content", "").strip() != current_question.strip()
    ]

    contextualized = current_question.strip()
    references: List[str] = []
    is_follow_up = False
    method = "none"
    context_used = False

    if history_for_resolution:
        topic_shift = _detect_topic_shift(current_question, history_for_resolution)

        if topic_shift:
            contextualized = current_question.strip()
            method = "topic-shift"
            context_used = False
        else:
            if CONVERSATION_AWARE_USE_LLM and llm is not None:
                contextualized, references, is_follow_up, method = _resolve_llm_based(
                    current_question, history_for_resolution, llm
                )
            else:
                contextualized, references, is_follow_up, method = _resolve_rule_based(
                    current_question, history_for_resolution
                )

            context_used = references or is_follow_up or method == "rule-based"
    else:
        method = "no-history"

    return ConversationContext(
        chat_id=chat_id,
        original_question=current_question,
        contextualized_question=contextualized if contextualized else current_question,
        references_resolved=references,
        is_follow_up=is_follow_up,
        topic_shift_detected=(method == "topic-shift"),
        context_used=context_used,
        relevant_previous_messages=history_for_resolution,
        resolution_method=method,
    )


def get_conversation_context_summary(context: ConversationContext) -> str:
    return (
        f"chat={context.chat_id} | follow_up={context.is_follow_up} | "
        f"topic_shift={context.topic_shift_detected} | "
        f"context_used={context.context_used} | method={context.resolution_method} | "
        f"orig='{context.original_question[:50]}' -> "
        f"ctx='{context.contextualized_question[:50]}'"
    )

