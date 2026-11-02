"""
Query Rewriting Module (Advanced RAG - First LLM Stage)
"""
from typing import Optional
from langchain_core.prompts import PromptTemplate

from app.common.logger import get_logger
from app.common.custom_exceptions import CustomException

logger = get_logger(__name__)

QUERY_REWRITE_PROMPT = """You are an expert query optimizer for document-based RAG systems.

Your task is to rewrite the user's question to maximize retrieval accuracy from a vector database.

Rules:
1. Preserve the EXACT intent and core question - do not change what is being asked.
2. Make it more specific and retrieval-friendly by adding relevant keywords that would appear in documents.
3. For ambiguous pronouns (it, this, that), infer from context if possible, else keep them neutral.
4. For questions about specific entities (names, numbers, dates), ensure they are explicit.
5. For multi-part questions, keep them coherent and focused.
6. Use natural language - optimize for semantic search with dense embeddings.
7. DO NOT answer the question. DO NOT add explanations, reasoning, or preamble.
8. DO NOT use quotes around your response.
9. Return ONLY the rewritten query text.

User Question:
{question}

Optimized Query for Retrieval:"""


def get_query_rewrite_prompt_template() -> PromptTemplate:
    """Return PromptTemplate for query rewriting."""
    return PromptTemplate(
        template=QUERY_REWRITE_PROMPT,
        input_variables=["question"]
    )


def rewrite_query(
    query: str,
    llm,
    language: Optional[str] = None
) -> str:
    """
    Rewrite/expand the user query using an LLM to improve retrieval quality.
    """
    if not query or not query.strip():
        return query

    try:
        prompt = get_query_rewrite_prompt_template().format(question=query.strip())

        if language and language.strip().lower() not in ("", "auto"):
            prompt = f"Respond in {language}. Keep the query optimized for document retrieval.\n\n{prompt}"

        response = llm.invoke(prompt)
        rewritten = response.content if hasattr(response, "content") else str(response)
        rewritten = rewritten.strip()

        # Clean up common LLM wrappers
        if rewritten.startswith('"') and rewritten.endswith('"'):
            rewritten = rewritten[1:-1].strip()
        if rewritten.startswith("'") and rewritten.endswith("'"):
            rewritten = rewritten[1:-1].strip()
        if rewritten.startswith("Optimized Query for Retrieval:") or rewritten.startswith("Rewritten Query:") or rewritten.startswith("Rewritten query:"):
            rewritten = rewritten.split(":", 1)[-1].strip()

        if not rewritten or len(rewritten) < 3:
            logger.warning("Query rewriting returned empty/invalid output; using original query.")
            return query.strip()

        logger.info(f"Query rewritten: '{query.strip()}' -> '{rewritten}'")
        return rewritten

    except Exception as e:
        logger.warning(f"Query rewriting failed, using original query: {e}")
        return query.strip()
