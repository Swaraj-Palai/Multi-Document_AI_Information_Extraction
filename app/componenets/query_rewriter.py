"""
Query Rewriting Module (Advanced RAG - First LLM Stage)

Sits between User Query and Embedding Model in the RAG pipeline.
Interprets the user's question and rewrites/expands it to improve
semantic retrieval before the query is embedded and searched in the Vector DB.

This implements the "extra LLM before the Embedding Model" shown in the
reference RAG pipeline diagram.
"""

from typing import Optional
from langchain_core.prompts import PromptTemplate

from app.common.logger import get_logger
from app.common.custom_exceptions import CustomException

logger = get_logger(__name__)

QUERY_REWRITE_PROMPT = """You are a query rewriting assistant for a document-based RAG system.

Your task is to analyze the user's question and rewrite it into a clearer,
more retrieval-friendly form so that semantic search over document chunks
returns the most relevant results.

Guidelines:
1. Preserve the original intent and meaning of the question.
2. Make the query more specific, complete, and unambiguous.
3. If the question is short or vague, expand it with likely related terms
   and context that would appear in relevant documents.
4. If the question contains multiple sub-questions, combine them into a
   single coherent retrieval query.
5. Do NOT answer the question. Do NOT add explanations.
6. Return ONLY the rewritten query text, nothing else.

User Question:
{question}

Rewritten Query:"""


def get_query_rewrite_prompt_template() -> PromptTemplate:
    """Return PromptTemplate for query rewriting (first LLM stage)."""
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

    Args:
        query: Raw user question.
        llm: Initialized LLM instance (e.g. ChatGroq).
        language: Optional target language hint for rewriting.

    Returns:
        Rewritten query string. Falls back to the original query if
        rewriting fails or produces empty output.
    """
    if not query or not query.strip():
        return query

    try:
        prompt = get_query_rewrite_prompt_template().format(question=query.strip())

        if language and language.strip().lower() not in ("", "auto"):
            prompt = f"Respond in {language}.\n\n{prompt}"

        response = llm.invoke(prompt)
        rewritten = response.content if hasattr(response, "content") else str(response)
        rewritten = rewritten.strip()

        # Clean up common LLM wrappers
        if rewritten.startswith('"') and rewritten.endswith('"'):
            rewritten = rewritten[1:-1].strip()
        if rewritten.startswith("Rewritten Query:") or rewritten.startswith("Rewritten query:"):
            rewritten = rewritten.split(":", 1)[-1].strip()

        if not rewritten:
            logger.warning("Query rewriting returned empty output; using original query.")
            return query.strip()

        logger.info(f"Query rewritten: '{query.strip()}' -> '{rewritten}'")
        return rewritten

    except Exception as e:
        logger.warning(f"Query rewriting failed, using original query: {e}")
        return query.strip()