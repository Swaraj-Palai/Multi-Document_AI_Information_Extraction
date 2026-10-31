"""
Evidence-grounded answer generation with verification
"""
import json
import re
from typing import Dict, Any, List, Optional

from app.common.logger import get_logger
from app.common.custom_exceptions import CustomException
from app.componenets.load_llm import load_llm

logger = get_logger(__name__)


GROUNDED_PROMPT = """You are an evidence-grounded document question-answering assistant.

CRITICAL RULES:
1. Use ONLY the supplied document evidence to answer document-specific questions.
2. Never invent names, marks, dates, codes, or relationships.
3. Every important factual claim must be traceable to the evidence.
4. If a field is absent, say it was not found.
5. If unclear, mark it as uncertain.
6. Do NOT confuse student names with parent names.
7. Do NOT assign values to incorrect table rows.
8. Do NOT invent individual grades from an overall grade.
9. Distinguish extracted information from calculated observations.
10. Never return Page 0 - use 1-based page numbers.
11. Cite page numbers and provide evidence quotes when possible.

EVIDENCE:
{evidence}

QUESTION:
{question}

Provide a structured, grounded answer. If information is not found in the evidence, clearly state: "The provided document does not contain sufficient evidence to answer this question."
"""


def verify_evidence(claim: str, evidence_text: str, page: int) -> Dict[str, Any]:
    """Verify that evidence supports the claim"""
    if not evidence_text or not claim:
        return {"verified": False, "reason": "Missing evidence or claim"}
    
    # Normalize for comparison
    claim_norm = re.sub(r'\s+', ' ', claim.lower())
    evidence_norm = re.sub(r'\s+', ' ', evidence_text.lower())
    
    # Check if key parts of claim appear in evidence
    claim_words = set(re.findall(r'\w+', claim_norm))
    evidence_words = set(re.findall(r'\w+', evidence_norm))
    
    if not claim_words:
        return {"verified": False, "reason": "Empty claim"}
    
    overlap = len(claim_words.intersection(evidence_words)) / len(claim_words)
    
    if overlap < 0.3:
        return {"verified": False, "reason": "Low evidence overlap"}
    
    return {"verified": True, "overlap": overlap, "page": page}


def generate_grounded_answer(
    question: str,
    evidence_chunks: List[Dict[str, Any]],
    language: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate evidence-grounded answer with verification
    """
    try:
        llm = load_llm()
        if not llm:
            raise CustomException("LLM not available")
        
        # Build evidence context
        evidence_parts = []
        for i, chunk in enumerate(evidence_chunks):
            page = chunk.get("page", "N/A")
            source = chunk.get("source", "Document")
            content = chunk.get("content", "")
            evidence_parts.append(f"[Evidence {i+1} - {source} Page {page}]:\n{content}")
        
        evidence_text = "\n\n---\n\n".join(evidence_parts)
        
        lang_instruction = ""
        if language and language.strip().lower() not in ("", "auto"):
            lang_instruction = f"Answer in {language}. "
        
        prompt = GROUNDED_PROMPT.format(
            evidence=evidence_text,
            question=lang_instruction + question
        )
        
        response = llm.invoke(prompt)
        answer = response.content if hasattr(response, "content") else str(response)
        
        return {
            "answer": answer.strip(),
            "sources": evidence_chunks,
            "verified": True
        }
    except Exception as e:
        logger.error(f"Error generating grounded answer: {e}")
        return {
            "answer": "An error occurred while generating the answer. Please try again.",
            "sources": [],
            "verified": False,
            "error": str(e)
        }
