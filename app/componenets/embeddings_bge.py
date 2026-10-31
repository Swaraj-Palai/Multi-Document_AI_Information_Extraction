"""
BGE-M3 Multilingual Embedding Service
Uses sentence-transformers for BGE-M3 with CPU-only inference
"""
import os
from typing import List, Optional, Any
from pathlib import Path

from app.common.logger import get_logger
from app.common.custom_exceptions import CustomException
from app.config.config import (
    BGE_M3_MODEL_PATH, BGE_M3_DEVICE, BGE_M3_BATCH_SIZE, 
    BGE_M3_MAX_LENGTH, EMBEDDING_PROVIDER
)

logger = get_logger(__name__)

_cached_bge_model = None
_cached_bge_path = None


class BGEM3Embeddings:
    """LangChain-compatible wrapper for BGE-M3"""
    def __init__(self, model=None):
        self.model = model
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        if self.model is None:
            raise CustomException("BGE-M3 model not initialized")
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=BGE_M3_BATCH_SIZE,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            if hasattr(embeddings, 'tolist'):
                return embeddings.tolist()
            return [e.tolist() if hasattr(e, 'tolist') else list(e) for e in embeddings]
        except Exception as e:
            logger.error(f"Error generating document embeddings: {e}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        if not text:
            return []
        result = self.embed_documents([text])
        return result[0] if result else []
    
    def __call__(self, text: str) -> List[float]:
        return self.embed_query(text)


def get_bge_m3_model():
    """Get or initialize cached BGE-M3 model"""
    global _cached_bge_model, _cached_bge_path
    
    if _cached_bge_model is not None:
        return _cached_bge_model
    
    try:
        model_path = Path(BGE_M3_MODEL_PATH)
        if model_path.exists() and (model_path / 'config.json').exists():
            logger.info(f"Loading BGE-M3 from local path: {model_path}")
            load_path = str(model_path)
        else:
            logger.info("Loading BGE-M3 from HuggingFace: BAAI/bge-m3")
            load_path = "BAAI/bge-m3"
        
        from sentence_transformers import SentenceTransformer
        logger.info(f"Initializing BGE-M3 (device={BGE_M3_DEVICE}, batch_size={BGE_M3_BATCH_SIZE})...")
        model = SentenceTransformer(load_path, device=BGE_M3_DEVICE)
        _cached_bge_model = model
        _cached_bge_path = load_path
        logger.info(f"BGE-M3 initialized successfully ({model.get_sentence_embedding_dimension()}-dim embeddings)")
        return model
    except Exception as e:
        error_msg = CustomException("Failed to initialize BGE-M3 model", e)
        logger.error(str(error_msg))
        raise


def get_bge_m3_embeddings() -> BGEM3Embeddings:
    """Get LangChain-compatible BGE-M3 embeddings instance"""
    model = get_bge_m3_model()
    return BGEM3Embeddings(model)


def is_bge_m3_available() -> bool:
    """Check if BGE-M3 is available/configured"""
    try:
        model_path = Path(BGE_M3_MODEL_PATH)
        return model_path.exists() or EMBEDDING_PROVIDER == "bge-m3"
    except:
        return False


