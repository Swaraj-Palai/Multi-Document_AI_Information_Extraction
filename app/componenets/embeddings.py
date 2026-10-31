from typing import Optional
from langchain_huggingface import HuggingFaceEmbeddings
from app.common.logger import get_logger
from app.common.custom_exceptions import CustomException
from app.config.config import EMBEDDING_MODEL, EMBEDDING_PROVIDER

logger = get_logger(__name__)

_cached_embedding_model: Optional[object] = None
_cached_model_name: Optional[str] = None


def get_embedding_model(model_name: Optional[str] = None) -> Optional[object]:
    """
    Get or initialize a cached embedding model.
    Supports BGE-M3 and HuggingFace embeddings.
    Reuses existing instance to avoid redundant model loads.
    """
    global _cached_embedding_model, _cached_model_name
    
    provider = EMBEDDING_PROVIDER
    target_model = model_name or EMBEDDING_MODEL

    if _cached_embedding_model is not None and _cached_model_name == f"{provider}:{target_model}":
        return _cached_embedding_model

    try:
        if provider == "bge-m3" or "bge-m3" in target_model.lower():
            logger.info(f"Initializing BGE-M3 embeddings (multilingual, 1024-dim)...")
            from app.componenets.embeddings_bge import get_bge_m3_embeddings
            model = get_bge_m3_embeddings()
            _cached_embedding_model = model
            _cached_model_name = f"bge-m3:{target_model}"
            logger.info("BGE-M3 embeddings initialized successfully.")
            return model
        else:
            logger.info(f"Initializing HuggingFace Embeddings model with: '{target_model}'...")
            model = HuggingFaceEmbeddings(
                model_name=target_model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
            _cached_embedding_model = model
            _cached_model_name = f"hf:{target_model}"
            logger.info(f"HuggingFace Embeddings model '{target_model}' initialized successfully.")
            return model
    except Exception as e:
        error_msg = CustomException(f"Failed to create embeddings with provider '{provider}'.", e)
        logger.error(str(error_msg))
        # Fallback to HuggingFace
        try:
            logger.info("Falling back to HuggingFace embeddings...")
            model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
            _cached_embedding_model = model
            _cached_model_name = "hf:fallback"
            return model
        except Exception as e2:
            logger.error(f"Fallback also failed: {e2}")
            return None

