import os
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from app.common.logger import get_logger
from app.common.custom_exceptions import CustomException
from app.config.config import DATA_PATH, ALLOWED_EXTENSIONS
from app.componenets.document_loader import create_text_chunks, load_single_document

logger = get_logger(__name__)


def load_pdf_files() -> List[Document]:
    """Load existing PDF files from DATA_PATH directory (backwards compatibility)."""
    try:
        if not os.path.exists(DATA_PATH):
            raise CustomException(f"Data path '{DATA_PATH}' does not exist.")
        logger.info(f"Loading PDF files from {DATA_PATH}")

        loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        if not documents:
            logger.warning(f"No PDF files found in {DATA_PATH}")
        else:
            logger.info(f"Successfully fetched {len(documents)} document pages.")
        return documents
    except Exception as e:
        error_msg = CustomException("Failed to load PDF files from directory", e)
        logger.error(str(error_msg))
        return []


def load_all_documents(folder: str) -> List[Document]:
    """Load all supported documents from a directory using universal loaders."""
    documents: List[Document] = []
    if not os.path.exists(folder):
        return documents
    allowed = {"." + e for e in ALLOWED_EXTENSIONS}  # e.g. ".pdf"
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if not os.path.isfile(file_path):
            continue
        ext = os.path.splitext(filename)[1].lower()
        if ext in allowed:
            try:
                docs = load_single_document(file_path, original_filename=filename)
                documents.extend(docs)
            except Exception as e:
                logger.warning(f"Could not load '{filename}': {e}")
    return documents


__all__ = ["load_pdf_files", "create_text_chunks", "load_single_document"]