import os
import re
from typing import List, Optional, Tuple

import pymupdf as fitz  # PyMuPDF: text-layer detection for scanned PDFs
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredFileLoader,
    CSVLoader,
    UnstructuredMarkdownLoader,
    UnstructuredHTMLLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.common.logger import get_logger
from app.common.custom_exceptions import CustomException
from app.config.config import CHUNK_SIZE, CHUNK_OVERLAP, OCR_TEXT_THRESHOLD
from app.componenets.ocr_engine import ocr_engine, OCRPageResult

logger = get_logger(__name__)


def clean_text(text: str) -> str:
    """Normalize whitespace and strip null/control characters from extracted text."""
    if not text:
        return ""
    # Remove null characters
    text = text.replace("\x00", "")
    # Replace multiple empty lines with maximum 2 newlines
    text = re.sub(r"\n\s*\n", "\n\n", text)
    # Replace weird space sequences with standard spaces
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


import hashlib
import uuid


def generate_document_id(filename: str = "") -> str:
    """Generate a unique document identifier like 'doc_8f31a92'.

    Random suffix ensures two different documents with the same filename
    never collide (filename alone is not reliable).
    """
    return "doc_" + uuid.uuid4().hex[:7]


def _pdf_has_text_layer(file_path: str, threshold: int = 50) -> Tuple[bool, List[int]]:
    """
    Inspect a PDF with PyMuPDF and report whether any page lacks an
    extractable text layer (i.e. is likely scanned / image-based).

    Returns (needs_ocr, page_indices_without_text).
    """
    pages_without_text: List[int] = []
    try:
        doc = fitz.open(file_path)
        for i, page in enumerate(doc):
            text = (page.get_text() or "").strip()
            if len(text) < threshold:
                pages_without_text.append(i)
        doc.close()
    except Exception as e:
        logger.warning(
            f"PyMuPDF text-layer inspection failed for '{file_path}': {e}"
        )
        # If we cannot inspect, be conservative and assume OCR may be needed.
        return True, []
    return len(pages_without_text) > 0, pages_without_text


def load_single_document(file_path: str, original_filename: Optional[str] = None, document_id: Optional[str] = None) -> List[Document]:
    """
    Load a PDF or DOCX file, extract text, clean it, and attach consistent metadata including document_id.
    """
    try:
        if not os.path.exists(file_path):
            raise CustomException(f"File not found: {file_path}")

        file_size = os.path.getsize(file_path)
        if file_size == 0:
            raise CustomException("Uploaded file is empty (0 bytes).")

        display_name = original_filename if original_filename else os.path.basename(file_path)
        ext = os.path.splitext(file_path)[1].lower()
        doc_id = document_id or generate_document_id(display_name)

        logger.info(f"Extracting content from '{display_name}' (type: {ext}, size: {file_size} bytes, id: {doc_id})")

        raw_docs: List[Document] = []

        if ext == ".pdf":
            # Determine whether the PDF is scanned / image-based.
            # Indra OCR (auto mode) uses the existing text layer when present
            # and applies OCR only to pages that lack extractable text.
            needs_ocr, sparse_pages = _pdf_has_text_layer(
                file_path, threshold=OCR_TEXT_THRESHOLD
            )

            if needs_ocr and ocr_engine.is_available():
                logger.info(
                    f"PDF '{display_name}' has {len(sparse_pages)} page(s) "
                    f"without a usable text layer; using Indra OCR (auto mode)."
                )
                try:
                    ocr_results: List[OCRPageResult] = ocr_engine.extract_pdf(
                        file_path
                    )
                    for result in ocr_results:
                        cleaned = clean_text(result.text)
                        if cleaned:
                            raw_docs.append(Document(
                                page_content=cleaned,
                                metadata={
                                    "source": display_name,
                                    "page": result.page_number,
                                    "file_type": "pdf",
                                    "document_id": doc_id,
                                    "extraction_method": result.extraction_method,
                                    "ocr_confidence": result.confidence,
                                    "ocr_engine": "indra"
                                }
                            ))
                except Exception as e:
                    logger.warning(f"OCR failed, falling back to text extraction: {e}")
                    # Fallback to PyPDF
                    try:
                        loader = PyPDFLoader(file_path)
                        pages = loader.load()
                        for idx, page in enumerate(pages):
                            page_num = page.metadata.get("page", idx)
                            cleaned = clean_text(page.page_content)
                            if cleaned:
                                raw_docs.append(Document(
                                    page_content=cleaned,
                                    metadata={
                                        "source": display_name,
                                        "page": int(page_num) + 1,
                                        "file_type": "pdf",
                                        "document_id": doc_id,
                                        "extraction_method": "text",
                                        "ocr_confidence": 0.0,
                                    }
                                ))
                    except Exception as e2:
                        raise CustomException(
                            f"OCR extraction failed for PDF '{display_name}'. "
                            f"The file may be corrupted or unsupported.", e2
                        )
            else:
                try:
                    loader = PyPDFLoader(file_path)
                    pages = loader.load()
                    for idx, page in enumerate(pages):
                        page_num = page.metadata.get("page", idx)
                        cleaned = clean_text(page.page_content)
                        if cleaned:
                            raw_docs.append(Document(
                                page_content=cleaned,
                                metadata={
                                    "source": display_name,
                                    "page": int(page_num) + 1,  # 1-indexed for display
                                    "file_type": "pdf",
                                    "document_id": doc_id,
                                    "extraction_method": "text",
                                    "ocr_confidence": 0.0,
                                }
                            ))
                except Exception as e:
                    raise CustomException(f"Failed to parse PDF document '{display_name}'. The file may be corrupted or password protected.", e)

        elif ext == ".docx":
            docx_text = ""
            # Try primary Docx2txtLoader
            try:
                loader = Docx2txtLoader(file_path)
                docx_docs = loader.load()
                for doc in docx_docs:
                    c = clean_text(doc.page_content)
                    if c:
                        docx_text += c + "\n\n"
            except Exception as e_docx2txt:
                logger.warning(f"Docx2txtLoader failed on '{display_name}', attempting python-docx fallback: {e_docx2txt}")

            # Fallback to python-docx if empty or failed
            if not docx_text.strip():
                try:
                    from docx import Document as DocxReader
                    doc_reader = DocxReader(file_path)
                    paragraphs = [p.text for p in doc_reader.paragraphs if p.text and p.text.strip()]
                    for table in doc_reader.tables:
                        for row in table.rows:
                            row_text = " | ".join(c.text.strip() for c in row.cells if c.text.strip())
                            if row_text:
                                paragraphs.append(row_text)
                    docx_text = "\n\n".join(paragraphs)
                except Exception as e_pydocx:
                    raise CustomException(f"Failed to parse DOCX document '{display_name}'. The file may be invalid or corrupted.", e_pydocx)

            cleaned = clean_text(docx_text)
            if cleaned:
                raw_docs.append(Document(
                    page_content=cleaned,
                    metadata={
                        "source": display_name,
                        "page": "N/A",  # DOCX does not have fixed pages
                        "file_type": "docx",
                        "document_id": doc_id
                    }
                ))

        elif ext == ".txt":
            try:
                loader = TextLoader(file_path, encoding="utf-8")
                pages = loader.load()
                for idx, page in enumerate(pages):
                    cleaned = clean_text(page.page_content)
                    if cleaned:
                        raw_docs.append(Document(
                            page_content=cleaned,
                            metadata={"source": display_name, "page": idx + 1,
                                      "file_type": "txt", "document_id": doc_id,
                                      "extraction_method": "text"}
                        ))
            except Exception as e:
                raise CustomException(f"Failed to parse TXT document '{display_name}'.", e)

        elif ext == ".csv":
            try:
                loader = CSVLoader(file_path, encoding="utf-8")
                pages = loader.load()
                for idx, page in enumerate(pages):
                    cleaned = clean_text(page.page_content)
                    if cleaned:
                        raw_docs.append(Document(
                            page_content=cleaned,
                            metadata={"source": display_name, "page": idx + 1,
                                      "file_type": "csv", "document_id": doc_id,
                                      "extraction_method": "text"}
                        ))
            except Exception as e:
                raise CustomException(f"Failed to parse CSV document '{display_name}'.", e)

        elif ext == ".md":
            try:
                loader = UnstructuredMarkdownLoader(file_path)
                pages = loader.load()
                for idx, page in enumerate(pages):
                    cleaned = clean_text(page.page_content)
                    if cleaned:
                        raw_docs.append(Document(
                            page_content=cleaned,
                            metadata={"source": display_name, "page": idx + 1,
                                      "file_type": "md", "document_id": doc_id,
                                      "extraction_method": "text"}
                        ))
            except Exception as e:
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        md_text = f.read()
                    cleaned = clean_text(md_text)
                    if cleaned:
                        raw_docs.append(Document(
                            page_content=cleaned,
                            metadata={"source": display_name, "page": 1,
                                      "file_type": "md", "document_id": doc_id,
                                      "extraction_method": "text"}
                        ))
                except Exception as e2:
                    raise CustomException(f"Failed to parse Markdown document '{display_name}'.", e2)

        elif ext == ".json":
            try:
                import json as _json
                with open(file_path, "r", encoding="utf-8") as f:
                    data = _json.load(f)
                json_text = _json.dumps(data, indent=2, ensure_ascii=False)
                cleaned = clean_text(json_text)
                if cleaned:
                    raw_docs.append(Document(
                        page_content=cleaned,
                        metadata={"source": display_name, "page": 1,
                                  "file_type": "json", "document_id": doc_id,
                                  "extraction_method": "text"}
                    ))
            except Exception as e:
                raise CustomException(f"Failed to parse JSON document '{display_name}'.", e)

        elif ext in (".html", ".htm"):
            try:
                loader = UnstructuredHTMLLoader(file_path)
                pages = loader.load()
                for idx, page in enumerate(pages):
                    cleaned = clean_text(page.page_content)
                    if cleaned:
                        raw_docs.append(Document(
                            page_content=cleaned,
                            metadata={"source": display_name, "page": idx + 1,
                                      "file_type": "html", "document_id": doc_id,
                                      "extraction_method": "text"}
                        ))
            except Exception as e:
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        html_text = f.read()
                    cleaned = clean_text(html_text)
                    if cleaned:
                        raw_docs.append(Document(
                            page_content=cleaned,
                            metadata={"source": display_name, "page": 1,
                                      "file_type": "html", "document_id": doc_id,
                                      "extraction_method": "text"}
                        ))
                except Exception as e2:
                    raise CustomException(f"Failed to parse HTML document '{display_name}'.", e2)

        elif ext == ".xml":
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    xml_text = f.read()
                cleaned = clean_text(xml_text)
                if cleaned:
                    raw_docs.append(Document(
                        page_content=cleaned,
                        metadata={"source": display_name, "page": 1,
                                  "file_type": "xml", "document_id": doc_id,
                                  "extraction_method": "text"}
                    ))
            except Exception as e:
                raise CustomException(f"Failed to parse XML document '{display_name}'.", e)

        elif ext == ".rtf":
            try:
                loader = UnstructuredFileLoader(file_path)
                pages = loader.load()
                for idx, page in enumerate(pages):
                    cleaned = clean_text(page.page_content)
                    if cleaned:
                        raw_docs.append(Document(
                            page_content=cleaned,
                            metadata={"source": display_name, "page": idx + 1,
                                      "file_type": "rtf", "document_id": doc_id,
                                      "extraction_method": "text"}
                        ))
            except Exception:
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        rtf_text = f.read()
                    cleaned = clean_text(rtf_text)
                    if cleaned:
                        raw_docs.append(Document(
                            page_content=cleaned,
                            metadata={"source": display_name, "page": 1,
                                      "file_type": "rtf", "document_id": doc_id,
                                      "extraction_method": "text"}
                        ))
                except Exception as e:
                    raise CustomException(f"Failed to parse RTF document '{display_name}'.", e)

        elif ext == ".epub":
            try:
                loader = UnstructuredFileLoader(file_path)
                pages = loader.load()
                for idx, page in enumerate(pages):
                    cleaned = clean_text(page.page_content)
                    if cleaned:
                        raw_docs.append(Document(
                            page_content=cleaned,
                            metadata={"source": display_name, "page": idx + 1,
                                      "file_type": "epub", "document_id": doc_id,
                                      "extraction_method": "text"}
                        ))
            except Exception as e:
                raise CustomException(f"Failed to parse EPUB document '{display_name}'.", e)

        elif ext == ".odt":
            try:
                loader = UnstructuredFileLoader(file_path)
                pages = loader.load()
                for idx, page in enumerate(pages):
                    cleaned = clean_text(page.page_content)
                    if cleaned:
                        raw_docs.append(Document(
                            page_content=cleaned,
                            metadata={"source": display_name, "page": idx + 1,
                                      "file_type": "odt", "document_id": doc_id,
                                      "extraction_method": "text"}
                        ))
            except Exception as e:
                raise CustomException(f"Failed to parse ODT document '{display_name}'.", e)

        elif ext == ".pptx":
            try:
                loader = UnstructuredFileLoader(file_path)
                pages = loader.load()
                for idx, page in enumerate(pages):
                    cleaned = clean_text(page.page_content)
                    if cleaned:
                        raw_docs.append(Document(
                            page_content=cleaned,
                            metadata={"source": display_name, "page": idx + 1,
                                      "file_type": "pptx", "document_id": doc_id,
                                      "extraction_method": "text"}
                        ))
            except Exception as e:
                raise CustomException(f"Failed to parse PPTX document '{display_name}'.", e)

        elif ext in (".xlsx", ".xls"):
            try:
                loader = UnstructuredFileLoader(file_path)
                pages = loader.load()
                for idx, page in enumerate(pages):
                    cleaned = clean_text(page.page_content)
                    if cleaned:
                        raw_docs.append(Document(
                            page_content=cleaned,
                            metadata={"source": display_name, "page": idx + 1,
                                      "file_type": ext.lstrip("."), "document_id": doc_id,
                                      "extraction_method": "text"}
                        ))
            except Exception as e:
                raise CustomException(f"Failed to parse Excel document '{display_name}'.", e)

        elif ext in (".yaml", ".yml"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                cleaned = clean_text(content)
                if cleaned:
                    raw_docs.append(Document(
                        page_content=cleaned,
                        metadata={"source": display_name, "page": 1,
                                  "file_type": ext.lstrip("."), "document_id": doc_id,
                                  "extraction_method": "text"}
                    ))
            except Exception as e:
                raise CustomException(f"Failed to parse YAML document '{display_name}'.", e)

        elif ext == ".toml":
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                cleaned = clean_text(content)
                if cleaned:
                    raw_docs.append(Document(
                        page_content=cleaned,
                        metadata={"source": display_name, "page": 1,
                                  "file_type": "toml", "document_id": doc_id,
                                  "extraction_method": "text"}
                    ))
            except Exception as e:
                raise CustomException(f"Failed to parse TOML document '{display_name}'.", e)

        elif ext == ".ini":
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                cleaned = clean_text(content)
                if cleaned:
                    raw_docs.append(Document(
                        page_content=cleaned,
                        metadata={"source": display_name, "page": 1,
                                  "file_type": "ini", "document_id": doc_id,
                                  "extraction_method": "text"}
                    ))
            except Exception as e:
                raise CustomException(f"Failed to parse INI document '{display_name}'.", e)

        elif ext == ".log":
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                cleaned = clean_text(content)
                if cleaned:
                    raw_docs.append(Document(
                        page_content=cleaned,
                        metadata={"source": display_name, "page": 1,
                                  "file_type": "log", "document_id": doc_id,
                                  "extraction_method": "text"}
                    ))
            except Exception as e:
                raise CustomException(f"Failed to parse LOG document '{display_name}'.", e)

        elif ext == ".sql":
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                cleaned = clean_text(content)
                if cleaned:
                    raw_docs.append(Document(
                        page_content=cleaned,
                        metadata={"source": display_name, "page": 1,
                                  "file_type": "sql", "document_id": doc_id,
                                  "extraction_method": "text"}
                    ))
            except Exception as e:
                raise CustomException(f"Failed to parse SQL document '{display_name}'.", e)

        # ---- E-books ----
        elif ext in (".epub", ".mobi", ".azw3", ".fb2"):
            try:
                loader = UnstructuredFileLoader(file_path)
                pages = loader.load()
                for idx, page in enumerate(pages):
                    cleaned = clean_text(page.page_content)
                    if cleaned:
                        raw_docs.append(Document(
                            page_content=cleaned,
                            metadata={"source": display_name, "page": idx + 1,
                                      "file_type": ext.lstrip("."), "document_id": doc_id,
                                      "extraction_method": "text"}
                        ))
            except Exception as e:
                raise CustomException(f"Failed to parse e-book document '{display_name}'.", e)

        # ---- OpenDocument Presentation / Spreadsheet ----
        elif ext in (".odp", ".ods"):
            try:
                loader = UnstructuredFileLoader(file_path)
                pages = loader.load()
                for idx, page in enumerate(pages):
                    cleaned = clean_text(page.page_content)
                    if cleaned:
                        raw_docs.append(Document(
                            page_content=cleaned,
                            metadata={"source": display_name, "page": idx + 1,
                                      "file_type": ext.lstrip("."), "document_id": doc_id,
                                      "extraction_method": "text"}
                        ))
            except Exception as e:
                raise CustomException(f"Failed to parse OpenDocument '{display_name}'.", e)

        # ---- Word legacy / macro / template ----
        elif ext in (".doc", ".docm", ".dotx"):
            try:
                loader = UnstructuredFileLoader(file_path)
                pages = loader.load()
                for idx, page in enumerate(pages):
                    cleaned = clean_text(page.page_content)
                    if cleaned:
                        raw_docs.append(Document(
                            page_content=cleaned,
                            metadata={"source": display_name, "page": idx + 1,
                                      "file_type": ext.lstrip("."), "document_id": doc_id,
                                      "extraction_method": "text"}
                        ))
            except Exception as e:
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                    cleaned = clean_text(content)
                    if cleaned:
                        raw_docs.append(Document(
                            page_content=cleaned,
                            metadata={"source": display_name, "page": 1,
                                      "file_type": ext.lstrip("."), "document_id": doc_id,
                                      "extraction_method": "text"}
                        ))
                except Exception as e2:
                    raise CustomException(f"Failed to parse Word document '{display_name}'.", e2)

        # ---- PowerPoint legacy / macro / show ----
        elif ext in (".ppt", ".pptm", ".ppsx"):
            try:
                loader = UnstructuredFileLoader(file_path)
                pages = loader.load()
                for idx, page in enumerate(pages):
                    cleaned = clean_text(page.page_content)
                    if cleaned:
                        raw_docs.append(Document(
                            page_content=cleaned,
                            metadata={"source": display_name, "page": idx + 1,
                                      "file_type": ext.lstrip("."), "document_id": doc_id,
                                      "extraction_method": "text"}
                        ))
            except Exception as e:
                raise CustomException(f"Failed to parse PowerPoint document '{display_name}'.", e)

        # ---- Excel macro / binary ----
        elif ext in (".xlsm", ".xlsb"):
            try:
                loader = UnstructuredFileLoader(file_path)
                pages = loader.load()
                for idx, page in enumerate(pages):
                    cleaned = clean_text(page.page_content)
                    if cleaned:
                        raw_docs.append(Document(
                            page_content=cleaned,
                            metadata={"source": display_name, "page": idx + 1,
                                      "file_type": ext.lstrip("."), "document_id": doc_id,
                                      "extraction_method": "text"}
                        ))
            except Exception as e:
                raise CustomException(f"Failed to parse Excel document '{display_name}'.", e)

        # ---- XPS ----
        elif ext == ".xps":
            try:
                loader = UnstructuredFileLoader(file_path)
                pages = loader.load()
                for idx, page in enumerate(pages):
                    cleaned = clean_text(page.page_content)
                    if cleaned:
                        raw_docs.append(Document(
                            page_content=cleaned,
                            metadata={"source": display_name, "page": idx + 1,
                                      "file_type": "xps", "document_id": doc_id,
                                      "extraction_method": "text"}
                        ))
            except Exception as e:
                raise CustomException(f"Failed to parse XPS document '{display_name}'.", e)

        # ---- Images (OCR) ----
        elif ext in (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif", ".webp"):
            try:
                if ocr_engine.is_available():
                    ocr_results = ocr_engine.extract_pdf(file_path)
                    for result in ocr_results:
                        cleaned = clean_text(result.text)
                        if cleaned:
                            raw_docs.append(Document(
                                page_content=cleaned,
                                metadata={"source": display_name, "page": result.page_number,
                                          "file_type": ext.lstrip("."), "document_id": doc_id,
                                          "extraction_method": "ocr",
                                          "ocr_confidence": result.confidence,
                                          "ocr_engine": "indra"}
                            ))
                else:
                    raw_docs.append(Document(
                        page_content=f"[Image file: {display_name} - OCR engine not available]",
                        metadata={"source": display_name, "page": 1,
                                  "file_type": ext.lstrip("."), "document_id": doc_id,
                                  "extraction_method": "metadata"}
                    ))
            except Exception as e:
                logger.warning(f"OCR failed for image '{display_name}': {e}")
                raw_docs.append(Document(
                    page_content=f"[Image file: {display_name} - OCR extraction failed]",
                    metadata={"source": display_name, "page": 1,
                              "file_type": ext.lstrip("."), "document_id": doc_id,
                              "extraction_method": "metadata"}
                ))

        # ---- Archives (ZIP, RAR, 7Z, TAR, GZ) ----
        elif ext in (".zip", ".rar", ".7z", ".tar", ".gz"):
            try:
                import zipfile
                import tarfile
                extracted_texts = []
                if ext == ".zip":
                    with zipfile.ZipFile(file_path, 'r') as zf:
                        for name in zf.namelist():
                            if name.endswith('/'):
                                continue
                            try:
                                data = zf.read(name)
                                text = data.decode('utf-8', errors='ignore')
                                cleaned = clean_text(text)
                                if cleaned:
                                    extracted_texts.append(f"--- File: {name} ---\n{cleaned}")
                            except Exception:
                                continue
                elif ext in (".tar", ".gz"):
                    with tarfile.open(file_path, 'r:*') as tf:
                        for member in tf.getmembers():
                            if member.isdir():
                                continue
                            try:
                                f = tf.extractfile(member)
                                if f is None:
                                    continue
                                data = f.read()
                                text = data.decode('utf-8', errors='ignore')
                                cleaned = clean_text(text)
                                if cleaned:
                                    extracted_texts.append(f"--- File: {member.name} ---\n{cleaned}")
                            except Exception:
                                continue
                else:
                    loader = UnstructuredFileLoader(file_path)
                    pages = loader.load()
                    for idx, page in enumerate(pages):
                        cleaned = clean_text(page.page_content)
                        if cleaned:
                            raw_docs.append(Document(
                                page_content=cleaned,
                                metadata={"source": display_name, "page": idx + 1,
                                          "file_type": ext.lstrip("."), "document_id": doc_id,
                                          "extraction_method": "text"}
                            ))

                if extracted_texts:
                    combined = "\n\n".join(extracted_texts)
                    raw_docs.append(Document(
                        page_content=combined,
                        metadata={"source": display_name, "page": 1,
                                  "file_type": ext.lstrip("."), "document_id": doc_id,
                                  "extraction_method": "archive"}
                    ))
                elif not raw_docs:
                    raw_docs.append(Document(
                        page_content=f"[Archive file: {display_name} - no text files found]",
                        metadata={"source": display_name, "page": 1,
                                  "file_type": ext.lstrip("."), "document_id": doc_id,
                                  "extraction_method": "metadata"}
                    ))
            except Exception as e:
                raise CustomException(f"Failed to parse archive '{display_name}'.", e)

        # ---- Email (EML, MSG, MBOX, PST) ----
        elif ext in (".eml", ".msg", ".mbox", ".pst"):
            try:
                import email
                from email import policy
                from email.parser import BytesParser
                email_texts = []
                if ext == ".eml":
                    with open(file_path, "rb") as f:
                        msg = BytesParser(policy=policy.default).parse(f)
                    subject = msg.get("subject", "")
                    sender = msg.get("from", "")
                    date = msg.get("date", "")
                    email_texts.append(f"From: {sender}")
                    email_texts.append(f"Subject: {subject}")
                    email_texts.append(f"Date: {date}")
                    if msg.is_multipart():
                        for part in msg.walk():
                            if part.get_content_type() == "text/plain":
                                email_texts.append(part.get_content())
                    else:
                        email_texts.append(msg.get_content())
                elif ext == ".mbox":
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                    email_texts.append(content)
                else:
                    loader = UnstructuredFileLoader(file_path)
                    pages = loader.load()
                    for idx, page in enumerate(pages):
                        cleaned = clean_text(page.page_content)
                        if cleaned:
                            raw_docs.append(Document(
                                page_content=cleaned,
                                metadata={"source": display_name, "page": idx + 1,
                                          "file_type": ext.lstrip("."), "document_id": doc_id,
                                          "extraction_method": "text"}
                            ))

                if email_texts:
                    combined = "\n\n".join(email_texts)
                    cleaned = clean_text(combined)
                    if cleaned:
                        raw_docs.append(Document(
                            page_content=cleaned,
                            metadata={"source": display_name, "page": 1,
                                      "file_type": ext.lstrip("."), "document_id": doc_id,
                                      "extraction_method": "email"}
                        ))
                elif not raw_docs:
                    raw_docs.append(Document(
                        page_content=f"[Email file: {display_name} - no readable content]",
                        metadata={"source": display_name, "page": 1,
                                  "file_type": ext.lstrip("."), "document_id": doc_id,
                                  "extraction_method": "metadata"}
                    ))
            except Exception as e:
                raise CustomException(f"Failed to parse email '{display_name}'.", e)

        # ---- ML Models (binary metadata) ----
        elif ext in (".h5", ".pt", ".pth", ".onnx", ".gguf", ".tflite"):
            try:
                file_size = os.path.getsize(file_path)
                raw_docs.append(Document(
                    page_content=f"[ML Model file: {display_name} - size: {file_size} bytes - binary model, no extractable text]",
                    metadata={"source": display_name, "page": 1,
                              "file_type": ext.lstrip("."), "document_id": doc_id,
                              "extraction_method": "metadata"}
                ))
            except Exception as e:
                raise CustomException(f"Failed to process ML model '{display_name}'.", e)

        # ---- Audio / Video (metadata only) ----
        elif ext in (".mp3", ".wav", ".mp4", ".mkv", ".mov"):
            try:
                file_size = os.path.getsize(file_path)
                raw_docs.append(Document(
                    page_content=f"[Audio/Video file: {display_name} - size: {file_size} bytes - speech-to-text not configured]",
                    metadata={"source": display_name, "page": 1,
                              "file_type": ext.lstrip("."), "document_id": doc_id,
                              "extraction_method": "metadata"}
                ))
            except Exception as e:
                raise CustomException(f"Failed to process media file '{display_name}'.", e)

        # ---- Universal fallback ----
        else:
            try:
                loader = UnstructuredFileLoader(file_path)
                pages = loader.load()
                for idx, page in enumerate(pages):
                    cleaned = clean_text(page.page_content)
                    if cleaned:
                        raw_docs.append(Document(
                            page_content=cleaned,
                            metadata={"source": display_name, "page": idx + 1,
                                      "file_type": ext.lstrip(".") if ext else "unknown",
                                      "document_id": doc_id, "extraction_method": "text"}
                        ))
            except Exception:
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                    cleaned = clean_text(content)
                    if cleaned:
                        raw_docs.append(Document(
                            page_content=cleaned,
                            metadata={"source": display_name, "page": 1,
                                      "file_type": ext.lstrip(".") if ext else "unknown",
                                      "document_id": doc_id, "extraction_method": "text"}
                        ))
                except Exception as e:
                    raise CustomException(f"Unsupported file format '{ext}'. Unable to extract text from '{display_name}'.", e)

        if not raw_docs:
            raise CustomException(f"Document '{display_name}' contains no readable text content.")

        logger.info(f"Extracted {len(raw_docs)} document segment(s) from '{display_name}'.")
        return raw_docs

    except CustomException:
        raise
    except Exception as e:
        error_msg = CustomException(f"Error while loading document '{file_path}'", e)
        logger.error(str(error_msg))
        raise error_msg


def create_text_chunks(
    documents: List[Document],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None
) -> List[Document]:
    """
    Split extracted documents into clean chunks while preserving metadata.
    """
    try:
        if not documents:
            raise CustomException("No documents provided to chunk.")

        c_size = chunk_size if chunk_size is not None else CHUNK_SIZE
        c_overlap = chunk_overlap if chunk_overlap is not None else CHUNK_OVERLAP

        logger.info(f"Chunking {len(documents)} document pages/sections (chunk_size={c_size}, chunk_overlap={c_overlap})")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=c_size,
            chunk_overlap=c_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

        chunks = splitter.split_documents(documents)

        # Filter out empty or whitespace-only chunks
        valid_chunks = [c for c in chunks if c.page_content and c.page_content.strip()]

        # Assign a sequential chunk_id within each document (keeps FAISS/metadata aligned)
        counters: dict = {}
        for chunk in valid_chunks:
            d_id = str(chunk.metadata.get("document_id", "unknown"))
            counters[d_id] = counters.get(d_id, 0) + 1
            chunk.metadata["chunk_id"] = counters[d_id]

        logger.info(f"Created {len(valid_chunks)} text chunks successfully.")

        if not valid_chunks:
            raise CustomException("Chunking resulted in 0 valid chunks.")

        return valid_chunks

    except CustomException:
        raise
    except Exception as e:
        error_msg = CustomException("Failed to create text chunks from document", e)
        logger.error(str(error_msg))
        raise error_msg
