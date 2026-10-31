"""
OCR Engine Module
-----------------
Wraps Indra OCR (PaddleOCR backend) to extract text from scanned / image-based
PDFs. Automatically detects whether a PDF page has a usable text layer or
requires OCR, falling back gracefully when OCR dependencies are unavailable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

from app.common.logger import get_logger
from app.common.custom_exceptions import CustomException
from app.config.config import OCR_DPI, OCR_ENABLED, OCR_MODE, OCR_THREADS

logger = get_logger(__name__)


@dataclass
class OCRPageResult:
    """Result for a single PDF page after extraction."""

    page_number: int
    extraction_method: str  # "text" | "ocr" | "error"
    text: str
    confidence: float = 0.0
    lines: List[dict] = field(default_factory=list)
    blocks: List[dict] = field(default_factory=list)
    markdown: str = ""


class OCREngine:
    """
    Singleton-style wrapper around Indra OCR.

    - _load_engine() initialises the CPU PaddleOCR engine once.
    - extract_pdf() runs Indra in auto mode: it uses the PDF text layer when
      present and applies OCR only to scanned / image pages.
    - is_available() reports whether the engine could be initialised.
    """

    def __init__(
        self,
        enabled: Optional[bool] = None,
        threads: Optional[int] = None,
        dpi: Optional[int] = None,
        mode: Optional[str] = None,
    ) -> None:
        self.enabled = OCR_ENABLED if enabled is None else enabled
        self.threads = OCR_THREADS if threads is None else threads
        self.dpi = OCR_DPI if dpi is None else dpi
        self.mode = OCR_MODE if mode is None else mode
        self._engine = None  # None = not loaded, False = load failed

    # ------------------------------------------------------------------ #
    # Lifecycle                                                          #
    # ------------------------------------------------------------------ #
    def _load_engine(self):
        if not self.enabled:
            logger.info("OCR is disabled via configuration.")
            return None

        if self._engine is None:
            try:
                import indra_ocr  # type: ignore

                logger.info(
                    "Initialising Indra OCR engine "
                    f"(threads={self.threads}, dpi={self.dpi}, mode={self.mode})..."
                )
                self._engine = indra_ocr.load(
                    threads=self.threads,
                    dpi=self.dpi,
                    mode=self.mode,
                )
                logger.info("Indra OCR engine initialised successfully.")
            except Exception as exc:
                logger.warning(
                    f"Indra OCR could not be initialised: {exc}. "
                    "Scanned PDFs will not be OCR'd."
                )
                self._engine = False  # sentinel: attempted and failed
        if self._engine is False:
            return None
        return self._engine

    def is_available(self) -> bool:
        engine = self._load_engine()
        return engine is not None and engine is not False

    # ------------------------------------------------------------------ #
    # Extraction                                                         #
    # ------------------------------------------------------------------ #
    def extract_pdf(self, pdf_path: str) -> List[OCRPageResult]:
        """
        Extract every page of a PDF using Indra OCR auto mode.

        Raises CustomException if the engine is unavailable or extraction
        fails on any page.
        """
        engine = self._load_engine()
        if engine is None or engine is False:
            raise CustomException("OCR engine is disabled or unavailable.")

        if not os.path.exists(pdf_path):
            raise CustomException(f"PDF file not found: {pdf_path}")

        results: List[OCRPageResult] = []

        try:
            for page_number, page in engine.convert(pdf_path):
                source = getattr(page, "source", "unknown") or "unknown"

                if source == "error":
                    raise CustomException(
                        f"OCR extraction failed on page {page_number}"
                    )

                text = getattr(page, "text", "") or ""
                markdown = getattr(page, "markdown", "") or ""

                line_objects = getattr(page, "lines", []) or []
                lines = [
                    {
                        "text": getattr(line, "text", ""),
                        "confidence": getattr(line, "conf", 0.0),
                        "bounding_box": getattr(line, "box", None),
                    }
                    for line in line_objects
                ]

                blocks = getattr(page, "blocks", []) or []

                confidences = [ln["confidence"] for ln in lines if ln["confidence"]]
                avg_confidence = (
                    sum(confidences) / len(confidences) if confidences else 0.0
                )

                results.append(
                    OCRPageResult(
                        page_number=int(page_number),
                        extraction_method=str(source),
                        text=text,
                        confidence=float(avg_confidence),
                        lines=lines,
                        blocks=blocks,
                        markdown=markdown,
                    )
                )

        except CustomException:
            raise
        except Exception as exc:
            error_msg = CustomException(
                f"Indra OCR extraction failed for '{pdf_path}'", exc
            )
            logger.error(str(error_msg))
            raise error_msg

        return results


# Module-level singleton for convenience
ocr_engine = OCREngine()