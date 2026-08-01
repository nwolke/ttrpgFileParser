"""PDF parser using PyMuPDF (fitz)."""

from pathlib import Path

import fitz  # PyMuPDF

from .base import BaseParser


class PDFParser(BaseParser):
    """Extract text from every page of a PDF file."""

    def extract_text(self, path: Path) -> str:
        try:
            doc = fitz.open(str(path))
        except Exception as exc:
            raise ValueError(f"Cannot open PDF '{path.name}': {exc}") from exc

        pages: list[str] = []
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text")
            if text.strip():
                pages.append(f"[Page {page_num}]\n{text.strip()}")

        doc.close()

        if not pages:
            raise ValueError(f"No extractable text found in '{path.name}' (may be scanned/image-only).")

        return "\n\n".join(pages)
