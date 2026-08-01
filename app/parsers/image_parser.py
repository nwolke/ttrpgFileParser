"""Image parser – uses Tesseract OCR via pytesseract."""

from pathlib import Path

from PIL import Image
import pytesseract

from .base import BaseParser


class ImageParser(BaseParser):
    """Extract text from an image file using OCR."""

    def extract_text(self, path: Path) -> str:
        try:
            img = Image.open(str(path))
        except Exception as exc:
            raise ValueError(f"Cannot open image '{path.name}': {exc}") from exc

        try:
            text = pytesseract.image_to_string(img)
        except Exception as exc:
            raise ValueError(
                f"OCR failed for '{path.name}': {exc}. "
                "Ensure Tesseract is installed (https://github.com/tesseract-ocr/tesseract)."
            ) from exc

        if not text.strip():
            raise ValueError(f"No text detected in image '{path.name}'.")

        return text.strip()
