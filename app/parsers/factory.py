"""Parser factory – returns the correct parser for a given file extension."""

from pathlib import Path

from .base import BaseParser
from .pdf_parser import PDFParser
from .image_parser import ImageParser
from .text_parser import TextParser

# Map lowercase extensions to parser instances (singletons).
_PARSERS: dict[str, BaseParser] = {
    # PDF
    ".pdf": PDFParser(),
    # Images
    ".png":  ImageParser(),
    ".jpg":  ImageParser(),
    ".jpeg": ImageParser(),
    ".gif":  ImageParser(),
    ".bmp":  ImageParser(),
    ".tiff": ImageParser(),
    ".tif":  ImageParser(),
    ".webp": ImageParser(),
    # Plain text / markup
    ".txt":  TextParser(),
    ".md":   TextParser(),
    ".rst":  TextParser(),
    ".csv":  TextParser(),
}

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(_PARSERS)


def get_parser(path: Path) -> BaseParser | None:
    """Return the parser for *path*, or ``None`` if unsupported."""
    return _PARSERS.get(path.suffix.lower())
