"""Plain-text and Markdown parser."""

from pathlib import Path

from .base import BaseParser

_ENCODINGS = ("utf-8", "utf-8-sig", "latin-1", "cp1252")


class TextParser(BaseParser):
    """Read text files, trying common encodings in order."""

    def extract_text(self, path: Path) -> str:
        for enc in _ENCODINGS:
            try:
                text = path.read_text(encoding=enc)
                if text.strip():
                    return text.strip()
                raise ValueError(f"File '{path.name}' is empty.")
            except (UnicodeDecodeError, UnicodeError):
                continue

        raise ValueError(
            f"Cannot decode '{path.name}' with any of the tried encodings: "
            + ", ".join(_ENCODINGS)
        )
