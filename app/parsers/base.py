"""Abstract base class for all file parsers."""

from abc import ABC, abstractmethod
from pathlib import Path


class BaseParser(ABC):
    """Extract plain text from a file so it can be indexed."""

    @abstractmethod
    def extract_text(self, path: Path) -> str:
        """Return the full text content of *path*.

        Raise ``ValueError`` with a descriptive message if the file cannot be
        parsed (e.g. encrypted PDF, corrupted image).
        """
