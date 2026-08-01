"""Tests for file parsers."""

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.parsers.factory import get_parser, SUPPORTED_EXTENSIONS
from app.parsers.text_parser import TextParser


# --------------------------------------------------------------------------- #
# Factory                                                                       #
# --------------------------------------------------------------------------- #

def test_supported_extensions_not_empty():
    assert len(SUPPORTED_EXTENSIONS) > 0


def test_get_parser_pdf():
    parser = get_parser(Path("book.pdf"))
    assert parser is not None
    assert type(parser).__name__ == "PDFParser"


def test_get_parser_txt():
    parser = get_parser(Path("notes.txt"))
    assert parser is not None
    assert type(parser).__name__ == "TextParser"


def test_get_parser_md():
    parser = get_parser(Path("readme.md"))
    assert parser is not None


def test_get_parser_image():
    for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tiff"):
        parser = get_parser(Path(f"image{ext}"))
        assert parser is not None, f"No parser for {ext}"
        assert type(parser).__name__ == "ImageParser"


def test_get_parser_unsupported():
    assert get_parser(Path("archive.zip")) is None
    assert get_parser(Path("binary.exe")) is None


def test_extension_case_insensitive():
    parser_lower = get_parser(Path("document.PDF"))
    parser_upper = get_parser(Path("document.pdf"))
    assert type(parser_lower) == type(parser_upper)


# --------------------------------------------------------------------------- #
# TextParser                                                                    #
# --------------------------------------------------------------------------- #

def test_text_parser_reads_file(tmp_path):
    f = tmp_path / "sample.txt"
    f.write_text("Hello from the test suite.", encoding="utf-8")
    parser = TextParser()
    result = parser.extract_text(f)
    assert "Hello from the test suite." in result


def test_text_parser_empty_file_raises(tmp_path):
    f = tmp_path / "empty.txt"
    f.write_text("   ", encoding="utf-8")
    parser = TextParser()
    with pytest.raises(ValueError, match="empty"):
        parser.extract_text(f)


def test_text_parser_multiline(tmp_path):
    content = textwrap.dedent("""\
        Line one
        Line two
        Line three
    """)
    f = tmp_path / "multi.txt"
    f.write_text(content, encoding="utf-8")
    parser = TextParser()
    result = parser.extract_text(f)
    assert "Line one" in result
    assert "Line three" in result


# --------------------------------------------------------------------------- #
# PDFParser – mocked fitz                                                       #
# --------------------------------------------------------------------------- #

def test_pdf_parser_extracts_pages():
    from app.parsers.pdf_parser import PDFParser

    mock_page = MagicMock()
    mock_page.get_text.return_value = "Crafting rules on page one."
    mock_doc = MagicMock()
    mock_doc.__iter__ = MagicMock(return_value=iter([mock_page]))
    mock_doc.__len__ = MagicMock(return_value=1)

    with patch("app.parsers.pdf_parser.fitz.open", return_value=mock_doc):
        parser = PDFParser()
        result = parser.extract_text(Path("fake.pdf"))

    assert "Crafting rules" in result


def test_pdf_parser_no_text_raises():
    from app.parsers.pdf_parser import PDFParser

    mock_page = MagicMock()
    mock_page.get_text.return_value = "   "
    mock_doc = MagicMock()
    mock_doc.__iter__ = MagicMock(return_value=iter([mock_page]))

    with patch("app.parsers.pdf_parser.fitz.open", return_value=mock_doc):
        parser = PDFParser()
        with pytest.raises(ValueError, match="No extractable text"):
            parser.extract_text(Path("blank.pdf"))


def test_pdf_parser_open_error_raises():
    from app.parsers.pdf_parser import PDFParser

    with patch("app.parsers.pdf_parser.fitz.open", side_effect=RuntimeError("bad file")):
        parser = PDFParser()
        with pytest.raises(ValueError, match="Cannot open PDF"):
            parser.extract_text(Path("corrupt.pdf"))


# --------------------------------------------------------------------------- #
# ImageParser – mocked pytesseract                                              #
# --------------------------------------------------------------------------- #

def test_image_parser_extracts_text(tmp_path):
    from app.parsers.image_parser import ImageParser
    from PIL import Image as PILImage

    img = PILImage.new("RGB", (100, 30), color="white")
    img_path = tmp_path / "test.png"
    img.save(img_path)

    with patch("app.parsers.image_parser.pytesseract.image_to_string", return_value="Magic item table"):
        parser = ImageParser()
        result = parser.extract_text(img_path)

    assert "Magic item table" in result


def test_image_parser_no_text_raises(tmp_path):
    from app.parsers.image_parser import ImageParser
    from PIL import Image as PILImage

    img = PILImage.new("RGB", (10, 10), color="black")
    img_path = tmp_path / "blank.png"
    img.save(img_path)

    with patch("app.parsers.image_parser.pytesseract.image_to_string", return_value="   "):
        parser = ImageParser()
        with pytest.raises(ValueError, match="No text detected"):
            parser.extract_text(img_path)
