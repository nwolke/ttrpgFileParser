"""Tests for the text chunker utility."""

import pytest

from app.chunker import chunk_text


def test_empty_string_returns_empty_list():
    assert chunk_text("") == []


def test_whitespace_only_returns_empty_list():
    assert chunk_text("   \n\n  ") == []


def test_short_text_returns_single_chunk():
    text = "Hello, adventurer!"
    result = chunk_text(text, chunk_size=500)
    assert len(result) == 1
    assert result[0] == text


def test_chunks_do_not_exceed_size_by_much():
    # Each individual segment (sentence) fits; verify no chunk is wildly over the limit.
    long_text = "  ".join([f"Sentence number {i}." for i in range(100)])
    chunks = chunk_text(long_text, chunk_size=200, overlap=20)
    for chunk in chunks:
        # Allow some slack because we join with spaces and overlap seeds.
        assert len(chunk) <= 400, f"Chunk too long ({len(chunk)} chars): {chunk[:80]}"


def test_all_content_is_covered():
    """No segment should be silently dropped."""
    words = [f"word{i}" for i in range(50)]
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size=100, overlap=0)
    joined = " ".join(chunks)
    for word in words:
        assert word in joined, f"'{word}' not found in any chunk"


def test_overlap_produces_more_chunks():
    text = "\n\n".join([f"Paragraph {i}: " + "A" * 100 for i in range(10)])
    chunks_no_overlap = chunk_text(text, chunk_size=200, overlap=0)
    chunks_overlap = chunk_text(text, chunk_size=200, overlap=50)
    # Overlap should produce at least as many chunks.
    assert len(chunks_overlap) >= len(chunks_no_overlap)


def test_paragraph_splitting():
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    chunks = chunk_text(text, chunk_size=500)
    joined = " ".join(chunks)
    assert "First paragraph" in joined
    assert "Second paragraph" in joined
    assert "Third paragraph" in joined
