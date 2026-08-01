"""Text chunking utility.

Splits a long document into overlapping chunks of approximately
``chunk_size`` characters to improve embedding quality and search recall.
"""

from __future__ import annotations

import re


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split *text* into overlapping chunks.

    Strategy:
    1. Split on paragraph boundaries (two or more newlines).
    2. If a paragraph is longer than *chunk_size*, split further on sentences.
    3. Greedily group segments until the running buffer would exceed
       *chunk_size*, then start a new chunk, back-tracking by *overlap*
       characters.

    Args:
        text: The full document text.
        chunk_size: Target character length for each chunk.
        overlap: Number of characters from the previous chunk to include at
            the start of the next chunk (provides contextual continuity).

    Returns:
        A list of non-empty string chunks.
    """
    if not text.strip():
        return []

    # Step 1: split into paragraphs.
    paragraphs = re.split(r"\n{2,}", text)

    # Step 2: further split very long paragraphs on sentence boundaries.
    sentence_end = re.compile(r"(?<=[.!?])\s+")
    segments: list[str] = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(para) <= chunk_size:
            segments.append(para)
        else:
            for sentence in sentence_end.split(para):
                s = sentence.strip()
                if s:
                    segments.append(s)

    # Step 3: greedily group segments into chunks with overlap.
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for seg in segments:
        seg_len = len(seg)
        if current and current_len + seg_len + 1 > chunk_size:
            chunk_text_str = " ".join(current)
            chunks.append(chunk_text_str)
            # Overlap: keep trailing characters as seed for next chunk.
            seed = chunk_text_str[-overlap:] if overlap else ""
            current = [seed, seg] if seed else [seg]
            current_len = len(seed) + seg_len + (1 if seed else 0)
        else:
            current.append(seg)
            current_len += seg_len + (1 if current_len else 0)

    if current:
        chunks.append(" ".join(current))

    return [c for c in chunks if c.strip()]
