"""Bound normalized speech by UTF-8 length, including Arabic punctuation."""

from __future__ import annotations

import re


def speech_chunks(text: str, *, max_bytes: int) -> list[str]:
    if max_bytes < 16:
        raise ValueError("Reference speaking-rate budget is too small")
    words = text.split()
    if not words:
        raise ValueError("Text must be nonempty")
    chunks: list[str] = []
    current = ""
    for word in words:
        if len(word.encode("utf-8")) > max_bytes:
            raise ValueError("A word exceeds the safe speech length; add spaces")
        candidate = f"{current} {word}" if current else word
        if len(candidate.encode("utf-8")) > max_bytes:
            chunks.append(current)
            current = word
        else:
            current = candidate
        if re.search(r"[،؛,.!?؟:;]$", word):
            chunks.append(current)
            current = ""
    if current:
        chunks.append(current)
    return chunks
