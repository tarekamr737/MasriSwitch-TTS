"""Exact and conservative near-text duplicate filtering."""

from __future__ import annotations

import re
from typing import Any


def _trigrams(text: str) -> set[tuple[str, ...]]:
    tokens = re.findall(r"\w+", text.casefold())
    return {tuple(tokens[i : i + 3]) for i in range(max(0, len(tokens) - 2))}


def dedupe_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    seen_audio: set[str] = set()
    seen_text: set[str] = set()
    accepted: list[dict[str, Any]] = []
    counts = {"exact_audio": 0, "exact_text": 0, "near_text": 0}
    # Comparing 3-grams is conservative; unique low-overlap text stays intact.
    trigram_index: dict[tuple[str, ...], set[int]] = {}
    grams_by_row: list[set[tuple[str, ...]]] = []
    for row in rows:
        audio_hash = str(row["audio_sha256"])
        text = str(row["text"]).casefold()
        if audio_hash in seen_audio:
            counts["exact_audio"] += 1
            continue
        if text in seen_text:
            counts["exact_text"] += 1
            continue
        grams = _trigrams(text)
        candidates: set[int] = set()
        for gram in grams:
            candidates.update(trigram_index.get(gram, set()))
        near = any(
            len(grams & grams_by_row[index]) / len(grams | grams_by_row[index]) >= 0.92
            for index in candidates
            if grams | grams_by_row[index]
        )
        if near:
            counts["near_text"] += 1
            continue
        index = len(accepted)
        accepted.append(row)
        grams_by_row.append(grams)
        for gram in grams:
            trigram_index.setdefault(gram, set()).add(index)
        seen_audio.add(audio_hash)
        seen_text.add(text)
    return accepted, counts
