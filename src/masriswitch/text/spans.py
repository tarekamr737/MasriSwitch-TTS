"""Unicode-script code-switch statistics."""

from __future__ import annotations

import re
from dataclasses import dataclass

_TOKEN = re.compile(r"[\u0600-\u06ff]+|[A-Za-z]+")
_ARABIC = re.compile(r"[\u0600-\u06ff]")
_LATIN = re.compile(r"[A-Za-z]")


@dataclass(frozen=True)
class SwitchStats:
    arabic_tokens: int
    latin_tokens: int
    english_ratio: float
    switch_count: int
    bucket: str


def analyze_switches(text: str) -> SwitchStats:
    languages = ["ar" if _ARABIC.search(token) else "en" for token in _TOKEN.findall(text)]
    ar = languages.count("ar")
    en = languages.count("en")
    ratio = en / (ar + en) if ar + en else 0.0
    switches = sum(a != b for a, b in zip(languages, languages[1:], strict=False))
    if not en:
        bucket = "ar_only"
    elif not ar:
        bucket = "en_only"
    elif ratio < 0.4:
        bucket = "cs_ar_dom"
    elif ratio <= 0.6:
        bucket = "cs_balanced"
    else:
        bucket = "cs_en_dom"
    return SwitchStats(ar, en, ratio, switches, bucket)
