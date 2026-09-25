"""Typed entity parser for inspectable normalization and evaluation."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Entity:
    kind: str
    value: str
    start: int
    end: int


_PATTERNS = [
    ("url", re.compile(r"https?://[^\s]+|\bwww\.[^\s]+", re.I)),
    ("phone", re.compile(r"(?<!\d)(?:\+20|0020|0)1[0125][\d\s-]{8,12}(?!\d)")),
    ("date", re.compile(r"(?<!\d)\d{1,2}[/-]\d{1,2}[/-]\d{4}(?!\d)")),
    ("time", re.compile(r"(?<!\d)\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?(?!\d)")),
    ("money", re.compile(r"(?<!\w)\d+(?:[.,]\d+)?\s*(?:EGP|USD|SAR|جنيه|دولار|ريال)\b", re.I)),
    ("percentage", re.compile(r"(?<!\d)\d+(?:[.,]\d+)?\s*%")),
    ("measurement", re.compile(r"(?<!\d)\d+(?:[.,]\d+)?\s*(?:kg|km|cm|GB|MB)\b", re.I)),
    ("acronym", re.compile(r"(?<![A-Za-z])(?:OTP|Wi-Fi|API|AI|URL|SMS|USB)(?![A-Za-z])", re.I)),
    ("id", re.compile(r"\b(?:ORD|ACC|ID)[-_]?[A-Z0-9]{3,}\b", re.I)),
    (
        "brand",
        re.compile(
            r"(?<![A-Za-z])(?:Vodafone|Visa|Amazon|Hilton|Microsoft|Vezeeta)(?![A-Za-z])", re.I
        ),
    ),
    ("english_term", re.compile(r"(?<![A-Za-z])[A-Za-z]+(?:[- ][A-Za-z]+)*(?![A-Za-z])")),
]


def parse_entities(text: str) -> list[Entity]:
    matches: list[Entity] = []
    occupied: set[int] = set()
    for kind, pattern in _PATTERNS:
        for match in pattern.finditer(text):
            if any(i in occupied for i in range(match.start(), match.end())):
                continue
            matches.append(Entity(kind, match.group(), match.start(), match.end()))
            occupied.update(range(match.start(), match.end()))
    return sorted(matches, key=lambda entity: entity.start)
