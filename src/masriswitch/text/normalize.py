"""Deterministic spoken-text normalization, separate from transcript cleaning."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from masriswitch.text.entities import Entity, parse_entities

_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_ONES = ["صفر", "واحد", "اتنين", "تلاتة", "أربعة", "خمسة", "ستة", "سبعة", "تمانية", "تسعة"]
_TEENS = [
    "عشرة",
    "حداشر",
    "اتناشر",
    "تلتاشر",
    "أربعتاشر",
    "خمستاشر",
    "ستاشر",
    "سبعتاشر",
    "تمنتاشر",
    "تسعتاشر",
]
_TENS = ["", "", "عشرين", "تلاتين", "أربعين", "خمسين", "ستين", "سبعين", "تمانين", "تسعين"]
_HUNDREDS = [
    "",
    "مية",
    "ميتين",
    "تلتمية",
    "أربعمية",
    "خمسمية",
    "ستمية",
    "سبعمية",
    "تمنمية",
    "تسعمية",
]
_ACRONYMS = {
    "OTP": "أو تي بي",
    "API": "إيه بي آي",
    "AI": "إيه آي",
    "URL": "يو آر إل",
    "SMS": "إس إم إس",
    "USB": "يو إس بي",
    "WI-FI": "واي فاي",
}
_CURRENCIES = {
    "EGP": "جنيه",
    "USD": "دولار",
    "SAR": "ريال",
    "جنيه": "جنيه",
    "دولار": "دولار",
    "ريال": "ريال",
}
_UNITS = {"KG": "كيلو جرام", "KM": "كيلومتر", "CM": "سنتيمتر", "GB": "جيجا بايت", "MB": "ميجا بايت"}


@dataclass(frozen=True)
class Normalization:
    original_text: str
    normalized_text: str
    entities: tuple[Entity, ...]


def number_words(value: int) -> str:
    if value < 0 or value >= 1_000_000:
        return " ".join(_ONES[int(digit)] for digit in str(value).lstrip("-"))
    if value < 10:
        return _ONES[value]
    if value < 20:
        return _TEENS[value - 10]
    if value < 100:
        ones, tens = value % 10, value // 10
        return f"{_ONES[ones]} و{_TENS[tens]}" if ones else _TENS[tens]
    if value < 1000:
        head, tail = divmod(value, 100)
        return _HUNDREDS[head] + (" و" + number_words(tail) if tail else "")
    head, tail = divmod(value, 1000)
    thousands = "ألف" if head == 1 else "ألفين" if head == 2 else number_words(head) + " ألف"
    return thousands + (" و" + number_words(tail) if tail else "")


def _decimal_words(value: str) -> str:
    parts = value.replace(",", ".").split(".")
    whole = number_words(int(parts[0]))
    if len(parts) == 1:
        return whole
    return whole + " فاصلة " + " ".join(_ONES[int(digit)] for digit in parts[1])


def _spoken(entity: Entity) -> str:
    value = entity.value
    if entity.kind == "phone":
        return " ".join(_ONES[int(digit)] for digit in value if digit.isdigit())
    if entity.kind == "money":
        match = re.fullmatch(r"(\d+(?:[.,]\d+)?)\s*(\w+)", value, re.I)
        if match:
            return _decimal_words(match[1]) + " " + _CURRENCIES.get(match[2].upper(), match[2])
    if entity.kind == "percentage":
        return _decimal_words(value.strip().rstrip("% ")) + " في المية"
    if entity.kind == "measurement":
        match = re.fullmatch(r"(\d+(?:[.,]\d+)?)\s*(\w+)", value, re.I)
        if match:
            return _decimal_words(match[1]) + " " + _UNITS.get(match[2].upper(), match[2])
    if entity.kind == "date":
        day, month, year = (int(part) for part in re.split(r"[/-]", value))
        if 1 <= day <= 31 and 1 <= month <= 12:
            return f"{number_words(day)} {number_words(month)} {number_words(year)}"
    if entity.kind == "time":
        match = re.fullmatch(r"(\d{1,2}):(\d{2})\s*(AM|PM)?", value, re.I)
        if match and int(match[1]) < 24 and int(match[2]) < 60:
            suffix = " صباحا" if (match[3] or "").upper() == "AM" else " مساء" if match[3] else ""
            return number_words(int(match[1])) + " و" + number_words(int(match[2])) + suffix
    if entity.kind == "acronym":
        return _ACRONYMS.get(value.upper(), value)
    if entity.kind == "id":
        return " ".join(_ONES[int(char)] if char.isdigit() else char for char in value)
    return value


def normalize_text(text: str) -> Normalization:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Text must be nonempty")
    clean = unicodedata.normalize("NFC", text.translate(_DIGITS))
    clean = re.sub("[\u0640\u200b-\u200f\u2060\ufeff]", "", clean)
    clean = " ".join(clean.split())
    entities = tuple(parse_entities(clean))
    pieces: list[str] = []
    cursor = 0
    for entity in entities:
        gap = clean[cursor : entity.start]
        pieces.append(re.sub(r"\b(\d{1,6})\b", lambda m: number_words(int(m[1])), gap))
        pieces.append(_spoken(entity))
        cursor = entity.end
    tail = clean[cursor:]
    pieces.append(re.sub(r"\b(\d{1,6})\b", lambda m: number_words(int(m[1])), tail))
    normalized = "".join(pieces)
    normalized = re.sub(r"\s+([،,.!?؟])", r"\1", normalized)
    normalized = " ".join(normalized.split())
    return Normalization(text, normalized, entities)
