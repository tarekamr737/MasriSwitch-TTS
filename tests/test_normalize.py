"""Hand-written golden spoken-text cases across entity classes."""

import pytest

from masriswitch.text.entities import parse_entities
from masriswitch.text.normalize import normalize_text


@pytest.mark.parametrize(
    "word",
    [
        "account",
        "access",
        "accuracy",
        "accumulation",
        "identification",
        "idiosyncratic",
        "ordinary",
    ],
)
def test_ordinary_english_words_are_not_spelled_as_ids(word: str) -> None:
    text = f"راجع ال {word} معايا."
    assert normalize_text(text).normalized_text == text
    assert all(entity.kind != "id" for entity in parse_entities(text))


@pytest.mark.parametrize("identifier", ["ORD123", "acc456", "ID-ABC", "ACC_AB12"])
def test_explicit_ids_remain_recognized(identifier: str) -> None:
    assert parse_entities(identifier)[0].kind == "id"


GOLDEN = [
    ("0", "صفر"),
    ("1", "واحد"),
    ("2", "اتنين"),
    ("3", "تلاتة"),
    ("4", "أربعة"),
    ("5", "خمسة"),
    ("6", "ستة"),
    ("7", "سبعة"),
    ("8", "تمانية"),
    ("9", "تسعة"),
    ("٠", "صفر"),
    ("١", "واحد"),
    ("٢", "اتنين"),
    ("٣", "تلاتة"),
    ("٤", "أربعة"),
    ("٥", "خمسة"),
    ("٦", "ستة"),
    ("٧", "سبعة"),
    ("٨", "تمانية"),
    ("٩", "تسعة"),
    ("10", "عشرة"),
    ("11", "حداشر"),
    ("12", "اتناشر"),
    ("13", "تلتاشر"),
    ("14", "أربعتاشر"),
    ("15", "خمستاشر"),
    ("16", "ستاشر"),
    ("17", "سبعتاشر"),
    ("18", "تمنتاشر"),
    ("19", "تسعتاشر"),
    ("20", "عشرين"),
    ("30", "تلاتين"),
    ("40", "أربعين"),
    ("50", "خمسين"),
    ("60", "ستين"),
    ("70", "سبعين"),
    ("80", "تمانين"),
    ("90", "تسعين"),
    ("100", "مية"),
    ("200", "ميتين"),
    ("OTP", "أو تي بي"),
    ("API", "إيه بي آي"),
    ("AI", "إيه آي"),
    ("URL", "يو آر إل"),
    ("SMS", "إس إم إس"),
    ("USB", "يو إس بي"),
    ("Wi-Fi", "واي فاي"),
    ("otp", "أو تي بي"),
    ("api", "إيه بي آي"),
    ("wi-fi", "واي فاي"),
    ("0 EGP", "صفر جنيه"),
    ("1 EGP", "واحد جنيه"),
    ("2 EGP", "اتنين جنيه"),
    ("3 USD", "تلاتة دولار"),
    ("4 SAR", "أربعة ريال"),
    ("5 جنيه", "خمسة جنيه"),
    ("6 دولار", "ستة دولار"),
    ("7 ريال", "سبعة ريال"),
    ("12.5 EGP", "اتناشر فاصلة خمسة جنيه"),
    ("499 EGP", "أربعمية وتسعة وتسعين جنيه"),
    ("0%", "صفر في المية"),
    ("1%", "واحد في المية"),
    ("2%", "اتنين في المية"),
    ("3%", "تلاتة في المية"),
    ("4%", "أربعة في المية"),
    ("5%", "خمسة في المية"),
    ("6%", "ستة في المية"),
    ("7%", "سبعة في المية"),
    ("8%", "تمانية في المية"),
    ("12.5%", "اتناشر فاصلة خمسة في المية"),
    ("1 kg", "واحد كيلو جرام"),
    ("2 kg", "اتنين كيلو جرام"),
    ("3 km", "تلاتة كيلومتر"),
    ("4 km", "أربعة كيلومتر"),
    ("5 cm", "خمسة سنتيمتر"),
    ("6 cm", "ستة سنتيمتر"),
    ("7 GB", "سبعة جيجا بايت"),
    ("8 GB", "تمانية جيجا بايت"),
    ("9 MB", "تسعة ميجا بايت"),
    ("12.5 MB", "اتناشر فاصلة خمسة ميجا بايت"),
    ("01/01/2000", "واحد واحد ألفين"),
    ("02/01/2000", "اتنين واحد ألفين"),
    ("03/01/2000", "تلاتة واحد ألفين"),
    ("04/01/2000", "أربعة واحد ألفين"),
    ("05/01/2000", "خمسة واحد ألفين"),
    ("06/01/2000", "ستة واحد ألفين"),
    ("07/01/2000", "سبعة واحد ألفين"),
    ("08/01/2000", "تمانية واحد ألفين"),
    ("09/01/2000", "تسعة واحد ألفين"),
    ("10/01/2000", "عشرة واحد ألفين"),
    ("1:00 AM", "واحد وصفر صباحا"),
    ("2:01 AM", "اتنين وواحد صباحا"),
    ("3:02 AM", "تلاتة واتنين صباحا"),
    ("4:03 AM", "أربعة وتلاتة صباحا"),
    ("5:04 AM", "خمسة وأربعة صباحا"),
    ("6:05 PM", "ستة وخمسة مساء"),
    ("7:06 PM", "سبعة وستة مساء"),
    ("8:07 PM", "تمانية وسبعة مساء"),
    ("9:08 PM", "تسعة وتمانية مساء"),
    ("10:09 PM", "عشرة وتسعة مساء"),
]


@pytest.mark.parametrize(("text", "expected"), GOLDEN)
def test_normalization_golden(text: str, expected: str) -> None:
    assert normalize_text(text).normalized_text == expected


def test_golden_count() -> None:
    assert len(GOLDEN) == 100


def test_mixed_sentence_and_url_protection() -> None:
    result = normalize_text("مساء الخير، الـ Premium Plan بـ 499 EGP. افتح https://x.test/123")
    assert "أربعمية وتسعة وتسعين جنيه" in result.normalized_text
    assert "https://x.test/123" in result.normalized_text
    assert [entity.kind for entity in result.entities] == ["english_term", "money", "url"]
