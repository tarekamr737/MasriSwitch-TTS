from __future__ import annotations

import pytest

from masriswitch.infer.chunks import speech_chunks
from masriswitch.text.normalize import normalize_text

LONG_TEXT = (
    "لو سمحت راجع الـ order رقم ORD123، واعمل update للـ delivery address قبل الساعة 14:30، "
    "وابعتلي confirmation بالـ email بعد خصم 12.5% من إجمالي 1499 EGP، "
    "ولو الدفع اترفض جرّب credit card تانية."
)


def test_expanded_arabic_sentence_is_bounded_without_losing_words() -> None:
    normalized = normalize_text(LONG_TEXT).normalized_text
    chunks = speech_chunks(normalized, max_bytes=113)
    assert len(chunks) > 3
    assert " ".join(chunks) == normalized
    assert all(len(chunk.encode("utf-8")) <= 113 for chunk in chunks)


def test_arabic_clause_boundary_and_unpunctuated_fallback() -> None:
    assert speech_chunks("أهلا بيك، عامل إيه؟", max_bytes=100) == ["أهلا بيك،", "عامل إيه؟"]
    text = "راجع ال account معايا " * 20
    chunks = speech_chunks(text, max_bytes=70)
    assert " ".join(chunks).split() == text.split()
    assert all(len(chunk.encode()) <= 70 for chunk in chunks)


def test_decimal_and_time_tokens_are_not_split() -> None:
    assert speech_chunks("12.5% at 14:30", max_bytes=40) == ["12.5% at 14:30"]


@pytest.mark.parametrize("text", ["", "أ" * 50])
def test_invalid_or_oversized_word_fails_before_inference(text: str) -> None:
    with pytest.raises(ValueError):
        speech_chunks(text, max_bytes=40)


def test_engine_emits_every_chunk_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    from types import SimpleNamespace

    import numpy as np

    from masriswitch.infer.engine import F5Engine

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(manual_seed=lambda _: None))
    engine = object.__new__(F5Engine)
    engine.files = SimpleNamespace(reference_audio="unused.wav", reference_text="reference")
    engine.seed, engine.nfe_steps, engine._chunk_bytes = 42, 16, 60
    engine._device, engine._model, engine._vocoder = None, None, None
    calls = []

    def infer(_reference, _transcript, text, *_args, **_kwargs):
        calls.append(text)
        return np.full(100, len(calls), dtype=np.float32), 24000, None

    engine._infer_process = infer
    text = "أهلا بيك، راجع ال account معايا."
    wave, rate = engine.synthesize(text)
    assert calls == speech_chunks(normalize_text(text).normalized_text, max_bytes=60)
    assert rate == 24000
    assert len(wave) == len(calls) * 100 + (len(calls) - 1) * 2880
    assert np.all(wave[:100] == 1)
    assert np.all(wave[-100:] == len(calls))
