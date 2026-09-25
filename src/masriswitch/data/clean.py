"""Conservative transcript and audio quality helpers."""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from dataclasses import dataclass

import numpy as np

_JUNK = re.compile("[\u0640\u200b-\u200f\u2060\ufeff]")


def clean_transcript(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = _JUNK.sub("", text)
    return " ".join(text.split())


def stable_id(source: str, upstream_id: str, audio_sha256: str) -> str:
    value = f"{source}\0{upstream_id}\0{audio_sha256}".encode()
    return hashlib.sha256(value).hexdigest()[:24]


@dataclass(frozen=True)
class AudioQuality:
    duration: float
    sample_rate: int
    rms_dbfs: float
    peak: float
    clip_ratio: float
    silence_ratio: float
    chars_per_sec: float


def audio_quality(samples: np.ndarray, sample_rate: int, text: str) -> AudioQuality:
    if sample_rate <= 0 or samples.ndim != 1 or samples.size == 0:
        raise ValueError("Audio must be nonempty mono PCM")
    duration = len(samples) / sample_rate
    rms = float(np.sqrt(np.mean(np.square(samples.astype(np.float64)))))
    return AudioQuality(
        duration=duration,
        sample_rate=sample_rate,
        rms_dbfs=20 * math.log10(max(rms, 1e-12)),
        peak=float(np.max(np.abs(samples))),
        clip_ratio=float(np.mean(np.abs(samples) >= 0.999)),
        silence_ratio=float(np.mean(np.abs(samples) < 0.001)),
        chars_per_sec=len(clean_transcript(text)) / duration,
    )


def rejection_reason(quality: AudioQuality, text: str) -> str | None:
    if not clean_transcript(text):
        return "empty_text"
    if quality.duration < 0.8 or quality.duration > 20:
        return "duration"
    if quality.clip_ratio > 0.001:
        return "clipping"
    if quality.silence_ratio > 0.35 or quality.rms_dbfs < -50:
        return "silence"
    return None
