"""Independent transcript and performance metrics."""

from __future__ import annotations

import math
import random
import re
from collections.abc import Callable, Sequence
from typing import TypeVar

import numpy as np

from masriswitch.text.entities import Entity

T = TypeVar("T")


def edit_distance(reference: Sequence[T], hypothesis: Sequence[T]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, left in enumerate(reference, 1):
        current = [i]
        for j, right in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def metric_tokens(text: str) -> list[str]:
    return re.findall(r"[\w]+", text.casefold())


def _contains_tokens(haystack: list[str], needle: list[str]) -> bool:
    return any(
        haystack[index : index + len(needle)] == needle
        for index in range(len(haystack) - len(needle) + 1)
    )


def word_error_rate(references: Sequence[str], hypotheses: Sequence[str]) -> float:
    if len(references) != len(hypotheses):
        raise ValueError("Reference/hypothesis count differs")
    gold = [metric_tokens(text) for text in references]
    denominator = sum(len(tokens) for tokens in gold)
    if not denominator:
        raise ValueError("Empty reference")
    return (
        sum(
            edit_distance(tokens, metric_tokens(hyp))
            for tokens, hyp in zip(gold, hypotheses, strict=True)
        )
        / denominator
    )


def arabic_cer(references: Sequence[str], hypotheses: Sequence[str]) -> float:
    arabic = re.compile(r"[\u0600-\u06ff]")
    gold = ["".join(arabic.findall(text)) for text in references]
    predicted = ["".join(arabic.findall(text)) for text in hypotheses]
    denominator = sum(map(len, gold))
    if not denominator:
        raise ValueError("No Arabic characters in references")
    return sum(edit_distance(a, b) for a, b in zip(gold, predicted, strict=True)) / denominator


def entity_accuracy(
    entities: Sequence[Sequence[Entity]], hypotheses: Sequence[str], kinds: set[str]
) -> float:
    if len(entities) != len(hypotheses):
        raise ValueError("Entity/hypothesis count differs")
    total = correct = 0
    for row_entities, hypothesis in zip(entities, hypotheses, strict=True):
        normalized_hyp = metric_tokens(hypothesis)
        for entity in row_entities:
            if entity.kind in kinds:
                total += 1
                correct += _contains_tokens(normalized_hyp, metric_tokens(entity.value))
    if not total:
        raise ValueError("No matching gold entities")
    return correct / total


def english_entity_error_rate(
    entities: Sequence[Sequence[Entity]], hypotheses: Sequence[str]
) -> float:
    """Wrong or missing ASCII-letter entities divided by reference entities."""
    if len(entities) != len(hypotheses):
        raise ValueError("Entity/hypothesis count differs")
    english = [
        [entity for entity in row if re.search(r"[A-Za-z]", entity.value)] for row in entities
    ]
    total = sum(map(len, english))
    if not total:
        raise ValueError("No English entities in references")
    correct = 0
    for row, hypothesis in zip(english, hypotheses, strict=True):
        tokens = metric_tokens(hypothesis)
        correct += sum(_contains_tokens(tokens, metric_tokens(entity.value)) for entity in row)
    return 1 - correct / total


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("Embedding shapes differ")
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 0:
        raise ValueError("Zero embedding")
    return float(np.dot(left, right) / denom)


def real_time_factor(latency_seconds: float, audio_seconds: float) -> float:
    if not math.isfinite(latency_seconds) or audio_seconds <= 0:
        raise ValueError("Invalid duration")
    return latency_seconds / audio_seconds


def bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[Sequence[float]], float],
    seed: int = 42,
    replicates: int = 2000,
) -> tuple[float, float]:
    if not values or replicates < 100:
        raise ValueError("Insufficient bootstrap input")
    rng = random.Random(seed)
    samples = sorted(statistic([rng.choice(values) for _ in values]) for _ in range(replicates))
    return samples[int(0.025 * replicates)], samples[int(0.975 * replicates)]


def bootstrap_ratio_ci(
    numerators: Sequence[float],
    denominators: Sequence[float],
    seed: int = 42,
    replicates: int = 2000,
) -> tuple[float, float]:
    """Prompt bootstrap for micro-averaged error rates with variable lengths."""
    if len(numerators) != len(denominators) or not numerators or replicates < 100:
        raise ValueError("Invalid bootstrap input")
    if any(
        n < 0 or d < 0 or not math.isfinite(n + d)
        for n, d in zip(numerators, denominators, strict=True)
    ):
        raise ValueError("Non-finite or negative bootstrap counts")
    if sum(denominators) <= 0:
        raise ValueError("Zero denominator")
    rng = random.Random(seed)
    size = len(numerators)
    samples: list[float] = []
    while len(samples) < replicates:
        indices = [rng.randrange(size) for _ in range(size)]
        bottom = sum(denominators[i] for i in indices)
        if bottom > 0:
            samples.append(sum(numerators[i] for i in indices) / bottom)
    samples.sort()
    return samples[int(0.025 * replicates)], samples[int(0.975 * replicates)]
