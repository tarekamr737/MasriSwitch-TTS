"""Deterministic update budgeting and memory probe policy."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

BATCH_CANDIDATES = (5600, 4800, 4000, 3200)


@dataclass(frozen=True)
class TrainingPlan:
    target_updates: int
    frames_per_gpu: int
    world_size: int
    grad_accumulation: int
    expected_updates_per_epoch: int
    epochs: int
    expected_total_updates: int


def plan_training(
    duration_seconds: float,
    frames_per_gpu: int,
    *,
    target_updates: int,
    world_size: int = 2,
    grad_accumulation: int = 1,
) -> TrainingPlan:
    if min(duration_seconds, frames_per_gpu, target_updates, world_size, grad_accumulation) <= 0:
        raise ValueError("Training plan inputs must be positive")
    frames = duration_seconds * 24000 / 256
    batches = math.ceil(frames / frames_per_gpu)
    updates_per_epoch = math.ceil(batches / (world_size * grad_accumulation))
    epochs = math.ceil(target_updates / updates_per_epoch)
    return TrainingPlan(
        target_updates,
        frames_per_gpu,
        world_size,
        grad_accumulation,
        updates_per_epoch,
        epochs,
        updates_per_epoch * epochs,
    )


def probe_frame_batch(
    probe: Callable[[int, bool, bool], tuple[bool, float]], minimum_free_gb: float = 1.0
) -> tuple[int, bool, bool]:
    """Probe callback performs 20 real forward/backward steps and returns success, free GB."""
    for checkpointing, optimizer_8bit in ((False, False), (True, False), (True, True)):
        for frames in BATCH_CANDIDATES:
            stable, free_gb = probe(frames, checkpointing, optimizer_8bit)
            if stable and free_gb >= minimum_free_gb:
                return frames, checkpointing, optimizer_8bit
    raise RuntimeError("All frame batches failed; pilot must stop")


def choose_checkpoint(
    candidates: list[dict[str, float]], baseline_ar_cer: float
) -> dict[str, float]:
    """English EER, then code-switch WER, with a 5% relative Arabic CER guardrail."""
    if baseline_ar_cer < 0:
        raise ValueError("Invalid baseline CER")
    eligible = [
        item
        for item in candidates
        if all(math.isfinite(item[key]) for key in ("english_eer", "cs_wer", "ar_cer"))
        and item["ar_cer"] <= baseline_ar_cer * 1.05
    ]
    if not eligible:
        raise ValueError("No checkpoint passes Arabic CER guardrail")
    return min(eligible, key=lambda item: (item["english_eer"], item["cs_wer"]))


def review_validation_progress(
    candidates: list[dict[str, float]], baseline_ar_cer: float
) -> dict[str, int | bool | None]:
    """Stop after three evaluations with no eligible EER or WER improvement."""
    if not candidates or baseline_ar_cer < 0:
        raise ValueError("Validation candidates and baseline CER are required")
    best_eer = math.inf
    best_wer = math.inf
    best_index: int | None = None
    stale = 0
    for index, item in enumerate(candidates):
        if not all(math.isfinite(item[key]) for key in ("english_eer", "cs_wer", "ar_cer")):
            raise ValueError("Non-finite validation metric")
        eligible = item["ar_cer"] <= baseline_ar_cer * 1.05
        improved = eligible and (item["english_eer"] < best_eer or item["cs_wer"] < best_wer)
        stale = 0 if improved else stale + 1
        if eligible:
            best_eer = min(best_eer, item["english_eer"])
            best_wer = min(best_wer, item["cs_wer"])
            if best_index is None or (
                item["english_eer"],
                item["cs_wer"],
            ) < (
                candidates[best_index]["english_eer"],
                candidates[best_index]["cs_wer"],
            ):
                best_index = index
    return {"best_index": best_index, "stale_evaluations": stale, "stop_early": stale >= 3}


def shortlist_validation_candidates(
    candidates: list[dict[str, float]], baseline_ar_cer: float, *, limit: int = 3
) -> list[int]:
    """Rank eligible 50-prompt checks for a bounded full-validation comparison."""
    if not candidates or baseline_ar_cer < 0 or limit <= 0:
        raise ValueError("Candidates, baseline CER, and positive limit are required")
    eligible = []
    for index, item in enumerate(candidates):
        if not all(math.isfinite(item[key]) for key in ("english_eer", "cs_wer", "ar_cer")):
            raise ValueError("Non-finite validation metric")
        if item["ar_cer"] <= baseline_ar_cer * 1.05:
            eligible.append(index)
    if not eligible:
        raise ValueError("No checkpoint passes Arabic CER guardrail")
    return sorted(
        eligible,
        key=lambda index: (candidates[index]["english_eer"], candidates[index]["cs_wer"], index),
    )[:limit]
