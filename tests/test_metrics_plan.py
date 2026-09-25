import hashlib

import numpy as np
import pytest

from masriswitch.eval.metrics import (
    arabic_cer,
    bootstrap_ci,
    bootstrap_ratio_ci,
    cosine_similarity,
    english_entity_error_rate,
    entity_accuracy,
    real_time_factor,
    word_error_rate,
)
from masriswitch.text.entities import Entity
from masriswitch.train.plan import (
    choose_checkpoint,
    plan_training,
    probe_frame_batch,
    review_validation_progress,
)
from masriswitch.train.verify import verify_e1_stage


def test_metrics_known_values() -> None:
    assert word_error_rate(["hello world"], ["hello there"]) == 0.5
    assert arabic_cer(["سلام"], ["سلم"]) == 0.25
    assert entity_accuracy([[Entity("acronym", "OTP", 0, 3)]], ["send OTP"], {"acronym"}) == 1.0
    assert entity_accuracy([[Entity("acronym", "OTP", 0, 3)]], ["notopting"], {"acronym"}) == 0.0
    entities = [[Entity("brand", "Visa", 0, 4), Entity("money", "25 EGP", 5, 11)]]
    assert english_entity_error_rate(entities, ["Visa for 25 EGP"]) == 0.0
    assert english_entity_error_rate(entities, ["Visa for 20 EGP"]) == 0.5
    assert english_entity_error_rate(entities, ["none"]) == 1.0
    with pytest.raises(ValueError):
        english_entity_error_rate([[]], ["none"])
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == 1.0
    assert real_time_factor(0.5, 2.0) == 0.25
    assert bootstrap_ci([1.0, 1.0, 1.0], lambda values: sum(values) / len(values)) == (
        1.0,
        1.0,
    )
    assert bootstrap_ratio_ci([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == (0.5, 0.5)
    with pytest.raises(ValueError, match="Zero denominator"):
        bootstrap_ratio_ci([0.0], [0.0])


def test_training_plan_and_batch_probe() -> None:
    plan = plan_training(3600, 5600, target_updates=500)
    assert plan.expected_total_updates >= 500
    assert plan.expected_total_updates - plan.expected_updates_per_epoch < 500
    calls: list[tuple[int, bool, bool]] = []

    def probe(frames: int, checkpointing: bool, optimizer_8bit: bool) -> tuple[bool, float]:
        calls.append((frames, checkpointing, optimizer_8bit))
        return frames <= 4000, 1.5

    assert probe_frame_batch(probe) == (4000, False, False)
    assert calls == [(5600, False, False), (4800, False, False), (4000, False, False)]


def test_checkpoint_selection_guardrail() -> None:
    candidates = [
        {"english_eer": 0.1, "cs_wer": 0.2, "ar_cer": 0.3},
        {"english_eer": 0.2, "cs_wer": 0.1, "ar_cer": 0.1},
    ]
    assert choose_checkpoint(candidates, 0.1) == candidates[1]
    with pytest.raises(ValueError):
        choose_checkpoint(candidates, 0.05)


def test_validation_progress_stops_after_three_stale_evaluations() -> None:
    candidates = [
        {"english_eer": 0.30, "cs_wer": 0.45, "ar_cer": 0.35},
        {"english_eer": 0.33, "cs_wer": 0.44, "ar_cer": 0.35},
        {"english_eer": 0.34, "cs_wer": 0.46, "ar_cer": 0.35},
        {"english_eer": 0.35, "cs_wer": 0.46, "ar_cer": 0.35},
        {"english_eer": 0.36, "cs_wer": 0.47, "ar_cer": 0.35},
    ]
    result = review_validation_progress(candidates, 0.35)
    assert result == {"best_index": 0, "stale_evaluations": 3, "stop_early": True}


def test_e1_stage_verification_rejects_eval_leakage() -> None:
    ids = [f"sample{i:04d}" for i in range(3414)]
    archive = {
        "archive_sha256": "b" * 64,
        "train_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
    }
    result = {
        "experiment": "E1",
        "target_updates": 8000,
        "updates_per_rank": [1000, 1000],
        "complete": False,
        "smoke_prompts_passed": 10,
        "pilot_passed": True,
        "training_source_ids": ["d1"],
        "full_archive_sha256": "b" * 64,
        "seed": 42,
        "checkpoint_sha256": "a" * 64,
        "training_sample_ids": ids,
    }
    assert verify_e1_stage(result, archive, expected_updates=1000) == "a" * 64
    with pytest.raises(ValueError, match="train archive"):
        verify_e1_stage(
            {**result, "training_sample_ids": ids[:-1] + ["eval-only"]},
            archive,
            expected_updates=1000,
        )
