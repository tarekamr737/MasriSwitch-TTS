import json
from pathlib import Path

import numpy as np
import pytest

from masriswitch.config import Paths, validate_configs
from masriswitch.data.clean import audio_quality, clean_transcript, rejection_reason, stable_id
from masriswitch.data.prepare import finalize_prepared
from masriswitch.data.registry import Source, assert_row_allowed, assert_source_allowed
from masriswitch.data.split import assign_splits
from masriswitch.eval.benchmark import generate_benchmark, write_benchmark
from masriswitch.eval.plan import build_e0_plan
from masriswitch.release.gate import _complete_evaluation, check_release
from masriswitch.text.spans import analyze_switches


def test_config_valid() -> None:
    validate_configs()


def test_license_gate_rejects_nc_and_bad_provenance() -> None:
    source = Source("d1", "example/repo", "a" * 40, None, "cc-by-4.0", True, True)
    assert_row_allowed(source, {"license": "cc-by-4.0", "source_type": "generated"})
    with pytest.raises(PermissionError):
        assert_row_allowed(source, {"license": "cc-by-nc-sa-4.0", "source_type": "generated"})
    with pytest.raises(PermissionError):
        assert_row_allowed(source, {"license": "cc-by-4.0", "source_type": "derived"})
    with pytest.raises(ValueError):
        assert_source_allowed(Source("d1", "x", "TBD", None, "cc-by-4.0", True, True))


def test_audio_quality_rejection() -> None:
    samples = np.zeros(24000, dtype=np.float32)
    quality = audio_quality(samples, 24000, "مرحبا")
    assert rejection_reason(quality, "مرحبا") == "silence"
    assert clean_transcript("مـر\u200bحبا  test") == "مرحبا test"
    assert stable_id("d1", "42", "ab") == stable_id("d1", "42", "ab")


def test_grouped_split_does_not_leak() -> None:
    rows = [
        {"sample_id": str(i), "speaker_id": str(i // 2), "domain": "support", "bucket": "cs_ar_dom"}
        for i in range(100)
    ]
    splits = assign_splits(rows)
    assert splits == assign_splits(reversed(rows))
    assert all(splits[str(i)] == splits[str(i + 1)] for i in range(0, 100, 2))
    assert set(splits.values()) == {"train", "val", "test"}


def test_switches_and_benchmark() -> None:
    assert analyze_switches("أهلا يا فندم").bucket == "ar_only"
    assert analyze_switches("أهلا Premium يا فندم").switch_count == 2
    rows, locked = generate_benchmark()
    assert len(rows) == 1200 and len(locked) == 300
    assert set(locked).issubset({row["id"] for row in rows})


def test_release_gate_fails_closed(tmp_path: Path) -> None:
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/sources.yaml").write_text(
        "sources:\n  d1:\n    repo: x/y\n    revision: abc\n    row_license: cc-by-4.0\n"
    )
    result = check_release(Paths(tmp_path))
    assert result["weight_publication_allowed"] is False
    assert "training_manifest_missing" in result["blockers"]


def test_release_gate_requires_matching_complete_e1_evaluation() -> None:
    metrics = {"cs_wer": 0.5, "english_eer": 0.4, "ar_cer": 0.3}
    baseline = {
        "stage": "E0",
        "complete": True,
        "overall": {"count": 489},
        "plan_sha256": "plan",
        "metrics": metrics,
    }
    e1 = {**baseline, "stage": "E1", "model_sha256": "checkpoint"}
    train = {"checkpoint_sha256": "checkpoint"}
    assert _complete_evaluation(baseline, {"E0": baseline, "E1": e1}, train)
    assert not _complete_evaluation(
        baseline, {"E0": baseline, "E1": {**e1, "overall": {"count": 50}}}, train
    )
    assert not _complete_evaluation(
        baseline, {"E0": baseline, "E1": {**e1, "model_sha256": "other"}}, train
    )
    assert not _complete_evaluation(
        baseline, {"E0": baseline, "E1": {**e1, "overall": "invalid"}}, train
    )


def test_release_gate_rejects_partial_training_split(tmp_path: Path) -> None:
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/sources.yaml").write_text(
        "sources:\n  d1:\n    repo: x/y\n    revision: abc\n"
        "    row_license: cc-by-4.0\n    release_safe: true\n"
    )
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "splits.json").write_text(
        json.dumps(
            {
                "complete": True,
                "rows": [
                    {"sample_id": "train-a", "split": "train", "source_id": "d1"},
                    {"sample_id": "train-b", "split": "train", "source_id": "d1"},
                ],
            }
        )
    )
    (artifacts / "train_manifest.json").write_text(
        json.dumps(
            {
                "training_source_ids": ["d1"],
                "training_sample_ids": ["train-a"],
                "checkpoint_sha256": "a" * 64,
                "smoke_prompts_passed": 10,
                "pilot_passed": True,
            }
        )
    )
    result = check_release(Paths(tmp_path))
    assert "training_sample_ids_not_proven_train_only" in result["blockers"]


def test_finalize_prepared_creates_report_directory(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    vocab = artifacts / "upstream" / "silma" / "vocab.txt"
    vocab.parent.mkdir(parents=True)
    vocab.write_text(" \n", encoding="utf-8")
    (artifacts / "data_audit.json").write_text(json.dumps({"rows_seen": 1}))
    wav = artifacts / "sample.wav"
    wav.write_bytes(b"RIFF")
    source = Source("d1", "example/repo", "a" * 40, None, "cc-by-4.0", True, True)
    accepted = [
        {
            "sample_id": "sample",
            "audio_sha256": "abc",
            "audio_path": str(wav),
            "text": "تجربة واحدة",
            "duration": 2.0,
            "chars_per_sec": 5.0,
            "domain": "test",
            "bucket": "ar_only",
        }
    ]
    result = finalize_prepared(
        source, Paths(tmp_path), accepted=accepted, selected=1, rejects={}, seed=42, complete=True
    )
    assert result["f5_train_rows"] == 1
    assert (tmp_path / "reports" / "DATA_AUDIT.md").is_file()


def test_e0_plan_uses_only_validation_and_locked_prompts(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    write_benchmark(artifacts)
    (artifacts / "splits.json").write_text(
        json.dumps(
            {
                "complete": True,
                "rows": [
                    {"sample_id": "train", "split": "train"},
                    {"sample_id": "val", "split": "val"},
                ],
            }
        ),
        encoding="utf-8",
    )
    (artifacts / "prepared_rows.jsonl").write_text(
        "".join(
            json.dumps(
                {"sample_id": sample_id, "domain": "test", "bucket": "ar_only", "text": text}
            )
            + "\n"
            for sample_id, text in (("train", "Do not evaluate"), ("val", "تجربة واحدة"))
        ),
        encoding="utf-8",
    )
    result = build_e0_plan(Paths(tmp_path))
    assert result["prompts"] == 301
    assert result["validation"] == 1
    assert result["locked_benchmark"] == 300
    rows = [
        json.loads(line)
        for line in (artifacts / "e0_eval_plan.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert "train" not in {row["id"] for row in rows}
