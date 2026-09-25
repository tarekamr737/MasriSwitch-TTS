"""The baseline aggregator must reject partial, altered, and duplicate outputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from masriswitch.config import Paths
from masriswitch.eval.aggregate import aggregate_e0, aggregate_evaluation
from masriswitch.eval.report import write_e0_report


def _fixture(tmp_path: Path) -> tuple[Paths, Path]:
    paths = Paths(tmp_path)
    paths.artifacts.mkdir()
    plan = [
        {
            "id": f"p{i}",
            "text": "الرقم Visa",
            "entities": [{"kind": "brand", "value": "Visa", "start": 6, "end": 10}],
            "set": "locked_benchmark",
            "bucket": "mixed",
            "domain": "banking",
        }
        for i in range(489)
    ]
    plan_path = paths.artifacts / "e0_eval_plan.jsonl"
    plan_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in plan), encoding="utf-8"
    )
    (paths.artifacts / "e0_eval_plan_manifest.json").write_text(
        json.dumps({"sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(), "prompts": 489}),
        encoding="utf-8",
    )
    rows = [
        {
            "id": row["id"],
            "reference_text": row["text"],
            "entities": row["entities"],
            "set": row["set"],
            "bucket": row["bucket"],
            "domain": row["domain"],
            "hypothesis": "الرقم فيزا",
            "english_decoder_hypothesis": "Visa",
            "sample_rate": 24000,
            "audio_seconds": 2.0,
            "synthesis_latency_seconds": 1.0,
            "speaker_similarity": 0.8,
            "peak_tts_vram_bytes": 100,
            "model_sha256": "f43256d0b78b8803c638aed0875da5a4b372b4a784690a0156e5baff14f7336c",
            "asr_sha256": "e76620f83d5f5b69efd3d87e3dc180c1bd21df9fbebacfd4335e5e1efcc018da",
            "speaker_model_sha256": (
                "0575cb64845e6b9a10db9bcb74d5ac32b326b8dc90352671d345e2ee3d0126a2"
            ),
            "seed": 42,
            "nfe_steps": 16,
        }
        for row in plan[:2]
    ]
    output = paths.artifacts / "rows.jsonl"
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    return paths, output


def test_aggregate_e0_partial_probe(tmp_path: Path) -> None:
    paths, output = _fixture(tmp_path)
    result = aggregate_e0(paths, output, max_samples=2)
    assert not result["complete"]
    assert result["overall"]["count"] == 2
    assert result["overall"]["english_entity_error_rate_en_decoder"] == 0
    assert result["overall"]["critical_entity_accuracy"] == 1
    assert result["overall"]["mean_rtf"] == 0.5
    assert result["metrics"]["cs_wer"] == 0.5
    assert result["metrics"]["ar_cer"] == "TBD"
    assert not (paths.artifacts / "baseline_metrics.json").exists()


def test_aggregate_e0_rejects_duplicate_rows(tmp_path: Path) -> None:
    paths, output = _fixture(tmp_path)
    rows = output.read_text(encoding="utf-8").splitlines()
    output.write_text(rows[0] + "\n" + rows[0] + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete or duplicated"):
        aggregate_e0(paths, output, max_samples=2)


def test_aggregate_e0_rejects_invalid_audio(tmp_path: Path) -> None:
    paths, output = _fixture(tmp_path)
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    rows[1]["audio_seconds"] = 0
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Invalid audio"):
        aggregate_e0(paths, output, max_samples=2)


def test_report_rejects_partial_probe(tmp_path: Path) -> None:
    paths, output = _fixture(tmp_path)
    result = aggregate_e0(paths, output, max_samples=2)
    with pytest.raises(ValueError, match="complete 489-prompt"):
        write_e0_report(paths, result)


def test_e1_validation_requires_checkpoint_hash_and_validation_ids(tmp_path: Path) -> None:
    paths, output = _fixture(tmp_path)
    plan_path = paths.artifacts / "e0_eval_plan.jsonl"
    plan = [json.loads(line) for line in plan_path.read_text(encoding="utf-8").splitlines()]
    for row in plan[:2]:
        row["set"] = "validation"
    plan_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in plan), encoding="utf-8"
    )
    (paths.artifacts / "e0_eval_plan_manifest.json").write_text(
        json.dumps({"sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(), "prompts": 489}),
        encoding="utf-8",
    )
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    checkpoint_sha = "a" * 64
    for row in rows:
        row["set"] = "validation"
        row["model_sha256"] = checkpoint_sha
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    result = aggregate_evaluation(
        paths, output, stage="E1", model_sha256=checkpoint_sha, subset="validation", max_samples=2
    )
    assert result["overall"]["count"] == 2
    assert (paths.artifacts / f"e1_validation_{checkpoint_sha[:12]}_metrics.json").is_file()
    with pytest.raises(ValueError, match="Unexpected model hash"):
        aggregate_evaluation(
            paths, output, stage="E1", model_sha256="b" * 64, subset="validation", max_samples=2
        )
