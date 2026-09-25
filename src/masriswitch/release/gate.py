"""Fail-closed release gate with machine-readable blockers."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from masriswitch.config import Paths
from masriswitch.data.registry import load_sources


def _read(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else None


def _has_metrics(data: dict[str, Any] | None) -> bool:
    if data is None:
        return False
    metrics = data.get("metrics", data)
    if not isinstance(metrics, dict):
        return False
    return all(
        isinstance(metrics.get(key), (int, float)) and math.isfinite(metrics[key])
        for key in ("cs_wer", "english_eer", "ar_cer")
    )


def _complete_evaluation(
    baseline: dict[str, Any] | None,
    evaluation: dict[str, Any] | None,
    train: dict[str, Any] | None,
) -> bool:
    if not baseline or not evaluation or not train:
        return False
    e0 = evaluation.get("E0")
    e1 = evaluation.get("E1")
    if not isinstance(e0, dict) or not isinstance(e1, dict):
        return False

    def has_489_rows(item: dict[str, Any]) -> bool:
        overall = item.get("overall")
        return isinstance(overall, dict) and overall.get("count") == 489

    return all(
        (
            baseline.get("complete") is True,
            baseline.get("stage") == "E0",
            has_489_rows(baseline),
            _has_metrics(baseline),
            e0.get("complete") is True,
            e0.get("stage") == "E0",
            has_489_rows(e0),
            e0.get("plan_sha256") == baseline.get("plan_sha256"),
            _has_metrics(e0),
            e1.get("complete") is True,
            e1.get("stage") == "E1",
            has_489_rows(e1),
            e1.get("plan_sha256") == baseline.get("plan_sha256"),
            e1.get("model_sha256") == train.get("checkpoint_sha256"),
            _has_metrics(e1),
        )
    )


def check_release(paths: Paths) -> dict[str, Any]:
    blockers: list[str] = []
    sources = load_sources(paths.root / "configs/sources.yaml")
    lock = _read(paths.artifacts / "source_lock.json")
    audit = _read(paths.artifacts / "data_audit.json")
    splits = _read(paths.artifacts / "splits.json")
    train = _read(paths.artifacts / "train_manifest.json")
    selection = _read(paths.artifacts / "checkpoint_selection.json")
    baseline = _read(paths.artifacts / "baseline_metrics.json")
    evaluation = _read(paths.artifacts / "eval_metrics.json")
    if not lock or not lock.get("verified"):
        blockers.append("source_lock_missing_or_unverified")
    elif not all(item.get("revision") for item in lock.get("sources", {}).values()):
        blockers.append("source_revision_missing")
    if not audit or not audit.get("complete") or not audit.get("release_safe"):
        blockers.append("data_audit_incomplete")
    if not splits or not splits.get("complete"):
        blockers.append("splits_incomplete")
    if not train:
        blockers.append("training_manifest_missing")
    else:
        source_ids = train.get("training_source_ids", [])
        if not source_ids:
            blockers.append("training_sources_missing")
        for source_id in source_ids:
            source = sources.get(source_id)
            if source is None or not source.release_safe:
                blockers.append(f"blocked_training_source:{source_id}")
        if not train.get("checkpoint_sha256"):
            blockers.append("checkpoint_hash_missing")
        if train.get("smoke_prompts_passed", 0) < 10:
            blockers.append("checkpoint_smoke_incomplete")
        if train.get("pilot_passed") is not True:
            blockers.append("pilot_not_verified")
        if not selection or selection.get("selected_checkpoint_sha256") != train.get(
            "checkpoint_sha256"
        ):
            blockers.append("metric_checkpoint_selection_missing")
        allowed_ids = {
            row["sample_id"]
            for row in (splits or {}).get("rows", [])
            if row.get("split") == "train" and row.get("source_id") in source_ids
        }
        training_ids = train.get("training_sample_ids", [])
        if (
            not isinstance(training_ids, list)
            or len(training_ids) != len(allowed_ids)
            or not all(isinstance(sample_id, str) for sample_id in training_ids)
            or len(set(training_ids)) != len(training_ids)
            or set(training_ids) != allowed_ids
        ):
            blockers.append("training_sample_ids_not_proven_train_only")
    if not _complete_evaluation(baseline, evaluation, train):
        blockers.append("measured_baseline_or_evaluation_missing")
    card_path = paths.reports / "MODEL_CARD_DRAFT.md"
    if not card_path.is_file():
        blockers.append("model_card_missing")
    else:
        card = card_path.read_text(encoding="utf-8").lower()
        if "tbd" in card or "limitations" not in card or "misuse" not in card:
            blockers.append("model_card_incomplete")
    notice_path = paths.root / "NOTICE"
    if not notice_path.is_file() or not all(
        term in notice_path.read_text(encoding="utf-8").lower()
        for term in ("silma", "abdelrahman", "cc by 4.0")
    ):
        blockers.append("attribution_missing")
    for path in paths.root.iterdir():
        if path.is_file() and path.suffix.lower() in {".wav", ".flac", ".pt", ".parquet"}:
            blockers.append(f"raw_or_checkpoint_in_repo_root:{path.name}")
    if (paths.root / ".env").is_file():
        blockers.append("unignored_env_file_present")
    result = {"weight_publication_allowed": not blockers, "blockers": blockers}
    paths.artifacts.mkdir(parents=True, exist_ok=True)
    (paths.artifacts / "release_gate.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result
