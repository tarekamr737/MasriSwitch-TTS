from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from masriswitch.config import Paths
from masriswitch.train.finalize import finalize_training_manifest


def _evidence(root: Path) -> tuple[Paths, Path, Path]:
    paths = Paths(root)
    paths.artifacts.mkdir()
    ids = [f"train-{index:04d}" for index in range(3414)]
    ids_sha = hashlib.sha256("\n".join(ids).encode()).hexdigest()
    archive_sha = "a" * 64
    (paths.artifacts / "kaggle_full_archive.json").write_text(
        json.dumps({"train_ids_sha256": ids_sha, "archive_sha256": archive_sha}),
        encoding="utf-8",
    )
    checkpoint = paths.artifacts / "selected.pt"
    checkpoint.write_bytes(b"selected checkpoint")
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    (paths.artifacts / "checkpoint_selection.json").write_text(
        json.dumps({"selected_checkpoint_sha256": checkpoint_sha}), encoding="utf-8"
    )
    result = paths.artifacts / "e1_1000_result.json"
    result.write_text(
        json.dumps(
            {
                "experiment": "E1",
                "complete": False,
                "target_updates": 8000,
                "updates_per_rank": [1000, 1000],
                "smoke_prompts_passed": 10,
                "pilot_passed": True,
                "training_source_ids": ["d1"],
                "training_sample_ids": ids,
                "full_archive_sha256": archive_sha,
                "seed": 42,
                "checkpoint_sha256": checkpoint_sha,
                "best_checkpoint": "/kaggle/working/model_last.pt",
            }
        ),
        encoding="utf-8",
    )
    return paths, result, checkpoint


def test_finalize_selected_checkpoint_binds_local_bytes(tmp_path: Path) -> None:
    paths, result, checkpoint = _evidence(tmp_path)
    manifest = finalize_training_manifest(paths, result, checkpoint)
    assert manifest["best_checkpoint"] == str(checkpoint.resolve())
    assert manifest["selected_updates"] == 1000
    assert len(manifest["training_sample_ids"]) == 3414
    assert json.loads((paths.artifacts / "train_manifest.json").read_text()) == manifest


def test_finalize_rejects_changed_checkpoint(tmp_path: Path) -> None:
    paths, result, checkpoint = _evidence(tmp_path)
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint hash mismatch"):
        finalize_training_manifest(paths, result, checkpoint)
    assert not (paths.artifacts / "train_manifest.json").exists()


def test_finalize_rejects_nonselected_stage(tmp_path: Path) -> None:
    paths, result, checkpoint = _evidence(tmp_path)
    (paths.artifacts / "checkpoint_selection.json").write_text(
        json.dumps({"selected_checkpoint_sha256": "b" * 64}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="metric-based checkpoint selection"):
        finalize_training_manifest(paths, result, checkpoint)
