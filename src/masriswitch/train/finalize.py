"""Bind a selected E1 stage to a local, hash-checked release checkpoint."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from masriswitch.config import Paths
from masriswitch.data.audit import sha256_file
from masriswitch.train.verify import verify_e1_stage


def _read(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def finalize_training_manifest(
    paths: Paths, stage_result: Path, checkpoint: Path
) -> dict[str, Any]:
    """Verify selected-stage evidence and checkpoint bytes before writing a manifest."""
    result = _read(stage_result)
    selection_path = paths.artifacts / "checkpoint_selection.json"
    selection = _read(selection_path)
    archive = _read(paths.artifacts / "kaggle_full_archive.json")
    updates = result.get("updates_per_rank")
    if (
        not isinstance(updates, list)
        or len(updates) != 2
        or type(updates[0]) is not int
        or updates[0] <= 0
        or updates[0] > 8000
        or updates[0] % 1000 != 0
    ):
        raise ValueError("Selected E1 update count is invalid")
    selected_sha = verify_e1_stage(result, archive, expected_updates=updates[0])
    if selection.get("selected_checkpoint_sha256") != selected_sha:
        raise ValueError("Stage result does not match metric-based checkpoint selection")
    progress_path = paths.artifacts / "e1_progress.json"
    progress = _read(progress_path)
    stages = progress.get("stages")
    if (
        not isinstance(stages, list)
        or not stages
        or not all(isinstance(stage, dict) for stage in stages)
    ):
        raise ValueError("Verified training endpoint is required")
    endpoint = stages[-1].get("updates")
    early_stop = progress.get("stop_early") is True
    if (
        type(endpoint) is not int
        or not updates[0] <= endpoint <= 8000
        or endpoint % 1000 != 0
        or (endpoint < 8000 and not early_stop)
        or (early_stop and progress.get("stale_evaluations", 0) < 3)
        or progress.get("locked_benchmark_used") is not False
        or not any(
            stage.get("updates") == updates[0] and stage.get("checkpoint_sha256") == selected_sha
            for stage in stages
        )
    ):
        raise ValueError("Training endpoint or declared early stop is unverified")
    resolved = checkpoint.resolve()
    if not resolved.is_relative_to(paths.artifacts.resolve()) or not resolved.is_file():
        raise ValueError("Selected checkpoint must be a file inside ignored artifacts")
    if sha256_file(resolved) != selected_sha:
        raise ValueError("Selected checkpoint hash mismatch")
    manifest = {
        **result,
        "complete": True,
        "selected_stage_reached_target": result["complete"],
        "training_endpoint_updates": endpoint,
        "stopped_early": early_stop,
        "best_checkpoint": str(resolved),
        "selected_updates": updates[0],
        "stage_result_sha256": sha256_file(stage_result),
        "checkpoint_selection_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
        "training_progress_sha256": sha256_file(progress_path),
    }
    output = paths.artifacts / "train_manifest.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
