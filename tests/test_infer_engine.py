from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from masriswitch.config import Paths
from masriswitch.infer.engine import approved_engine_files


def _write_manifest(root: Path) -> None:
    artifacts = root / "artifacts"
    artifacts.mkdir()
    checkpoint = artifacts / "selected.pt"
    checkpoint.write_bytes(b"checkpoint")
    (artifacts / "train_manifest.json").write_text(
        json.dumps(
            {
                "best_checkpoint": str(checkpoint),
                "checkpoint_sha256": hashlib.sha256(b"checkpoint").hexdigest(),
                "experiment": "E1",
            }
        ),
        encoding="utf-8",
    )


def test_public_engine_rejects_private_reference(tmp_path: Path) -> None:
    _write_manifest(tmp_path)
    reference_dir = tmp_path / "artifacts" / "reference"
    reference_dir.mkdir()
    (reference_dir / "approved.json").write_text(
        json.dumps(
            {
                "filename": "sample.wav",
                "sha256": "placeholder",
                "transcript": "sample",
                "speaker_consent_documented": False,
                "public_use_approved": False,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(PermissionError, match="not approved"):
        approved_engine_files(Paths(tmp_path))


def test_public_engine_rejects_reference_path_escape(tmp_path: Path) -> None:
    _write_manifest(tmp_path)
    reference_dir = tmp_path / "artifacts" / "reference"
    reference_dir.mkdir()
    (reference_dir / "approved.json").write_text(
        json.dumps(
            {
                "filename": "../../outside.wav",
                "sha256": "placeholder",
                "transcript": "sample",
                "speaker_consent_documented": True,
                "public_use_approved": True,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Reference path"):
        approved_engine_files(Paths(tmp_path))
