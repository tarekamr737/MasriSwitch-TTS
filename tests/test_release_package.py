"""The local release bundle is created only from a gated, hash-checked checkpoint."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from masriswitch.config import Paths
from masriswitch.release import package as release_package


def test_release_bundle_requires_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        release_package,
        "check_release",
        lambda paths: {"weight_publication_allowed": False, "blockers": ["unmeasured"]},
    )
    with pytest.raises(PermissionError, match="unmeasured"):
        release_package.package_release(Paths(tmp_path))


def test_release_bundle_hashes_selected_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = Paths(tmp_path)
    checkpoint = paths.artifacts / "selected.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"selected checkpoint")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    (paths.artifacts / "train_manifest.json").write_text(
        json.dumps({"best_checkpoint": str(checkpoint), "checkpoint_sha256": digest})
    )
    for path in (
        paths.root / "LICENSE",
        paths.root / "NOTICE",
        paths.reports / "MODEL_CARD_DRAFT.md",
        paths.reports / "EVALUATION.md",
        paths.reports / "DATA_AUDIT.md",
        paths.artifacts / "upstream/silma/config.yaml",
        paths.artifacts / "upstream/silma/vocab.txt",
        paths.artifacts / "data_audit.json",
        paths.artifacts / "eval_metrics.json",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("test")
    monkeypatch.setattr(
        release_package,
        "check_release",
        lambda paths: {"weight_publication_allowed": True, "blockers": []},
    )
    result = release_package.package_release(paths)
    bundle = Path(result["bundle"])
    assert (bundle / "model.pt").read_bytes() == checkpoint.read_bytes()
    assert result["files_sha256"]["model.pt"] == digest
    assert (bundle / "README.md").read_text() == "test"
    assert release_package.package_release(paths)["checkpoint_sha256"] == digest
