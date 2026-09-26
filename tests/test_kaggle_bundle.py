from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import pytest

from scripts.build_kaggle_bundle import build_bundle


@pytest.mark.parametrize("stage", ["probe", "full-probe"])
def test_probe_bundle_does_not_require_its_own_result(tmp_path: Path, stage: str) -> None:
    required = [
        "src/masriswitch/__init__.py",
        "configs/train_e1.yaml",
        "pyproject.toml",
        "artifacts/data_audit.json",
        "artifacts/upstream/silma/vocab.txt",
        "scripts/archive_pilot.py",
        "scripts/kaggle_train_probe.py",
        "scripts/kaggle_train_pilot.py",
        "artifacts/kaggle_pilot_archive.json",
        "artifacts/train_patch.json",
    ]
    if stage == "full-probe":
        required += [
            "artifacts/train_probe.json",
            "scripts/archive_full.py",
            "scripts/kaggle_full_probe.py",
            "scripts/kaggle_train_e1.py",
            "artifacts/pilot_500_result.json",
            "artifacts/kaggle_full_archive.json",
        ]
    for name in required:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture", encoding="utf-8")
    private = tmp_path / "artifacts/reference/private.wav"
    private.parent.mkdir(parents=True)
    private.write_bytes(b"must not be bundled")
    output = tmp_path / "artifacts/bundle.zip"
    result = build_bundle(tmp_path, output, stage)
    with ZipFile(output) as archive:
        assert set(archive.namelist()) == set(required)
    assert result["files"] == len(required)
    missing_result = "train_probe.json" if stage == "probe" else "full_probe_result.json"
    assert not (tmp_path / "artifacts" / missing_result).exists()
