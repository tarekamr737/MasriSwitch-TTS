from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import pytest


@pytest.mark.parametrize("fault", [None, "consent", "hash", "path"])
def test_smoke_reference_requires_bound_consent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str | None
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    module = importlib.import_module("kaggle_product_smoke")
    audio = tmp_path / "voice.wav"
    audio.write_bytes(b"reference")
    approval = {
        "filename": audio.name,
        "sha256": hashlib.sha256(audio.read_bytes()).hexdigest(),
        "transcript": "test transcript",
        "speaker_consent_documented": True,
        "public_use_approved": True,
    }
    if fault == "consent":
        approval["public_use_approved"] = False
    elif fault == "hash":
        audio.write_bytes(b"changed")
    elif fault == "path":
        approval["filename"] = "../outside.wav"
    path = tmp_path / "approved.json"
    path.write_text(json.dumps(approval), encoding="utf-8")
    if fault:
        with pytest.raises((PermissionError, ValueError)):
            module.approved_reference(path)
    else:
        assert module.approved_reference(path) == (
            audio.resolve(),
            approval["sha256"],
            approval["transcript"],
        )
