from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import pytest

from masriswitch.infer.hosted import restore_reference


@pytest.mark.parametrize("fault", [None, "consent", "hash", "part"])
def test_hosted_reference_secrets(tmp_path: Path, fault: str | None) -> None:
    wave = b"test wave bytes"
    approval = {
        "sha256": hashlib.sha256(wave).hexdigest(),
        "transcript": "approved transcript",
        "speaker_consent_documented": fault != "consent",
        "public_use_approved": True,
        "parts": 1,
    }
    environment = {
        "REFERENCE_APPROVAL_JSON": json.dumps(approval),
        "REFERENCE_WAV_B64_0": base64.b64encode(wave).decode(),
    }
    if fault == "hash":
        environment["REFERENCE_WAV_B64_0"] = base64.b64encode(b"changed").decode()
    elif fault == "part":
        del environment["REFERENCE_WAV_B64_0"]
    if fault:
        with pytest.raises((KeyError, ValueError, PermissionError)):
            restore_reference(tmp_path, environment)
        assert not (tmp_path / "fixed_reference.wav").exists()
    else:
        path, digest, transcript = restore_reference(tmp_path, environment)
        assert path.read_bytes() == wave
        assert digest == approval["sha256"]
        assert transcript == approval["transcript"]
