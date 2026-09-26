"""Pinned public model download and private fixed-reference provisioning."""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from masriswitch.infer.engine import EngineFiles

MODEL_REPO = "Tarek737/MasriSwitch-TTS"
MODEL_REVISION = "de6cd6819e719876909437cc33ca7086cff369c9"
MANIFEST_SHA = "9ac9d66331adb847995fb3bc6afbd28980a5863d8ada0a3e2da9a91a68a3b0f4"
MODEL_SHA = "558e2ab53e1b5450bcd1a1c30683a1b3e6be1234b7e198b74209f362693eaf92"
VOCOS_REVISION = "0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21"
VOCOS_HASHES = {
    "config.yaml": "da9033922f969a47f0c160010226919e59f27761fd5066f3828d46de6650b0fc",
    "pytorch_model.bin": "97ec976ad1fd67a33ab2682d29c0ac7df85234fae875aefcc5fb215681a91b2a",
}


def restore_reference(root: Path, environment: Mapping[str, str]) -> tuple[Path, str, str]:
    """Restore consented WAV bytes from write-only hosting secrets, never logs."""
    approval = json.loads(environment["REFERENCE_APPROVAL_JSON"])
    if (
        approval.get("speaker_consent_documented") is not True
        or approval.get("public_use_approved") is not True
    ):
        raise PermissionError("Fixed reference is not approved")
    count = approval.get("parts")
    if not isinstance(count, int) or not 1 <= count <= 16:
        raise ValueError("Invalid reference secret part count")
    encoded = "".join(environment[f"REFERENCE_WAV_B64_{i}"] for i in range(count))
    if len(encoded) > 2_000_000:
        raise ValueError("Reference is too large")
    wave = base64.b64decode(encoded, validate=True)
    digest = hashlib.sha256(wave).hexdigest()
    if digest != approval["sha256"] or not str(approval["transcript"]).strip():
        raise ValueError("Reference hash or transcript is invalid")
    root.mkdir(parents=True, exist_ok=True)
    reference = root / "fixed_reference.wav"
    reference.write_bytes(wave)
    return reference, digest, str(approval["transcript"])


def hosted_engine_files(root: Path, environment: Mapping[str, str]) -> EngineFiles:
    from huggingface_hub import hf_hub_download

    from masriswitch.data.audit import sha256_file

    reference, reference_sha, transcript = restore_reference(root / "reference", environment)

    def download(repo: str, revision: str, name: str, folder: str, digest: str) -> Path:
        path = Path(hf_hub_download(repo, name, revision=revision, local_dir=root / folder))
        if sha256_file(path) != digest:
            raise ValueError(f"Pinned file hash mismatch: {name}")
        return path

    manifest = download(MODEL_REPO, MODEL_REVISION, "release_manifest.json", "model", MANIFEST_SHA)
    hashes = json.loads(manifest.read_text(encoding="utf-8"))["files_sha256"]
    for name in ("model.pt", "config.yaml", "vocab.txt"):
        download(MODEL_REPO, MODEL_REVISION, name, "model", hashes[name])
    for name, digest in VOCOS_HASHES.items():
        download("charactr/vocos-mel-24khz", VOCOS_REVISION, name, "vocos", digest)
    files = EngineFiles(
        checkpoint=root / "model/model.pt",
        checkpoint_sha256=MODEL_SHA,
        vocab=root / "model/vocab.txt",
        silma_config=root / "model/config.yaml",
        vocoder_dir=root / "vocos",
        reference_audio=reference,
        reference_sha256=reference_sha,
        reference_text=transcript,
        model_id="Tarek737/MasriSwitch-TTS (experimental E1)",
    )
    files.validate()
    return files
