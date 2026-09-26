"""Build an allowlisted Space bundle and deploy only on free ZeroGPU hardware."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    artifacts = root / "artifacts"
    bundle = artifacts / "space_bundle"
    bundle.mkdir(exist_ok=True)
    files = [*sorted((root / "deploy/space").glob("*")), root / "pyproject.toml", root / "LICENSE"]
    files += sorted((root / "src/masriswitch").rglob("*.py"))
    allow = []
    for source in files:
        relative = (
            Path(source.name)
            if source.parent == root / "deploy/space"
            else source.relative_to(root)
        )
        target = bundle / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        allow.append(relative.as_posix())
    approval = json.loads((artifacts / "reference/approved.json").read_text(encoding="utf-8"))
    if not approval["speaker_consent_documented"] or not approval["public_use_approved"]:
        raise PermissionError("Public reference consent is required")
    wave = (artifacts / "reference/consented.wav").read_bytes()
    if hashlib.sha256(wave).hexdigest() != approval["sha256"]:
        raise ValueError("Reference hash mismatch")
    encoded = base64.b64encode(wave).decode("ascii")
    parts = [encoded[i : i + 64_000] for i in range(0, len(encoded), 64_000)]
    secret_approval = {
        key: approval[key]
        for key in ("sha256", "transcript", "speaker_consent_documented", "public_use_approved")
    }
    secret_approval["parts"] = len(parts)
    manifest = {name: hashlib.sha256((bundle / name).read_bytes()).hexdigest() for name in allow}
    evidence = {
        "repo_id": "Tarek737/MasriSwitch-TTS-Demo",
        "hardware": "zero-a10g",
        "files_sha256": manifest,
        "reference_sha256": approval["sha256"],
        "secret_parts": len(parts),
        "status": "prepared",
    }
    output = artifacts / "space_deployment.json"
    output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    if args.dry_run:
        print(json.dumps({"status": "prepared", "files": len(allow), "hardware": "zero-a10g"}))
        return
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])
    if api.whoami()["name"] != "Tarek737":
        raise PermissionError("Unexpected Hugging Face account")
    if not api.repo_exists(evidence["repo_id"], repo_type="space"):
        api.create_repo(
            evidence["repo_id"],
            repo_type="space",
            space_sdk="gradio",
            space_hardware="zero-a10g",
            private=False,
        )
    runtime = api.get_space_runtime(evidence["repo_id"])
    if (runtime.requested_hardware or runtime.hardware) != "zero-a10g":
        raise RuntimeError("Space must use free ZeroGPU; no paid fallback is allowed")
    for index, value in enumerate(parts):
        api.add_space_secret(evidence["repo_id"], f"REFERENCE_WAV_B64_{index}", value)
    api.add_space_secret(
        evidence["repo_id"],
        "REFERENCE_APPROVAL_JSON",
        json.dumps(secret_approval, ensure_ascii=False),
    )
    commit = api.upload_folder(
        repo_id=evidence["repo_id"],
        repo_type="space",
        folder_path=bundle,
        allow_patterns=allow,
        commit_message="Deploy fixed consented voice on free ZeroGPU",
    )
    evidence.update(status="submitted", revision=commit.oid)
    output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "submitted", "revision": commit.oid, "url": commit.repo_url}))


if __name__ == "__main__":
    main()
