"""Verify a pinned public HF release and smoke-test it in a fresh Kaggle runtime."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from kaggle_e0_eval import VOCOS_REV, VOCOS_SHA, _sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="Tarek737/MasriSwitch-TTS")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision) or not re.fullmatch(
        r"[0-9a-f]{64}", args.checkpoint_sha256
    ):
        raise ValueError("Immutable HF commit and selected checkpoint SHA256 are required")
    if not 1 <= args.max_samples <= 10:
        raise ValueError("Release smoke is limited to 1–10 prompts")
    if args.dry_run:
        print(
            json.dumps(
                {"repo_id": args.repo_id, "revision": args.revision, "planned": args.max_samples}
            )
        )
        return
    project = Path(__file__).resolve().parents[1]
    if sys.version_info[:2] != (3, 10):
        subprocess.run(["uv", "python", "install", "3.10"], check=True, timeout=300)
        venv = Path("/tmp/masriswitch-hf-smoke-venv310")
        subprocess.run(["uv", "venv", "--python", "3.10", str(venv)], check=True, timeout=60)
        py = str(venv / "bin/python")
        subprocess.run(
            ["uv", "pip", "install", "--python", py, str(project) + "[serve,train]"],
            check=True,
            timeout=900,
        )
        os.execv(py, [py, __file__, *sys.argv[1:]])

    from huggingface_hub import hf_hub_download

    root = Path("/tmp/masriswitch-e1-eval")
    silma = root / "silma"
    output = Path("/kaggle/working/hf_release_smoke.json")
    if output.exists():
        previous = json.loads(output.read_text(encoding="utf-8"))
        if (
            args.resume
            and previous.get("repo_id") == args.repo_id
            and previous.get("revision") == args.revision
            and previous.get("checkpoint_sha256") == args.checkpoint_sha256
            and previous.get("api_prompts") == args.max_samples
            and previous.get("seed") == args.seed
            and previous.get("passed") is True
        ):
            print("HF_RELEASE_SMOKE_ALREADY_COMPLETE", flush=True)
            return
        raise ValueError("Existing HF smoke evidence does not match requested release")
    manifest_path = Path(
        hf_hub_download(
            args.repo_id, "release_manifest.json", revision=args.revision, local_dir=silma
        )
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    hashes = manifest.get("files_sha256", {})
    if (
        manifest.get("weight_publication_allowed") is not True
        or manifest.get("checkpoint_sha256") != args.checkpoint_sha256
        or hashes.get("model.pt") != args.checkpoint_sha256
        or not {"model.pt", "config.yaml", "vocab.txt"}.issubset(hashes)
    ):
        raise ValueError("Release manifest does not bind the selected checkpoint")
    for name, expected in hashes.items():
        if Path(name).name != name or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError("Invalid release filename or hash")
        path = Path(hf_hub_download(args.repo_id, name, revision=args.revision, local_dir=silma))
        if _sha256(path) != expected:
            raise ValueError(f"Downloaded release hash mismatch: {name}")
        print("HF_RELEASE_FILE_VERIFIED", name, flush=True)
    for name in ("config.yaml", "pytorch_model.bin"):
        path = Path(
            hf_hub_download(
                "charactr/vocos-mel-24khz", name, revision=VOCOS_REV, local_dir=root / "vocos"
            )
        )
        if name == "pytorch_model.bin" and _sha256(path) != VOCOS_SHA:
            raise ValueError("Vocos hash mismatch")
    command = [
        sys.executable,
        str(project / "scripts/kaggle_product_smoke.py"),
        "--checkpoint-sha256",
        args.checkpoint_sha256,
        "--max-samples",
        str(args.max_samples),
        "--seed",
        str(args.seed),
    ]
    if args.resume:
        command.append("--resume")
    subprocess.run(command, check=True, timeout=900)
    smoke = json.loads(Path("/kaggle/working/product_smoke.json").read_text(encoding="utf-8"))
    if smoke.get("passed") is not True or smoke.get("checkpoint_sha256") != args.checkpoint_sha256:
        raise ValueError("Downloaded checkpoint smoke failed")
    evidence = {
        **smoke,
        "repo_id": args.repo_id,
        "revision": args.revision,
        "release_manifest_sha256": _sha256(manifest_path),
        "verified_files": sorted(hashes),
        "runtime": "fresh private Kaggle session; Python 3.10",
    }
    output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print("HF_RELEASE_SMOKE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
