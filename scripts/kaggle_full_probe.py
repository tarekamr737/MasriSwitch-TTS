"""Bounded 20-update rehearsal on all 3,414 audited E1 train rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
from pathlib import Path
from typing import Any

from kaggle_train_pilot import _install_python310
from kaggle_train_probe import (
    FRAMES,
    SILMA_SHA,
    SOURCE_REV,
    _download_archive,
    _probe_once,
    _sha256,
    _stage_code,
)


def _extract_full(archive_path: Path, output: Path, arrow_sha: str) -> tuple[Path, Path]:
    output.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        files = archive.getmembers()
        if len(files) != 3417:
            raise ValueError("Full archive must have 3,414 WAVs and 3 metadata files")
        for item in files:
            path = Path(item.name)
            if not item.isfile() or path.is_absolute() or ".." in path.parts:
                raise ValueError("Unsafe full archive member")
            if path.parts[0] not in {"finetuning_project_char", "wavs"}:
                raise ValueError("Unexpected full archive path")
        archive.extractall(output, filter="data")
    dataset = output / "finetuning_project_char"
    wavs = output / "wavs"
    if _sha256(dataset / "raw.arrow") != arrow_sha or len(list(wavs.glob("*.wav"))) != 3414:
        raise ValueError("Full Arrow hash or audio count mismatch")
    return dataset, wavs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--archive-url", required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--arrow-sha256", required=True)
    parser.add_argument("--archive-manifest", type=Path, required=True)
    parser.add_argument("--patch-manifest", type=Path, required=True)
    parser.add_argument("--probe-result", type=Path, required=True)
    parser.add_argument("--pilot-result", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=3414)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.max_samples != 3414 or args.seed != 42:
        raise ValueError("Full probe requires locked 3,414 train rows and seed 42")
    pilot = json.loads(args.pilot_result.read_text(encoding="utf-8"))
    probe = json.loads(args.probe_result.read_text(encoding="utf-8"))
    if (
        pilot.get("complete") is not True
        or pilot.get("updates_per_rank") != [500, 500]
        or pilot.get("smoke_prompts_passed") != 10
        or probe.get("passed") is not True
        or probe.get("selected", {}).get("updates_per_rank") != [20, 20]
    ):
        raise ValueError("Passing 500-update pilot and memory probe are required")
    manifest = json.loads(args.archive_manifest.read_text(encoding="utf-8"))
    if (
        manifest.get("source_revision") != SOURCE_REV
        or manifest.get("train_only") is not True
        or manifest.get("train_rows") != 3414
        or manifest.get("archive_sha256") != args.archive_sha256
        or manifest.get("train_arrow_sha256") != args.arrow_sha256
    ):
        raise ValueError("Full archive manifest changed")
    if args.dry_run:
        print(json.dumps({"train_rows": 3414, "updates_per_attempt": 20, "frames": FRAMES}))
        return
    result_path = Path("/kaggle/working/masriswitch_full_probe_result.json")
    if args.resume and result_path.is_file():
        previous = json.loads(result_path.read_text(encoding="utf-8"))
        if previous.get("full_archive_sha256") != args.archive_sha256:
            raise ValueError("Resume archive hash changed")
        if previous.get("passed") is True:
            print("MASRISWITCH_FULL_PROBE_RESUMED", flush=True)
            return
    if sys.version_info[:2] != (3, 10):
        _install_python310(args.root, Path(__file__))
    import torch

    if torch.cuda.device_count() != 2 or any(
        "T4" not in torch.cuda.get_device_name(i) for i in (0, 1)
    ):
        raise RuntimeError("Two T4 GPUs are required")
    archive = Path("/tmp/masriswitch-full-data.tar.gz")
    _download_archive(args.archive_url, archive, args.archive_sha256)
    dataset, wavs = _extract_full(archive, Path("/tmp/masriswitch-full-data"), args.arrow_sha256)
    ids_hash = hashlib.sha256(
        "\n".join(sorted(path.stem for path in wavs.glob("*.wav"))).encode()
    ).hexdigest()
    if ids_hash != manifest["train_ids_sha256"]:
        raise ValueError("Full train IDs changed")
    expected_patch = json.loads(args.patch_manifest.read_text(encoding="utf-8"))
    stage, model = _stage_code(args.root, expected_patch)
    attempts: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    for checkpointing, optimizer_8bit in ((False, False), (True, False), (True, True)):
        for frames in FRAMES:
            stable, free_gb, result = _probe_once(
                stage,
                model,
                dataset,
                wavs,
                frames,
                checkpointing,
                optimizer_8bit,
                args.seed,
                len(attempts),
            )
            attempts.append(result)
            if stable and free_gb >= 1.0:
                selected = result
                break
        if selected:
            break
    evidence = {
        "source_revision": SOURCE_REV,
        "silma_sha256": SILMA_SHA,
        "full_archive_sha256": args.archive_sha256,
        "train_arrow_sha256": args.arrow_sha256,
        "patch_hashes": expected_patch,
        "pilot_checkpoint_sha256": pilot["checkpoint_sha256"],
        "seed": args.seed,
        "target_updates_per_attempt": 20,
        "attempts": attempts,
        "selected": selected,
        "passed": selected is not None,
    }
    result_path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    if not selected:
        raise RuntimeError("Full data probe failed")
    print("MASRISWITCH_FULL_PROBE_PASSED", json.dumps(selected), flush=True)


if __name__ == "__main__":
    main()
