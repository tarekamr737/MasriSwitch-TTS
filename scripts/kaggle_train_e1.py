"""Resumable, exact-update E1 training on the audited full D1 split."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from kaggle_full_probe import _extract_full
from kaggle_train_pilot import _install_python310, _smoke_checkpoint
from kaggle_train_probe import SILMA_SHA, SOURCE_REV, _download_archive, _sha256, _stage_code


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
    parser.add_argument("--full-probe-result", type=Path, required=True)
    parser.add_argument("--resume-checkpoint-url")
    parser.add_argument("--resume-checkpoint-sha256")
    parser.add_argument("--max-samples", type=int, default=3414)
    parser.add_argument("--max-updates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.max_samples != 3414 or args.seed != 42 or not 1 <= args.max_updates <= 8000:
        raise ValueError("E1 requires 3,414 locked train rows, seed 42, and at most 8,000 updates")
    archive_manifest = json.loads(args.archive_manifest.read_text(encoding="utf-8"))
    pilot = json.loads(args.pilot_result.read_text(encoding="utf-8"))
    full_probe = json.loads(args.full_probe_result.read_text(encoding="utf-8"))
    expected_patch = json.loads(args.patch_manifest.read_text(encoding="utf-8"))
    selected = full_probe.get("selected")
    if (
        archive_manifest.get("source_revision") != SOURCE_REV
        or archive_manifest.get("train_only") is not True
        or archive_manifest.get("train_rows") != 3414
        or archive_manifest.get("archive_sha256") != args.archive_sha256
        or archive_manifest.get("train_arrow_sha256") != args.arrow_sha256
        or pilot.get("complete") is not True
        or pilot.get("updates_per_rank") != [500, 500]
        or pilot.get("smoke_prompts_passed") != 10
        or full_probe.get("passed") is not True
        or full_probe.get("full_archive_sha256") != args.archive_sha256
        or full_probe.get("patch_hashes") != expected_patch
        or not isinstance(selected, dict)
        or selected.get("updates_per_rank") != [20, 20]
        or selected.get("finite_loss") is not True
        or selected.get("minimum_free_gb", 0) < 1.0
    ):
        raise ValueError("Verified 500-update pilot and full-data 20-update probe are required")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "rows": 3414,
                    "max_updates": args.max_updates,
                    "frames_per_gpu": selected["frames_per_gpu"],
                    "resume": args.resume,
                }
            )
        )
        return
    if sys.version_info[:2] != (3, 10):
        _install_python310(args.root, Path(__file__))
    import torch
    from datasets import Dataset

    from masriswitch.train.plan import plan_training

    if torch.cuda.device_count() != 2 or any(
        "T4" not in torch.cuda.get_device_name(i) for i in (0, 1)
    ):
        raise RuntimeError("Two T4 GPUs are required")
    archive = Path("/tmp/masriswitch-full-data.tar.gz")
    _download_archive(args.archive_url, archive, args.archive_sha256)
    dataset, wavs = _extract_full(archive, Path("/tmp/masriswitch-full-data"), args.arrow_sha256)
    ids = sorted(path.stem for path in wavs.glob("*.wav"))
    if hashlib.sha256("\n".join(ids).encode()).hexdigest() != archive_manifest["train_ids_sha256"]:
        raise ValueError("E1 train IDs changed")
    stage, model = _stage_code(args.root, expected_patch)
    frames = int(selected["frames_per_gpu"])
    duration = sum(Dataset.from_file(str(dataset / "raw.arrow"))["duration"])
    plan = plan_training(duration, frames, target_updates=8000)
    checkpoints = Path("/kaggle/working/masriswitch_e1_checkpoints")
    checkpoints.mkdir(parents=True, exist_ok=True)
    pretrained = checkpoints / "pretrained_model.pt"
    if not pretrained.exists():
        try:
            os.link(os.path.realpath(model), pretrained)
        except OSError:
            shutil.copy2(model, pretrained)
    if _sha256(pretrained) != SILMA_SHA:
        raise ValueError("Staged pretrained checkpoint hash changed")
    previous = checkpoints / "model_last.pt"
    if args.resume_checkpoint_url:
        if not args.resume or not args.resume_checkpoint_sha256 or previous.exists():
            raise ValueError("Resume checkpoint arguments conflict")
        _download_archive(args.resume_checkpoint_url, previous, args.resume_checkpoint_sha256)
    if previous.exists() and not args.resume:
        raise ValueError("Checkpoint exists; pass --resume")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(stage / "src") + os.pathsep + str(stage.parents[2] / "src")
    env["MASRISWITCH_DATA_DIR"] = str(dataset)
    env["MASRISWITCH_AUDIO_ROOT"] = str(wavs)
    env["MASRISWITCH_CHECKPOINT_DIR"] = str(checkpoints)
    env["MASRISWITCH_MAX_UPDATES"] = str(args.max_updates)
    env["MASRISWITCH_SEED"] = str(args.seed)
    env["MASRISWITCH_PROBE_LOG"] = "/kaggle/working/masriswitch_e1_update"
    env["MASRISWITCH_CHECKPOINT_ACTIVATIONS"] = "1" if selected["checkpoint_activations"] else "0"
    env["WANDB_DISABLED"] = "true"
    command = [
        str(Path(sys.executable).with_name("accelerate")),
        "launch",
        "--multi_gpu",
        "--num_processes",
        "2",
        "--mixed_precision",
        "fp16",
        str(stage / "src" / "f5_tts" / "train" / "finetune_cli.py"),
        "--exp_name",
        "F5TTS_v1_Base",
        "--dataset_name",
        "masriswitch_e1",
        "--learning_rate",
        "1e-5",
        "--batch_size_per_gpu",
        str(frames),
        "--batch_size_type",
        "frame",
        "--max_samples",
        "64",
        "--grad_accumulation_steps",
        "1",
        "--max_grad_norm",
        "1.0",
        "--epochs",
        str(plan.epochs + 1),
        "--num_warmup_updates",
        "50",
        "--save_per_updates",
        "500",
        "--last_per_updates",
        "500",
        "--keep_last_n_checkpoints",
        "0",
        "--finetune",
        "--pretrain",
        str(model),
        "--tokenizer",
        "custom",
        "--tokenizer_path",
        str(dataset / "vocab.txt"),
    ]
    if selected["optimizer_8bit"]:
        command.append("--bnb_optimizer")
    log = Path("/kaggle/working/masriswitch_e1.log")
    started = time.perf_counter()
    import threading

    finished = threading.Event()

    def report_progress() -> None:
        while not finished.wait(60):
            rank_updates: list[str] = []
            for rank in (0, 1):
                path = Path(f"/kaggle/working/masriswitch_e1_update_{rank}.csv")
                lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
                rank_updates.append(lines[-1].split(",")[0] if lines else "waiting")
            print("E1_TRAINING_PROGRESS", *rank_updates, flush=True)

    watcher = threading.Thread(target=report_progress, daemon=True)
    watcher.start()
    try:
        with log.open("w", encoding="utf-8") as stream:
            process = subprocess.run(
                command, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=21600
            )
    finally:
        finished.set()
        watcher.join(timeout=1)
    if process.returncode:
        tail = log.read_text(encoding="utf-8", errors="replace")[-2000:]
        raise RuntimeError(f"E1 stopped at update <= {args.max_updates}: {tail}")
    rank_logs = [Path(f"/kaggle/working/masriswitch_e1_update_{rank}.csv") for rank in (0, 1)]
    updates = []
    for path in rank_logs:
        values = [line.split(",") for line in path.read_text(encoding="utf-8").splitlines()]
        if not values or any(not math.isfinite(float(row[2])) for row in values):
            raise FloatingPointError("E1 loss was missing or non-finite")
        updates.append(int(values[-1][0]))
    if updates != [args.max_updates, args.max_updates] or not previous.is_file():
        raise RuntimeError("E1 checkpoint or exact update count missing")
    smoke = _smoke_checkpoint(previous, args.root)
    result = {
        "experiment": "E1",
        "complete": args.max_updates == 8000,
        "target_updates": 8000,
        "updates_per_rank": updates,
        "best_checkpoint": str(previous),
        "checkpoint_sha256": _sha256(previous),
        "training_source_ids": ["d1"],
        "training_sample_ids": ids,
        "full_archive_sha256": args.archive_sha256,
        "pilot_passed": True,
        "smoke_prompts_passed": smoke,
        "seed": args.seed,
        "elapsed_seconds": time.perf_counter() - started,
    }
    pretrained.unlink(missing_ok=True)
    Path("/kaggle/working/masriswitch_e1_result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print("MASRISWITCH_E1_CHECKPOINT_READY", args.max_updates, flush=True)


if __name__ == "__main__":
    main()
