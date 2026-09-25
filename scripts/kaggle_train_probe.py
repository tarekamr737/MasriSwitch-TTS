"""Bounded 20-update two-T4 training probe with automatic memory fallback."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

F5_COMMIT = "c96c3aeed84f5e02aa54dc42c1193537ead39837"
SILMA_REV = "226dd7a65cadf51f9a6dbe3953fc89003b3844d5"
SILMA_SHA = "f43256d0b78b8803c638aed0875da5a4b372b4a784690a0156e5baff14f7336c"
SILMA_CLI_SHA = "62020a4c77fb6a8375c6afa207264d0a57f671012c73e5c1e7c19bbcc87afac4"
SOURCE_REV = "eae9a87c17e91e3f59a9696d5f4ff3eb51502e82"
FRAMES = (5600, 4800, 4000, 3200)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _install_python310(root: Path) -> None:
    subprocess.run(["uv", "python", "install", "3.10"], check=True)
    venv = Path("/tmp/masriswitch-train-venv310")
    subprocess.run(["uv", "venv", "--python", "3.10", str(venv)], check=True)
    py = str(venv / "bin" / "python")
    subprocess.run(
        ["uv", "pip", "install", "--python", py, "-e", f"{root}[data,train]"], check=True
    )
    os.execv(py, [py, __file__, *sys.argv[1:]])


def _download_archive(url: str, destination: Path, expected_sha: str) -> None:
    import time

    import requests

    if not url.startswith("https://www.kaggleusercontent.com/"):
        raise ValueError("Only signed private Kaggle output URLs are accepted")
    destination.parent.mkdir(parents=True, exist_ok=True)
    started = last_report = time.monotonic()
    transferred = 0
    print("PRIVATE_ARCHIVE_DOWNLOAD_START", destination.name, flush=True)
    with requests.get(url, stream=True, timeout=(30, 180)) as response:
        response.raise_for_status()
        with destination.open("wb") as stream:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    stream.write(chunk)
                    transferred += len(chunk)
                    now = time.monotonic()
                    if now - last_report >= 30:
                        print("PRIVATE_ARCHIVE_DOWNLOAD_BYTES", transferred, flush=True)
                        last_report = now
    if _sha256(destination) != expected_sha:
        raise ValueError("Private archive hash mismatch")
    print(
        "PRIVATE_ARCHIVE_DOWNLOAD_VERIFIED",
        destination.name,
        transferred,
        int(time.monotonic() - started),
        flush=True,
    )


def _extract_archive(archive_path: Path, output: Path, arrow_sha: str) -> tuple[Path, Path]:
    output.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        files = archive.getmembers()
        if len(files) != 346:
            raise ValueError("Pilot archive must have 343 WAVs and 3 metadata files")
        for item in files:
            path = Path(item.name)
            if not item.isfile() or path.is_absolute() or ".." in path.parts:
                raise ValueError("Unsafe pilot archive member")
            if path.parts[0] not in {"pilot_char", "wavs"}:
                raise ValueError("Unexpected pilot archive path")
        archive.extractall(output, filter="data")
    pilot = output / "pilot_char"
    wavs = output / "wavs"
    if _sha256(pilot / "raw.arrow") != arrow_sha or len(list(wavs.glob("*.wav"))) != 343:
        raise ValueError("Pilot Arrow hash or audio count mismatch")
    return pilot, wavs


def _stage_code(root: Path, expected_patch: dict[str, str]) -> tuple[Path, Path]:
    import signal
    import threading
    import time

    from huggingface_hub import hf_hub_download

    from masriswitch.config import Paths
    from masriswitch.train.patch_f5 import stage_training_code

    upstream = root / "artifacts" / "upstream"
    upstream.mkdir(parents=True, exist_ok=True)
    f5 = upstream / "f5-tts"
    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            "1.1.7",
            "https://github.com/SWivid/F5-TTS.git",
            str(f5),
        ],
        check=True,
    )
    actual_commit = subprocess.check_output(["git", "-C", str(f5), "rev-parse", "HEAD"], text=True)
    if actual_commit.strip() != F5_COMMIT:
        raise ValueError("F5 1.1.7 commit changed")
    silma = upstream / "silma"
    silma.mkdir(exist_ok=True)
    cli = Path(
        hf_hub_download(
            "silma-ai/silma-tts", "finetune_cli.py", revision=SILMA_REV, local_dir=silma
        )
    )
    if _sha256(cli) != SILMA_CLI_SHA:
        raise ValueError("SILMA training CLI hash changed")
    print("SILMA_MODEL_DOWNLOAD_START", flush=True)
    started = time.monotonic()
    finished = threading.Event()

    def heartbeat() -> None:
        while not finished.wait(30):
            print("SILMA_MODEL_DOWNLOAD_WAIT", int(time.monotonic() - started), flush=True)

    watcher = threading.Thread(target=heartbeat, daemon=True)
    watcher.start()
    previous_handler = signal.getsignal(signal.SIGALRM)

    def download_timeout(_signum: int, _frame: object) -> None:
        raise TimeoutError("SILMA model download exceeded 20 minutes")

    signal.signal(signal.SIGALRM, download_timeout)
    signal.alarm(1200)
    try:
        model = Path(hf_hub_download("silma-ai/silma-tts", "model.pt", revision=SILMA_REV))
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
        finished.set()
        watcher.join(timeout=1)
    print("SILMA_MODEL_DOWNLOAD_COMPLETE", int(time.monotonic() - started), flush=True)
    if _sha256(model) != SILMA_SHA:
        raise ValueError("SILMA checkpoint hash changed")
    actual_patch = stage_training_code(Paths(root))
    if actual_patch != expected_patch:
        raise ValueError("Staged F5 patch hashes differ from reviewed local patch")
    return root / "artifacts" / "staged" / "f5-tts", model


def _probe_once(
    stage: Path,
    model: Path,
    pilot: Path,
    wavs: Path,
    frames: int,
    checkpointing: bool,
    optimizer_8bit: bool,
    seed: int,
    attempt: int,
) -> tuple[bool, float, dict[str, Any]]:
    root = Path("/kaggle/working")
    log_prefix = root / f"masriswitch_probe_{attempt}"
    process_log = root / f"masriswitch_probe_attempt_{attempt}.log"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(stage / "src") + os.pathsep + str(stage.parents[2] / "src")
    env["MASRISWITCH_DATA_DIR"] = str(pilot)
    env["MASRISWITCH_AUDIO_ROOT"] = str(wavs)
    env["MASRISWITCH_CHECKPOINT_DIR"] = "/tmp/masriswitch-probe-checkpoints"
    env["MASRISWITCH_MAX_UPDATES"] = "20"
    env["MASRISWITCH_NO_FINAL_SAVE"] = "1"
    env["MASRISWITCH_SEED"] = str(seed)
    env["MASRISWITCH_PROBE_LOG"] = str(log_prefix)
    env["MASRISWITCH_CHECKPOINT_ACTIVATIONS"] = "1" if checkpointing else "0"
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
        "masriswitch_pilot",
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
        "3",
        "--num_warmup_updates",
        "50",
        "--save_per_updates",
        "100000",
        "--last_per_updates",
        "100000",
        "--keep_last_n_checkpoints",
        "0",
        "--finetune",
        "--pretrain",
        str(model),
        "--tokenizer",
        "custom",
        "--tokenizer_path",
        str(pilot / "vocab.txt"),
    ]
    if optimizer_8bit:
        command.append("--bnb_optimizer")
    start = time.perf_counter()
    print(
        "PROBE_ATTEMPT",
        json.dumps(
            {"frames": frames, "checkpointing": checkpointing, "optimizer_8bit": optimizer_8bit}
        ),
        flush=True,
    )
    with process_log.open("w", encoding="utf-8") as stream:
        try:
            completed = subprocess.run(
                command, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=1200
            )
            exit_code = completed.returncode
        except subprocess.TimeoutExpired:
            exit_code = 124
    elapsed = time.perf_counter() - start
    tail = process_log.read_text(encoding="utf-8", errors="replace")[-5000:]
    memory_logs = [root / f"masriswitch_probe_{attempt}_{rank}.csv" for rank in (0, 1)]
    rows: list[list[tuple[int, int, float]]] = []
    for path in memory_logs:
        if not path.is_file():
            rows.append([])
            continue
        rows.append(
            [
                (int(update), int(free), float(loss))
                for line in path.read_text(encoding="utf-8").splitlines()
                for update, free, loss in [line.split(",")]
            ]
        )
    if "NaN or Inf training loss" in tail:
        raise FloatingPointError("Pilot probe produced NaN or Inf loss")
    oom = "out of memory" in tail.lower() or "cuda error" in tail.lower()
    if exit_code and not oom:
        raise RuntimeError(f"Non-memory training failure (exit {exit_code}): {tail[-1800:]}")
    complete = exit_code == 0 and all(len(rank_rows) >= 20 for rank_rows in rows)
    finite = all(math.isfinite(loss) for rank_rows in rows for _, _, loss in rank_rows)
    free_gb = min((free for rank_rows in rows for _, free, _ in rank_rows), default=0) / 2**30
    result = {
        "frames_per_gpu": frames,
        "checkpoint_activations": checkpointing,
        "optimizer_8bit": optimizer_8bit,
        "exit_code": exit_code,
        "oom": oom,
        "updates_per_rank": [len(rank_rows) for rank_rows in rows],
        "minimum_free_gb": free_gb,
        "finite_loss": finite,
        "elapsed_seconds": elapsed,
        "log_tail": tail[-600:],
    }
    print("PROBE_RESULT", json.dumps(result), flush=True)
    return complete and finite, free_gb, result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--archive-url", required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--arrow-sha256", required=True)
    parser.add_argument("--archive-manifest", type=Path, required=True)
    parser.add_argument("--patch-manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=343)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.max_samples != 343:
        raise ValueError("Training probe requires the locked 343-row pilot subset")
    if args.dry_run:
        print(json.dumps({"pilot_rows": 343, "updates_per_attempt": 20, "frames": FRAMES}))
        return
    result_path = Path("/kaggle/working/masriswitch_probe_result.json")
    if args.resume and result_path.is_file():
        previous = json.loads(result_path.read_text(encoding="utf-8"))
        if previous.get("pilot_archive_sha256") != args.archive_sha256:
            raise ValueError("Resume archive hash changed")
        if previous.get("passed") is True:
            print("MASRISWITCH_PROBE_RESUMED", json.dumps(previous["selected"]), flush=True)
            return
    if sys.version_info[:2] != (3, 10):
        _install_python310(args.root)
    import torch

    if torch.cuda.device_count() != 2 or any(
        "T4" not in torch.cuda.get_device_name(i) for i in (0, 1)
    ):
        raise RuntimeError("Two T4 GPUs are required")
    archive = Path("/tmp/masriswitch-pilot-data.tar.gz")
    manifest = json.loads(args.archive_manifest.read_text(encoding="utf-8"))
    if (
        manifest.get("source_revision") != SOURCE_REV
        or manifest.get("train_only") is not True
        or manifest.get("pilot_rows") != 343
        or manifest.get("archive_sha256") != args.archive_sha256
        or manifest.get("pilot_arrow_sha256") != args.arrow_sha256
    ):
        raise ValueError("Pilot archive manifest is incomplete or changed")
    _download_archive(args.archive_url, archive, args.archive_sha256)
    pilot, wavs = _extract_archive(archive, Path("/tmp/masriswitch-pilot-data"), args.arrow_sha256)
    actual_ids_hash = hashlib.sha256(
        "\n".join(sorted(path.stem for path in wavs.glob("*.wav"))).encode()
    ).hexdigest()
    if actual_ids_hash != manifest.get("pilot_ids_sha256"):
        raise ValueError("Pilot train sample IDs changed")
    expected_patch = json.loads(args.patch_manifest.read_text(encoding="utf-8"))
    stage, model = _stage_code(args.root, expected_patch)
    attempts: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    for checkpointing, optimizer_8bit in ((False, False), (True, False), (True, True)):
        for frames in FRAMES:
            stable, free_gb, result = _probe_once(
                stage,
                model,
                pilot,
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
        "pilot_archive_sha256": args.archive_sha256,
        "pilot_arrow_sha256": args.arrow_sha256,
        "patch_hashes": expected_patch,
        "seed": args.seed,
        "target_updates_per_attempt": 20,
        "attempts": attempts,
        "selected": selected,
        "passed": selected is not None,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
    }
    result_path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    if not selected:
        raise RuntimeError("No frame batch passed 20 updates with at least 1 GiB free")
    print("MASRISWITCH_PROBE_PASSED", json.dumps(selected), flush=True)


if __name__ == "__main__":
    main()
