"""Resume-safe 500-update Kaggle pilot, gated by a measured two-GPU probe."""

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

from kaggle_train_probe import (
    SILMA_REV,
    SILMA_SHA,
    SOURCE_REV,
    _download_archive,
    _extract_archive,
    _sha256,
    _stage_code,
)

VOCOS_REV = "0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21"
VOCOS_SHA = "97ec976ad1fd67a33ab2682d29c0ac7df85234fae875aefcc5fb215681a91b2a"
SILMA_CONFIG_SHA = "6c748bacc3ee115d4e61453775fde0c2f12f2828bcc409c5dbbbae4a7f88bc1a"
SILMA_VOCAB_SHA = "5c2ffc48802a52bbdf715dacf1d6519d3fee96e391aef690261963a692b8e661"
REFERENCE_SHA = "b6b88232c3b851a3d9833c66349c71b2527de346b467ccba944809bad776c7ae"
REFERENCE_COMMIT = "96ec4beedb0766fcbddbf8b49b696c4049574ed6"
REFERENCE_TEXT = (
    "ويدقق النظر في القرآن الكريم وسائر الكتب السماوية "
    "ويتبع مسالك الرسل العظام عليهم الصلاة والسلام."
)
SMOKE_TEXTS = (
    "ممكن تبعتلي OTP على الموبايل؟",
    "الـ Premium Plan بتاعك هيتجدد بكرة.",
    "حضرتك هتلاقي تفاصيل Online Banking في الحساب.",
    "طلبك من Amazon اتأكد خلاص.",
    "خدمة Wi-Fi شغالة دلوقتي.",
    "ممكن تراجع رقم الطلب ORD12345؟",
    "عندنا ميعاد الساعة 4:30 PM.",
    "الرصيد الحالي 499 EGP.",
    "حجز Hilton متاح يوم الخميس.",
    "شكراً لاتصالك، يومك سعيد.",
)


def _install_python310(root: Path, entrypoint: Path | None = None) -> None:
    subprocess.run(["uv", "python", "install", "3.10"], check=True)
    venv = Path("/tmp/masriswitch-train-venv310")
    subprocess.run(["uv", "venv", "--python", "3.10", str(venv)], check=True)
    py = str(venv / "bin" / "python")
    subprocess.run(
        ["uv", "pip", "install", "--python", py, "-e", f"{root}[data,train]"], check=True
    )
    os.execv(py, [py, str(entrypoint or Path(__file__)), *sys.argv[1:]])


def _smoke_checkpoint(checkpoint: Path, root: Path) -> int:
    """Reload saved EMA bytes and synthesize 10 private, finite 24 kHz samples."""
    import numpy as np
    import requests
    import torch
    import yaml
    from f5_tts.infer.utils_infer import infer_process, load_model, load_vocoder
    from f5_tts.model import DiT
    from huggingface_hub import hf_hub_download

    silma = root / "artifacts" / "upstream" / "silma"
    config_file = Path(
        hf_hub_download("silma-ai/silma-tts", "config.yaml", revision=SILMA_REV, local_dir=silma)
    )
    if _sha256(config_file) != SILMA_CONFIG_SHA or _sha256(silma / "vocab.txt") != SILMA_VOCAB_SHA:
        raise ValueError("SILMA config or vocabulary hash changed")
    vocos = Path("/tmp/masriswitch-pilot-vocos")
    vocos.mkdir(parents=True, exist_ok=True)
    model_file = Path(
        hf_hub_download(
            "charactr/vocos-mel-24khz",
            "pytorch_model.bin",
            revision=VOCOS_REV,
            local_dir=vocos,
        )
    )
    hf_hub_download("charactr/vocos-mel-24khz", "config.yaml", revision=VOCOS_REV, local_dir=vocos)
    if _sha256(model_file) != VOCOS_SHA:
        raise ValueError("Vocos hash changed")
    url = (
        "https://raw.githubusercontent.com/SILMA-AI/silma-tts/"
        f"{REFERENCE_COMMIT}/src/silma_tts/infer/ref_audio_samples/ar.ref.24k.wav"
    )
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    reference = Path("/tmp/masriswitch-private-reference.wav")
    reference.write_bytes(response.content)
    if _sha256(reference) != REFERENCE_SHA:
        raise ValueError("Private reference hash changed")
    torch.cuda.set_device(0)
    config = yaml.safe_load((silma / "config.yaml").read_text(encoding="utf-8"))
    vocoder = load_vocoder("vocos", is_local=True, local_path=str(vocos), device="cuda:0")
    model = load_model(
        DiT,
        config["model"]["arch"],
        str(checkpoint),
        mel_spec_type="vocos",
        vocab_file=str(silma / "vocab.txt"),
        use_ema=True,
        device="cuda:0",
    )
    for index, text in enumerate(SMOKE_TEXTS):
        torch.manual_seed(42)
        audio, sample_rate, _ = infer_process(
            str(reference),
            REFERENCE_TEXT,
            text,
            model,
            vocoder,
            nfe_step=16,
            cross_fade_duration=0,
            device="cuda:0",
        )
        wave = np.asarray(audio)
        if sample_rate != 24000 or wave.size == 0 or not np.isfinite(wave).all():
            raise RuntimeError(f"Pilot checkpoint smoke failed at prompt {index}")
    reference.unlink(missing_ok=True)
    return len(SMOKE_TEXTS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--archive-url", required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--arrow-sha256", required=True)
    parser.add_argument("--archive-manifest", type=Path, required=True)
    parser.add_argument("--patch-manifest", type=Path, required=True)
    parser.add_argument("--probe-result", type=Path, required=True)
    parser.add_argument("--resume-checkpoint-url")
    parser.add_argument("--resume-checkpoint-sha256")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=343)
    parser.add_argument("--max-updates", type=int, default=500)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.max_samples != 343 or not 1 <= args.max_updates <= 500:
        raise ValueError("Pilot may use only 343 train rows and at most 500 updates")
    probe = json.loads(args.probe_result.read_text(encoding="utf-8"))
    selected = probe.get("selected")
    if (
        probe.get("passed") is not True
        or not isinstance(selected, dict)
        or selected.get("updates_per_rank") != [20, 20]
        or selected.get("finite_loss") is not True
        or selected.get("minimum_free_gb", 0) < 1.0
        or probe.get("pilot_archive_sha256") != args.archive_sha256
        or probe.get("pilot_arrow_sha256") != args.arrow_sha256
        or probe.get("silma_sha256") != SILMA_SHA
        or probe.get("source_revision") != SOURCE_REV
    ):
        raise ValueError("A passing pinned 20-update probe is required")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "updates": args.max_updates,
                    "rows": 343,
                    "frames_per_gpu": selected["frames_per_gpu"],
                    "minimum_free_gb": selected["minimum_free_gb"],
                }
            )
        )
        return
    if sys.version_info[:2] != (3, 10):
        _install_python310(args.root)
    import torch
    from datasets import Dataset

    from masriswitch.train.plan import plan_training

    if torch.cuda.device_count() != 2 or any(
        "T4" not in torch.cuda.get_device_name(i) for i in (0, 1)
    ):
        raise RuntimeError("Two T4 GPUs are required")
    manifest = json.loads(args.archive_manifest.read_text(encoding="utf-8"))
    if (
        manifest.get("source_revision") != SOURCE_REV
        or manifest.get("train_only") is not True
        or manifest.get("pilot_rows") != 343
        or manifest.get("archive_sha256") != args.archive_sha256
        or manifest.get("pilot_arrow_sha256") != args.arrow_sha256
    ):
        raise ValueError("Pilot archive manifest changed")
    archive = Path("/tmp/masriswitch-pilot-data.tar.gz")
    _download_archive(args.archive_url, archive, args.archive_sha256)
    pilot, wavs = _extract_archive(archive, Path("/tmp/masriswitch-pilot-data"), args.arrow_sha256)
    actual_ids_hash = hashlib.sha256(
        "\n".join(sorted(path.stem for path in wavs.glob("*.wav"))).encode()
    ).hexdigest()
    if actual_ids_hash != manifest["pilot_ids_sha256"]:
        raise ValueError("Pilot IDs changed")
    expected_patch = json.loads(args.patch_manifest.read_text(encoding="utf-8"))
    stage, model = _stage_code(args.root, expected_patch)
    frames = int(selected["frames_per_gpu"])
    duration = sum(Dataset.from_file(str(pilot / "raw.arrow"))["duration"])
    # Keep the scheduler/epoch budget fixed across the 250 -> 500 resume boundary.
    plan = plan_training(duration, frames, target_updates=500)
    checkpoints = Path("/kaggle/working/masriswitch_pilot_checkpoints")
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
            raise ValueError("Resume checkpoint arguments are incomplete or conflicting")
        _download_archive(args.resume_checkpoint_url, previous, args.resume_checkpoint_sha256)
    if previous.exists() and not args.resume:
        raise ValueError("Checkpoint exists; pass --resume")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(stage / "src") + os.pathsep + str(stage.parents[2] / "src")
    env["MASRISWITCH_DATA_DIR"] = str(pilot)
    env["MASRISWITCH_AUDIO_ROOT"] = str(wavs)
    env["MASRISWITCH_CHECKPOINT_DIR"] = str(checkpoints)
    env["MASRISWITCH_MAX_UPDATES"] = str(args.max_updates)
    env["MASRISWITCH_SEED"] = str(args.seed)
    env["MASRISWITCH_PROBE_LOG"] = "/kaggle/working/masriswitch_pilot_update"
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
        str(plan.epochs + 1),
        "--num_warmup_updates",
        "50",
        "--save_per_updates",
        "250",
        "--last_per_updates",
        "250",
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
    if selected["optimizer_8bit"]:
        command.append("--bnb_optimizer")
    log = Path("/kaggle/working/masriswitch_pilot.log")
    started = time.perf_counter()
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.run(
            command, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=7200
        )
    if process.returncode:
        tail = log.read_text(encoding="utf-8", errors="replace")[-2000:]
        raise RuntimeError(f"Pilot failed at update <= {args.max_updates}: {tail}")
    rank_logs = [Path(f"/kaggle/working/masriswitch_pilot_update_{rank}.csv") for rank in (0, 1)]
    updates = []
    for path in rank_logs:
        values = [line.split(",") for line in path.read_text(encoding="utf-8").splitlines()]
        if not values or any(not math.isfinite(float(row[2])) for row in values):
            raise FloatingPointError("Pilot loss was missing or non-finite")
        updates.append(int(values[-1][0]))
    if updates != [args.max_updates, args.max_updates] or not previous.is_file():
        raise RuntimeError("Pilot checkpoint or exact update count missing")
    result = {
        "experiment": "E1-pilot",
        "complete": args.max_updates == 500,
        "target_updates": 500,
        "updates_per_rank": updates,
        "checkpoint": str(previous),
        "checkpoint_sha256": _sha256(previous),
        "selected_probe": selected,
        "source_revision": SOURCE_REV,
        "pilot_archive_sha256": args.archive_sha256,
        "seed": args.seed,
        "elapsed_seconds": time.perf_counter() - started,
        "smoke_prompts_passed": _smoke_checkpoint(previous, args.root),
    }
    pretrained.unlink(missing_ok=True)
    Path("/kaggle/working/masriswitch_pilot_result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print("MASRISWITCH_PILOT_CHECKPOINT_READY", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
