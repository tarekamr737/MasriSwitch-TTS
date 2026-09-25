"""Private two-prompt SILMA inference smoke test; never publish its reference voice."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SILMA_REV = "226dd7a65cadf51f9a6dbe3953fc89003b3844d5"
VOCOS_REV = "0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21"
SILMA_SHA = "f43256d0b78b8803c638aed0875da5a4b372b4a784690a0156e5baff14f7336c"
VOCOS_SHA = "97ec976ad1fd67a33ab2682d29c0ac7df85234fae875aefcc5fb215681a91b2a"
REFERENCE_SHA = "b6b88232c3b851a3d9833c66349c71b2527de346b467ccba944809bad776c7ae"
REFERENCE_COMMIT = "96ec4beedb0766fcbddbf8b49b696c4049574ed6"
REFERENCE_TEXT = (
    "ويدقق النظر في القرآن الكريم وسائر الكتب السماوية "
    "ويتبع مسالك الرسل العظام عليهم الصلاة والسلام."
)
PROMPTS = ("حضرتك تقدر تراجع الباقة دلوقتي.", "يا فندم، Visa card جاهزة دلوقتي.")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _run(*args: str) -> None:
    subprocess.run(args, check=True)


def _inference() -> None:
    import numpy as np
    import requests
    import torch
    import yaml
    from f5_tts.infer.utils_infer import infer_process, load_model, load_vocoder
    from f5_tts.model import DiT
    from huggingface_hub import hf_hub_download

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA T4 is required for this smoke test")
    root = Path("/tmp/masriswitch-e0-smoke")
    root.mkdir(parents=True, exist_ok=True)
    silma_dir = root / "silma"
    vocos_dir = root / "vocos"
    for directory in (silma_dir, vocos_dir):
        directory.mkdir(exist_ok=True)
    files = (
        ("silma-ai/silma-tts", SILMA_REV, "model.pt", silma_dir, SILMA_SHA),
        ("charactr/vocos-mel-24khz", VOCOS_REV, "pytorch_model.bin", vocos_dir, VOCOS_SHA),
    )
    for repo, revision, filename, directory, expected in files:
        path = Path(hf_hub_download(repo, filename, revision=revision, local_dir=directory))
        if _sha256(path) != expected:
            raise ValueError(f"Pinned {filename} hash mismatch")
    for repo, revision, filename, directory in (
        ("silma-ai/silma-tts", SILMA_REV, "config.yaml", silma_dir),
        ("silma-ai/silma-tts", SILMA_REV, "vocab.txt", silma_dir),
        ("charactr/vocos-mel-24khz", VOCOS_REV, "config.yaml", vocos_dir),
    ):
        hf_hub_download(repo, filename, revision=revision, local_dir=directory)
    url = (
        "https://raw.githubusercontent.com/SILMA-AI/silma-tts/"
        f"{REFERENCE_COMMIT}/src/silma_tts/infer/ref_audio_samples/ar.ref.24k.wav"
    )
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    reference = root / "private_reference.wav"
    reference.write_bytes(response.content)
    if _sha256(reference) != REFERENCE_SHA:
        raise ValueError("Private reference hash mismatch")
    config = yaml.safe_load((silma_dir / "config.yaml").read_text())
    start = time.perf_counter()
    vocoder = load_vocoder("vocos", is_local=True, local_path=str(vocos_dir))
    model = load_model(
        DiT,
        config["model"]["arch"],
        str(silma_dir / "model.pt"),
        mel_spec_type="vocos",
        vocab_file=str(silma_dir / "vocab.txt"),
        use_ema=True,
    )
    load_seconds = time.perf_counter() - start
    outputs = []
    for prompt in PROMPTS:
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        audio, sample_rate, _ = infer_process(
            str(reference),
            REFERENCE_TEXT,
            prompt,
            model,
            vocoder,
            nfe_step=8,
            cross_fade_duration=0,
        )
        latency = time.perf_counter() - start
        samples = np.asarray(audio)
        if samples.size == 0 or not np.isfinite(samples).all():
            raise RuntimeError("Invalid smoke-test audio")
        outputs.append(
            {
                "prompt": prompt,
                "sample_rate": int(sample_rate),
                "duration_seconds": len(samples) / sample_rate,
                "latency_seconds": latency,
                "peak_vram_bytes": torch.cuda.max_memory_allocated(),
            }
        )
    result = {
        "private_reference_only": True,
        "model_sha256": SILMA_SHA,
        "reference_sha256": REFERENCE_SHA,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "load_seconds": load_seconds,
        "nfe_steps": 8,
        "outputs": outputs,
    }
    Path("/kaggle/working/masriswitch_e0_smoke.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False)
    )
    print("E0_SMOKE_PASSED", json.dumps(result, ensure_ascii=False), flush=True)
    reference.unlink()


def main() -> None:
    if sys.version_info[:2] != (3, 10):
        _run("uv", "python", "install", "3.10")
        venv = Path("/tmp/masriswitch-f5-venv310")
        _run("uv", "venv", "--python", "3.10", str(venv))
        py = str(venv / "bin" / "python")
        _run(
            "uv",
            "pip",
            "install",
            "--python",
            py,
            "f5-tts==1.1.7",
            "torch==2.6.0",
            "torchaudio==2.6.0",
            "transformers==4.46.3",
            "huggingface-hub==0.35.3",
            "datasets==3.6.0",
        )
        os.execv(py, [py, __file__])
    _inference()


if __name__ == "__main__":
    main()
