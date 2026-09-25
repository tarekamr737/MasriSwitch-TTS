"""Private, resumable E0 generation and pinned independent ASR evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SILMA_REV = "226dd7a65cadf51f9a6dbe3953fc89003b3844d5"
VOCOS_REV = "0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21"
ASR_REV = "0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf"
SILMA_SHA = "f43256d0b78b8803c638aed0875da5a4b372b4a784690a0156e5baff14f7336c"
VOCOS_SHA = "97ec976ad1fd67a33ab2682d29c0ac7df85234fae875aefcc5fb215681a91b2a"
ASR_SHA = "e76620f83d5f5b69efd3d87e3dc180c1bd21df9fbebacfd4335e5e1efcc018da"
SPEAKER_REV = "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"
SPEAKER_SHA = "0575cb64845e6b9a10db9bcb74d5ac32b326b8dc90352671d345e2ee3d0126a2"
REFERENCE_SHA = "b6b88232c3b851a3d9833c66349c71b2527de346b467ccba944809bad776c7ae"
REFERENCE_COMMIT = "96ec4beedb0766fcbddbf8b49b696c4049574ed6"
REFERENCE_TEXT = (
    "ويدقق النظر في القرآن الكريم وسائر الكتب السماوية "
    "ويتبع مسالك الرسل العظام عليهم الصلاة والسلام."
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_plan(
    path: Path, expected_sha: str, max_samples: int, subset: str = "all"
) -> list[dict[str, Any]]:
    if _sha256(path) != expected_sha:
        raise ValueError("Evaluation plan hash changed")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != 489 or len({row["id"] for row in rows}) != 489:
        raise ValueError("Expected 189 validation plus 300 locked benchmark prompts")
    if subset == "validation":
        rows = [row for row in rows if row["set"] == "validation"]
    elif subset != "all":
        raise ValueError("Unknown evaluation subset")
    if max_samples < 1 or max_samples > len(rows):
        raise ValueError("max_samples outside evaluation plan")
    return rows[:max_samples]


def _install_python310() -> None:
    subprocess.run(["uv", "python", "install", "3.10"], check=True)
    venv = Path("/tmp/masriswitch-eval-venv310")
    subprocess.run(["uv", "venv", "--python", "3.10", str(venv)], check=True)
    py = str(venv / "bin" / "python")
    subprocess.run(
        [
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
            "faster-whisper==1.2.0",
            "speechbrain==1.0.3",
        ],
        check=True,
    )
    os.execv(py, [py, __file__, *sys.argv[1:]])


def _hf_file(repo: str, revision: str, filename: str, directory: Path, sha: str | None) -> Path:
    from huggingface_hub import hf_hub_download

    path = Path(hf_hub_download(repo, filename, revision=revision, local_dir=directory))
    if sha and _sha256(path) != sha:
        raise ValueError(f"Pinned {filename} hash mismatch")
    print("EVAL_MODEL_FILE_READY", repo, filename, flush=True)
    return path


def _load_models(
    root: Path,
    checkpoint_url: str | None,
    checkpoint_sha: str | None,
    checkpoint_path: Path | None,
) -> tuple[Any, Any, Any, Any, Any, Path]:
    import requests
    import torch
    import torchaudio
    import yaml
    from f5_tts.infer.utils_infer import load_model, load_vocoder
    from f5_tts.model import DiT
    from faster_whisper import WhisperModel
    from huggingface_hub import snapshot_download
    from speechbrain.inference.speaker import EncoderClassifier

    if torch.cuda.device_count() < 2:
        raise RuntimeError("Two T4 GPUs are required for the fixed evaluation protocol")
    silma = root / "silma"
    vocos = root / "vocos"
    asr = root / "asr"
    for directory in (silma, vocos, asr):
        directory.mkdir(parents=True, exist_ok=True)
    model_path = silma / "model.pt"
    if checkpoint_path:
        resolved = checkpoint_path.resolve()
        if not checkpoint_sha or not resolved.is_relative_to(Path("/kaggle/working")):
            raise ValueError("Pinned Kaggle E1 checkpoint path and hash are required")
        if not resolved.is_file() or _sha256(resolved) != checkpoint_sha:
            raise ValueError("Local E1 checkpoint hash changed")
        model_path = resolved
        print("E1_CHECKPOINT_HASH_VERIFIED", resolved.stat().st_size, flush=True)
    elif checkpoint_url:
        if not checkpoint_sha or not checkpoint_url.startswith(
            "https://www.kaggleusercontent.com/"
        ):
            raise ValueError("Pinned private E1 checkpoint URL and hash are required")
        downloaded = 0
        started = time.monotonic()
        with requests.get(checkpoint_url, stream=True, timeout=(30, 180)) as response:
            response.raise_for_status()
            with model_path.open("wb") as stream:
                for chunk in response.iter_content(1024 * 1024):
                    if chunk:
                        stream.write(chunk)
                        downloaded += len(chunk)
                        if downloaded // (256 * 1024 * 1024) != (downloaded - len(chunk)) // (
                            256 * 1024 * 1024
                        ):
                            print(
                                "E1_CHECKPOINT_DOWNLOADED_MB",
                                downloaded // (1024 * 1024),
                                flush=True,
                            )
                        if time.monotonic() - started > 1200:
                            raise TimeoutError("Private checkpoint download exceeded 20 minutes")
        if _sha256(model_path) != checkpoint_sha:
            raise ValueError("E1 checkpoint hash changed")
        print("E1_CHECKPOINT_HASH_VERIFIED", downloaded, flush=True)
    else:
        _hf_file("silma-ai/silma-tts", SILMA_REV, "model.pt", silma, SILMA_SHA)
    _hf_file("charactr/vocos-mel-24khz", VOCOS_REV, "pytorch_model.bin", vocos, VOCOS_SHA)
    _hf_file("dropbox-dash/faster-whisper-large-v3-turbo", ASR_REV, "model.bin", asr, ASR_SHA)
    for repo, revision, filename, directory in (
        ("silma-ai/silma-tts", SILMA_REV, "config.yaml", silma),
        ("silma-ai/silma-tts", SILMA_REV, "vocab.txt", silma),
        ("charactr/vocos-mel-24khz", VOCOS_REV, "config.yaml", vocos),
        ("dropbox-dash/faster-whisper-large-v3-turbo", ASR_REV, "config.json", asr),
        ("dropbox-dash/faster-whisper-large-v3-turbo", ASR_REV, "tokenizer.json", asr),
        ("dropbox-dash/faster-whisper-large-v3-turbo", ASR_REV, "vocabulary.json", asr),
        ("dropbox-dash/faster-whisper-large-v3-turbo", ASR_REV, "preprocessor_config.json", asr),
    ):
        _hf_file(repo, revision, filename, directory, None)
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
    print("EVAL_REFERENCE_HASH_VERIFIED", flush=True)
    config = yaml.safe_load((silma / "config.yaml").read_text(encoding="utf-8"))
    vocoder = load_vocoder("vocos", is_local=True, local_path=str(vocos), device="cuda:0")
    model = load_model(
        DiT,
        config["model"]["arch"],
        str(model_path),
        mel_spec_type="vocos",
        vocab_file=str(silma / "vocab.txt"),
        use_ema=True,
        device="cuda:0",
    )
    whisper = WhisperModel(str(asr), device="cuda", device_index=1, compute_type="float16")
    print("EVAL_TTS_ASR_READY", flush=True)
    torch.cuda.set_device(0)
    speaker_dir = Path(
        snapshot_download(
            "speechbrain/spkrec-ecapa-voxceleb",
            revision=SPEAKER_REV,
            local_dir=root / "speaker",
            allow_patterns=[
                "embedding_model.ckpt",
                "classifier.ckpt",
                "mean_var_norm_emb.ckpt",
                "hyperparams.yaml",
                "label_encoder.txt",
            ],
        )
    )
    if _sha256(speaker_dir / "embedding_model.ckpt") != SPEAKER_SHA:
        raise ValueError("Speaker embedding model hash mismatch")
    print("EVAL_SPEAKER_HASH_VERIFIED", flush=True)
    speaker = EncoderClassifier.from_hparams(
        source=str(speaker_dir), savedir=str(root / "speaker_saved"), run_opts={"device": "cpu"}
    )
    ref_wave, ref_rate = torchaudio.load(reference)
    ref_16k = torchaudio.functional.resample(ref_wave.mean(dim=0), ref_rate, 16000)
    with torch.no_grad():
        reference_embedding = speaker.encode_batch(ref_16k.unsqueeze(0)).flatten()
    return model, vocoder, whisper, speaker, reference_embedding, reference


def _evaluate(
    rows: list[dict[str, Any]],
    resume: bool,
    seed: int,
    stage: str,
    subset: str,
    checkpoint_url: str | None,
    checkpoint_sha: str | None,
    checkpoint_path: Path | None,
) -> None:
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio
    from f5_tts.infer.utils_infer import infer_process

    root = Path(f"/tmp/masriswitch-{stage.lower()}-eval")
    root.mkdir(parents=True, exist_ok=True)
    output = Path(f"/kaggle/working/{stage.lower()}_{subset}_eval_rows.jsonl")
    existing: dict[str, dict[str, Any]] = {}
    if output.exists():
        if not resume:
            raise ValueError("Evaluation output exists; pass --resume")
        for line in output.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            if item["id"] in existing or item.get("model_sha256") != (checkpoint_sha or SILMA_SHA):
                raise ValueError("Resume rows are duplicate or use a different checkpoint")
            existing[item["id"]] = item
        if not set(existing).issubset({row["id"] for row in rows}):
            raise ValueError("Resume output contains unexpected prompt IDs")
    model, vocoder, whisper, speaker, reference_embedding, reference = _load_models(
        root, checkpoint_url, checkpoint_sha, checkpoint_path
    )
    generated = root / "generated.wav"
    try:
        with output.open("a", encoding="utf-8", newline="\n") as stream:
            for index, row in enumerate(rows, 1):
                if row["id"] in existing:
                    continue
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.reset_peak_memory_stats(0)
                start = time.perf_counter()
                audio, sr, _ = infer_process(
                    str(reference),
                    REFERENCE_TEXT,
                    row["text"],
                    model,
                    vocoder,
                    nfe_step=16,
                    cross_fade_duration=0,
                    device="cuda:0",
                )
                synth_latency = time.perf_counter() - start
                wave = np.asarray(audio, dtype=np.float32)
                if sr != 24000 or wave.size == 0 or not np.isfinite(wave).all():
                    raise RuntimeError(f"Invalid audio for {row['id']}")
                sf.write(generated, wave, sr, subtype="PCM_16")
                start = time.perf_counter()
                segments, info = whisper.transcribe(
                    str(generated),
                    language="ar",
                    task="transcribe",
                    beam_size=5,
                    vad_filter=False,
                    condition_on_previous_text=False,
                )
                hypothesis = " ".join(segment.text.strip() for segment in segments).strip()
                english_segments, _ = whisper.transcribe(
                    str(generated),
                    language="en",
                    task="transcribe",
                    beam_size=5,
                    vad_filter=False,
                    condition_on_previous_text=False,
                )
                english_hypothesis = " ".join(
                    segment.text.strip() for segment in english_segments
                ).strip()
                asr_latency = time.perf_counter() - start
                start = time.perf_counter()
                wave_16k = torchaudio.functional.resample(torch.from_numpy(wave), sr, 16000)
                with torch.no_grad():
                    generated_embedding = speaker.encode_batch(wave_16k.unsqueeze(0)).flatten()
                similarity = float(
                    torch.nn.functional.cosine_similarity(
                        reference_embedding, generated_embedding, dim=0
                    ).item()
                )
                if not np.isfinite(similarity):
                    raise RuntimeError(f"Invalid speaker similarity for {row['id']}")
                speaker_latency = time.perf_counter() - start
                item = {
                    "id": row["id"],
                    "set": row["set"],
                    "domain": row["domain"],
                    "bucket": row["bucket"],
                    "reference_text": row["text"],
                    "entities": row["entities"],
                    "hypothesis": hypothesis,
                    "english_decoder_hypothesis": english_hypothesis,
                    "asr_language": info.language,
                    "sample_rate": sr,
                    "audio_seconds": wave.size / sr,
                    "synthesis_latency_seconds": synth_latency,
                    "asr_latency_seconds": asr_latency,
                    "speaker_similarity": similarity,
                    "speaker_latency_seconds": speaker_latency,
                    "peak_tts_vram_bytes": torch.cuda.max_memory_allocated(0),
                    "model_sha256": checkpoint_sha or SILMA_SHA,
                    "asr_sha256": ASR_SHA,
                    "speaker_model_sha256": SPEAKER_SHA,
                    "seed": seed,
                    "nfe_steps": 16,
                }
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")
                stream.flush()
                if index % 10 == 0 or index == len(rows):
                    print(f"{stage}_EVAL_PROGRESS", index, len(rows), flush=True)
    finally:
        generated.unlink(missing_ok=True)
        reference.unlink(missing_ok=True)
    print(f"{stage}_EVAL_COMPLETE", len(rows), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--max-samples", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--subset", choices=("all", "validation"), default="all")
    parser.add_argument("--eval-stage", choices=("E0", "E1"), default="E0")
    parser.add_argument("--checkpoint-url")
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--checkpoint-sha256")
    args = parser.parse_args()
    if args.eval_stage == "E1" and (
        bool(args.checkpoint_url) == bool(args.checkpoint_path) or not args.checkpoint_sha256
    ):
        raise ValueError("E1 requires one pinned checkpoint URL or path and SHA256")
    if args.eval_stage == "E0" and (
        args.checkpoint_url or args.checkpoint_path or args.checkpoint_sha256
    ):
        raise ValueError("E0 must use the untouched SILMA checkpoint")
    rows = _load_plan(args.plan, args.plan_sha256, args.max_samples, args.subset)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "stage": args.eval_stage,
                    "subset": args.subset,
                    "planned": len(rows),
                    "seed": args.seed,
                    "gpu": "T4x2",
                }
            )
        )
        return
    if sys.version_info[:2] != (3, 10):
        _install_python310()
    _evaluate(
        rows,
        args.resume,
        args.seed,
        args.eval_stage,
        args.subset,
        args.checkpoint_url,
        args.checkpoint_sha256,
        args.checkpoint_path,
    )


if __name__ == "__main__":
    main()
