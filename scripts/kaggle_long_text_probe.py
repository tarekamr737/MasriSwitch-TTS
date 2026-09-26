"""One-prompt, inference-only comparison of legacy and bounded Arabic chunking."""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

PROMPT = (
    "لو سمحت راجع الـ order رقم ORD123، واعمل update للـ delivery address قبل الساعة 14:30، "
    "وابعتلي confirmation بالـ email بعد خصم 12.5% من إجمالي 1499 EGP، "
    "ولو الدفع اترفض جرّب credit card تانية."
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, choices=[1], default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--chunk-steps", type=int, choices=[16, 32], default=16)
    parser.add_argument("--baseline-json", type=Path)
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps({"prompts": 1, "variants": ["legacy", "chunked"], "training": False}))
        return
    project = Path(__file__).resolve().parents[1]
    if sys.version_info[:2] != (3, 10):
        subprocess.run(["uv", "python", "install", "3.10"], check=True, timeout=300)
        py = "/tmp/masriswitch-long-probe-venv/bin/python"
        subprocess.run(["uv", "venv", "--python", "3.10", str(Path(py).parents[1])], check=True)
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                py,
                str(project),
                "f5-tts==1.1.7",
                "torch==2.8.0",
                "torchaudio==2.8.0",
                "transformers==4.46.3",
                "datasets==3.6.0",
                "faster-whisper==1.2.0",
            ],
            check=True,
            timeout=900,
        )
        os.execv(py, [py, __file__, *sys.argv[1:]])
    import numpy as np
    import soundfile as sf
    import torch
    from faster_whisper import WhisperModel
    from kaggle_e0_eval import ASR_REV, ASR_SHA, _hf_file, _sha256

    from masriswitch.eval.metrics import arabic_cer, english_entity_error_rate, word_error_rate
    from masriswitch.infer.chunks import speech_chunks
    from masriswitch.infer.engine import F5Engine
    from masriswitch.infer.hosted import MODEL_REVISION, MODEL_SHA, hosted_engine_files
    from masriswitch.text.normalize import normalize_text

    output = Path("/kaggle/working")
    result_file = output / f"long_text_{args.chunk_steps}_probe.json"
    if result_file.exists():
        result = json.loads(result_file.read_text(encoding="utf-8"))
        if args.resume and result["seed"] == args.seed and result["checkpoint_sha256"] == MODEL_SHA:
            print("LONG_TEXT_PROBE_ALREADY_COMPLETE", flush=True)
            return
        raise ValueError("Existing probe evidence does not match")
    approval = json.loads(
        (project / "artifacts/reference/approved.json").read_text(encoding="utf-8")
    )
    approval["parts"] = 1
    environment = {
        "REFERENCE_APPROVAL_JSON": json.dumps(approval),
        "REFERENCE_WAV_B64_0": base64.b64encode(
            (project / "artifacts/reference/consented.wav").read_bytes()
        ).decode(),
    }
    root = Path("/tmp/masriswitch-long-probe")
    engine = F5Engine(
        hosted_engine_files(root, environment),
        seed=args.seed,
        nfe_steps=args.chunk_steps,
        device="cuda:0",
    )
    asr = root / "asr"
    for name in (
        "model.bin",
        "config.json",
        "tokenizer.json",
        "vocabulary.json",
        "preprocessor_config.json",
    ):
        _hf_file(
            "dropbox-dash/faster-whisper-large-v3-turbo",
            ASR_REV,
            name,
            asr,
            ASR_SHA if name == "model.bin" else None,
        )
    whisper = WhisperModel(str(asr), device="cuda", device_index=1, compute_type="float16")
    normalized = normalize_text(PROMPT)
    result = {
        "prompt": PROMPT,
        "normalized": normalized.normalized_text,
        "seed": args.seed,
        "checkpoint_sha256": MODEL_SHA,
        "model_revision": MODEL_REVISION,
        "reference_sha256": engine.files.reference_sha256,
        "asr_sha256": ASR_SHA,
        "nfe_steps_by_variant": {"legacy": 16, "chunked": args.chunk_steps},
        "chunk_bytes_limit": engine._chunk_bytes,
        "chunks": speech_chunks(normalized.normalized_text, max_bytes=engine._chunk_bytes),
        "diagnostic_only": True,
        "training": False,
        "variants": {},
    }
    variants = ["legacy", "chunked"]
    if args.baseline_json:
        baseline = json.loads(args.baseline_json.read_text(encoding="utf-8"))
        for key in (
            "prompt",
            "normalized",
            "seed",
            "checkpoint_sha256",
            "reference_sha256",
            "asr_sha256",
        ):
            if baseline[key] != result[key]:
                raise ValueError(f"Baseline evidence differs: {key}")
        if baseline.get("nfe_steps") != 16:
            raise ValueError("Expected verified 16-step legacy baseline")
        result["baseline_json_sha256"] = _sha256(args.baseline_json)
        result["variants"]["legacy"] = baseline["variants"]["legacy"]
        variants = ["chunked"]
    for variant in variants:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        start = time.perf_counter()
        if variant == "legacy":
            wave, rate, _ = engine._infer_process(
                str(engine.files.reference_audio),
                engine.files.reference_text,
                normalized.normalized_text,
                engine._model,
                engine._vocoder,
                nfe_step=16,
                cross_fade_duration=0,
                device="cuda:0",
            )
        else:
            wave, rate = engine.synthesize(PROMPT)
        latency = time.perf_counter() - start
        assert rate == 24000 and len(wave) and np.isfinite(wave).all()
        assert np.max(np.abs(wave)) > 0
        path = output / f"long_text_{variant}_{args.chunk_steps}.wav"
        sf.write(path, wave, rate, subtype="PCM_16")
        hypotheses = {}
        for language in ("ar", "en"):
            segments, _ = whisper.transcribe(
                str(path),
                language=language,
                beam_size=5,
                vad_filter=False,
                condition_on_previous_text=False,
            )
            hypotheses[language] = " ".join(segment.text.strip() for segment in segments)
        metrics = {
            "audio_seconds": len(wave) / rate,
            "synthesis_seconds": latency,
            "wav_sha256": _sha256(path),
            "hypotheses": hypotheses,
            "wer_ar": word_error_rate([normalized.normalized_text], [hypotheses["ar"]]),
            "cer_ar": arabic_cer([normalized.normalized_text], [hypotheses["ar"]]),
            "eer_en": english_entity_error_rate([normalized.entities], [hypotheses["en"]]),
        }
        result["variants"][variant] = metrics
        print("LONG_TEXT_VARIANT_COMPLETE", variant, json.dumps(metrics), flush=True)
    result_file.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("LONG_TEXT_PROBE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
