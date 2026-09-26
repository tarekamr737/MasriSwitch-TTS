"""Bounded private API/demo smoke using the final evaluator's cached model files."""

from __future__ import annotations

import argparse
import json
from io import BytesIO
from pathlib import Path

from kaggle_e0_eval import REFERENCE_COMMIT, REFERENCE_SHA, REFERENCE_TEXT, _sha256


def approved_reference(path: Path) -> tuple[Path, str, str]:
    approval = json.loads(path.read_text(encoding="utf-8"))
    if (
        approval.get("speaker_consent_documented") is not True
        or approval.get("public_use_approved") is not True
    ):
        raise PermissionError("Public reference consent is required")
    reference = (path.parent / str(approval["filename"])).resolve()
    if reference.parent != path.parent.resolve():
        raise ValueError("Reference must stay inside its approval directory")
    digest = str(approval["sha256"])
    transcript = str(approval["transcript"])
    if _sha256(reference) != digest or not transcript.strip():
        raise ValueError("Approved reference hash or transcript is invalid")
    return reference, digest, transcript


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reference-approval", type=Path)
    args = parser.parse_args()
    if not 1 <= args.max_samples <= 10:
        raise ValueError("Private smoke is limited to 1–10 prompts")
    if len(args.checkpoint_sha256) != 64:
        raise ValueError("Expected selected checkpoint SHA256")
    approved = approved_reference(args.reference_approval) if args.reference_approval else None
    reference_sha = approved[1] if approved else REFERENCE_SHA
    if args.dry_run:
        print(json.dumps({"private_only": True, "planned": args.max_samples, "seed": args.seed}))
        return

    import numpy as np
    import requests
    import soundfile as sf
    from fastapi.testclient import TestClient

    from masriswitch.api.app import create_app
    from masriswitch.infer.demo import create_demo
    from masriswitch.infer.engine import EngineFiles, F5Engine

    output = Path("/kaggle/working/product_smoke.json")
    if output.exists():
        previous = json.loads(output.read_text(encoding="utf-8"))
        if (
            args.resume
            and previous.get("checkpoint_sha256") == args.checkpoint_sha256
            and previous.get("api_prompts") == args.max_samples
            and previous.get("seed") == args.seed
            and previous.get("reference_sha256") == reference_sha
            and previous.get("public_reference_approved") is bool(approved)
            and previous.get("passed") is True
        ):
            print("PRIVATE_PRODUCT_SMOKE_ALREADY_COMPLETE", flush=True)
            return
        raise ValueError("Existing smoke evidence differs; refusing to overwrite")
    root = Path("/tmp/masriswitch-e1-eval")
    if approved:
        reference, reference_sha, reference_text = approved
    else:
        reference = root / "private_product_reference.wav"
        reference_text = REFERENCE_TEXT
        url = (
            "https://raw.githubusercontent.com/SILMA-AI/silma-tts/"
            f"{REFERENCE_COMMIT}/src/silma_tts/infer/ref_audio_samples/ar.ref.24k.wav"
        )
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        reference.write_bytes(response.content)
    try:
        files = EngineFiles(
            checkpoint=root / "silma/model.pt",
            checkpoint_sha256=args.checkpoint_sha256,
            vocab=root / "silma/vocab.txt",
            silma_config=root / "silma/config.yaml",
            vocoder_dir=root / "vocos",
            reference_audio=reference,
            reference_sha256=reference_sha,
            reference_text=reference_text,
            model_id="E1-selected-private-test",
        )
        engine = F5Engine(files, seed=args.seed)
        prompts = [
            "أهلا بيك في خدمة العملاء.",
            "ممكن تعمل reset لل password؟",
            "ابعت OTP على الموبايل.",
            "ال meeting بكرة الساعة عشرة.",
            "عايز أعمل update لل account.",
            "ال order جاهز للتوصيل.",
            "راجع ال invoice قبل الدفع.",
            "ممكن تبعت email بالتفاصيل؟",
            "هنراجع ال request ونرد عليك.",
            "شكرا لاتصالك، يومك سعيد.",
        ][: args.max_samples]
        durations = []
        with TestClient(create_app(engine)) as client:
            assert client.get("/health").json()["ready"] is True
            assert client.get("/model-info").json()["fixed_reference_only"] is True
            normalized = client.post("/normalize", json={"text": "ابعت OTP"})
            assert normalized.json()["normalized_text"] == "ابعت أو تي بي"
            for index, prompt in enumerate(prompts, 1):
                result = client.post("/synthesize", json={"text": prompt})
                assert result.status_code == 200
                assert result.headers["content-type"] == "audio/wav"
                assert result.headers["x-ai-generated"] == "true"
                wave, rate = sf.read(BytesIO(result.content))
                assert rate == 24000 and wave.size > 0 and np.isfinite(wave).all()
                assert np.max(np.abs(wave)) > 0
                durations.append(wave.size / rate)
                print("PRIVATE_API_SMOKE_PROGRESS", index, len(prompts), flush=True)
        demo = create_demo(engine)
        callback = next(iter(demo.fns.values())).fn
        rate, wave = callback(prompts[0])
        assert rate == 24000 and wave.size > 0 and np.isfinite(wave).all()
        assert np.max(np.abs(wave)) > 0
        evidence = {
            "passed": True,
            "private_only": True,
            "public_reference_approved": bool(approved),
            "checkpoint_sha256": args.checkpoint_sha256,
            "reference_sha256": reference_sha,
            "api_prompts": len(prompts),
            "api_audio_seconds": durations,
            "gradio_callback_prompts": 1,
            "sample_rate": 24000,
            "seed": args.seed,
        }
        output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
        print("PRIVATE_PRODUCT_SMOKE_COMPLETE", flush=True)
    finally:
        if not approved:
            reference.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
