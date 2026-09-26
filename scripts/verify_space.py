"""Run one bounded real generation and save operational evidence on D."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import requests
import soundfile as sf
from gradio_client import Client


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--long-text", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / "artifacts"
    name = "space_long_text_smoke" if args.long_text else "space_smoke"
    target = root / name
    target.mkdir(exist_ok=True)
    url = "https://tarek737-masriswitch-tts-demo.hf.space"
    response = requests.get(url + "/config", timeout=30)
    response.raise_for_status()
    config = response.json()
    assert not any(
        component["type"] in {"file", "uploadbutton"}
        or (component["type"] == "audio" and component["props"].get("interactive") is not False)
        for component in config["components"]
    )
    client = Client(url, hf_token=False, download_files=str(target), verbose=False)
    prompt = "راجع ال account معايا."
    if args.long_text:
        prompt = json.loads((root / "long_text_diagnostic.json").read_text(encoding="utf-8"))[
            "original"
        ]
    start = time.perf_counter()
    job = client.submit(prompt, api_name="/synthesize")
    result = job.result(timeout=180)
    path = Path(result)
    wave, rate = sf.read(path)
    assert rate == 24000 and wave.size and np.isfinite(wave).all()
    assert np.max(np.abs(wave)) > 0
    evidence = {
        "passed": True,
        "url": url,
        "prompt": prompt,
        "audio_file": str(path.relative_to(root)),
        "audio_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "sample_rate": rate,
        "audio_seconds": len(wave) / rate,
        "request_seconds": time.perf_counter() - start,
        "public_anonymous_request": True,
        "no_reference_upload": True,
        "ai_generated": True,
        "runtime_gradio_version": config["version"],
        "requests": 1,
    }
    (root / f"{name}.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(evidence, ensure_ascii=True))


if __name__ == "__main__":
    main()
