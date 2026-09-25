"""Select an E1 checkpoint using validation metrics only."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from masriswitch.train.plan import choose_checkpoint


def _metric(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Missing or non-finite {name}")
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline", type=Path, default=Path("artifacts/e0_validation_metrics.json")
    )
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/checkpoint_selection.json"))
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    if (
        baseline.get("stage") != "E0"
        or baseline.get("subset") != "validation"
        or baseline.get("overall", {}).get("count") != 189
    ):
        raise ValueError("Complete E0 validation comparison is required")
    baseline_ar_cer = _metric(baseline["metrics"]["overall_ar_cer"], "E0 Arabic CER")
    scored: list[dict[str, float]] = []
    identities: list[str] = []
    for path in args.candidate:
        candidate = json.loads(path.read_text(encoding="utf-8"))
        sha = candidate.get("model_sha256")
        if (
            candidate.get("stage") != "E1"
            or candidate.get("subset") != "validation"
            or candidate.get("plan_sha256") != baseline.get("plan_sha256")
            or candidate.get("overall", {}).get("count") != 189
            or not isinstance(sha, str)
            or len(sha) != 64
            or sha in identities
        ):
            raise ValueError(f"Incomplete or duplicate E1 validation candidate: {path}")
        metrics = candidate["metrics"]
        identities.append(sha)
        scored.append(
            {
                "english_eer": _metric(metrics["english_eer"], "English EER"),
                "cs_wer": _metric(metrics["cs_wer"], "code-switch WER"),
                "ar_cer": _metric(metrics["overall_ar_cer"], "Arabic CER"),
                "candidate_index": float(len(scored)),
            }
        )
    selected = choose_checkpoint(scored, baseline_ar_cer)
    index = int(selected["candidate_index"])
    result = {
        "selection_set": "189 validation prompts; locked benchmark excluded",
        "metric_order": ["english_eer", "cs_wer"],
        "arabic_cer_guardrail": "overall validation Arabic CER <= 1.05 * E0 validation",
        "baseline_ar_cer": baseline_ar_cer,
        "candidates": [
            {"checkpoint_sha256": sha, **metrics}
            for sha, metrics in zip(identities, scored, strict=True)
        ],
        "selected_checkpoint_sha256": identities[index],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selected_checkpoint_sha256": identities[index]}))


if __name__ == "__main__":
    main()
