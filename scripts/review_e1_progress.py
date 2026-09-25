"""Apply the declared three-evaluation early-stop rule to measured E1 stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from masriswitch.train.plan import review_validation_progress
from masriswitch.train.verify import verify_e1_stage


def _read(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--metrics", type=Path, action="append", required=True)
    parser.add_argument("--archive", type=Path, default=Path("artifacts/kaggle_full_archive.json"))
    parser.add_argument(
        "--baseline", type=Path, default=Path("artifacts/e0_validation_50_metrics.json")
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/e1_progress.json"))
    args = parser.parse_args()
    if len(args.result) != len(args.metrics):
        raise ValueError("A metrics file is required for each E1 stage")
    archive = _read(args.archive)
    baseline = _read(args.baseline)
    if baseline.get("overall", {}).get("count") != 50 or baseline.get("subset") != "validation":
        raise ValueError("Frozen 50-prompt E0 baseline is required")
    candidates = []
    stages = []
    for index, (result_path, metrics_path) in enumerate(
        zip(args.result, args.metrics, strict=True), 1
    ):
        updates = index * 1000
        result = _read(result_path)
        metrics = _read(metrics_path)
        sha = verify_e1_stage(result, archive, expected_updates=updates)
        if (
            metrics.get("model_sha256") != sha
            or metrics.get("stage") != "E1"
            or metrics.get("subset") != "validation"
            or metrics.get("overall", {}).get("count") != 50
            or metrics.get("plan_sha256") != baseline.get("plan_sha256")
        ):
            raise ValueError(f"Mismatched E1 validation metrics: {metrics_path}")
        item = {
            "english_eer": metrics["metrics"]["english_eer"],
            "cs_wer": metrics["metrics"]["cs_wer"],
            "ar_cer": metrics["metrics"]["overall_ar_cer"],
        }
        candidates.append(item)
        stages.append({"updates": updates, "checkpoint_sha256": sha, **item})
    review = review_validation_progress(candidates, baseline["metrics"]["overall_ar_cer"])
    payload = {
        "selection_subset": "first 50 frozen validation prompts",
        "locked_benchmark_used": False,
        "stages": stages,
        **review,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"stale_evaluations": review["stale_evaluations"], "stop_early": review["stop_early"]}
        )
    )


if __name__ == "__main__":
    main()
