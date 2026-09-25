"""Derive the E0 validation comparison from the frozen 489-prompt run."""

from __future__ import annotations

import json
from pathlib import Path

from masriswitch.config import Paths
from masriswitch.eval.aggregate import aggregate_evaluation

SILMA_SHA = "f43256d0b78b8803c638aed0875da5a4b372b4a784690a0156e5baff14f7336c"


def main() -> None:
    paths = Paths(Path.cwd())
    plan = [
        json.loads(line)
        for line in (paths.artifacts / "e0_eval_plan.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    validation_plan = [row for row in plan if row["set"] == "validation"]
    validation_ids = {row["id"] for row in validation_plan}
    if len(validation_ids) != 189:
        raise ValueError("Frozen validation plan changed")
    all_rows = [
        json.loads(line)
        for line in (paths.artifacts / "e0_eval_rows.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    if len(all_rows) != 489 or len({row["id"] for row in all_rows}) != 489:
        raise ValueError("Complete E0 rows are required")
    validation_rows = [row for row in all_rows if row["id"] in validation_ids]
    if {row["id"] for row in validation_rows} != validation_ids:
        raise ValueError("E0 validation rows are incomplete")
    output = paths.artifacts / "e0_validation_eval_rows.jsonl"
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in validation_rows),
        encoding="utf-8",
    )
    metrics = aggregate_evaluation(
        paths, output, stage="E0", model_sha256=SILMA_SHA, subset="validation", max_samples=189
    )
    (paths.artifacts / "e0_validation_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    first_ids = {row["id"] for row in validation_plan[:50]}
    first_rows = [row for row in validation_rows if row["id"] in first_ids]
    first_output = paths.artifacts / "e0_validation_50_eval_rows.jsonl"
    first_output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in first_rows),
        encoding="utf-8",
    )
    first_metrics = aggregate_evaluation(
        paths, first_output, stage="E0", model_sha256=SILMA_SHA, subset="validation", max_samples=50
    )
    (paths.artifacts / "e0_validation_50_metrics.json").write_text(
        json.dumps(first_metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"rows": len(validation_rows), "metrics": metrics["metrics"]}))


if __name__ == "__main__":
    main()
