"""Freeze validation and locked benchmark prompts before model evaluation."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from masriswitch.config import Paths
from masriswitch.text.entities import parse_entities


def build_e0_plan(paths: Paths, max_samples: int | None = None) -> dict[str, Any]:
    artifacts = paths.artifacts
    splits = json.loads((artifacts / "splits.json").read_text(encoding="utf-8"))
    benchmark = json.loads((artifacts / "benchmark_manifest.json").read_text(encoding="utf-8"))
    benchmark_path = artifacts / "masriswitch_bench_v1.jsonl"
    if (
        not splits.get("complete")
        or benchmark["sha256"] != hashlib.sha256(benchmark_path.read_bytes()).hexdigest()
    ):
        raise ValueError("Complete splits and unchanged benchmark are required")
    val_ids = {row["sample_id"] for row in splits["rows"] if row["split"] == "val"}
    train_ids = {row["sample_id"] for row in splits["rows"] if row["split"] == "train"}
    validation = []
    for line in (artifacts / "prepared_rows.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["sample_id"] in val_ids:
            validation.append(
                {
                    "id": row["sample_id"],
                    "set": "validation",
                    "domain": row["domain"],
                    "bucket": row["bucket"],
                    "text": row["text"],
                    "entities": [entity.__dict__ for entity in parse_entities(row["text"])],
                }
            )
    if len(validation) != len(val_ids) or any(row["id"] in train_ids for row in validation):
        raise ValueError("Validation prompt IDs do not match held-out split")
    locked = set(json.loads((artifacts / "benchmark_locked_ids.json").read_text())["ids"])
    benchmark_rows = []
    for line in benchmark_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["id"] in locked:
            benchmark_rows.append({**row, "set": "locked_benchmark"})
    if len(benchmark_rows) != len(locked) or len(locked) != 300:
        raise ValueError("Locked benchmark IDs changed")
    rows = sorted(validation + benchmark_rows, key=lambda row: (row["set"], row["id"]))
    if max_samples is not None:
        if max_samples < 1:
            raise ValueError("max_samples must be positive")
        rows = rows[:max_samples]
    output = artifacts / "e0_eval_plan.jsonl"
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    result = {
        "prompts": len(rows),
        "validation": sum(row["set"] == "validation" for row in rows),
        "locked_benchmark": sum(row["set"] == "locked_benchmark" for row in rows),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    (artifacts / "e0_eval_plan_manifest.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result
