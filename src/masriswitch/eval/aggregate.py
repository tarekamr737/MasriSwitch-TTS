"""Validate generated evaluation rows and compute reproducible metrics."""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any

from masriswitch.config import Paths
from masriswitch.eval.metrics import (
    arabic_cer,
    bootstrap_ci,
    bootstrap_ratio_ci,
    edit_distance,
    english_entity_error_rate,
    entity_accuracy,
    metric_tokens,
    real_time_factor,
    word_error_rate,
)
from masriswitch.text.entities import Entity


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _entities(row: dict[str, Any]) -> list[Entity]:
    return [Entity(**item) for item in row["entities"]]


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    references = [row["reference_text"] for row in rows]
    arabic = [row["hypothesis"] for row in rows]
    english = [row["english_decoder_hypothesis"] for row in rows]
    entities = [_entities(row) for row in rows]
    latencies = [row["synthesis_latency_seconds"] for row in rows]
    rtfs = [
        real_time_factor(latency, row["audio_seconds"])
        for latency, row in zip(latencies, rows, strict=True)
    ]
    similarities = [row["speaker_similarity"] for row in rows]
    result: dict[str, Any] = {
        "count": len(rows),
        "wer_ar_decoder": word_error_rate(references, arabic),
        "arabic_cer_ar_decoder": arabic_cer(references, arabic),
        "mean_speaker_similarity": statistics.mean(similarities),
        "mean_synthesis_latency_seconds": statistics.mean(latencies),
        "p95_synthesis_latency_seconds": sorted(latencies)[math.ceil(0.95 * len(latencies)) - 1],
        "mean_rtf": statistics.mean(rtfs),
        "peak_tts_vram_bytes": max(row["peak_tts_vram_bytes"] for row in rows),
        "invalid_audio_count": 0,
    }
    numeric_kinds = {"phone", "date", "time", "money", "id"}
    english_kinds = {"brand", "acronym"}
    numeric_count = sum(entity.kind in numeric_kinds for group in entities for entity in group)
    english_count = sum(entity.kind in english_kinds for group in entities for entity in group)
    if numeric_count + english_count:
        numeric_correct = (
            entity_accuracy(entities, arabic, numeric_kinds) * numeric_count
            if numeric_count
            else 0.0
        )
        english_correct = (
            entity_accuracy(entities, english, english_kinds) * english_count
            if english_count
            else 0.0
        )
        result["critical_entity_accuracy"] = (numeric_correct + english_correct) / (
            numeric_count + english_count
        )
        result["critical_entity_count"] = numeric_count + english_count
    else:
        result["critical_entity_accuracy"] = "TBD"
        result["critical_entity_count"] = 0
    if any(
        any(
            any("A" <= char <= "Z" or "a" <= char <= "z" for char in entity.value)
            for entity in group
        )
        for group in entities
    ):
        result["english_entity_error_rate_en_decoder"] = english_entity_error_rate(
            entities, english
        )
    else:
        result["english_entity_error_rate_en_decoder"] = "TBD"
    if len(rows) >= 20:
        result["mean_rtf_95ci"] = bootstrap_ci(rtfs, statistics.mean)
        result["mean_speaker_similarity_95ci"] = bootstrap_ci(similarities, statistics.mean)
    return result


def aggregate_evaluation(
    paths: Paths,
    rows_path: Path,
    *,
    stage: str,
    model_sha256: str,
    subset: str = "all",
    max_samples: int = 489,
) -> dict[str, Any]:
    """Reject partial, duplicated, or changed evaluation rows."""
    if stage not in {"E0", "E1"} or subset not in {"all", "validation"}:
        raise ValueError("Unknown evaluation stage or subset")
    if not re.fullmatch(r"[0-9a-f]{64}", model_sha256):
        raise ValueError("Invalid model SHA256")
    plan_path = paths.artifacts / "e0_eval_plan.jsonl"
    manifest = json.loads(
        (paths.artifacts / "e0_eval_plan_manifest.json").read_text(encoding="utf-8")
    )
    if _sha256(plan_path) != manifest["sha256"] or manifest["prompts"] != 489:
        raise ValueError("Evaluation plan changed")
    plan = [json.loads(line) for line in plan_path.read_text(encoding="utf-8").splitlines()]
    selected_plan = [row for row in plan if subset == "all" or row["set"] == "validation"]
    if not 1 <= max_samples <= len(selected_plan):
        raise ValueError("max_samples outside evaluation subset")
    expected = {row["id"]: row for row in selected_plan[:max_samples]}
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()]
    if (
        len(rows) != max_samples
        or len(expected) != max_samples
        or {row["id"] for row in rows} != set(expected)
    ):
        raise ValueError("Evaluation rows are incomplete or duplicated")
    for row in rows:
        target = expected[row["id"]]
        if any(
            row[key] != target[source]
            for key, source in (
                ("reference_text", "text"),
                ("entities", "entities"),
                ("set", "set"),
                ("bucket", "bucket"),
                ("domain", "domain"),
            )
        ):
            raise ValueError(f"Evaluation row changed: {row['id']}")
        for key in (
            "audio_seconds",
            "synthesis_latency_seconds",
            "speaker_similarity",
            "peak_tts_vram_bytes",
        ):
            if not isinstance(row.get(key), (float, int)) or not math.isfinite(row[key]):
                raise ValueError(f"Invalid {key}: {row['id']}")
        if (
            row["sample_rate"] != 24000
            or row["audio_seconds"] <= 0
            or row["synthesis_latency_seconds"] < 0
        ):
            raise ValueError(f"Invalid audio or duration: {row['id']}")
        if not -1 <= row["speaker_similarity"] <= 1:
            raise ValueError(f"Invalid speaker similarity: {row['id']}")
        if row["model_sha256"] != model_sha256:
            raise ValueError("Unexpected model hash")
        if (
            row["asr_sha256"] != "e76620f83d5f5b69efd3d87e3dc180c1bd21df9fbebacfd4335e5e1efcc018da"
            or row["speaker_model_sha256"]
            != "0575cb64845e6b9a10db9bcb74d5ac32b326b8dc90352671d345e2ee3d0126a2"
            or row["seed"] != 42
            or row["nfe_steps"] != 16
        ):
            raise ValueError("Evaluation protocol changed")
    ordered = [next(row for row in rows if row["id"] == key) for key in expected]
    result = {
        "stage": stage,
        "subset": subset,
        "model_sha256": model_sha256,
        "complete": subset == "all" and max_samples == 489,
        "plan_sha256": manifest["sha256"],
        "rows_sha256": _sha256(rows_path),
        "protocol": {
            "tts": (
                "SILMA v1; 16 NFE steps; fixed private upstream reference"
                if stage == "E0"
                else "E1; 16 NFE steps; fixed private upstream reference"
            ),
            "asr": (
                "Whisper Turbo CTranslate2; Arabic pass for WER/CER and numeric CEA, "
                "English pass for EER and brand/acronym CEA"
            ),
            "speaker": "SpeechBrain ECAPA cosine to the same private reference",
            "seed": 42,
        },
        "overall": _metrics(ordered),
        "by_set": {
            name: _metrics([row for row in ordered if row["set"] == name])
            for name in sorted({row["set"] for row in ordered})
        },
        "by_bucket": {
            name: _metrics([row for row in ordered if row["bucket"] == name])
            for name in sorted({row["bucket"] for row in ordered})
        },
    }
    switched = [row for row in ordered if row["bucket"] not in {"simple", "ar_only"}]
    arabic_only = [row for row in ordered if row["bucket"] in {"simple", "ar_only"}]
    result["metrics"] = {
        "cs_wer": (
            word_error_rate(
                [row["reference_text"] for row in switched],
                [row["hypothesis"] for row in switched],
            )
            if switched
            else "TBD"
        ),
        "ar_cer": (
            arabic_cer(
                [row["reference_text"] for row in arabic_only],
                [row["hypothesis"] for row in arabic_only],
            )
            if arabic_only
            else "TBD"
        ),
        "overall_wer": result["overall"]["wer_ar_decoder"],
        "overall_ar_cer": result["overall"]["arabic_cer_ar_decoder"],
        "code_switch_prompts": len(switched),
        "arabic_only_prompts": len(arabic_only),
        "english_eer": result["overall"]["english_entity_error_rate_en_decoder"],
        "critical_entity_accuracy": result["overall"]["critical_entity_accuracy"],
        "speaker_similarity": result["overall"]["mean_speaker_similarity"],
        "mean_rtf": result["overall"]["mean_rtf"],
    }
    if result["complete"]:
        cs_gold = [metric_tokens(row["reference_text"]) for row in switched]
        cs_errors = [
            edit_distance(gold, metric_tokens(row["hypothesis"]))
            for gold, row in zip(cs_gold, switched, strict=True)
        ]
        result["metrics"]["cs_wer_95ci"] = bootstrap_ratio_ci(
            cs_errors, [len(gold) for gold in cs_gold]
        )
        arabic_pattern = re.compile(r"[\u0600-\u06ff]")
        ar_gold = ["".join(arabic_pattern.findall(row["reference_text"])) for row in arabic_only]
        ar_errors = [
            edit_distance(gold, "".join(arabic_pattern.findall(row["hypothesis"])))
            for gold, row in zip(ar_gold, arabic_only, strict=True)
        ]
        result["metrics"]["ar_cer_95ci"] = bootstrap_ratio_ci(
            ar_errors, [len(gold) for gold in ar_gold]
        )
        english_counts = [
            sum(bool(re.search(r"[A-Za-z]", entity.value)) for entity in _entities(row))
            for row in ordered
        ]
        english_errors = [
            english_entity_error_rate([_entities(row)], [row["english_decoder_hypothesis"]]) * count
            if count
            else 0.0
            for row, count in zip(ordered, english_counts, strict=True)
        ]
        result["metrics"]["english_eer_95ci"] = bootstrap_ratio_ci(english_errors, english_counts)
    if stage == "E0" and result["complete"]:
        paths.artifacts.mkdir(parents=True, exist_ok=True)
        (paths.artifacts / "baseline_metrics.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    elif stage == "E1":
        paths.artifacts.mkdir(parents=True, exist_ok=True)
        (paths.artifacts / f"e1_{subset}_{model_sha256[:12]}_metrics.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return result


def aggregate_e0(paths: Paths, rows_path: Path, *, max_samples: int = 489) -> dict[str, Any]:
    """Validate the untouched SILMA baseline against the fixed plan."""
    return aggregate_evaluation(
        paths,
        rows_path,
        stage="E0",
        model_sha256="f43256d0b78b8803c638aed0875da5a4b372b4a784690a0156e5baff14f7336c",
        max_samples=max_samples,
    )
