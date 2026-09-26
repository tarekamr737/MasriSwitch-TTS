"""Write the locked E0/E1 comparison only from verified measured artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from masriswitch.config import Paths
from masriswitch.eval.metrics import edit_distance, metric_tokens
from masriswitch.release.gate import _complete_evaluation


def _read(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def _row(label: str, item: dict[str, Any]) -> str:
    metrics = item["metrics"]
    overall = item["overall"]
    return (
        f"| {label} | {_percent(metrics['cs_wer'])} | {_percent(metrics['english_eer'])} | "
        f"{_percent(metrics['ar_cer'])} | {_percent(metrics['critical_entity_accuracy'])} | "
        f"{metrics['speaker_similarity']:.3f} | {metrics['mean_rtf']:.3f} | "
        f"{overall['invalid_audio_count']} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e1-metrics", type=Path, required=True)
    parser.add_argument("--e1-rows", type=Path, required=True)
    args = parser.parse_args()
    paths = Paths(Path.cwd())
    e0 = _read(paths.artifacts / "baseline_metrics.json")
    e1 = _read(args.e1_metrics)
    train = _read(paths.artifacts / "train_manifest.json")
    selection = _read(paths.artifacts / "checkpoint_selection.json")
    progress = _read(paths.artifacts / "e1_progress.json")
    if selection.get("selected_checkpoint_sha256") != train.get("checkpoint_sha256"):
        raise ValueError("Checkpoint selection does not match the final training manifest")
    if not _complete_evaluation(e0, {"E0": e0, "E1": e1}, train):
        raise ValueError("Complete, matching E0/E1 evaluations and selected checkpoint required")
    if _sha256(args.e1_rows) != e1.get("rows_sha256") or _sha256(
        paths.artifacts / "e0_eval_rows.jsonl"
    ) != e0.get("rows_sha256"):
        raise ValueError("Evaluation row hashes changed")
    rows = [json.loads(line) for line in args.e1_rows.read_text(encoding="utf-8").splitlines()]
    if len(rows) != 489 or len({row["id"] for row in rows}) != 489:
        raise ValueError("E1 evaluation rows are incomplete")
    locked = [row for row in rows if row["set"] == "locked_benchmark"]
    if len(locked) != 300:
        raise ValueError("Locked benchmark count changed")
    ranked = sorted(
        locked,
        key=lambda row: edit_distance(
            metric_tokens(row["reference_text"]), metric_tokens(row["hypothesis"])
        )
        / max(1, len(metric_tokens(row["reference_text"]))),
        reverse=True,
    )
    m0, m1 = e0["metrics"], e1["metrics"]
    cs_gain = 1 - m1["cs_wer"] / m0["cs_wer"]
    eer_gain = 1 - m1["english_eer"] / m0["english_eer"]
    arabic_guardrail = m1["ar_cer"] <= m0["ar_cer"] * 1.05
    latency_guardrail = m1["mean_rtf"] <= m0["mean_rtf"] * 1.10
    lines = [
        "# Evaluation",
        "",
        "Measured on the same 189 validation and 300 locked benchmark prompts, "
        "with the same private reference voice and 16-step inference protocol.",
        "",
        "| Model | Code-switch WER | English EER | Arabic-only CER | "
        "Critical entity accuracy | Speaker cosine | Mean RTF | Invalid audio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        _row("E0 SILMA", e0),
        _row("E1 selected", e1),
        "",
        f"Relative code-switch WER reduction: {_percent(cs_gain)}; "
        f"English EER reduction: {_percent(eer_gain)}.",
        f"Arabic-only CER ≤5% relative regression: {arabic_guardrail}; "
        f"mean RTF ≤10% regression: {latency_guardrail}.",
        f"Accuracy target (≥15% code-switch WER or ≥25% English EER reduction): "
        f"{cs_gain >= 0.15 or eer_gain >= 0.25}.",
        "",
        "## Prompt bootstrap 95% intervals",
        "",
        "Intervals use 2,000 prompt resamples with seed 42.",
        "",
    ]
    for label, metrics in (("E0", m0), ("E1", m1)):
        lines.append(
            f"- {label}: code-switch WER {_percent(metrics['cs_wer_95ci'][0])}–"
            f"{_percent(metrics['cs_wer_95ci'][1])}; English EER "
            f"{_percent(metrics['english_eer_95ci'][0])}–"
            f"{_percent(metrics['english_eer_95ci'][1])}; Arabic-only CER "
            f"{_percent(metrics['ar_cer_95ci'][0])}–"
            f"{_percent(metrics['ar_cer_95ci'][1])}."
        )
    lines += [
        "",
        "## Validation-only checkpoint selection",
        "",
        "The frozen 50-prompt screening advanced three eligible checkpoints to all "
        "189 validation prompts. Selection minimized English EER, then code-switch WER, "
        "subject to overall Arabic CER ≤1.05 × E0 validation. "
        "The 300 locked benchmark prompts were excluded from selection.",
        "",
        "| Updates | English EER | Code-switch WER | Overall Arabic CER | Selected |",
        "|---|---:|---:|---:|---|",
    ]
    updates = {stage["checkpoint_sha256"]: stage["updates"] for stage in progress["stages"]}
    for candidate in selection["candidates"]:
        sha = candidate["checkpoint_sha256"]
        lines.append(
            f"| {updates[sha]:,} | {_percent(candidate['english_eer'])} | "
            f"{_percent(candidate['cs_wer'])} | {_percent(candidate['ar_cer'])} | "
            f"{'yes' if sha == train['checkpoint_sha256'] else 'no'} |"
        )
    lines += [
        "",
        f"Training endpoint: {train['training_endpoint_updates']:,} updates; "
        f"selected checkpoint: {train['selected_updates']:,} updates. "
        f"Declared early stop: {train['stopped_early']}.",
        "",
        "## Highest-error locked prompts",
        "",
    ]
    for row in ranked[:3]:
        lines.append(
            f"- `{row['id']}` reference: {row['reference_text']}  "
            f"ASR hypothesis: {row['hypothesis']}"
        )
    lines += [
        "",
        "## Limits and provenance",
        "",
        "These are independent ASR proxy scores, not human judgments of pronunciation "
        "or naturalness. The training audio is synthetic and has no speaker IDs. "
        "The fixed SILMA sample was used privately; no public voice consent was established.",
        f"Plan SHA256: `{e0['plan_sha256']}`. Selected checkpoint SHA256: "
        f"`{train['checkpoint_sha256']}`.",
        "Machine-readable results: `artifacts/eval_metrics.json`.",
        "",
    ]
    (paths.artifacts / "eval_metrics.json").write_text(
        json.dumps({"E0": e0, "E1": e1}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    output = paths.reports / "EVALUATION.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"report": str(output), "checkpoint_sha256": train["checkpoint_sha256"]}))


if __name__ == "__main__":
    main()
