"""Render the measured E0 result into a concise human-readable report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from masriswitch.config import Paths


def _percent(value: float | str) -> str:
    return f"{100 * value:.2f}%" if isinstance(value, (int, float)) else value


def _line(name: str, metrics: dict[str, Any]) -> str:
    return (
        f"| {name} | {metrics['count']} | {_percent(metrics['wer_ar_decoder'])} | "
        f"{_percent(metrics['arabic_cer_ar_decoder'])} | "
        f"{_percent(metrics['english_entity_error_rate_en_decoder'])} | "
        f"{_percent(metrics['critical_entity_accuracy'])} | "
        f"{metrics['mean_speaker_similarity']:.3f} | {metrics['mean_rtf']:.3f} |"
    )


def write_e0_report(paths: Paths, result: dict[str, Any]) -> Path:
    if result.get("complete") is not True or result.get("overall", {}).get("count") != 489:
        raise ValueError("A complete 489-prompt E0 result is required")
    rows = [_line("Overall", result["overall"])]
    rows.extend(_line(name, result["by_set"][name]) for name in sorted(result["by_set"]))
    bucket_rows = [_line(name, item) for name, item in sorted(result["by_bucket"].items())]
    overall = result["overall"]
    summary = result["metrics"]
    report = "\n".join(
        [
            "# E0 SILMA baseline",
            "",
            "Measured on 189 held-out validation prompts and 300 locked MasriSwitch-Bench prompts.",
            "The same fixed SILMA upstream sample was used for private tests only. "
            "Public synthesis",
            "still requires a documented, consented voice.",
            "",
            "| Set | Prompts | WER | Arabic CER | English EER | "
            "Critical entity accuracy | Speaker cosine | Mean RTF |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
            *rows,
            "",
            f"Code-switch WER: {_percent(summary['cs_wer'])} on "
            f"{summary['code_switch_prompts']} prompts; Arabic-only CER: "
            f"{_percent(summary['ar_cer'])} on {summary['arabic_only_prompts']} prompts.",
            "",
            "## By code-switch bucket",
            "",
            "| Bucket | Prompts | WER | Arabic CER | English EER | "
            "Critical entity accuracy | Speaker cosine | Mean RTF |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
            *bucket_rows,
            "",
            "## Timing and uncertainty",
            "",
            f"Mean synthesis latency: {overall['mean_synthesis_latency_seconds']:.3f} s; "
            f"p95: {overall['p95_synthesis_latency_seconds']:.3f} s; "
            f"peak allocated TTS VRAM: {overall['peak_tts_vram_bytes'] / 2**30:.3f} GiB.",
            f"Mean RTF 95% prompt-bootstrap CI: {overall['mean_rtf_95ci'][0]:.3f}–"
            f"{overall['mean_rtf_95ci'][1]:.3f}; mean speaker cosine CI: "
            f"{overall['mean_speaker_similarity_95ci'][0]:.3f}–"
            f"{overall['mean_speaker_similarity_95ci'][1]:.3f} (2,000 resamples, seed 42).",
            f"Code-switch WER 95% CI: {_percent(summary['cs_wer_95ci'][0])}–"
            f"{_percent(summary['cs_wer_95ci'][1])}; Arabic-only CER CI: "
            f"{_percent(summary['ar_cer_95ci'][0])}–"
            f"{_percent(summary['ar_cer_95ci'][1])}; English EER CI: "
            f"{_percent(summary['english_eer_95ci'][0])}–"
            f"{_percent(summary['english_eer_95ci'][1])} (prompt bootstrap, 2,000 resamples).",
            f"Invalid audio outputs: {overall['invalid_audio_count']} of 489.",
            "",
            "## Protocol and limits",
            "",
            "SILMA v1 uses 16 NFE steps at 24 kHz. WER and Arabic CER use a pinned independent",
            "Whisper Turbo Arabic decoder. English EER uses a second English decoder on the same",
            "audio; this may miss correctly spoken English that the ASR transcribes differently.",
            "Critical entity accuracy requires exact token matches: numeric entities use the",
            "Arabic decoder, and brands/acronyms use the English decoder. These automated scores",
            "are ASR proxies, not human pronunciation or naturalness judgments. The table's WER "
            "includes all 489 prompts; code-switch WER excludes the simple/Arabic-only buckets. "
            "Synthetic source speech does not "
            "establish",
            "real Egyptian speaker diversity.",
            "",
            f"Plan SHA256: `{result['plan_sha256']}`. "
            f"Result rows SHA256: `{result['rows_sha256']}`.",
            "Machine-readable metrics: `artifacts/baseline_metrics.json`.",
            "",
        ]
    )
    paths.reports.mkdir(parents=True, exist_ok=True)
    output = paths.reports / "BASELINE.md"
    output.write_text(report, encoding="utf-8")
    return output


def write_e0_report_from_artifact(paths: Paths | None = None) -> Path:
    paths = paths or Paths()
    result = json.loads((paths.artifacts / "baseline_metrics.json").read_text(encoding="utf-8"))
    return write_e0_report(paths, result)
