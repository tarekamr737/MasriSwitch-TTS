"""Resumable D1 preparation; only row-licensed audio reaches F5 metadata."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from masriswitch.config import Paths
from masriswitch.data.audit import sha256_file
from masriswitch.data.clean import audio_quality, clean_transcript, rejection_reason, stable_id
from masriswitch.data.dedupe import dedupe_rows
from masriswitch.data.prepare_f5 import export_metadata
from masriswitch.data.registry import Source, assert_row_allowed, assert_source_allowed
from masriswitch.data.split import assign_splits
from masriswitch.text.spans import analyze_switches


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Missing required evidence: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid evidence: {path}")
    return data


def _decode_and_trim(blob: bytes) -> tuple[np.ndarray, int]:
    with sf.SoundFile(BytesIO(blob)) as file:
        samples = file.read(dtype="float32", always_2d=True)
        sample_rate = file.samplerate
    mono = samples.mean(axis=1)
    if sample_rate != 24000:
        from math import gcd

        from scipy.signal import resample_poly

        factor = gcd(sample_rate, 24000)
        mono = resample_poly(mono, 24000 // factor, sample_rate // factor).astype(np.float32)
        sample_rate = 24000
    active = np.flatnonzero(np.abs(mono) >= 0.005)
    if active.size:
        pad = int(0.25 * sample_rate)
        mono = mono[max(0, int(active[0]) - pad) : min(len(mono), int(active[-1]) + pad + 1)]
    return mono, sample_rate


def prepare_data(
    source: Source,
    paths: Paths,
    *,
    max_samples: int | None = None,
    resume: bool = False,
    seed: int = 42,
) -> dict[str, Any]:
    assert_source_allowed(source)
    audit = _load_json(paths.artifacts / "data_audit.json")
    if not audit.get("complete") or not audit.get("release_safe"):
        raise ValueError("Complete release-safe data audit is required before preparation")
    if audit.get("revision") != source.revision:
        raise ValueError("Audit revision differs from source registry")
    from datasets import Audio, load_dataset

    data_dir = paths.artifacts / "data" / "finetuning_project_char"
    wav_dir = data_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    ledger = paths.artifacts / "prepared_rows.jsonl"
    existing: dict[str, dict[str, Any]] = {}
    if resume and ledger.is_file():
        for line in ledger.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if Path(row["audio_path"]).is_file():
                existing[str(row["upstream_id"])] = row
    elif ledger.is_file():
        raise ValueError("Preparation exists; pass --resume to continue")
    selected = 0
    rejects: Counter[str] = Counter()
    accepted = list(existing.values())
    with ledger.open("a", encoding="utf-8", newline="\n") as log:
        for split in ("train", "test"):
            dataset = load_dataset(
                source.repo, source.config, split=split, revision=source.revision, streaming=True
            )
            dataset = dataset.cast_column("audio", Audio(decode=False))
            for row in dataset:
                if max_samples is not None and selected >= max_samples:
                    break
                try:
                    assert_row_allowed(source, row)
                except PermissionError:
                    continue
                selected += 1
                upstream_id = str(row["id"])
                if upstream_id in existing:
                    continue
                text = clean_transcript(str(row["text"]))
                blob = row["audio"].get("bytes")
                if blob is None:
                    rejects["missing_audio"] += 1
                    continue
                raw_hash = hashlib.sha256(blob).hexdigest()
                try:
                    samples, sample_rate = _decode_and_trim(blob)
                    quality = audio_quality(samples, sample_rate, text)
                except (RuntimeError, ValueError, sf.LibsndfileError):
                    rejects["decode"] += 1
                    continue
                reason = rejection_reason(quality, text)
                if reason:
                    rejects[reason] += 1
                    continue
                sample_id = stable_id(source.id, upstream_id, raw_hash)
                audio_path = wav_dir / f"{sample_id}.wav"
                sf.write(audio_path, samples, sample_rate, subtype="PCM_16")
                prepared = {
                    "sample_id": sample_id,
                    "source_id": source.id,
                    "upstream_id": upstream_id,
                    "source_split": split,
                    "license": row["license"],
                    "text": text,
                    "domain": row.get("domain", "unknown"),
                    "bucket": analyze_switches(text).bucket,
                    "audio_path": str(audio_path),
                    "audio_sha256": raw_hash,
                    "processed_sha256": sha256_file(audio_path),
                    "duration": quality.duration,
                    "rms_dbfs": quality.rms_dbfs,
                    "clip_ratio": quality.clip_ratio,
                    "silence_ratio": quality.silence_ratio,
                    "chars_per_sec": quality.chars_per_sec,
                }
                log.write(json.dumps(prepared, ensure_ascii=False) + "\n")
                log.flush()
                accepted.append(prepared)
                if max_samples is not None and selected >= max_samples:
                    break
            if max_samples is not None and selected >= max_samples:
                break
    complete = max_samples is None
    if complete and selected != source.expected_rows:
        raise ValueError(f"Selected row count changed: {selected} != {source.expected_rows}")
    return finalize_prepared(
        source,
        paths,
        accepted=accepted,
        selected=selected,
        rejects=dict(rejects),
        seed=seed,
        complete=complete,
    )


def _write_f5(rows: list[dict[str, Any]], data_dir: Path, vocab: Path) -> dict[str, Any]:
    from datasets.arrow_writer import ArrowWriter

    metadata_count = export_metadata(rows, data_dir / "metadata.csv")
    with ArrowWriter(path=str(data_dir / "raw.arrow"), writer_batch_size=64) as writer:
        for row in rows:
            writer.write(
                {"audio_path": row["audio_path"], "text": row["text"], "duration": row["duration"]}
            )
        writer.finalize()
    (data_dir / "duration.json").write_text(
        json.dumps({"duration": [row["duration"] for row in rows]}), encoding="utf-8"
    )
    (data_dir / "vocab.txt").write_bytes(vocab.read_bytes())
    return {
        "rows": metadata_count,
        "metadata_sha256": sha256_file(data_dir / "metadata.csv"),
        "arrow_sha256": sha256_file(data_dir / "raw.arrow"),
        "duration_seconds": sum(row["duration"] for row in rows),
    }


def _pilot_subset(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        strata[str(row["domain"]), str(row["bucket"])].append(row)
    pilot: list[dict[str, Any]] = []
    for group in strata.values():
        ordered = sorted(
            group, key=lambda row: hashlib.sha256(f"{seed}:{row['sample_id']}".encode()).hexdigest()
        )
        pilot.extend(ordered[: max(1, round(len(ordered) * 0.1))])
    return sorted(pilot, key=lambda row: row["sample_id"])


def finalize_prepared(
    source: Source,
    paths: Paths,
    *,
    accepted: list[dict[str, Any]],
    selected: int,
    rejects: dict[str, int],
    seed: int,
    complete: bool,
) -> dict[str, Any]:
    if not accepted:
        raise ValueError("No quality-accepted rows to prepare")
    bounds: list[float] | None = None
    if complete:
        bounds = np.percentile([row["chars_per_sec"] for row in accepted], [1, 99]).tolist()
        filtered = [row for row in accepted if bounds[0] <= row["chars_per_sec"] <= bounds[1]]
        rejects["chars_per_sec_outlier"] = len(accepted) - len(filtered)
    else:
        filtered = accepted
    data_dir = paths.artifacts / "data" / "finetuning_project_char"
    unique, duplicate_counts = dedupe_rows(filtered)
    assignments = assign_splits(unique, seed)
    for row in unique:
        row["split"] = assignments[row["sample_id"]]
    train_rows = [row for row in unique if row["split"] == "train"]
    vocab = paths.artifacts / "upstream" / "silma" / "vocab.txt"
    if not vocab.is_file():
        raise ValueError("Pinned SILMA vocab.txt must be downloaded first")
    full_export = _write_f5(train_rows, data_dir, vocab)
    pilot_rows = _pilot_subset(train_rows, seed)
    pilot_export = _write_f5(pilot_rows, paths.artifacts / "data" / "pilot_char", vocab)
    result = {
        "source_id": source.id,
        "revision": source.revision,
        "seed": seed,
        "complete": complete,
        "selected_rows": selected,
        "decoded_accepted": len(accepted),
        "quality_rejections": rejects,
        "chars_per_sec_bounds": bounds,
        "duplicates": duplicate_counts,
        "unique_rows": len(unique),
        "split_counts": dict(Counter(assignments.values())),
        "speaker_disjoint": False,
        "f5_train_rows": full_export["rows"],
        "f5_metadata_sha256": full_export["metadata_sha256"],
        "f5_arrow_sha256": full_export["arrow_sha256"],
        "training_duration_seconds": full_export["duration_seconds"],
        "pilot_rows": pilot_export["rows"],
        "pilot_duration_seconds": pilot_export["duration_seconds"],
        "pilot_sample_ids": [row["sample_id"] for row in pilot_rows],
        "rows": [
            {"sample_id": row["sample_id"], "split": row["split"], "source_id": source.id}
            for row in unique
        ],
    }
    (paths.artifacts / "splits.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if complete:
        target_outliers = sum(row["duration"] < 1.5 or row["duration"] > 15 for row in unique)
        total_seconds = sum(row["duration"] for row in unique)
        paths.reports.mkdir(parents=True, exist_ok=True)
        (paths.reports / "DATA_AUDIT.md").write_text(
            "# Data audit\n\n"
            f"Pinned D1 revision: `{source.revision}`. Full source rows: "
            f"{_load_json(paths.artifacts / 'data_audit.json')['rows_seen']}; "
            f"selected CC BY 4.0 generated rows: {selected}. "
            "All other source rows were excluded before processing.\n\n"
            f"Decoded quality-accepted: {len(accepted)}; quality rejections: "
            f"{json.dumps(rejects, sort_keys=True)}. "
            f"Chars/sec p1/p99 bounds: {bounds}. "
            f"Duplicates: {json.dumps(duplicate_counts, sort_keys=True)}.\n\n"
            f"Final clips: {len(unique)} ({total_seconds / 3600:.3f} hours); "
            f"outside 1.5–15 s target: {target_outliers}. "
            f"Split train/val/test: {result['split_counts']}. "
            "No speaker IDs were provided, so splits are not speaker-disjoint.\n",
            encoding="utf-8",
        )
    return {key: value for key, value in result.items() if key not in {"rows", "pilot_sample_ids"}}


def finalize_existing(source: Source, paths: Paths) -> dict[str, Any]:
    previous = _load_json(paths.artifacts / "splits.json")
    if not previous.get("complete") or previous.get("revision") != source.revision:
        raise ValueError("Complete preparation at pinned revision is required")
    ledger = paths.artifacts / "prepared_rows.jsonl"
    accepted = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    return finalize_prepared(
        source,
        paths,
        accepted=accepted,
        selected=int(previous["selected_rows"]),
        rejects={
            key: value
            for key, value in previous["quality_rejections"].items()
            if key != "chars_per_sec_outlier"
        },
        seed=int(previous["seed"]),
        complete=True,
    )
