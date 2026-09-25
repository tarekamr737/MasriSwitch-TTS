"""Read-only Hugging Face source inspection and small audio decode checks."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

from masriswitch.config import Paths
from masriswitch.data.registry import Source, assert_row_allowed, assert_source_allowed


def audit_source(source: Source, paths: Paths, max_samples: int | None = None) -> dict[str, Any]:
    assert_source_allowed(source)
    try:
        import soundfile as sf
        from datasets import Audio, load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the data extras before auditing") from exc

    counts: Counter[str] = Counter()
    selected = 0
    sampled_audio = 0
    rows_seen = 0
    for split in ("train", "test"):
        dataset = load_dataset(
            source.repo, source.config, split=split, revision=source.revision, streaming=True
        )
        dataset = dataset.cast_column("audio", Audio(decode=False))
        for row in dataset:
            rows_seen += 1
            counts[str(row.get("license", "missing"))] += 1
            try:
                assert_row_allowed(source, row)
            except PermissionError:
                continue
            selected += 1
            if sampled_audio < 12:
                audio = row["audio"]
                blob = audio.get("bytes")
                if blob is None:
                    raise ValueError("Expected embedded audio bytes")
                with sf.SoundFile(BytesIO(blob)) as file:
                    if file.frames == 0 or file.channels != 1 or file.samplerate != 24000:
                        raise ValueError("Invalid D1 sample audio")
                sampled_audio += 1
            if max_samples is not None and rows_seen >= max_samples:
                break
        if max_samples is not None and rows_seen >= max_samples:
            break
    complete = max_samples is None
    if complete and source.expected_rows is not None and selected != source.expected_rows:
        raise ValueError(f"D1 selected {selected}, expected {source.expected_rows}")
    evidence: dict[str, Any] = {
        "source": source.id,
        "repo": source.repo,
        "revision": source.revision,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "rows_seen": rows_seen,
        "selected_rows": selected,
        "selected_license": source.row_license,
        "license_counts": dict(counts),
        "audio_decode_checks": sampled_audio,
        "complete": complete,
        "release_safe": source.release_safe and complete,
    }
    paths.artifacts.mkdir(parents=True, exist_ok=True)
    (paths.artifacts / "data_audit.json").write_text(
        json.dumps(evidence, indent=2), encoding="utf-8"
    )
    paths.reports.mkdir(parents=True, exist_ok=True)
    (paths.reports / "DATA_AUDIT.md").write_text(
        f"# Data audit\n\nD1 revision: `{source.revision}`.\n\n"
        f"Rows examined: {rows_seen}; selected CC BY 4.0 generated rows: {selected}.\n\n"
        f"Audio decode checks: {sampled_audio}. Complete: {complete}.\n",
        encoding="utf-8",
    )
    return evidence


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
