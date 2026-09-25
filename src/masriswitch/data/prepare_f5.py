"""Export F5's filename|text metadata after policy checks."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any


def export_metadata(rows: Iterable[dict[str, Any]], output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            audio = Path(str(row["audio_path"]))
            text = str(row["text"])
            if "|" in text or "\n" in text or not text.strip():
                raise ValueError("Invalid F5 transcript")
            if not audio.is_file() or audio.suffix.lower() != ".wav":
                raise ValueError(f"Missing WAV: {audio}")
            file.write(f"{audio.stem}|{text}\n")
            count += 1
    return count
