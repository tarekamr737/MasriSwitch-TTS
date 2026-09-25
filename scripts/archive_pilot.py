"""Archive only audited train-split pilot audio for isolated Kaggle GPU runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def archive_pilot(root: Path, destination: Path) -> dict[str, Any]:
    from datasets import Dataset

    artifacts = root / "artifacts"
    splits = json.loads((artifacts / "splits.json").read_text(encoding="utf-8"))
    if not splits.get("complete") or splits.get("revision") != (
        "eae9a87c17e91e3f59a9696d5f4ff3eb51502e82"
    ):
        raise ValueError("Pinned complete D1 split is required")
    train_ids = {row["sample_id"] for row in splits["rows"] if row["split"] == "train"}
    pilot_ids = set(splits["pilot_sample_ids"])
    if not pilot_ids or not pilot_ids.issubset(train_ids):
        raise ValueError("Pilot IDs must be a nonempty train-only subset")
    data = artifacts / "data" / "pilot_char"
    expected = {
        "raw.arrow": data / "raw.arrow",
        "duration.json": data / "duration.json",
        "vocab.txt": data / "vocab.txt",
    }
    for path in expected.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    rows = Dataset.from_file(str(data / "raw.arrow"))
    if len(rows) != len(pilot_ids):
        raise ValueError("Pilot Arrow count differs from locked IDs")
    wav_root = artifacts / "data" / "finetuning_project_char" / "wavs"
    wav_files: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for row in rows:
        sample_id = Path(str(row["audio_path"])).stem
        if sample_id not in pilot_ids or sample_id in seen:
            raise ValueError("Pilot Arrow contains missing or duplicate train ID")
        wav = wav_root / f"{sample_id}.wav"
        if not wav.is_file():
            raise FileNotFoundError(wav)
        wav_files.append((sample_id, wav))
        seen.add(sample_id)
    if {item[0] for item in wav_files} != pilot_ids:
        raise ValueError("Pilot Arrow does not exactly match locked IDs")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(destination, "w:gz") as archive:
        for name, path in expected.items():
            archive.add(path, arcname=f"pilot_char/{name}")
        for sample_id, wav in sorted(wav_files):
            archive.add(wav, arcname=f"wavs/{sample_id}.wav")
    result = {
        "source_id": "d1",
        "source_revision": splits["revision"],
        "train_only": True,
        "pilot_rows": len(pilot_ids),
        "archive_sha256": _sha256(destination),
        "archive_bytes": destination.stat().st_size,
        "pilot_arrow_sha256": _sha256(data / "raw.arrow"),
        "pilot_ids_sha256": hashlib.sha256("\n".join(sorted(pilot_ids)).encode()).hexdigest(),
    }
    destination.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(archive_pilot(args.root, args.output)))


if __name__ == "__main__":
    main()
