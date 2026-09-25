"""Archive only the audited D1 train split for an isolated Kaggle E1 run."""

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


def archive_full(root: Path, destination: Path) -> dict[str, Any]:
    from datasets import Dataset

    artifacts = root / "artifacts"
    splits = json.loads((artifacts / "splits.json").read_text(encoding="utf-8"))
    if not splits.get("complete") or splits.get("revision") != (
        "eae9a87c17e91e3f59a9696d5f4ff3eb51502e82"
    ):
        raise ValueError("Pinned complete D1 split is required")
    train_ids = {row["sample_id"] for row in splits["rows"] if row["split"] == "train"}
    if len(train_ids) != 3414 or splits.get("f5_train_rows") != 3414:
        raise ValueError("Expected exactly 3,414 audited train rows")
    data = artifacts / "data" / "finetuning_project_char"
    expected = {
        "raw.arrow": data / "raw.arrow",
        "duration.json": data / "duration.json",
        "vocab.txt": data / "vocab.txt",
    }
    for path in expected.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    rows = Dataset.from_file(str(data / "raw.arrow"))
    if len(rows) != len(train_ids):
        raise ValueError("Train Arrow count differs from locked split")
    wav_root = data / "wavs"
    wav_files: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for row in rows:
        sample_id = Path(str(row["audio_path"])).stem
        if sample_id not in train_ids or sample_id in seen:
            raise ValueError("Train Arrow contains held-out or duplicate ID")
        wav = wav_root / f"{sample_id}.wav"
        if not wav.is_file():
            raise FileNotFoundError(wav)
        wav_files.append((sample_id, wav))
        seen.add(sample_id)
    if seen != train_ids:
        raise ValueError("Train Arrow does not exactly match split IDs")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(destination, "w:gz") as archive:
        for name, path in expected.items():
            archive.add(path, arcname=f"finetuning_project_char/{name}")
        for sample_id, wav in sorted(wav_files):
            archive.add(wav, arcname=f"wavs/{sample_id}.wav")
    result = {
        "source_id": "d1",
        "source_revision": splits["revision"],
        "train_only": True,
        "train_rows": len(train_ids),
        "archive_sha256": _sha256(destination),
        "archive_bytes": destination.stat().st_size,
        "train_arrow_sha256": _sha256(data / "raw.arrow"),
        "train_ids_sha256": hashlib.sha256("\n".join(sorted(train_ids)).encode()).hexdigest(),
    }
    destination.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-samples", type=int, default=3414)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_samples != 3414 or args.seed != 42:
        raise ValueError("Full E1 archive requires the locked train split and seed")
    if args.dry_run:
        print(json.dumps({"train_rows": 3414, "destination": str(args.output)}))
        return
    if args.output.exists() and not args.resume:
        raise ValueError("Archive exists; pass --resume")
    if args.output.exists() and args.resume:
        manifest = args.output.with_suffix(".json")
        if not manifest.is_file():
            raise ValueError("Resume manifest missing")
        result = json.loads(manifest.read_text(encoding="utf-8"))
        if result["archive_sha256"] != _sha256(args.output):
            raise ValueError("Resume archive hash changed")
        print(json.dumps(result))
        return
    print(json.dumps(archive_full(args.root, args.output)))


if __name__ == "__main__":
    main()
