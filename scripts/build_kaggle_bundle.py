"""Package only reproducible source and small evidence for private Kaggle runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def build_bundle(root: Path, output: Path, stage: str = "e1") -> dict[str, object]:
    if stage not in {"probe", "pilot", "full-probe", "e1", "eval"}:
        raise ValueError("Unknown Kaggle bundle stage")
    source = root / "src" / "masriswitch"
    paths = sorted(source.rglob("*.py"))
    paths += sorted((root / "configs").glob("*.yaml"))
    paths += [
        root / "pyproject.toml",
        root / "artifacts" / "data_audit.json",
        root / "artifacts" / "upstream" / "silma" / "vocab.txt",
    ]
    if stage in {"probe", "pilot", "full-probe", "e1"}:
        paths += [
            root / "scripts" / "archive_pilot.py",
            root / "scripts" / "kaggle_train_probe.py",
            root / "scripts" / "kaggle_train_pilot.py",
            root / "artifacts" / "kaggle_pilot_archive.json",
            root / "artifacts" / "train_patch.json",
        ]
    if stage in {"pilot", "full-probe", "e1"}:
        paths.append(root / "artifacts" / "train_probe.json")
    if stage in {"full-probe", "e1"}:
        paths += [
            root / "scripts" / "archive_full.py",
            root / "scripts" / "kaggle_full_probe.py",
            root / "scripts" / "kaggle_train_e1.py",
            root / "artifacts" / "pilot_500_result.json",
            root / "artifacts" / "kaggle_full_archive.json",
        ]
    if stage == "e1":
        paths.append(root / "artifacts" / "full_probe_result.json")
    if stage in {"e1", "eval"}:
        paths += [
            root / "scripts" / "kaggle_e0_eval.py",
            root / "artifacts" / "e0_eval_plan.jsonl",
            root / "artifacts" / "e0_eval_plan_manifest.json",
        ]
    if stage == "eval":
        paths.append(root / "scripts" / "kaggle_product_smoke.py")
        paths.append(root / "scripts" / "kaggle_hf_smoke.py")
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("Required source, config, audit, or vocab is missing")
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, path.relative_to(root).as_posix())
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    return {
        "bundle": str(output),
        "stage": stage,
        "sha256": digest,
        "files": len(paths),
        "bytes": output.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/kaggle_bundle.zip"))
    parser.add_argument(
        "--stage", choices=("probe", "pilot", "full-probe", "e1", "eval"), default="e1"
    )
    args = parser.parse_args()
    print(json.dumps(build_bundle(Path.cwd(), args.output, args.stage)))


if __name__ == "__main__":
    main()
