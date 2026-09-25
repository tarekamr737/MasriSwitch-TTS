"""Build a hash-checked local model release bundle after the release gate passes."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from masriswitch.config import Paths
from masriswitch.data.audit import sha256_file
from masriswitch.release.gate import check_release


def package_release(paths: Paths) -> dict[str, Any]:
    gate = check_release(paths)
    if gate["weight_publication_allowed"] is not True:
        raise PermissionError("Release gate blocked: " + ", ".join(gate["blockers"]))
    train = json.loads((paths.artifacts / "train_manifest.json").read_text(encoding="utf-8"))
    checkpoint = Path(train["best_checkpoint"]).resolve()
    if not checkpoint.is_relative_to(paths.artifacts.resolve()):
        raise ValueError("Selected checkpoint must be inside ignored artifacts")
    if sha256_file(checkpoint) != train["checkpoint_sha256"]:
        raise ValueError("Selected checkpoint hash mismatch")
    destination = paths.artifacts / "release_bundle"
    destination.mkdir(parents=True, exist_ok=True)
    model = destination / "model.pt"
    if model.exists():
        if sha256_file(model) != train["checkpoint_sha256"]:
            raise ValueError("Existing release model has another hash")
    else:
        try:
            os.link(checkpoint, model)
        except OSError:
            shutil.copy2(checkpoint, model)
    files = {
        "README.md": paths.reports / "MODEL_CARD_DRAFT.md",
        "EVALUATION.md": paths.reports / "EVALUATION.md",
        "DATA_AUDIT.md": paths.reports / "DATA_AUDIT.md",
        "LICENSE": paths.root / "LICENSE",
        "NOTICE": paths.root / "NOTICE",
        "config.yaml": paths.artifacts / "upstream" / "silma" / "config.yaml",
        "vocab.txt": paths.artifacts / "upstream" / "silma" / "vocab.txt",
        "training_args.json": paths.artifacts / "train_manifest.json",
        "data_manifest.json": paths.artifacts / "data_audit.json",
        "eval_results.json": paths.artifacts / "eval_metrics.json",
    }
    for name, source in files.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, destination / name)
    hashes = {name: sha256_file(destination / name) for name in ["model.pt", *files]}
    manifest = {
        "checkpoint_sha256": train["checkpoint_sha256"],
        "files_sha256": hashes,
        "weight_publication_allowed": True,
    }
    (destination / "release_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return {"bundle": str(destination), **manifest}
