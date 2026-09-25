"""Strict configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    root: Path = ROOT

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    @property
    def reports(self) -> Path:
        return self.root / "reports"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def validate_configs(root: Path = ROOT) -> None:
    sources = load_yaml(root / "configs/sources.yaml").get("sources")
    if not isinstance(sources, dict) or "d1" not in sources or "silma" not in sources:
        raise ValueError("Source registry must contain d1 and silma")
    for key, source in sources.items():
        if not isinstance(source, dict) or not source.get("repo"):
            raise ValueError(f"Invalid source {key}")
        if source.get("release_safe") and not (source.get("license") or source.get("row_license")):
            raise ValueError(f"Missing license for {key}")
    data = load_yaml(root / "configs/data.yaml")
    split = data.get("split")
    if not isinstance(split, list) or len(split) != 3 or abs(sum(split) - 1) > 1e-9:
        raise ValueError("Data split must sum to one")
    for name in ("train_pilot", "train_e1", "train_e2"):
        train = load_yaml(root / f"configs/{name}.yaml")
        if train.get("updates", 0) <= 0 or train.get("seed") is None:
            raise ValueError(f"Invalid {name} config")
    load_yaml(root / "configs/eval.yaml")
