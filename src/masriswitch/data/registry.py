"""Row-level source permissions; unknown provenance is rejected."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from masriswitch.config import ROOT, load_yaml


@dataclass(frozen=True)
class Source:
    id: str
    repo: str
    revision: str
    license: str | None
    row_license: str | None
    release_safe: bool
    enabled: bool
    config: str | None = None
    expected_rows: int | None = None
    expected_sha256: str | None = None


def load_sources(path: Path = ROOT / "configs/sources.yaml") -> dict[str, Source]:
    raw: dict[str, Any] = load_yaml(path)["sources"]
    return {
        key: Source(
            id=key,
            repo=value["repo"],
            revision=str(value.get("revision", "TBD")),
            license=value.get("license"),
            row_license=value.get("row_license"),
            release_safe=value.get("release_safe") is True,
            enabled=value.get("enabled", True) is True,
            config=value.get("config"),
            expected_rows=value.get("expected_rows"),
            expected_sha256=value.get("expected_sha256"),
        )
        for key, value in raw.items()
    }


def assert_source_allowed(source: Source, *, research: bool = False) -> None:
    if not source.enabled and not research:
        raise PermissionError(f"{source.id}: disabled source")
    if not source.release_safe and not research:
        raise PermissionError(f"{source.id}: source is not release-safe")
    if source.revision == "TBD":
        raise ValueError(f"{source.id}: revision must be pinned before fetching")
    if not source.license and not source.row_license:
        raise PermissionError(f"{source.id}: unknown license")


def assert_row_allowed(source: Source, row: Mapping[str, Any], *, research: bool = False) -> None:
    assert_source_allowed(source, research=research)
    if not research and source.row_license:
        actual = str(row.get("license", "")).lower().strip()
        if actual != source.row_license:
            raise PermissionError(f"{source.id}: row license {actual!r} is not allowed")
        if source.id == "d1" and row.get("source_type") != "generated":
            raise PermissionError("d1: row is not author-created generated provenance")
