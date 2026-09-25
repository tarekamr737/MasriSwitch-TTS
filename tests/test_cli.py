"""Stable training commands pass only pinned, bounded arguments to Kaggle scripts."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from masriswitch import cli
from masriswitch.config import Paths


def test_train_dry_run_uses_pinned_manifest(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "kaggle_full_archive.json").write_text(
        json.dumps({"archive_sha256": "a" * 64, "train_arrow_sha256": "b" * 64})
    )
    commands: list[list[str]] = []

    def run(command: list[str], *, check: bool) -> SimpleNamespace:
        assert check is False
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli, "Paths", lambda: Paths(tmp_path))
    monkeypatch.setattr(cli.subprocess, "run", run)
    assert cli.main(["train", "--dry-run", "--max-updates", "1000"]) == 0
    command = commands[0]
    assert command[1] == str(tmp_path / "scripts" / "kaggle_train_e1.py")
    assert command[command.index("--archive-sha256") + 1] == "a" * 64
    assert command[command.index("--arrow-sha256") + 1] == "b" * 64
    assert command[-1] == "--dry-run"
