"""Pin Hub revisions and hashes before downloading bulk sources."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download

from masriswitch.config import Paths
from masriswitch.data.audit import sha256_file
from masriswitch.data.registry import load_sources


def lock_sources(paths: Paths) -> dict[str, Any]:
    sources = load_sources(paths.root / "configs/sources.yaml")
    api = HfApi()
    result: dict[str, Any] = {"verified": False, "sources": {}}
    for source_id in ("silma", "vocos", "d1"):
        source = sources[source_id]
        repo_type = "dataset" if source_id == "d1" else "model"
        info = (
            api.model_info(source.repo, revision=source.revision, files_metadata=True)
            if repo_type == "model"
            else api.dataset_info(source.repo, revision=source.revision, files_metadata=True)
        )
        if info.sha != source.revision:
            raise ValueError(f"Revision changed for {source_id}: {info.sha}")
        files: dict[str, Any] = {}
        if info.siblings is None:
            raise ValueError(f"No file metadata returned for {source_id}")
        for sibling in info.siblings:
            files[sibling.rfilename] = {
                "size": sibling.size,
                "remote_sha256": sibling.lfs.sha256 if sibling.lfs else None,
                "downloaded_sha256": "TBD",
            }
        small = {
            "silma": ("README.md", "config.yaml", "vocab.txt", "finetune_cli.py"),
            "vocos": ("README.md", "config.yaml"),
            "d1": ("README.md", "LICENSE.md"),
        }[source_id]
        output = paths.artifacts / "upstream" / source_id
        output.mkdir(parents=True, exist_ok=True)
        for filename in small:
            cached = Path(
                hf_hub_download(
                    source.repo, filename, repo_type=repo_type, revision=source.revision
                )
            )
            target = output / filename
            shutil.copyfile(cached, target)
            files[filename]["downloaded_sha256"] = sha256_file(target)
        bulk_name = {"silma": "model.pt", "vocos": "pytorch_model.bin"}.get(source_id)
        if bulk_name and (output / bulk_name).is_file():
            files[bulk_name]["downloaded_sha256"] = sha256_file(output / bulk_name)
        if source_id == "silma" and files["model.pt"]["remote_sha256"] != source.expected_sha256:
            raise ValueError("SILMA model.pt remote SHA256 differs from pinned expected hash")
        if (
            source_id == "vocos"
            and files["pytorch_model.bin"]["remote_sha256"] != source.expected_sha256
        ):
            raise ValueError("Vocos remote SHA256 differs from pinned expected hash")
        result["sources"][source_id] = {
            "repo": source.repo,
            "revision": source.revision,
            "license": source.license or "mixed; row-level filter required",
            "files": files,
        }
    f5 = sources["f5_tts"]
    f5_path = paths.artifacts / "upstream" / "f5-tts"
    commit = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={f5_path.as_posix()}",
            "-C",
            str(f5_path),
            "rev-parse",
            "HEAD",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != f5.revision:
        raise ValueError("F5-TTS checkout does not match pinned 1.1.7 commit")
    result["sources"]["f5_tts"] = {
        "repo": f5.repo,
        "revision": commit,
        "tag": "1.1.7",
        "license": f5.license,
    }
    result["verified"] = all(
        result["sources"][key]["files"][name]["downloaded_sha256"] == sources[key].expected_sha256
        for key, name in (("silma", "model.pt"), ("vocos", "pytorch_model.bin"))
    )
    paths.artifacts.mkdir(parents=True, exist_ok=True)
    (paths.artifacts / "source_lock.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result
