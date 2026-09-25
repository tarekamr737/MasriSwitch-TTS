"""Download exact inference/training weights only after the data pipeline passes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

from masriswitch.config import Paths
from masriswitch.data.audit import sha256_file
from masriswitch.data.lock import lock_sources
from masriswitch.data.registry import load_sources


def bootstrap_model(paths: Paths) -> dict[str, Any]:
    splits_path = paths.artifacts / "splits.json"
    if not splits_path.is_file() or not json.loads(splits_path.read_text(encoding="utf-8")).get(
        "complete"
    ):
        raise ValueError("Complete data preparation is required before model download")
    sources = load_sources(paths.root / "configs/sources.yaml")
    downloaded: dict[str, str] = {}
    for source_id, filename in (("silma", "model.pt"), ("vocos", "pytorch_model.bin")):
        source = sources[source_id]
        output = paths.artifacts / "upstream" / source_id
        output.mkdir(parents=True, exist_ok=True)
        path = Path(
            hf_hub_download(source.repo, filename, revision=source.revision, local_dir=output)
        )
        actual_hash = sha256_file(path)
        if actual_hash != source.expected_sha256:
            raise ValueError(f"{source_id} downloaded hash differs from pinned SHA256")
        downloaded[source_id] = actual_hash
    locked = lock_sources(paths)
    if not locked["verified"]:
        raise ValueError("Bulk files downloaded but source lock is not verified")
    return {"verified": True, "downloaded_sha256": downloaded}
