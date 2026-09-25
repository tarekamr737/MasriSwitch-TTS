"""Fail-closed verification for downloaded private E1 stage evidence."""

from __future__ import annotations

import hashlib
import re
from typing import Any


def verify_e1_stage(
    result: dict[str, Any], archive: dict[str, Any], *, expected_updates: int
) -> str:
    """Return the pinned checkpoint hash only for a complete bounded stage."""
    ids = result.get("training_sample_ids")
    checkpoint_sha = result.get("checkpoint_sha256")
    if (
        result.get("experiment") != "E1"
        or result.get("target_updates") != 8000
        or result.get("updates_per_rank") != [expected_updates, expected_updates]
        or result.get("complete") is not (expected_updates == 8000)
        or result.get("smoke_prompts_passed") != 10
        or result.get("pilot_passed") is not True
        or result.get("training_source_ids") != ["d1"]
        or result.get("full_archive_sha256") != archive.get("archive_sha256")
        or result.get("seed") != 42
        or not isinstance(checkpoint_sha, str)
        or not re.fullmatch(r"[0-9a-f]{64}", checkpoint_sha)
        or not isinstance(ids, list)
        or len(ids) != 3414
        or not all(isinstance(item, str) for item in ids)
        or len(set(ids)) != 3414
    ):
        raise ValueError("E1 stage evidence is incomplete or changed")
    ids_hash = hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()
    if ids_hash != archive.get("train_ids_sha256"):
        raise ValueError("E1 training IDs differ from the audited train archive")
    return checkpoint_sha
