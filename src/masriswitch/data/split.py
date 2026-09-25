"""Deterministic grouped, stratified split assignment."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable
from typing import Any


def assign_splits(rows: Iterable[dict[str, Any]], seed: int = 42) -> dict[str, str]:
    strata: dict[tuple[str, str], dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        sample_id = str(row["sample_id"])
        group = str(row.get("speaker_id") or row.get("session_id") or sample_id)
        key = (str(row.get("domain", "unknown")), str(row.get("bucket", "unknown")))
        strata[key][group].append(sample_id)
    assignments: dict[str, str] = {}
    for key, groups in sorted(strata.items()):
        ordered = sorted(
            groups,
            key=lambda group: hashlib.sha256(f"{seed}:{key}:{group}".encode()).hexdigest(),
        )
        total = sum(len(ids) for ids in groups.values())
        targets = [total * 0.9, total * 0.95]
        seen = 0
        for group in ordered:
            split = "train" if seen < targets[0] else "val" if seen < targets[1] else "test"
            assignments.update({sample_id: split for sample_id in groups[group]})
            seen += len(groups[group])
    return assignments
