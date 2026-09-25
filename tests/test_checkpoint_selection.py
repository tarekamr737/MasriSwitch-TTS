from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scripts.select_e1_checkpoint import main


def _inputs(root: Path) -> tuple[Path, Path]:
    artifacts = root / "artifacts"
    artifacts.mkdir()
    (artifacts / "e0_validation_metrics.json").write_text(
        json.dumps(
            {
                "stage": "E0",
                "subset": "validation",
                "overall": {"count": 189},
                "plan_sha256": "plan",
                "metrics": {"overall_ar_cer": 0.35},
            }
        ),
        encoding="utf-8",
    )
    (artifacts / "e1_progress.json").write_text(
        json.dumps(
            {
                "locked_benchmark_used": False,
                "selection_subset": "first 50 frozen validation prompts",
                "shortlist_checkpoint_sha256": ["a" * 64, "b" * 64],
            }
        ),
        encoding="utf-8",
    )
    paths = (artifacts / "candidate_a.json", artifacts / "candidate_b.json")
    for path, sha, eer, wer, cer in (
        (paths[0], "a" * 64, 0.30, 0.40, 0.35),
        (paths[1], "b" * 64, 0.20, 0.30, 0.38),
    ):
        path.write_text(
            json.dumps(
                {
                    "stage": "E1",
                    "subset": "validation",
                    "overall": {"count": 189},
                    "plan_sha256": "plan",
                    "model_sha256": sha,
                    "metrics": {
                        "english_eer": eer,
                        "cs_wer": wer,
                        "overall_ar_cer": cer,
                    },
                }
            ),
            encoding="utf-8",
        )
    return paths


def test_selection_requires_full_shortlist_and_arabic_guardrail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate_a, candidate_b = _inputs(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["select_e1_checkpoint", "--candidate", str(candidate_a), "--candidate", str(candidate_b)],
    )
    main()
    selected = json.loads((tmp_path / "artifacts/checkpoint_selection.json").read_text())
    assert selected["selected_checkpoint_sha256"] == "a" * 64
    monkeypatch.setattr(sys, "argv", ["select_e1_checkpoint", "--candidate", str(candidate_a)])
    with pytest.raises(ValueError, match="frozen 50-prompt shortlist"):
        main()
