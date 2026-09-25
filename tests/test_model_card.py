from __future__ import annotations

import pytest

from scripts.render_model_card import _training_status


def test_card_discloses_early_stop_and_selected_update() -> None:
    status = _training_status(
        {"selected_updates": 1000},
        {"stages": [{"updates": 1000}, {"updates": 5000}], "stop_early": True},
    )
    assert "5,000 of 8,000" in status
    assert "update 1,000" in status
    assert "stopped after three stale" in status


def test_card_rejects_unverified_early_stop() -> None:
    with pytest.raises(ValueError, match="unverified"):
        _training_status(
            {"selected_updates": 1000},
            {"stages": [{"updates": 5000}], "stop_early": False},
        )
