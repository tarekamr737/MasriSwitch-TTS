from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from masriswitch.api.app import create_app


class FakeEngine:
    model_id = "test-checkpoint"

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        assert text == "أهلا API"
        return np.zeros(2400, dtype=np.float32), 24000


def test_api_requires_approved_engine() -> None:
    client = TestClient(create_app())
    assert client.get("/health").json()["ready"] is False
    assert client.get("/model-info").json()["fixed_reference_only"] is True
    assert client.post("/synthesize", json={"text": "أهلا"}).status_code == 503
    assert client.post("/normalize", json={"text": ""}).status_code == 422
    assert client.post("/synthesize", json={"text": "x" * 501}).status_code == 422


def test_api_fixed_engine_wav_and_normalization() -> None:
    client = TestClient(create_app(FakeEngine()))
    normalized = client.post("/normalize", json={"text": "ابعت OTP"})
    assert normalized.status_code == 200
    assert normalized.json()["normalized_text"] == "ابعت أو تي بي"
    response = client.post("/synthesize", json={"text": "أهلا API"})
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.headers["x-ai-generated"] == "true"
    assert response.content[:4] == b"RIFF"
