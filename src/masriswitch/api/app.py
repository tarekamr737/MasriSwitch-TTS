"""FastAPI service with fixed server-side reference voice."""

from __future__ import annotations

from io import BytesIO

import soundfile as sf
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from masriswitch.infer.engine import SynthesisEngine
from masriswitch.text.normalize import normalize_text


class TextRequest(BaseModel):
    text: str = Field(min_length=1, max_length=500)


def create_app(engine: SynthesisEngine | None = None) -> FastAPI:
    app = FastAPI(title="MasriSwitch-TTS", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str | bool]:
        return {"ready": engine is not None, "status": "ready" if engine else "unavailable"}

    @app.get("/model-info")
    def model_info() -> dict[str, str | bool]:
        return {
            "model_id": engine.model_id if engine else "TBD",
            "fixed_reference_only": True,
            "ai_generated_audio": True,
        }

    @app.post("/normalize")
    def normalize(request: TextRequest) -> dict[str, object]:
        result = normalize_text(request.text)
        return {
            "normalized_text": result.normalized_text,
            "entities": [entity.__dict__ for entity in result.entities],
        }

    @app.post("/synthesize")
    def synthesize(request: TextRequest) -> Response:
        if engine is None:
            raise HTTPException(status_code=503, detail="Approved voice or checkpoint unavailable")
        try:
            audio, sample_rate = engine.synthesize(request.text)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        output = BytesIO()
        sf.write(output, audio, sample_rate, format="WAV", subtype="PCM_16")
        return Response(
            output.getvalue(), media_type="audio/wav", headers={"X-AI-Generated": "true"}
        )

    return app
