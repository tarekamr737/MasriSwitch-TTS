"""Typed F5 engine backed by a selected, hash-checked local checkpoint."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import soundfile as sf

from masriswitch.config import Paths, load_yaml
from masriswitch.data.audit import sha256_file
from masriswitch.infer.chunks import speech_chunks
from masriswitch.text.normalize import normalize_text


class SynthesisEngine(Protocol):
    model_id: str

    def synthesize(self, text: str) -> tuple[np.ndarray, int]: ...


@dataclass(frozen=True)
class EngineFiles:
    checkpoint: Path
    checkpoint_sha256: str
    vocab: Path
    silma_config: Path
    vocoder_dir: Path
    reference_audio: Path
    reference_sha256: str
    reference_text: str
    model_id: str

    def validate(self) -> None:
        for path in (
            self.checkpoint,
            self.vocab,
            self.silma_config,
            self.vocoder_dir / "config.yaml",
            self.vocoder_dir / "pytorch_model.bin",
            self.reference_audio,
        ):
            if not path.is_file():
                raise FileNotFoundError(path)
        if sha256_file(self.checkpoint) != self.checkpoint_sha256:
            raise ValueError("Checkpoint hash mismatch")
        if sha256_file(self.reference_audio) != self.reference_sha256:
            raise ValueError("Reference voice hash mismatch")
        if not self.reference_text.strip():
            raise ValueError("Reference transcript is required")


def approved_engine_files(paths: Paths) -> EngineFiles:
    approval_path = paths.artifacts / "reference" / "approved.json"
    manifest_path = paths.artifacts / "train_manifest.json"
    if not approval_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("Approved reference and trained checkpoint are required")
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        approval.get("speaker_consent_documented") is not True
        or approval.get("public_use_approved") is not True
    ):
        raise PermissionError("Reference voice is not approved for public synthesis")
    reference = (approval_path.parent / str(approval["filename"])).resolve()
    if reference.parent != approval_path.parent.resolve():
        raise ValueError("Reference path must remain inside artifacts/reference")
    checkpoint = Path(str(manifest["best_checkpoint"])).resolve()
    if not checkpoint.is_relative_to(paths.artifacts.resolve()):
        raise ValueError("Checkpoint must remain inside ignored artifacts")
    files = EngineFiles(
        checkpoint=checkpoint,
        checkpoint_sha256=str(manifest["checkpoint_sha256"]),
        vocab=paths.artifacts / "upstream" / "silma" / "vocab.txt",
        silma_config=paths.artifacts / "upstream" / "silma" / "config.yaml",
        vocoder_dir=paths.artifacts / "upstream" / "vocos",
        reference_audio=reference,
        reference_sha256=str(approval["sha256"]),
        reference_text=str(approval["transcript"]),
        model_id=str(manifest["experiment"]),
    )
    files.validate()
    return files


class F5Engine:
    def __init__(
        self, files: EngineFiles, *, seed: int = 42, nfe_steps: int = 16, device: str | None = None
    ) -> None:
        files.validate()
        self.model_id = files.model_id
        self.files = files
        self.seed = seed
        self.nfe_steps = nfe_steps
        self._device = device
        duration = sf.info(files.reference_audio).duration
        if not 0 < duration < 12:
            raise ValueError("Reference duration must be between zero and twelve seconds")
        self._chunk_bytes = min(160, int(len(files.reference_text.encode("utf-8")) / duration * 6))
        from f5_tts.infer.utils_infer import infer_process, load_model, load_vocoder
        from f5_tts.model import DiT

        self._infer_process = infer_process
        config = load_yaml(files.silma_config)
        device_options = {"device": device} if device else {}
        self._vocoder = load_vocoder(
            "vocos", is_local=True, local_path=str(files.vocoder_dir), **device_options
        )
        self._model = load_model(
            DiT,
            config["model"]["arch"],
            str(files.checkpoint),
            mel_spec_type="vocos",
            vocab_file=str(files.vocab),
            use_ema=True,
            **device_options,
        )

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        if len(text) > 500:
            raise ValueError("Text must contain at most 500 characters")
        normalized = normalize_text(text).normalized_text
        chunks = speech_chunks(normalized, max_bytes=self._chunk_bytes)
        import torch

        waves: list[np.ndarray] = []
        sample_rate = 24000
        for index, chunk in enumerate(chunks):
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            audio, rate, _ = self._infer_process(
                str(self.files.reference_audio),
                self.files.reference_text,
                chunk,
                self._model,
                self._vocoder,
                nfe_step=self.nfe_steps,
                cross_fade_duration=0,
                **({"device": self._device} if self._device else {}),
            )
            if (
                audio is None
                or not np.isfinite(audio).all()
                or len(audio) == 0
                or int(rate) != sample_rate
            ):
                raise RuntimeError("Synthesis returned invalid audio")
            if index:
                waves.append(np.zeros(int(sample_rate * 0.12), dtype=np.float32))
            waves.append(np.asarray(audio, dtype=np.float32))
        return np.concatenate(waves), sample_rate


def sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()
