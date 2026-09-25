# TASKS.md — MasriSwitch-TTS

Execute top-to-bottom. Keep each item tiny; mark `[x]` only with evidence.

## M0 — Scaffold
- [x] Create package/layout, `pyproject.toml`, Makefile, LICENSE, NOTICE, `.gitignore`, `.env.example` (files present).
- [x] Add config schemas + CLI skeleton (`masriswitch validate-config` passed).
- [x] Add ruff/mypy/pytest and make `make check` green (equivalent venv commands passed: ruff, mypy, 127 tests, config validation; GNU make is unavailable on this Windows host).

## M1 — Sources + data
- [x] Implement source registry + fail-closed license gate (`tests/test_pipeline.py`).
- [x] Pin SILMA model files/revision/hashes and F5-TTS `1.1.7` (`artifacts/source_lock.json`; both local model byte hashes verified).
- [x] Load D1 and assert `license == cc-by-4.0`; report actual selected rows (`artifacts/data_audit.json`: 3,944 eligible / 9,617 total).
- [x] Implement audio/text audit and write `DATA_AUDIT.md` (`reports/DATA_AUDIT.md`: 3,786 final clips, measured filters).
- [x] Implement clean/dedupe/90-5-5 stratified split (`artifacts/splits.json`: 3,414/189/183; speaker IDs unavailable).
- [x] Export F5-compatible training metadata (reloadable full/pilot Arrow files: 3,414/343 rows).

## M2 — Text intelligence
- [x] Implement code-switch span/bucket detector (`tests/test_pipeline.py`).
- [x] Implement deterministic entity parser + text normalizer (`src/masriswitch/text/`, golden tests).
- [x] Add ≥100 normalization golden tests (`tests/test_normalize.py`: 100 cases).
- [x] Generate MasriSwitch-Bench v1 (1,200 prompts) + locked 300 test IDs (`artifacts/benchmark_manifest.json`).

## M3 — Baseline

Status 2026-09-25: pinned SILMA/Vocos model hashes verified locally on D and
in private Kaggle CPU output. Kaggle Python 3.10/F5 1.1.7 import preflight passed.
The complete private 489-prompt E0 evaluation produced finite 24 kHz audio
(`artifacts/e0_eval_rows.jsonl`, `artifacts/baseline_metrics.json`).
- [x] Bootstrap SILMA/F5 environment from pinned artifacts.
- [x] Run E0 on validation + locked benchmark (189 + 300 prompts; `artifacts/e0_eval_rows.jsonl`).
- [x] Implement WER/CER, EER, CEA, speaker-similarity, latency/RTF metrics (`eval/metrics.py`, `eval/aggregate.py`; 5-prompt all-metric preflight and unit tests passed).
- [x] Save `baseline_metrics.json` + `BASELINE.md` (code-switch WER 61.38%, English EER 51.60%, Arabic-only CER 38.57%).

## M4 — Training
- [x] Implement Kaggle T4x2 batch auto-probe + resumable state (`artifacts/train_probe.json`: 20 finite updates per rank at 5,600 frames/GPU, ≥5.76 GiB free; Kaggle private probe v5).
- [x] Run 500-update pilot; verify checkpoint reload/inference (`artifacts/pilot_500_result.json`: both ranks reached 500 updates; 10 private inference prompts passed).
- [ ] Run E1 to target 8k updates with periodic eval (1,000, 2,000, and 3,000-update checkpoints and 50-prompt validation measured; the 3,000-update check has one stale evaluation, below the three-evaluation early-stop threshold; bounded 4,000-update train/validation stage running).
- [ ] Select best E1 checkpoint by declared metric order.
- [ ] If compute permits, run E2 with 10% Common Voice replay.

E2 is held at the source gate: `common_voice_ar` is disabled and its exact
revision/terms are still `TBD` in `configs/sources.yaml`. No Common Voice audio
has been downloaded or trained on.

The private 3,000-update Kaggle session completed after the user renewed tool
access. Both ranks reached exactly 3,000 updates, the 3,414 audited train IDs
matched, and 10 private checkpoint reload/inference prompts passed
(`artifacts/e1_3000_result.json`). Its 50-prompt validation had zero invalid
outputs (`artifacts/e1_validation_c3eac52f7456_metrics.json`). The prior
notebook redirected training logs to a Kaggle file. The next-stage bundle adds
minute-by-minute update logging, download progress, and a 20-minute SILMA
model download cap.

## M5 — Product
- [ ] Build typed inference engine around best checkpoint.
- [ ] Add FastAPI health/model-info/normalize/synthesize.
- [ ] Add minimal Gradio demo with fixed approved reference voice.
- [ ] Add Dockerfile/local run instructions only if they stay lightweight.

## M6 — Evaluation + release
- [ ] Run final locked evaluation E0 vs E1 (+E2 if available).
- [ ] Write `EVALUATION.md` with real tables/CIs and failure examples.
- [ ] Implement `release-check`; verify no blocked data touched training.
- [ ] Create GitHub README with exact reproduce/Kaggle commands.
- [ ] Generate HF model card, benchmark data card, NOTICE/attributions.
- [x] Publish text-only benchmark (`https://huggingface.co/datasets/Tarek737/MasriSwitch-Bench`; downloaded SHA256 matched local manifest; `artifacts/hf_benchmark_release.json`).
- [ ] Publish weights only if release gate passes.
- [ ] Smoke-test downloaded HF artifact from a clean environment.

GitHub publication is currently gated by an invalid local `gh` token for
`tarekamr737` (`gh auth status`, 2026-09-25). Code and release artifacts can
continue to be prepared on D before authentication is renewed.
