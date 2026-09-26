# TASKS.md — MasriSwitch-TTS

Execute top-to-bottom. Keep each item tiny; mark `[x]` only with evidence.

## M0 — Scaffold
- [x] Create package/layout, `pyproject.toml`, Makefile, LICENSE, NOTICE, `.gitignore`, `.env.example` (files present).
- [x] Add config schemas + CLI skeleton (`masriswitch validate-config` passed).
- [x] Add ruff/mypy/pytest and make `make check` green (equivalent venv commands passed: ruff, mypy, 140 tests, config validation; GNU make is unavailable on this Windows host).

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
- [x] Run E1 to target 8k updates with periodic eval (the declared three-stale-check early stop fired at 5,000 updates; all 1,000 through 5,000-update stages and their 50-prompt validations were verified in `artifacts/e1_progress.json`).
- [x] Select best E1 checkpoint by declared metric order (`artifacts/checkpoint_selection.json`: update 2,000 won the complete 189-prompt validation comparison; locked benchmark excluded).
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
The 4,000-update stage also passed both-rank exact updates, audited train IDs,
and ten private synthesis prompts (`artifacts/e1_4000_result.json`). Its
50-prompt validation had zero invalid audio; English EER 34.74%, code-switch
WER 44.03%, and overall Arabic CER 37.91%. This exceeds the 5% Arabic CER
guardrail relative to the same E0 subset and is the second stale check.
The first 5,000-update submission was cancelled before training because its
resume URL and expected checkpoint hash did not match. A corrected private
session (`tarekamr/masriswitch-e1-5000-corrected`) completed with the verified
4,000-update checkpoint hash; no invalid-resume training occurred. The 5,000
checkpoint also passed exact updates on both ranks, audited train IDs, and ten
private synthesis prompts. Its 50-prompt English EER was 32.63%, code-switch
WER 44.18%, and overall Arabic CER 37.01%; the Arabic guardrail failed. The
third stale validation triggered the stop rule. Full 189-prompt validation of
the frozen 1,000/2,000/3,000 short list completed. The 2,000-update checkpoint
won by English EER (36.39%), followed by code-switch WER (45.47%); overall
Arabic CER (37.23%) passed the relative 5% guardrail. Full metrics have the
`_189_metrics.json` suffix; the original 50-prompt screening files are preserved.
The selected checkpoint downloaded directly to D and its SHA256 matched.
`artifacts/train_manifest.json` binds the verified checkpoint to the audited
training data, update 2,000 selection, and update 5,000 early-stop endpoint.
Final 489-prompt evaluation and a private API/Gradio smoke completed in
`tarekamr/masriswitch-e1-final-489-product-smoke`. All 489 evaluation outputs,
ten FastAPI WAV responses, and one Gradio callback were valid. E1 did not meet
the accuracy target: code-switch WER 61.34%, English EER 52.53%, Arabic-only
CER 38.03%. Reports and the model card disclose the experimental outcome.

## M5 — Product
- [x] Build typed inference engine around best checkpoint (`artifacts/product_smoke.json`: selected checkpoint hash verified; real private inference passed).
- [x] Add FastAPI health/model-info/normalize/synthesize (unit tests and ten real private 24 kHz WAV requests passed).
- [x] Add minimal Gradio demo with fixed approved reference voice (`artifacts/approved_voice_product_smoke.json`: user consent documented; ten real API requests and one Gradio callback passed with the approved reference).
- [x] Add lightweight local run instructions (`README.md`: API/demo commands, D-drive caches, and approved-reference configuration).

## M6 — Evaluation + release
- [x] Run final locked evaluation E0 vs E1 (`artifacts/eval_metrics.json`: same 489-prompt plan, including 300 locked prompts, zero invalid outputs).
- [x] Write `EVALUATION.md` with real tables/CIs and failure examples (includes validation-only selection table and the missed accuracy target).
- [x] Implement `release-check`; verify no blocked data touched training (`artifacts/release_gate.json`: passed; exactly the 3,414 audited D1 train IDs).
- [x] Create GitHub README with exact reproduce/Kaggle commands (fresh probe stages tested; measured final table included).
- [x] Generate HF model card, benchmark data card, NOTICE/attributions (model card generated from measured artifacts; benchmark card already published).
- [x] Publish text-only benchmark (`https://huggingface.co/datasets/Tarek737/MasriSwitch-Bench`; downloaded SHA256 matched local manifest; `artifacts/hf_benchmark_release.json`).
- [x] Publish weights only if release gate passes (`Tarek737/MasriSwitch-TTS`, commit `de6cd6819e719876909437cc33ca7086cff369c9`; remote checkpoint SHA256 matches the selected model).
- [x] Smoke-test downloaded HF artifact from a clean environment (`artifacts/hf_release_smoke.json`: immutable release, all 13 file hashes, ten API requests and one Gradio callback passed in fresh Kaggle Python 3.10).

Authentication was verified on 2026-09-26: Hugging Face `Tarek737` has repository
write permission and GitHub `tarekamr737` has a valid login. The active HF
credential is available under the ignored D-drive cache. The code and measured
reports are public at `https://github.com/tarekamr737/MasriSwitch-TTS`.
The experimental model is published at `https://huggingface.co/Tarek737/MasriSwitch-TTS`.
Its checkpoint SHA256 matches the selected model. A fresh private Kaggle
session (`tarekamr/masriswitch-hf-release-smoke`, v2) verified all 13 release
files at the immutable published commit and passed ten API prompts plus one
Gradio callback using the approved user reference. The dry-run, stored source,
and downloaded output checks passed. No further training is running.

The user supplied their own 8.4935-second recording and explicitly confirmed
public demo consent and the exact transcript on 2026-09-26. The mono 24 kHz
PCM16 reference and approval evidence are under ignored `artifacts/reference/`.
Audio checks and the real local checkpoint/reference configuration passed.
The upstream sample remains private-test-only. The bounded inference check with
the new approved voice passed in release-smoke v2. Version 1 verified all
13 published files but failed before synthesis on a datasets/pyarrow import
incompatibility; the training extra now pins the previously working datasets
3.6.0 version. No additional training is needed.
