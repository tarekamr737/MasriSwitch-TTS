# CPU check evidence

Local Windows virtual environment: Python 3.10.20 in `D:\MasriSwitch-TTS\artifacts\python`.
Dependencies and caches are under `D:\MasriSwitch-TTS`.

2026-09-25: `ruff format --check src tests scripts`, `ruff check src tests scripts`,
`mypy src`, `pytest -q`, and `masriswitch validate-config` passed; 136 tests.
The added inference tests verify that a private reference and a path escaping
`artifacts/reference` cannot enable public synthesis.
The disabled Gradio demo constructed successfully (`Blocks`, six components).
`release-check` failed closed as expected while the selected train manifest,
final E1 evaluation, and completed model card are missing.
The selected-checkpoint finalizer tests cover local checkpoint hash binding,
metric-selection identity, and the complete audited train-ID set.
The release bundle now records `model.pt` as a portable relative checkpoint
path in `training_args.json`; the bundled model bytes remain hash-checked.
The model card renderer now states both the actual training endpoint and the
selected checkpoint update, and requires verified early-stop evidence if the
run ends before 8,000 updates.
GNU make is not installed
on this host, so the equivalent commands in the Makefile were run directly.

The 20-row preparation probe exported a reloadable F5 Arrow dataset. Full
preparation passed: 3,786 final clips, with 3,414 train, 189 validation, and
183 test rows. The 343-row pilot Arrow file reloads and contains only train
sample IDs. The two-T4 probe completed 20 finite updates per rank with at
least 5.76 GiB free. The pilot completed 500 updates per rank via a 250-update
resume, and its saved checkpoint passed 10 private inference smoke prompts.
The pilot checkpoint SHA256 is
`163987cddd8cbed5916d85228234fd42e7605fd9d6237fff098fd8afc5b44ec2`.
The Kaggle full training archive has 3,414 train-only rows and the same sorted
train-ID hash as the local archive (`artifacts/kaggle_full_archive.json`).
The bounded full-data probe completed 20 finite updates per GPU at 5,600
frames/GPU, with at least 6.12 GiB free (`artifacts/full_probe_result.json`).
The first E1 stage completed exactly 1,000 updates per GPU and passed 10
checkpoint reload/inference prompts (`artifacts/e1_1000_result.json`). Its
checkpoint SHA256 is
`377ae2f8a9cfc949544e8eddd743fd3ddebce872bea6b10a14170d6ebd3a49b4`.
The 50-prompt E1 validation check completed with zero invalid outputs.
English EER was 32.63% versus E0 37.89%; code-switch WER was 44.48% versus
E0 43.43%; overall Arabic CER was 35.12% versus E0 34.88% on those same
prompts (`artifacts/e1_validation_377ae2f8a9cf_metrics.json`). A private
resume to 2,000 updates completed and passed the same exact-update, train-ID,
and 10-prompt checkpoint checks (`artifacts/e1_2000_result.json`). Its SHA256 is
`558e2ab53e1b5450bcd1a1c30683a1b3e6be1234b7e198b74209f362693eaf92`.
The first 2,000-update validation session stalled after dependency setup and
was cancelled; it produced no evaluation rows. The bounded retry completed
all 50 prompts with zero invalid audio. At 2,000 updates English EER was
34.74%, code-switch WER 43.43%, and overall Arabic CER 35.26% on the same
validation subset (`artifacts/e1_validation_558e2ab53e1b_metrics.json`).
The 3,000-update resume now runs training and local-checkpoint validation in
one private Kaggle notebook, avoiding a second checkpoint transfer.
That notebook completed: both ranks reached exactly 3,000 updates, all 3,414
train IDs matched the audit, ten private checkpoint reload/inference prompts
passed, and 50 validation prompts produced zero invalid outputs. Its checkpoint
SHA256 is `c3eac52f7456d78367d4bec8209ab5c7cd2876226c0a4dc6a4ffdc25993c08e4`.
English EER was 40.00%, code-switch WER 43.58%, and overall Arabic CER 35.45%
(`artifacts/e1_validation_c3eac52f7456_metrics.json`). The bounded early-stop
review has one stale evaluation and does not stop E1 yet.
The 4,000-update stage also passed exact-update, audited train-ID, and
ten-prompt private checkpoint checks. Its checkpoint SHA256 is
`02f0a3ab57ff5489ee7eb23438a1e8b6477564f12b34ed69b67797c3cb1648d6`.
All 50 validation outputs were valid. English EER was 34.74%, code-switch WER
44.03%, and overall Arabic CER 37.91%; the Arabic guardrail fails on this
subset. The early-stop review now records two consecutive stale checks.

Kaggle CPU Python 3.10.20 environment imported F5-TTS 1.1.7, PyTorch
2.6.0+cu124, torchaudio 2.6.0, Transformers 4.46.3, Accelerate 1.15.0,
and Datasets 3.6.0. The pinned SILMA/Vocos byte hashes were verified on
Kaggle and again under D. Private T4 E0 completed all 489 prompts;
`artifacts/baseline_metrics.json` and `reports/BASELINE.md` hold the values.
