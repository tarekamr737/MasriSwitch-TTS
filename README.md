# MasriSwitch-TTS

Egyptian Arabic ↔ English code-switched TTS built from [SILMA TTS v1](https://huggingface.co/silma-ai/silma-tts)
and F5-TTS 1.1.7. **Experimental:** the completed E1 fine-tune did not meet
the declared accuracy-improvement target. See [evaluation](reports/EVALUATION.md)
for measured results, confidence intervals, and failure examples.

[Model weights and card](https://huggingface.co/Tarek737/MasriSwitch-TTS) ·
[Text benchmark](https://huggingface.co/datasets/Tarek737/MasriSwitch-Bench)

## Architecture

Text → deterministic entity/number normalization → Unicode code-switch analysis
→ pinned SILMA/F5 model → 24 kHz waveform. Data takes a separate fail-closed path:
source lock → row-level license gate → audio/text quality audit → dedupe → grouped
90/5/5 split → F5 Arrow training dataset.

## Reproduction

Python 3.10 and Git are required. Keep caches on a drive with free space.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
export HF_HOME="$PWD/artifacts/hf-cache"
export HF_DATASETS_CACHE="$PWD/artifacts/hf-cache/datasets"
export TMPDIR="$PWD/artifacts/tmp"
mkdir -p "$TMPDIR"
make setup
make check
git clone --depth 1 --branch 1.1.7 https://github.com/SWivid/F5-TTS.git artifacts/upstream/f5-tts
masriswitch lock-sources
make audit-data
make prepare-data
make bootstrap
make benchmark
python -c "from masriswitch.config import Paths; from masriswitch.train.patch_f5 import stage_training_code; stage_training_code(Paths())"
python -c "from masriswitch.config import Paths; from masriswitch.eval.plan import build_e0_plan; build_e0_plan(Paths())"
```

On Windows PowerShell, activate `.venv\Scripts\Activate.ps1` and set the
equivalent variables to paths on D:. `make check` can be run as its five
commands in `Makefile` when GNU make is unavailable.

Training runs in a private two-T4 Kaggle notebook. Package the source with
`python scripts/build_kaggle_bundle.py --stage pilot` or `--stage e1`, attach a
signed private archive URL as
`MASRISWITCH_ARCHIVE_URL`, then use `make train-pilot` or `make train` in that
runtime. `make train` is capped at 1,000 updates by default; use
`masriswitch train --max-updates N` for later bounded stages. Resume requires
`--resume`, `MASRISWITCH_RESUME_CHECKPOINT_URL`, and
`MASRISWITCH_RESUME_CHECKPOINT_SHA256`. The matching archive, patch, pilot, and
full-data probe manifests must be present under `artifacts/`. Both commands
support `--dry-run`, `--max-samples`, and `--seed`; `make eval` aggregates pinned
evaluation rows. No signed URLs or checkpoint bytes belong in the repository.

### First-run Kaggle preparation

Create the audited archives on the machine holding the prepared audio:

```bash
python scripts/archive_pilot.py --root . --output artifacts/pilot.tar.gz
cp artifacts/pilot.tar.json artifacts/kaggle_pilot_archive.json
python scripts/archive_full.py --root . --output artifacts/full_train.tar.gz --dry-run
python scripts/archive_full.py --root . --output artifacts/full_train.tar.gz
cp artifacts/full_train.tar.json artifacts/kaggle_full_archive.json
python scripts/build_kaggle_bundle.py --stage probe --output artifacts/probe.zip
```

Keep archives private. Transfer the probe bundle to a fresh private Kaggle
T4×2 session, extract it, and set `MASRISWITCH_ARCHIVE_URL` to the pilot
archive's private download URL. From the extracted project, run:

```bash
python -m pip install --no-deps -e .
PILOT_SHA=$(python -c "import json; print(json.load(open('artifacts/kaggle_pilot_archive.json'))['archive_sha256'])")
ARROW_SHA=$(python -c "import json; print(json.load(open('artifacts/kaggle_pilot_archive.json'))['pilot_arrow_sha256'])")
python scripts/kaggle_train_probe.py --root . \
  --archive-url "$MASRISWITCH_ARCHIVE_URL" --archive-sha256 "$PILOT_SHA" \
  --arrow-sha256 "$ARROW_SHA" --archive-manifest artifacts/kaggle_pilot_archive.json \
  --patch-manifest artifacts/train_patch.json --max-samples 343 --seed 42 --dry-run
# Repeat without --dry-run to perform the bounded 20-update probe.
```

Download `/kaggle/working/masriswitch_probe_result.json` as local
`artifacts/train_probe.json`. Only after it passes, build `--stage pilot` and
run `masriswitch train-pilot --dry-run`, then `masriswitch train-pilot` in a
fresh private session, installing the extracted project there with the same
`python -m pip install --no-deps -e .` command. Save its
`masriswitch_pilot_result.json` as `artifacts/pilot_500_result.json`.

Next build `--stage full-probe`, use the full archive URL, and run:

```bash
FULL_SHA=$(python -c "import json; print(json.load(open('artifacts/kaggle_full_archive.json'))['archive_sha256'])")
ARROW_SHA=$(python -c "import json; print(json.load(open('artifacts/kaggle_full_archive.json'))['train_arrow_sha256'])")
python scripts/kaggle_full_probe.py --root . \
  --archive-url "$MASRISWITCH_ARCHIVE_URL" --archive-sha256 "$FULL_SHA" \
  --arrow-sha256 "$ARROW_SHA" --archive-manifest artifacts/kaggle_full_archive.json \
  --patch-manifest artifacts/train_patch.json --probe-result artifacts/train_probe.json \
  --pilot-result artifacts/pilot_500_result.json --max-samples 3414 --seed 42 --dry-run
# Repeat without --dry-run for the bounded full-data probe.
```

Save `masriswitch_full_probe_result.json` as `artifacts/full_probe_result.json` before building
`--stage e1`. Run training in 1,000-update stages and resume from the previous
checkpoint URL and verified hash. Each probe/training session supplies its
own Python 3.10 environment; install this project before using its CLI.
Stop on failed probes, invalid audio, or three stale validation checks.

For the selected E1 evaluation, build an `--stage eval` bundle and extract it
in a private Kaggle T4×2 session. Set `E1_CHECKPOINT_URL` to the selected
checkpoint's private output URL, then run from the extracted project:

```bash
python scripts/kaggle_e0_eval.py \
  --plan artifacts/e0_eval_plan.jsonl \
  --plan-sha256 26dc087cabc32544c401eebcaaf804185901cbd829358f6c795e0d232ced9185 \
  --eval-stage E1 --subset all --max-samples 489 --seed 42 \
  --checkpoint-url "$E1_CHECKPOINT_URL" \
  --checkpoint-sha256 558e2ab53e1b5450bcd1a1c30683a1b3e6be1234b7e198b74209f362693eaf92 \
  --dry-run
# Repeat the same command without --dry-run after the plan check passes.
# Add --resume only when resuming matching, persisted output rows.
```

For E0, use the same evaluation command with `--eval-stage E0` and omit both
checkpoint arguments. The evaluator downloads the pinned untouched SILMA
weights. Download `e0_all_eval_rows.jsonl` as `artifacts/e0_eval_rows.jsonl`
and run `make eval` to aggregate the baseline before training.

After downloading `e1_all_eval_rows.jsonl` into local `artifacts/`:

```bash
masriswitch eval --stage E1 --subset all --max-samples 489 \
  --rows artifacts/e1_all_eval_rows.jsonl \
  --checkpoint-sha256 558e2ab53e1b5450bcd1a1c30683a1b3e6be1234b7e198b74209f362693eaf92
python scripts/finalize_training.py --result artifacts/e1_2000_result.json \
  --checkpoint artifacts/checkpoints/e1_2000.pt
python scripts/finalize_evaluation.py \
  --e1-metrics artifacts/e1_all_558e2ab53e1b_metrics.json \
  --e1-rows artifacts/e1_all_eval_rows.jsonl
python scripts/render_model_card.py
make release-check
make release
```

The finalizer requires the verified stage result, audited archive manifest,
and validation-only `checkpoint_selection.json`. These are generated by the
training, `review_e1_progress.py`, and `select_e1_checkpoint.py` stages; they
must accompany a resumed run. The commands above describe the selected run
and will fail closed if its evidence or hashes are missing.

## Measured evidence

At pinned D1 revision `eae9a87c17e91e3f59a9696d5f4ff3eb51502e82`, the
source audit read 9,617 rows: 3,944 author-created `cc-by-4.0` rows passed the
license/provenance filter; 5,673 `cc-by-nc-sa-4.0` rows were excluded.
The [published MasriSwitch-Bench v1](https://huggingface.co/datasets/Tarek737/MasriSwitch-Bench)
contains 1,200 text-only prompts with 300 locked test IDs. The downloaded
JSONL SHA256 matches the local manifest.

| Experiment | Code-switch WER | English EER | Arabic CER | Status |
|---|---:|---:|---:|---|
| E0 SILMA baseline | 61.38% | 51.60% | 38.57% | 489 prompts measured |
| E1 fine-tune | 61.34% | 52.53% | 38.03% | 489 prompts measured; update 2,000 selected |
| E2 replay | TBD | TBD | TBD | Optional |

E1 reduced code-switch WER by only 0.06% relative; English EER worsened by
1.80% relative. It missed the target of ≥15% WER or ≥25% EER reduction.
Arabic-only CER and latency guardrails passed, with zero invalid audio.
The selected model also passed ten private FastAPI synthesis requests and a
Gradio generation callback. A user-provided fixed voice now has documented
public demo consent; ten API requests and a Gradio callback passed using it
with a freshly downloaded, hash-verified HF release. The recording and consent
record are kept in ignored artifacts and are not distributed with the code.

## Licenses and limitations

Repository code: Apache-2.0. SILMA model weights: Apache-2.0; SILMA source
code: MIT. F5-TTS 1.1.7 code: MIT. Eligible D1 rows: CC BY 4.0, attributed
to Abdelrahman R. Hashem; the mixed-license dataset as a whole is **not** a
release-safe training source. See [NOTICE](NOTICE) and
[data audit](reports/DATA_AUDIT.md).

D1 is synthetic speech and is not evidence of real Egyptian speaker or acoustic
diversity. No human naturalness study or real-speech external evaluation exists
yet. Benchmark prompts are templated and text only. Audio must be disclosed as
AI-generated. Public synthesis accepts only a fixed server-side voice with
documented consent; arbitrary voice uploads are not supported. SILMA's example
audio is limited to private smoke tests because its speaker consent for public
reuse was not documented.

The 1,000- through 5,000-update E1 checkpoints passed exact-update and private
inference checks. Their 50-prompt validation results are interim and are not
the locked benchmark result. The 3,000-, 4,000-, and 5,000-update checks did
not improve the eligible primary metrics, so the declared early-stop rule ended
training at 5,000 updates. The 1,000-, 2,000-, and 3,000-update checkpoints
were compared on all 189 validation prompts. Update 2,000 won by English
EER (36.39%), with code-switch WER 45.47% and overall Arabic CER 37.23%.
These validation values are separate from the final results table above.

The release gate blocks weight publication until source hashes, measured E0/E1
results, checkpoint smoke tests, attribution, and policy checks pass.
After the gate passes, `make release` builds a hash-checked local bundle under
ignored `artifacts/release_bundle/` for review and upload.

## Local API and demo

### Browser demo

Open [MasriSwitch-TTS Demo](https://huggingface.co/spaces/Tarek737/MasriSwitch-TTS-Demo).
When its status is Running, enter a short sentence (up to 200 characters),
click **Generate speech**, wait for the free GPU queue, and play/download the WAV.
Try `ال order جاهز للتوصيل.` or `ممكن تعمل reset لل password؟`.
Audio is AI-generated using a fixed, consented voice. Pronunciation and numbers
can be wrong; the model is experimental. Free GPU quotas apply.

The Space uses PyTorch/torchaudio 2.8.0 for ZeroGPU compatibility; benchmark
results were measured with 2.6.0. `deploy/space/` contains the pinned runtime
and app. `scripts/deploy_space.py --dry-run` builds an allowlisted source bundle;
deployment requires `HF_TOKEN` and the local approved reference. Voice bytes are
stored as runtime secrets and are never uploaded as repository files.
See [quality review](reports/QUALITY_REVIEW.md) for the corrected ID-normalization
bug and the decision to gather listening feedback before another training run.

### Local setup

Install the serving and inference dependencies into the Python 3.10 environment:

```bash
python -m pip install -e ".[train,serve]"
make bootstrap
make api   # http://127.0.0.1:8000/docs
# In another terminal using the same environment:
make demo  # http://127.0.0.1:7860
```

On Windows, use `masriswitch api` and `masriswitch demo`. Before installing or
downloading, keep `HF_HOME`, `HF_DATASETS_CACHE`, `PIP_CACHE_DIR`, `TMP`, and
`TEMP` under `D:\MasriSwitch-TTS\artifacts\` as shown in `.env.example`.
Set these variables in the shell; `.env.example` is documentation and is not
loaded automatically.

Synthesis requires `artifacts/train_manifest.json`, its selected checkpoint,
the bootstrapped SILMA/Vocos files, and a consented fixed reference recording.
Put that recording in `artifacts/reference/`, then create `approved.json`
beside it with the following fields, using the recording's real SHA256 and
exact transcript. Set the approval flags to true only after documenting the
speaker's consent for public use:

```json
{
  "filename": "consented.wav",
  "sha256": "REPLACE_WITH_RECORDING_SHA256",
  "transcript": "REPLACE_WITH_EXACT_TRANSCRIPT",
  "speaker_consent_documented": false,
  "public_use_approved": false
}
```

Until that approval exists, `/health` reports unavailable, `/normalize` works,
and `/synthesize` returns 503. `/model-info` identifies the model and discloses
AI-generated audio. The Gradio demo accepts text only and keeps generation
disabled. The upstream example voice is authorized for private tests only.
