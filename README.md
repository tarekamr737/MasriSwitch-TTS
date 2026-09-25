# MasriSwitch-TTS

Egyptian Arabic ↔ English code-switched TTS built from [SILMA TTS v1](https://huggingface.co/silma-ai/silma-tts)
and F5-TTS 1.1.7. This repository is under active development; no fine-tuned
weights or performance claims are published yet.

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
masriswitch lock-sources
make audit-data
make prepare-data
make benchmark
make release-check
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
| E1 fine-tune | TBD | TBD | TBD | 3,000/8,000 updates verified; interim 50-prompt validation measured |
| E2 replay | TBD | TBD | TBD | Optional |

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

The 1,000-, 2,000-, and 3,000-update E1 checkpoints passed exact-update and
private inference checks. Their 50-prompt validation results are interim and
are not the locked benchmark result. The 3,000-update check worsened on both
English EER and code-switch WER relative to the previous E1 best; the declared
early-stop rule has not fired.

The release gate blocks weight publication until source hashes, measured E0/E1
results, checkpoint smoke tests, attribution, and policy checks pass.
After the gate passes, `make release` builds a hash-checked local bundle under
ignored `artifacts/release_bundle/` for review and upload.
