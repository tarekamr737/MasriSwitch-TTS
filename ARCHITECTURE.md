# ARCHITECTURE.md — MasriSwitch-TTS

## 1. System flow
```text
Raw text
  -> Text Normalizer / Entity Parser
  -> language-span + code-switch analysis
  -> SILMA/F5 inference
  -> 24 kHz waveform
  -> API/CLI/Gradio
```

Training:
```text
Source registry -> license gate -> audit -> clean -> dedupe -> stratified split
-> F5 dataset prep -> SILMA checkpoint -> pilot -> full fine-tune
-> checkpoint evaluation -> best-model selection -> release gate
```

## 2. Repository layout
```text
.
├── AGENTS.md
├── PRODUCT.md
├── ARCHITECTURE.md
├── TASKS.md
├── README.md
├── LICENSE
├── NOTICE
├── Makefile
├── pyproject.toml
├── .env.example
├── .gitignore
├── configs/
│   ├── sources.yaml
│   ├── data.yaml
│   ├── train_pilot.yaml
│   ├── train_e1.yaml
│   ├── train_e2.yaml
│   └── eval.yaml
├── src/masriswitch/
│   ├── cli.py
│   ├── config.py
│   ├── data/
│   │   ├── registry.py
│   │   ├── audit.py
│   │   ├── clean.py
│   │   ├── dedupe.py
│   │   ├── split.py
│   │   └── prepare_f5.py
│   ├── text/
│   │   ├── normalize.py
│   │   ├── entities.py
│   │   └── spans.py
│   ├── train/
│   │   ├── bootstrap_silma.py
│   │   ├── plan.py
│   │   └── run.py
│   ├── infer/
│   │   ├── engine.py
│   │   └── service.py
│   ├── eval/
│   │   ├── benchmark.py
│   │   ├── generate.py
│   │   ├── asr.py
│   │   ├── entities.py
│   │   ├── speaker.py
│   │   ├── latency.py
│   │   └── aggregate.py
│   ├── api/
│   │   ├── app.py
│   │   └── schemas.py
│   └── release/
│       ├── gate.py
│       └── hf.py
├── tests/
├── notebooks/
│   └── kaggle_train.py
├── reports/
└── artifacts/                 # gitignored
```

`notebooks/kaggle_train.py` is a percent-format/lightweight Kaggle entry point, not business logic. All reusable logic stays in `src/`.

## 3. Source registry and reproducibility
`configs/sources.yaml` is authoritative. Each source records:
- id, URL/repo, revision;
- license asserted by source;
- redistribution constraints;
- training/eval permission in this project;
- expected row count where known;
- checksum/hash when available.

At first successful fetch, create `artifacts/source_lock.json` with resolved commit/revision and SHA256s.

Base model:
- HF repo: `silma-ai/silma-tts`
- checkpoint: `model.pt`
- upstream reported SHA256: `f43256d0b78b8803c638aed0875da5a4b372b4a784690a0156e5baff14f7336c`
- F5 code: tag `1.1.7`
- copy SILMA `vocab.txt`, `config.yaml`, and patched `finetune_cli.py`.
Never mutate upstream files in-place without preserving their hashes; stage them under `artifacts/upstream/`.

## 4. Data pipeline

### 4.1 Audit
Before audio processing:
1. resolve/pin revision;
2. inspect dataset card/license;
3. verify required fields;
4. count selected rows;
5. sample decode;
6. write license/provenance decision.

A source with contradictory/unknown terms is `blocked`, not “probably okay”.

### 4.2 Cleaning
Use `ffmpeg`/`torchaudio`:
- mono 24 kHz;
- PCM WAV for training cache;
- boundary-silence trim to max ~250 ms;
- no default denoise;
- record original and processed hashes.

Quality fields:
`duration,sample_rate,rms_dbfs,peak,clip_ratio,silence_ratio,char_count,word_count,chars_per_sec`.

Start with conservative rejection rules:
- duration <0.8 or >20 s;
- empty transcript;
- `clip_ratio > 0.001`;
- `silence_ratio > 0.35`;
- extreme `chars_per_sec` outside corpus p1/p99 after a first-pass audit.
Do not invent an SNR threshold unless a reliable estimator is implemented.

### 4.3 Transcript cleanup
Training cleanup:
- NFC;
- strip tatweel + zero-width chars;
- canonical whitespace/punctuation;
- keep Arabic diacritics only when source is consistent; otherwise strip them consistently;
- preserve English tokens;
- do not expand digits unless transcript/audio alignment requires it.

Inference normalization is separate and fully unit-tested.

### 4.4 Dedupe / split
Dedupe order:
1. exact audio SHA256;
2. exact normalized text;
3. near-text duplicate detection using normalized token 3-grams / MinHash, threshold ~0.92; co-locate near-duplicates.

Split **90/5/5** train/val/test, seed 42, stratified by domain and code-switch bucket. Group by speaker/session/source where metadata exists. If speaker IDs do not exist, state that splits are not speaker-disjoint.

Never train on the locked MasriSwitch-Bench final test set.

## 5. Code-switch analysis
For every transcript compute:
- Arabic token count;
- Latin token count;
- English-token ratio;
- switch count;
- bucket: `ar_only`, `cs_ar_dom`, `cs_balanced`, `cs_en_dom`, `en_only`.

Use Unicode-script rules + a small exception lexicon; do not add a heavyweight language model.

## 6. Text normalizer
Pipeline:
1. sanitize Unicode;
2. protect URLs/emails/IDs;
3. identify typed entities;
4. normalize entity to spoken form;
5. restore protected literals where needed;
6. normalize punctuation/chunking;
7. emit `normalized_text` + structured entity spans.

Keep normalization deterministic. Golden tests must cover at least 100 hand-written cases, including:
`499 EGP`, `12.5%`, `25/09/2026`, `4:30 PM`, Egyptian mobile numbers, OTP, Wi-Fi, API, order IDs, measurements and mixed Arabic clitics.

## 7. Training

### 7.1 Hardware
Target: free Kaggle **T4 x2**, 16 GB VRAM each.
Use Hugging Face cache outside committed paths. Kaggle sessions may end; every run must resume safely.

### 7.2 Environment
- Python 3.10
- F5-TTS `1.1.7`
- PyTorch/CUDA version compatible with Kaggle image
- `accelerate`
- `datasets`, `huggingface_hub`, `soundfile`, `torchaudio`, `jiwer`, `rapidfuzz`
- `ruff`, `mypy`, `pytest`

Lock top-level direct dependencies in `pyproject.toml`; record `pip freeze` in each train manifest, not Git.

### 7.3 SILMA architecture
Use upstream config exactly unless an experiment explicitly changes it:
- DiT dim 768
- depth 18
- heads 12
- ff_mult 2
- text_dim 512
- text_mask_padding true
- conv_layers 4
- char tokenizer / SILMA vocab
- 24 kHz
- 100 mels
- hop 256 / win 1024 / FFT 1024
- Vocos

### 7.4 Pilot
Purpose: prove data/config/resume before spending GPU quota.
- subset: deterministic 10% of E1 train, stratified;
- 2 GPUs, DDP, fp16;
- full fine-tune;
- LR `1e-5`;
- max grad norm `1.0`;
- warmup `50` updates;
- target `500` optimizer updates;
- save at 250 and 500;
- evaluate 50 benchmark prompts.
Abort on NaN/Inf, bad checkpoint reload, or invalid audio.

### 7.5 Full E1/E2
Default:
- full fine-tune, fp16, DDP 2 GPUs;
- LR `1e-5`;
- max grad norm `1.0`;
- warmup = 5% of target updates, capped at 500;
- target **8,000 optimizer updates**;
- checkpoint every 500;
- keep last 2 + current best;
- fixed seed 42;
- no augmentation by default.

Frame batch auto-probe on each Kaggle runtime:
`5600 -> 4800 -> 4000 -> 3200` per GPU.
Run 20 forward/backward steps; choose the largest stable value with ≥1 GB free VRAM. If all fail, enable gradient checkpointing and retry. Use 8-bit Adam only as the final memory fallback and record the change.

Because upstream F5 training is epoch-oriented, `train/plan.py` must compute the epochs needed to approximate 8,000 optimizer updates from measured dataset duration/batch plan and record expected vs actual updates.

Evaluate checkpoints at least every 1,000 updates. If all primary metrics plateau or worsen for 3 evaluations, stop early. Select best checkpoint by:
1. lowest English Entity Error Rate;
2. then lowest code-switch WER;
3. reject candidates violating Arabic CER regression guardrail.

### 7.6 Kaggle persistence
Keep large files out of the repo.
After each checkpoint interval:
- atomically write checkpoint;
- write `train_state.json`;
- prune old checkpoints;
- optionally sync best/last checkpoint to a private HF repo when `HF_TOKEN` exists.
Without a token, save best/last as Kaggle notebook outputs for manual persistence.

Never depend on more than one session to be available.

## 8. Evaluation

### 8.1 Generation protocol
- Same fixed reference voice/audio/text for E0/E1/E2.
- Same seed, speed, NFE/inference settings.
- Generate the locked 300 benchmark prompts plus validation prompts.
- Cache generated WAVs by model hash + prompt ID + inference config hash.

### 8.2 Intelligibility
Independent evaluator: `openai/whisper-large-v3-turbo` via `faster-whisper` where practical.
Normalize hypothesis/reference using a metric-only normalizer, separate from TTS TN.
Report:
- WER overall;
- CER Arabic;
- WER by code-switch bucket/domain;
- bootstrap 95% confidence intervals.

### 8.3 Entity metrics
Use benchmark gold spans.
- English EER = wrong/missing English entities / total English entities.
- CEA = exactly-correct critical entities / total critical entities.
Use normalized token matching plus entity-type-specific comparison (digits, money, date, acronym).

### 8.4 Speaker similarity
Use one pinned speaker-verification embedding model; record repo/revision.
Compare generated audio against fixed approved reference audio.
Report cosine mean/median and distribution; do not overinterpret it as voice quality.

### 8.5 Naturalness
Automated MOS predictors are secondary only.
Primary naturalness evidence, when collected, is blinded human MOS with saved anonymous scores. If no human study exists, write `not_measured`.

### 8.6 Performance
Benchmark warm + cold runs:
- RTF;
- time-to-first-audio / end-to-end latency;
- peak GPU memory;
- audio duration;
- invalid/empty output rate.
Use equal hardware/inference settings for E0/E1/E2.

## 9. Serving
FastAPI:
- `GET /health`
- `GET /model-info`
- `POST /normalize`
- `POST /synthesize`

Request limits:
- UTF-8 text;
- default max 500 chars;
- fixed server-side approved reference voice for public demo;
- explicit error codes;
- no arbitrary file paths.

Return WAV bytes or a JSON metadata envelope + streamed audio in a later optional milestone. Do not claim true streaming until TTFA is measured from chunked output.

## 10. Testing / CI
CI is CPU-only:
- ruff format/check;
- mypy;
- pytest;
- config/schema validation;
- 1–2 second synthetic WAV fixtures only;
- no model downloads in normal CI.

Mark network/GPU tests separately.
Target ≥80% coverage for project-owned deterministic modules; coverage is not a substitute for behavioral tests.

## 11. Release gate
`masriswitch release-check` verifies:
- all training sources `release_safe=true`;
- source revisions/hashes present;
- no blocked source IDs in train manifest;
- required attribution in NOTICE/model card;
- measured E0/E1 metrics exist;
- success guardrails pass or model card clearly says experimental;
- no secrets/raw third-party audio;
- checkpoint loads and generates 10 smoke prompts;
- model card limitations + misuse policy present.

HF model repo includes:
`README.md`, `LICENSE`, `NOTICE`, `config.yaml`, `vocab.txt`, pruned best checkpoint (prefer safe serialization), `training_args.json`, `data_manifest.json`, `eval_results.json`.

Benchmark HF dataset contains text + annotations only, no third-party audio.
