# PRODUCT.md — MasriSwitch-TTS

## 1. Product
**MasriSwitch-TTS** is an Egyptian Arabic ↔ English code-switched text-to-speech system optimized for real-world voice-agent text: Egyptian dialect mixed with English product names, technical terms, acronyms, dates, money, phone numbers, IDs, and brands.

Target sentence:
> مساء الخير يا فندم، الـ Premium Plan بتاع حضرتك هيتجدد يوم 25 September بقيمة 499 جنيه.

This is not “another Arabic TTS demo.” The product goal is reliable pronunciation and natural switching inside one utterance.

## 2. Users / use cases
Primary:
- Egyptian customer-support and sales voice agents.
- Telecom, banking, e-commerce, restaurant, booking, and healthcare assistants.
- Developers needing an open, reproducible Arabic-English TTS baseline.

Secondary:
- Accessibility and local assistants.
- Research on Arabic-English code-switching.

## 3. Base model
Use **`silma-ai/silma-tts`** as the only default base.
- Architecture: F5-TTS-compatible DiT, ~150M parameters.
- Native Arabic + English.
- Model weights: Apache-2.0.
- Fine-tuning compatibility: **F5-TTS v1.1.7**.
- Audio: 24 kHz, Vocos, 100 mel bins.
- SILMA config: dim=768, depth=18, heads=12, ff_mult=2, text_dim=512, char tokenizer.
- Base file: `model.pt`; verify SHA256 against upstream before training.
- Record the resolved Hugging Face revision and every downloaded file hash in `artifacts/source_lock.json`.

Upstream:
- https://huggingface.co/silma-ai/silma-tts
- https://github.com/SILMA-AI/silma-tts
- https://github.com/SWivid/F5-TTS/tree/1.1.7

Do not default to `SWivid/F5-TTS` pretrained weights because the official weights are CC-BY-NC.

## 4. Data policy

### 4.1 Release-safe default training set
**D1 — Code-switch core**
- HF: `abdo1819/arabic-english-code-switching-synthetic-asr`
- config: `synthetic`
- select only `license == "cc-by-4.0"`
- expected public count: **3,944 rows**; assert and report actual count.
- Keep only author-created/domain-oriented rows; exclude ArE-CSTD-derived NC/SA rows.
- Audio is 24 kHz synthetic ar-EG; measure duration after filtering.
- Purpose: Egyptian Arabic-English switching and domain terminology.

**D2 — Arabic stability replay (optional E2 only)**
- Mozilla Common Voice Scripted Speech **27.0 Arabic**, CC0-1.0.
- Use only validated clips.
- Prefer rows explicitly carrying an Egypt/Egyptian variant/accent if metadata supports it.
- If Egypt cannot be verified, treat this as Arabic stability data, not Egyptian-dialect data.
- Sample a capped replay subset equal to **10% of E1 training duration**, speaker-balanced where IDs exist.
- Do not re-host Common Voice audio.

### 4.2 Evaluation / research-only sources
These are disabled for release-safe training by default:
- **ArzEn Speech Corpus**: 12 h / 6,216 utterances, real Egyptian Arabic-English conversational speech. Treat as noncommercial/research-only unless corpus terms are independently verified for the intended use. Excellent external evaluation source.
- `MohamedRashad/arabic-english-code-switching`: GPL-tagged aggregate with mixed upstream provenance. Research-only by default.
- `MohamedGomaa30/EGYSpeak`: card says CC-BY-4.0, but provenance must be independently audited before training because it derives from an upstream Kaggle source and contains ASR-generated transcripts. Default: BLOCKED.
- ArE-CSTD: do not use for release-safe audio training; its card/body licensing includes noncommercial/share-alike restrictions.

The loader must implement `release_safe: true|false` and refuse blocked sources unless an explicit research config is used.

## 5. Data quality contract
For every candidate clip:
- decode successfully;
- mono; resample to 24 kHz;
- duration target 1.5–15.0 s; hard reject <0.8 s or >20 s;
- trim excessive boundary silence but retain ≤250 ms each side;
- reject empty/garbled transcript;
- Unicode NFC; remove tatweel/zero-width junk; normalize whitespace;
- preserve Arabic + Latin text; no transliteration;
- reject transcript/audio duplicates;
- compute clipping ratio, RMS, silence ratio, duration, chars/sec, words/sec;
- reject obvious clipping, near-silence, or pathological text/audio ratio;
- never aggressively denoise or pitch-shift by default.

Create stable sample IDs from source + upstream ID + content hash.

## 6. Text-normalization product feature
Build a deterministic pre-TTS normalizer, separate from training cleanup.

It must support:
- Arabic/Western digits;
- EGP/USD/SAR and decimal amounts;
- dates and clock times;
- phone/account/order numbers;
- percentages and measurements;
- common English acronyms: OTP, Wi-Fi, API, AI, URL, SMS, USB;
- product/brand lexicon hooks;
- Arabic clitics around English tokens (`الـ`, `و`, `بـ`);
- safe punctuation/chunking.

Never “translate” English tokens. Normalization output must be testable and reversible enough to inspect.

## 7. Benchmark
Generate a versioned, text-only **MasriSwitch-Bench v1** with deterministic templates, not an LLM dependency.

**1,200 prompts**:
- telecom 200
- banking/fintech 200
- e-commerce/delivery 200
- restaurants/hospitality 200
- appointments/healthcare 200
- technical/support 200

Within each domain include:
- 25% simple Egyptian Arabic
- 50% Arabic-dominant code-switch
- 15% balanced code-switch
- 10% entity-heavy stress tests

Tag entities: `money,date,time,phone,id,brand,acronym,english_term,measurement,url`.
Keep a locked 300-prompt final test subset never used for tuning decisions.

## 8. Experiments
- **E0**: untouched SILMA TTS baseline.
- **E1**: full fine-tune on release-safe D1.
- **E2**: E1 recipe + 10% Common Voice Arabic stability replay.
- **E3 (research only)**: best recipe + real Egyptian/code-switch data after explicit license gate. Never publish E3 weights automatically.

Only E0/E1 are mandatory.

## 9. Success metrics
Primary comparison is E1/E2 vs E0 on the same fixed prompts/reference voice.

Required:
- Arabic CER and overall WER from an independent multilingual ASR evaluator.
- Code-switched WER.
- **English Entity Error Rate (EER)**: incorrect/missing English entities ÷ reference English entities.
- **Critical Entity Accuracy (CEA)** over tagged money/date/time/phone/ID/brand/acronym entities.
- Speaker-similarity cosine score to the fixed reference voice.
- invalid-audio rate.
- Real-Time Factor (RTF), end-to-end latency, and peak VRAM.

Release target:
- ≥15% relative improvement in code-switched WER **or** ≥25% relative reduction in English EER vs E0;
- Arabic-only CER regression ≤5% relative;
- zero invalid outputs on the locked benchmark;
- no material latency regression (>10%) at equal inference settings.

Human evaluation is recommended but never fabricated:
- Egyptian naturalness MOS 1–5;
- code-switch naturalness MOS 1–5;
- pronunciation correctness for critical entities.

## 10. Non-goals
- Training a foundation TTS model from scratch.
- Supporting every Arabic dialect.
- Celebrity/person voice cloning.
- Production telephony integration.
- Paid inference APIs.
- Claiming commercial readiness before license and robustness gates pass.

## 11. Release
GitHub:
- code license: Apache-2.0;
- no raw third-party data/checkpoints;
- reproducible Kaggle recipe and measured results.

Hugging Face:
- model: `Tarek737/MasriSwitch-TTS`;
- benchmark dataset: `Tarek737/MasriSwitch-Bench`;
- optional Space: fixed approved reference voice only.

Weight publication is allowed only if `release_gate.json` proves every training source is release-safe and attribution requirements are satisfied. Otherwise publish code/config/results only.
