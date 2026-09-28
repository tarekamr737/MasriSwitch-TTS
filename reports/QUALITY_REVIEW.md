# Quality review — 2026-09-26

## Confirmed serving bug and correction

The ID recognizer accepted any word beginning with `ACC`, `ORD`, or `ID`,
ignoring case. Consequently the spoken-text normalizer expanded ordinary
words such as `account` and `access` into individual letters. An identifier
now requires a separator after its prefix or at least one numeric character.
Real identifiers such as `ORD123`, `acc456`, and `ID-ABC` remain recognized.
Eleven regression cases cover ordinary words and explicit identifiers.

An offline scan of the frozen 189 validation prompts found five affected
rows: access, accumulation, idiosyncratic, accuracy, and identification.
After the correction, all five retain their original text. No locked prompts
were used to choose this correction. The exact IDs and plan hash are recorded
in `artifacts/quality_diagnostics.json`.

## What these checks establish

This fixes text corruption before synthesis in the API and demo. It does not
establish a measured improvement in acoustic quality. The published evaluator
used raw prompt text, so this serving bug does not explain away the original
E1 accuracy result. Published metrics, model weights, and checkpoint selection
remain the original experiment's results. No training was started.

## Decision on further training

Do not extend the stopped E1 run: it already reached its declared three-stale-
check stopping rule, and later checkpoints failed the Arabic guardrail. Its
selected checkpoint reduced code-switch WER by only 0.06% relative; English
EER worsened by 1.80% relative. The evidence does not justify more updates with
the same data and recipe. It also does not establish the cause of the failure.

Recommended next experiment, not yet run:

1. Compare untouched SILMA and selected E1 on fresh development prompts with
   the same approved voice, normalization, chunking, seed, and 16-step runtime.
   Cover short and long utterances, language switches, IDs, times, and prices.
   Keep both the old locked benchmark and a new evaluation holdout out of tuning.
2. Save paired audio and annotate wrong, omitted, repeated, or rushed words.
   Combine independent ASR scoring with blind human listening; forced-language
   ASR alone cannot establish pronunciation quality.
3. Inspect audio/transcript alignment, Egyptian pronunciation, English coverage,
   and entity examples in the audited synthetic training set. Limited real-speaker
   diversity is a plausible constraint, not a measured cause. Any additional
   speech needs verified redistribution/training terms and documented consent.
4. If that review supports a changed dataset or recipe, start a separate bounded
   correctness probe and pilot. Advance to a longer run only after valid audio
   and validation improvement satisfy the declared gates. Select by validation
   metrics and measure once on the held-out evaluation set.

No new training was started. E2 remains blocked by unverified Common Voice
source terms/revision; it requires a verified source audit before training.

## Deployment environment

The public demo uses free ZeroGPU with pinned PyTorch/torchaudio 2.8.0,
required by that hosting runtime. The published E0/E1 evaluation used 2.6.0.
The deployment checks therefore verify operational compatibility, not numerical
equivalence with the published benchmark. The model remains experimental.
Gradio SDK 5.25.2 avoids the host's Gradio 5.35 MCP extra, whose Pydantic
requirement conflicts with F5-TTS 1.1.7. The Linux dependency set was resolved
before deployment (`artifacts/space_resolved.txt`); model code is unchanged.
An anonymous live request containing `account` passed with finite nonzero
24 kHz audio (`artifacts/space_smoke.json`). This confirms the serving path
works after the correction; pronunciation quality still needs listening.

## Reported long-sentence degradation

The user reported reasonable speech at the start of the 192-character challenge
and severe degradation later. After normalization this prompt is 259 characters
and 416 UTF-8 bytes. The pinned upstream splitter ignores Arabic commas and
does not split an oversized clause at word boundaries. It therefore emits one
416-byte chunk despite its 256-byte budget, estimating about 22 seconds of new
speech plus the 8.4935-second reference.

The serving correction recognizes Arabic punctuation and enforces a byte limit
even without punctuation. The limit is the smaller of 160 bytes and six seconds
estimated from the fixed reference's speaking rate (113 bytes for this voice).
The challenge becomes five pieces; all words are preserved, and the generated
pieces are joined in order with 120 ms pauses. The six-second estimate is a
length heuristic, not a measured phoneme-duration guarantee.

A private two-generation comparison used the same published checkpoint,
approved voice, seed 42, PyTorch 2.8.0, and 16 inference steps. Both WAV hashes
were verified after download to D (`artifacts/long_text_probe.json`).

| One reported prompt only | Legacy, 16 steps | Chunked, 16 steps | Chunked, 32 steps |
|---|---:|---:|---:|
| Arabic-decoder WER | 80.85% | 76.60% | 78.72% |
| Arabic character error | 59.87% | 66.24% | 66.24% |
| English entity error | 25.00% | 25.00% | 25.00% |
| Generated seconds | 21.792 | 21.984 | 21.984 |
| Synthesis seconds, T4 | 4.761 | 6.271 | 12.759 |

The split version's transcript recovers the final instruction about trying
another credit card, but aggregate proxy results are mixed and ID/time errors
remain. Forced Arabic transcription often renders English words in Arabic
letters and spoken numbers as digits. These are diagnostic results, not a new
benchmark or proof of improved naturalness.

The previously queued single 32-step chunked pass completed and was verified
on 2026-09-28. It reused the hash-verified 16-step legacy baseline and preserved
the checkpoint, approved reference, prompt, five chunks, ASR model, and seed.
The downloaded WAV hash matched, audio was finite/nonzero mono 24 kHz, and all
three proxy error metrics were recomputed locally. WER was worse than chunked
16-step inference, CER/EER were unchanged, and synthesis took 2.03 times as long.
Keep the live setting at 16 steps. This one-prompt result does not establish
which step count is best across other prompts.

Evidence: `artifacts/long_text_32_probe.json`,
`artifacts/long_text_32_verification.json`. Result SHA256:
`fd62f346de0ff90f78ad23e949a2736d597331190c33f94cd7101837c5f9ba57`;
WAV SHA256: `a6b7f8b2ac07853a4489550c1c51a6ccbd69bf95405246b0b22fe0eb4b78358f`.

The verified 16-step repair was deployed at Space commit
`2c34711b41a388e41a311ee4827944f00ed82cc5`. An anonymous request for the exact
reported sentence produced a finite nonzero 24 kHz WAV lasting 21.984 seconds,
with 7.888 seconds total request time (one measurement including queue/network).
Remote engine/chunker hashes and the downloaded WAV hash were verified.
The fresh live audio is recorded in `artifacts/space_long_text_smoke.json`.
User listening assessment is pending; the model remains experimental.
