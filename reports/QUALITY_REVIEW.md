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
check stopping rule, and later checkpoints failed the Arabic guardrail.
The next useful evidence is human listening on new sentences with the approved
demo voice. Preserve exact inputs and describe wrong, missing, or rushed words.
If those errors justify another experiment, freeze a validation-only comparison
of inference settings before any new training; keep the original locked set
out of tuning. Any claimed improvement needs a separate, measured evaluation.
E2 also remains blocked by unverified Common Voice source terms/revision.

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

| One reported prompt only | Legacy | Chunked, 16 steps |
|---|---:|---:|
| Arabic-decoder WER | 80.85% | 76.60% |
| Arabic character error | 59.87% | 66.24% |
| English entity error | 25.00% | 25.00% |
| Generated seconds | 21.792 | 21.984 |
| Synthesis seconds, T4 | 4.761 | 6.271 |

The split version's transcript recovers the final instruction about trying
another credit card, but aggregate proxy results are mixed and ID/time errors
remain. Forced Arabic transcription often renders English words in Arabic
letters and spoken numbers as digits. These are diagnostic results, not a new
benchmark or proof of improved naturalness. A single additional chunked pass
at the upstream default of 32 steps is pending; the legacy baseline is reused.
Kaggle reported its maximum two-batch-GPU-session limit on a retry. No 32-step
result is available, and that setting was not promoted.

The verified 16-step repair was deployed at Space commit
`2c34711b41a388e41a311ee4827944f00ed82cc5`. An anonymous request for the exact
reported sentence produced a finite nonzero 24 kHz WAV lasting 21.984 seconds,
with 7.888 seconds total request time (one measurement including queue/network).
Remote engine/chunker hashes and the downloaded WAV hash were verified.
The fresh live audio is recorded in `artifacts/space_long_text_smoke.json`.
User listening assessment is pending; the model remains experimental.
