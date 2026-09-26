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
