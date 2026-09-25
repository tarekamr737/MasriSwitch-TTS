# MasriSwitch-TTS model card draft

## Status

Experimental. Fine-tuned checkpoint: TBD. Fine-tuned metrics: TBD.
Weights must not be published until `artifacts/release_gate.json` allows it.

## Base and data

Base: SILMA TTS v1, Apache-2.0 weights. F5-TTS 1.1.7, MIT code. Eligible D1:
3,944 generated CC BY 4.0 rows at pinned revision
`eae9a87c17e91e3f59a9696d5f4ff3eb51502e82`; 5,673 noncommercial rows
excluded. Attribution: Abdelrahman R. Hashem, *Synthetic Arabic-English
Code-Switched Speech for ASR* (2026).

## Evaluation

E0 on 189 validation + 300 locked benchmark prompts: code-switch WER 61.38%,
English EER 51.60%, Arabic-only CER 38.57%. See `reports/BASELINE.md` for
full measured protocol, CIs, and limitations. E1: TBD. Locked benchmark: 300
of 1,200 text-only prompts. Independent ASR: pinned Whisper Turbo
`0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf`; speaker embedding:
SpeechBrain ECAPA `0f99f2d0ebe89ac095bcc5903c4dd8f72b367286`.
The SILMA upstream reference sample was used for private evaluation only;
public reference voice: TBD. Human MOS: not_measured.

## Limitations and misuse policy

Synthetic training audio does not establish real-world Egyptian dialect,
speaker, or acoustic robustness. Numeric/entity pronunciation may be wrong;
do not use for consequential decisions without human verification. Disclose
AI-generated audio. Only synthesize a custom voice with documented speaker
consent. Do not impersonate, deceive, or commit fraud.
