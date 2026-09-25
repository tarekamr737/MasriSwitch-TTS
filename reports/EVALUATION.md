# Evaluation

Status: E0 completed on 489 prompts with a pinned independent ASR evaluator;
E1 comparison and checkpoint selection: TBD. See `reports/BASELINE.md` for the
full E0 protocol, group tables, timing, and limitations.

| Model | Code-switch WER (412) | English EER | Arabic-only CER (77) |
|---|---:|---:|---:|
| E0 SILMA | 61.38% (95% CI 58.76–64.10%) | 51.60% (47.14–56.08%) | 38.57% (34.53–42.45%) |
| E1 | TBD | TBD | TBD |

The intervals are prompt bootstraps with 2,000 resamples and seed 42. The
reference sample is restricted to private evaluation. Human naturalness and
pronunciation scores have not been measured.

Checkpoint selection uses only the 189 validation prompts. E0 on this subset
has English EER 36.88%, code-switch WER 45.11%, and overall Arabic CER 36.35%
(`artifacts/e0_validation_metrics.json`). Only two validation prompts are
Arabic-only, so the selection guardrail uses overall Arabic CER on validation;
the final comparison will separately measure Arabic-only CER on the fixed
489-prompt plan. The 300 locked benchmark prompts are excluded from selection.
To bound GPU use, every 1,000-update checkpoint is screened on the same frozen
first 50 validation prompts. The three eligible checkpoints with lowest
English EER, then code-switch WER, advance to the full 189-prompt validation
comparison. The final checkpoint is chosen from those three by the same metric
order and Arabic CER guardrail; locked benchmark prompts never influence it.

At 1,000 E1 updates, an early 50-prompt validation check measured English EER
32.63% versus E0 37.89%, code-switch WER 44.48% versus E0 43.43%, and overall
Arabic CER 35.12% versus E0 34.88% on those same prompts. All 50 outputs were
valid. These are interim selection signals, not the final locked comparison.
At 2,000 updates on the same 50 prompts, English EER was 34.74%, code-switch
WER 43.43%, and overall Arabic CER 35.26%, with zero invalid outputs.
At 3,000 updates, English EER was 40.00%, code-switch WER 43.58%, and overall
Arabic CER 35.45%, again with zero invalid outputs. The checkpoint passed the
audited train-ID, exact-update, finite-loss, and ten-prompt private smoke
checks. This is the first validation check with neither EER nor code-switch
WER improving over previous E1 checks (`artifacts/e1_progress.json`).
At 4,000 updates, English EER was 34.74%, code-switch WER 44.03%, and overall
Arabic CER 37.91%, with zero invalid outputs. The Arabic CER fails the 5%
relative guardrail on these 50 validation prompts. This is the second stale
check under the predeclared early-stop rule.
