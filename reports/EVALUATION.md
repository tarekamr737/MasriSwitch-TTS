# Evaluation

Measured on the same 189 validation and 300 locked benchmark prompts, with the same private reference voice and 16-step inference protocol.

| Model | Code-switch WER | English EER | Arabic-only CER | Critical entity accuracy | Speaker cosine | Mean RTF | Invalid audio |
|---|---:|---:|---:|---:|---:|---:|---:|
| E0 SILMA | 61.38% | 51.60% | 38.57% | 10.19% | 0.728 | 0.308 | 0 |
| E1 selected | 61.34% | 52.53% | 38.03% | 10.19% | 0.728 | 0.305 | 0 |

Relative code-switch WER reduction: 0.06%; English EER reduction: -1.80%.
Arabic-only CER ≤5% relative regression: True; mean RTF ≤10% regression: True.
Accuracy target (≥15% code-switch WER or ≥25% English EER reduction): False.

## Prompt bootstrap 95% intervals

Intervals use 2,000 prompt resamples with seed 42.

- E0: code-switch WER 58.76%–64.10%; English EER 47.14%–56.08%; Arabic-only CER 34.53%–42.45%.
- E1: code-switch WER 58.75%–63.98%; English EER 48.28%–56.95%; Arabic-only CER 34.01%–41.81%.

## Validation-only checkpoint selection

The frozen 50-prompt screening advanced three eligible checkpoints to all 189 validation prompts. Selection minimized English EER, then code-switch WER, subject to overall Arabic CER ≤1.05 × E0 validation. The 300 locked benchmark prompts were excluded from selection.

| Updates | English EER | Code-switch WER | Overall Arabic CER | Selected |
|---|---:|---:|---:|---|
| 1,000 | 37.38% | 45.87% | 37.29% | no |
| 2,000 | 36.39% | 45.47% | 37.23% | yes |
| 3,000 | 37.13% | 45.44% | 37.73% | no |

Training endpoint: 5,000 updates; selected checkpoint: 2,000 updates. Declared early stop: True.

## Highest-error locked prompts

- `msb1_telecom_ar_dominant_042` reference: حضرتك هتلاقي تفاصيل Premium Plan في الباقة رقم 1042.  ASR hypothesis: حضاقي تصل بريميوم في مبيع باقة رقم ثم فوزين فونسين
- `msb1_banking_balanced_016` reference: حضرتك، credit card مرتبط بـ Online Banking رقم 1016 من النهارده.  ASR hypothesis: كردت كار برون لائن بانكين رقمتازن سيسكتين نهارب
- `msb1_banking_balanced_022` reference: حضرتك، credit card مرتبط بـ Online Banking رقم 1022 من النهارده.  ASR hypothesis: ترجمة نانسي قنقر

## Limits and provenance

These are independent ASR proxy scores, not human judgments of pronunciation or naturalness. The training audio is synthetic and has no speaker IDs. The fixed SILMA sample was used privately; no public voice consent was established.
Plan SHA256: `26dc087cabc32544c401eebcaaf804185901cbd829358f6c795e0d232ced9185`. Selected checkpoint SHA256: `558e2ab53e1b5450bcd1a1c30683a1b3e6be1234b7e198b74209f362693eaf92`.
Machine-readable results: `artifacts/eval_metrics.json`.
