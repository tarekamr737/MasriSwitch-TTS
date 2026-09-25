# E0 SILMA baseline

Measured on 189 held-out validation prompts and 300 locked MasriSwitch-Bench prompts.
The same fixed SILMA upstream sample was used for private tests only. Public synthesis
still requires a documented, consented voice.

| Set | Prompts | WER | Arabic CER | English EER | Critical entity accuracy | Speaker cosine | Mean RTF |
|---|---:|---:|---:|---:|---:|---:|---:|
| Overall | 489 | 60.62% | 52.11% | 51.60% | 10.19% | 0.728 | 0.308 |
| locked_benchmark | 300 | 74.72% | 67.58% | 68.68% | 9.87% | 0.735 | 0.316 |
| validation | 189 | 44.84% | 36.35% | 36.88% | 20.00% | 0.716 | 0.295 |

Code-switch WER: 61.38% on 412 prompts; Arabic-only CER: 38.57% on 77 prompts.

## By code-switch bucket

| Bucket | Prompts | WER | Arabic CER | English EER | Critical entity accuracy | Speaker cosine | Mean RTF |
|---|---:|---:|---:|---:|---:|---:|---:|
| ar_dominant | 150 | 75.59% | 63.17% | 57.33% | 52.00% | 0.742 | 0.317 |
| ar_only | 2 | 15.38% | 2.56% | TBD | TBD | 0.762 | 0.217 |
| balanced | 45 | 76.86% | 86.47% | 53.33% | 14.29% | 0.737 | 0.316 |
| cs_ar_dom | 158 | 41.79% | 32.75% | 33.86% | 33.33% | 0.723 | 0.284 |
| cs_balanced | 28 | 68.37% | 88.94% | 46.99% | 0.00% | 0.676 | 0.355 |
| cs_en_dom | 1 | 100.00% | 375.00% | 100.00% | TBD | 0.608 | 0.574 |
| entity_heavy | 30 | 96.07% | 178.06% | 97.22% | 0.83% | 0.681 | 0.286 |
| simple | 75 | 56.46% | 40.73% | TBD | TBD | 0.741 | 0.327 |

## Timing and uncertainty

Mean synthesis latency: 1.256 s; p95: 1.810 s; peak allocated TTS VRAM: 0.457 GiB.
Mean RTF 95% prompt-bootstrap CI: 0.302–0.315; mean speaker cosine CI: 0.724–0.731 (2,000 resamples, seed 42).
Code-switch WER 95% CI: 58.76%–64.10%; Arabic-only CER CI: 34.53%–42.45%; English EER CI: 47.14%–56.08% (prompt bootstrap, 2,000 resamples).
Invalid audio outputs: 0 of 489.

## Protocol and limits

SILMA v1 uses 16 NFE steps at 24 kHz. WER and Arabic CER use a pinned independent
Whisper Turbo Arabic decoder. English EER uses a second English decoder on the same
audio; this may miss correctly spoken English that the ASR transcribes differently.
Critical entity accuracy requires exact token matches: numeric entities use the
Arabic decoder, and brands/acronyms use the English decoder. These automated scores
are ASR proxies, not human pronunciation or naturalness judgments. The table's WER includes all 489 prompts; code-switch WER excludes the simple/Arabic-only buckets. Synthetic source speech does not establish
real Egyptian speaker diversity.

Plan SHA256: `26dc087cabc32544c401eebcaaf804185901cbd829358f6c795e0d232ced9185`. Result rows SHA256: `bb125e7c1c9907a0b7af6f4a385d027b6e48ec92b08746a2616a3dab91ecd16b`.
Machine-readable metrics: `artifacts/baseline_metrics.json`.
