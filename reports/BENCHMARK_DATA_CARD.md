---
license: cc-by-4.0
language:
  - ar
  - en
task_categories:
  - text-to-speech
tags:
  - egyptian-arabic
  - code-switching
  - benchmark
configs:
  - config_name: default
    data_files:
      - split: test
        path: masriswitch_bench_v1.jsonl
---

# MasriSwitch-Bench v1

MasriSwitch-Bench is a **text-only** prompt set for evaluating Egyptian Arabic and English code-switched speech synthesis. It contains no audio, speaker recordings, or personal data. Prompts were generated deterministically by this project's templates with seed 42.

## Contents

The JSONL file has 1,200 prompts: 200 each for telecom, banking, ecommerce, hospitality, healthcare, and technical support. Each domain has 50 simple Arabic, 100 Arabic-dominant code-switched, 30 balanced, and 20 entity-heavy prompts. A separate ID list fixes 300 prompts as the final locked benchmark. The remaining prompts are available for development; the locked IDs should not guide model selection.

Each row contains `id`, `version`, `domain`, `bucket`, `text`, `entities`, `critical_entity_count`, and `detected_switch_bucket`. The `entities` field gives parser-derived entity kind, value, and character offsets. Entity tags are automatic and may be incomplete.

The generator is `src/masriswitch/eval/benchmark.py`. Recreate locally with `make benchmark` (or `python -m masriswitch.cli benchmark`). The generated JSONL SHA256 is `3ee41bd7668bfb9d6480256d8929f29384ee0e4cfb933508ea4765991ebf6350`. The manifest is `artifacts/benchmark_manifest.json` in the project workspace.

## Use and limitations

The text is templated. Scores on this set do not establish natural conversational quality, real speaker diversity, or production readiness. An ASR-based pronunciation score may mishear correctly spoken words, especially when Arabic and English appear in one utterance. Human listening evaluation is recommended before deployment.

The dataset does not grant rights to any reference voice or generated audio. Synthesis using a person's voice requires consent, and generated audio should be disclosed as AI-generated.

## Citation and license

MasriSwitch-TTS contributors, *MasriSwitch-Bench v1*, 2026. The benchmark text and metadata are released under CC BY 4.0. Project code is Apache 2.0. This card does not cover the separate D1 training dataset or SILMA model weights.
