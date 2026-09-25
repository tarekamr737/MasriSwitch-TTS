# AGENTS.md — MasriSwitch-TTS

## Mission
Build MasriSwitch-TTS end to end: a reproducible Egyptian Arabic ↔ English code-switched TTS system for voice-agent use cases, fine-tuned from SILMA TTS v1 (F5-TTS architecture), trained/evaluated on free resources, and releasable on GitHub + Hugging Face.

## Read order
1. Read this file once.
2. Read only the relevant section of `PRODUCT.md` / `ARCHITECTURE.md`.
3. Execute the next unchecked item in `TASKS.md`.
4. Update `TASKS.md` and evidence files; do not rewrite specs unless required.

## Agent rules
- Work autonomously; do not delegate.
- Prefer executable code/configs over prose or notebooks.
- Minimize tokens: no long narration, no restating specs, no duplicate docs.
- Search with `rg`/targeted reads; never repeatedly reread whole files.
- Make the smallest coherent change that completes a checklist item.
- Batch related edits/tests in one pass.
- Never invent metrics, dataset counts, licenses, paths, or successful runs.
- `TBD` is mandatory until measured.
- Pin model/dataset revisions and record hashes before expensive work.
- Fail closed on unknown/contradictory licenses.
- Never commit raw third-party audio, secrets, checkpoints, caches, or PII.
- Never train on test/eval data.
- Never silently replace a failed dependency/model/dataset.
- Prefer deterministic scripts with seeds over manual notebook steps.
- Use Python 3.10, type hints, `pathlib`, structured logging, dataclasses/Pydantic where useful.
- Keep modules small; enforce SOLID/DRY/KISS/YAGNI. No premature abstractions.
- Public APIs must have tests. Pure transformations need unit tests.
- Every expensive stage must support `--dry-run`, `--max-samples`, resume, and deterministic seed.
- All generated artifacts live under ignored `artifacts/`; durable summaries live under `reports/`.
- Use environment variables for tokens. Provide `.env.example`; never print secrets.
- Do not add paid services. Kaggle + Hugging Face + local CPU/GPU are the target stack.
- Do not require W&B; TensorBoard/local JSONL is default. W&B is optional.
- Public demo uses a fixed approved reference voice; no arbitrary voice-cloning upload.
- Clearly disclose AI-generated audio and require consent for any custom voice.

## Definition of done
The project is done only when:
- `make check` passes: format, lint, types, unit tests, config validation.
- Data audit is reproducible and license policy passes.
- Baseline and fine-tuned evaluations exist with real values.
- At least E0 and E1 are completed; E2 is completed if compute permits.
- Best checkpoint is selected by declared metrics, not listening preference alone.
- FastAPI inference works and a minimal Gradio demo works.
- Kaggle training can resume after session interruption.
- GitHub README contains architecture, exact reproduction commands, results, limitations, and licenses.
- Hugging Face model card/data card are generated from measured artifacts.
- Release gate blocks weight publication if any training source is not release-safe.

## Required commands
Expose these stable commands (Makefile or equivalent):
- `make setup`
- `make check`
- `make audit-data`
- `make prepare-data`
- `make benchmark`
- `make train-pilot`
- `make train`
- `make eval`
- `make demo`
- `make api`
- `make release-check`
- `make release`

## Evidence contract
Each stage writes machine-readable evidence:
- `artifacts/source_lock.json`
- `artifacts/data_audit.json`
- `artifacts/splits.json`
- `artifacts/baseline_metrics.json`
- `artifacts/train_manifest.json`
- `artifacts/eval_metrics.json`
- `artifacts/release_gate.json`
Human summaries:
- `reports/DATA_AUDIT.md`
- `reports/BASELINE.md`
- `reports/EVALUATION.md`
- `reports/MODEL_CARD_DRAFT.md`

## Stop conditions
Stop an expensive action, not the whole project, when:
- license cannot be verified;
- required dataset terms need human acceptance;
- Kaggle/HF credentials are absent;
- a checksum/revision changed unexpectedly;
- pilot produces NaN/Inf, corrupt audio, or repeated OOM after fallback settings.
Continue all independent offline work and record the blocker in `TASKS.md`.
