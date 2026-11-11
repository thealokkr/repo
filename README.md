# CloudServe Support Automation Capstone

A self-contained, CLI-first support-ticket automation capstone.

## Repository layout

- `src/cloudserve_support/` — application package.
- `data/kb/` — packaged source support articles.
- `data/index/` — rebuildable Chroma vector store.
- `data/tickets/` — development and validation fixtures.
- `data/evaluation/` — reference responses for evaluation only.
- `data/incoming/` — convenient location for new ticket files; processing commands will accept arbitrary input paths.
- `data/runtime/` — SQLite decision log and local state.
- `outputs/` — per-run results, metrics, and exception reports.
- `tests/` — unit, integration, and acceptance tests.
- `scripts/` — repeatable setup and demo helpers.
- `.agents/skills/` — development-only Codex workflows; omitted from the recipient ZIP.
- `.workarea/` — development-only feature specs, queue state, implementation logs, and review packets; omitted from the recipient ZIP.
- `scripts/ingest_kb.py` — rebuilds the passage index from the packaged KB using OpenAI embeddings.

See [Agent.md](Agent.md) for repository conventions, data handling rules, and scope.

## Python and Chroma setup

The project uses Python 3.13 and `uv`. The Chroma dependency is declared in `pyproject.toml` and pinned by `uv.lock`.
`jsonschema` is declared for output-contract validation; `pytest` is a development dependency for the test suite.

From a clean checkout, run the following from the repository root. Copy the example configuration to the ignored local `.env`; the default is offline and contains no credential:

```bash
uv sync --frozen
cp .env.example .env
uv run python scripts/init_runtime_db.py
uv run python scripts/init_output_schemas.py
uv run cloudserve --help
```

On Windows PowerShell, use `Copy-Item .env.example .env`. Add `OPENAI_API_KEY` to `.env` only when using provider-backed retrieval, generation, classification, post-run judging, or KB ingestion. `.env` is ignored by Git; never commit it or include it in a distributable archive. Without a key, ticket batches still complete in offline mode and route tickets without available retrieval evidence to human review.

To create a distributable source archive, run:

```bash
uv run python scripts/package_source.py --output dist/cloudserve-support.zip
```

The packager creates and verifies a ZIP rooted at `cloudserve-support/`. It includes the source code, locked dependencies, packaged KB, rebuildable vector index, fixtures, policy, schemas, tests, and setup example. It excludes `.env` files except `.env.example`, development-only `.agents/`, `.workarea/`, and `.codex/` directories, virtual environments, caches, build output, runtime SQLite files, and generated run exports. Use this command instead of recursively zipping the working directory.

To activate the environment manually, use:

```bash
source .venv/bin/activate
```

On Windows PowerShell, activate with:

```powershell
uv sync --frozen
.venv\Scripts\Activate.ps1
```

The installed `cloudserve` command provides the CLI and help. Run the unit/integration/acceptance suite with the single documented test command:

```bash
uv run cloudserve --help
uv run pytest
```

The single test command for the repository is `uv run pytest`.

## Process a ticket batch

The repository includes a frozen routing policy calibrated from the development fixtures. To reproduce calibration, write a local copy under the ignored `outputs/` directory, then process tickets with arbitrary input, output, and database paths:

```bash
uv run --env-file .env cloudserve calibrate --development-input data/tickets/development_tickets.json --output outputs/frozen_routing_policy.json
uv run --env-file .env cloudserve run --input data/tickets/development_tickets.json --output-root outputs/development --database outputs/development.sqlite3 --policy outputs/frozen_routing_policy.json --label "development batch"
```

The run command has no interactive prompts or retry steps. Each invocation creates a unique run directory under `--output-root` with `results.jsonl`, `decision_log.jsonl`, `metrics.json`, and `exceptions.jsonl`. It also writes the SQLite run, ticket, and audit rows to `--database`. `--label` is optional metadata. Exit status `0` means the batch and exports completed, including when a ticket has a recorded degradation or malformed-record result; exit status `1` means startup, persistence, or export finalization failed. Calibration errors return `2`.

The routing calibration command uses the development ticket text after normalization, excludes invalid records, and never reads fixture `labels` or outcome `history`. It selects a confidence floor from the 20th percentile of non-protected development confidence scores, clamps the floor to `[0.65, 0.98]`, and writes the frozen policy with an operational-input hash and calibration counts. The offline classifier's confidence correction is separately fitted and cross-checked using development intent labels; runtime classification applies only the frozen correction and never reads labels. Do not recalibrate with validation data. A validation run reads the same frozen policy and records its full provenance in every result and audit event.

The public route enum remains `draft_for_review`, `escalate`, and `degrade`. For evaluation documentation only, fixture label `auto_respond` maps to `draft_for_review`, and fixture label `escalate` maps to `escalate`. Stage 3 prompt term `draft_for_agent_review` maps to `draft_for_review`; `escalate_for_human_review` maps to `escalate`. The `must_not_auto_respond` fixture flag is a safety oracle, not a route value. None of these fixture fields are read by the operational classifier or router. The processing path excludes `labels` and evaluation-only `history` from route inputs and persisted ticket context.

Each input item receives exactly one result row. Provider outages, timeouts, rate limits, and invalid model JSON produce a failed `degrade` result plus an exception record for that ticket; later tickets still run, and a fully processed batch exits with status `0`. A startup or final-export failure exits with status `1`.

Before draft eligibility, a separate deterministic guardrail checks operational ticket subject, body, and descriptive context for protected intent and symptom-only requests. It can escalate a ticket even when the classifier reports an ordinary category. Evaluation labels and outcome history do not enter this check.

The classifier defaults to deterministic offline mode. Runtime retrieval uses the configured embedding model and therefore needs `OPENAI_API_KEY` and network access. If embeddings or the packaged index are unavailable, the result records an evidence gap and eligible tickets route to human review without a fabricated source. `LLM_PROVIDER` selects the classifier provider. `DRAFT_GENERATION_PROVIDER` selects the draft provider and defaults to `LLM_PROVIDER` for compatibility; set it to `OPENAI` with `OPENAI_API_KEY` to use OpenAI drafting while keeping classification offline. This sends ticket text and eligible retrieved passages to OpenAI Chat Completions. `OPENAI_LLM_MODEL` defaults to `gpt-6-luna`. The separate post-run judge uses `OPENAI_QUALITY_JUDGE_MODEL`, also defaulting to `gpt-6-luna`, regardless of either inference-provider setting. The generation prompt treats ticket and source text as untrusted data and accepts only verbatim source excerpts for factual claims; the composer labels general possibilities as unconfirmed. The model selects only an allow-listed missing-detail category for clarification; the application renders a fixed question and omits it when the requested detail is already present. Provider-authored free-form question text is rejected, so it cannot add unsupported assertions or request credentials. The offline provider selects a deterministic source excerpt and uses the same safe composition. Provider errors or invalid structured output produce a `degrade` result for that ticket and do not stop later tickets. A frozen routing policy is tied to its classifier provider and model; changing the classifier requires calibrating a new policy from development data.

Classifier intent values are validated against the public result schema. The offline classifier uses the versioned `offline-rule-based-v5` contract, which distinguishes the detailed support categories in the development fixtures, including API keys, rate limits, SSO, database, webhook, deployment/rollback, data export/residency, and billing/quota requests. Explicit former-employee account-compromise cases are labeled as security incidents, and direct feature requests remain feature requests when they mention the roadmap. Changes to those rules require a newly calibrated and frozen policy so its classifier and confidence provenance remains explicit.

For classifier calibration or batch commands that use values from `.env`, pass `--env-file .env` to `uv run` (for example, `uv run --env-file .env cloudserve run ...`). Otherwise export those variables in the shell.

After calibration, run development evaluation and inspect its report, then run validation once with the same frozen policy and thresholds. This procedure uses offline classification by default; with no API key the retrieval step records unavailable evidence and escalates safely. If you choose to rebuild the KB index, set `OPENAI_API_KEY` in `.env` first and run the ingestion command in the section below; this makes external embedding API requests and may incur provider charges.

### Reproducible development and validation gates

For a development evaluation run with retrieval enabled, configure a valid `OPENAI_API_KEY` in the ignored local `.env` file and keep `LLM_PROVIDER=OFFLINE` there. This enables query embeddings for retrieval while classification remains deterministic and offline. Do not put the key in a command, commit it, or include `.env` in an archive. The following check confirms the settings are loaded without printing the key or making a provider request:

```bash
uv run --env-file .env python -c 'from cloudserve_support.config import Settings; settings = Settings.from_env(); assert settings.provider == "OFFLINE", "Set LLM_PROVIDER=OFFLINE in .env"; assert settings.openai_api_key, "Set OPENAI_API_KEY in the ignored .env file"; print("offline classifier and embedding credential configured")'
```

The development evaluation makes external embedding API requests and may incur provider charges. Use the checked-in `data/policy/frozen_routing_policy.json` for both runs; it was calibrated from the development set and is covered by the acceptance test as frozen and evaluation-field independent. Do not run calibration between the development and validation gates. Run the development command with `.env` loaded, so the embedding credential is available without passing it on the command line. The CLI prints an end-of-run summary with predicted intent, urgency, and route counts; per-intent support, predictions, correct counts, precision/recall; answerability and protected-case summaries; and post-run label matches. Capture the run command's exit code, then run the post-run comparator against the run ID printed by the CLI:

Run the development command first and reconcile every input, result, decision, and SQLite audit record. Label comparisons remain visible diagnostics and do not require exact equality. The criteria-specific development tasks evaluate the blocking thresholds before validation; comparable-group metrics and their under-five-point target are reported but do not gate progression under PRD v2.6.

```bash
set +e
uv run --env-file .env cloudserve run --input data/tickets/development_tickets.json --output-root outputs/development-evaluation --database outputs/development-evaluation.sqlite3 --policy data/policy/frozen_routing_policy.json --label "development evaluation"
run_exit=$?
set -e
uv run --env-file .env python scripts/compare_development_run.py --input data/tickets/development_tickets.json --run-dir outputs/development-evaluation/<RUN_ID> --database outputs/development-evaluation.sqlite3 --cli-exit-code "$run_exit"
```

Replace `<RUN_ID>` with the ID printed by `cloudserve run`. The comparator opens evaluation labels only after confirming the run is terminal and loading its persisted operational records. It writes per-ticket mismatches to `label_comparison.json` beside the run exports. Label mismatches do not fail the CS-020 run-integrity gate. Exit `1` means the CLI exit status, record reconciliation, evaluation boundary, frozen provenance, or automatic-send safety check failed; exit `2` means the comparator could not complete.

The comparator verifies each persisted policy-file, threshold, classifier-implementation, and classifier-model fingerprint against the current run design. When comparing a later gate with a completed development gate, pass its `metrics.json` with `--expected-provenance`; a difference fails reconciliation.

Only after all blocking development criteria have passed, run validation once with the same frozen policy and credential configuration. The group-quality target remains reported but does not block this transition:

```bash
uv run --env-file .env cloudserve run --input data/tickets/validation_tickets.json --output-root outputs/validation-evaluation --database outputs/validation-evaluation.sqlite3 --policy data/policy/frozen_routing_policy.json --label "validation evaluation"
```

The `.env.example` default intentionally leaves `OPENAI_API_KEY` empty. With that default, or whenever query embeddings are unavailable, the application safely records unavailable retrieval evidence and routes without a source. That credential-free offline mode is useful for smoke runs, but it cannot establish whether retrieval and citation-quality criteria pass. A configured key does not bypass packaged-index freshness checks: schema-3 validation compares the current canonical passage records with the packaged Chroma records before retrieval. The current packaged index passes the local retrieval check.

For each completed run, verify `metrics.json` reports `evaluation.integrity.status` as `complete` and `evaluation.run_observed.technical.decision_log_reconciliation.value` as `1`. Reconcile the input count with `results.jsonl`, `decision_log.jsonl`, SQLite `ticket_runs`, and SQLite `audit_events`; confirm `exceptions.jsonl` contains any per-ticket failures. The four-ticket acceptance smoke test performs these checks using an input file outside the repository and temporary output/database paths.

Each `run` writes mechanical post-processing metrics. These include per-class classification results, latency, private-data output checks, decision/audit coverage, confidence calibration, group-quality comparisons, citation resolution, source provenance, exact claim-to-passage matches, and coverage. Under PRD v2.6, the scored p95 processing-latency target excludes draft-generation model wait while including retrieval and durable outcome/audit persistence; generation-inclusive latency is reported separately. Group variation retains its under-five-point target but is report-only and non-gating. Fixture labels and history are read only after operational decisions have been persisted and never appear in `results.jsonl` or decision logs. Mechanical citation/source-integrity checks do not establish semantic correctness or hallucination.

After a full development or validation run, use the separate post-run judge command. It selects exactly 20 distinct generated drafts reproducibly, sends only the ticket request, draft, cited eligible passages, and a matching reference response/rubric when available, then writes `quality_judgments.jsonl` and `evaluation.criteria.llm_judge` in `metrics.json`:

```bash
uv run --env-file .env cloudserve evaluate-quality \
  --run-dir outputs/development-evaluation/<RUN_ID> \
  --input data/tickets/development_tickets.json \
  --ground-truth data/evaluation/ground_truth_responses.json
```

The judge uses `OPENAI_QUALITY_JUDGE_MODEL=gpt-6-luna`, independently of `LLM_PROVIDER` and `OPENAI_LLM_MODEL`. Keep `LLM_PROVIDER=OFFLINE` for deterministic operational classification. Set `OPENAI_API_KEY` in the ignored `.env`; judging ticket and response text makes an external OpenAI request and may incur charges. The report records the fixed sample seed, input and reference hashes, model and prompt/rubric identity, selected ticket IDs, all numerators and denominators, and the before/after result and decision-log hashes. The three model-estimated criteria are mean response quality >=4.0/5, citation support >=95%, and unsupported factual claims in <=5% of judged responses. Results are estimates, not human ratings or actual customer satisfaction. A completed judge command returns status 0 even when a target is `fail`; fewer than 20 valid judgments are marked incomplete and return status 1. The ticket-processing run remains complete either way.

Run the development judge before the development criteria gate. Only after the final development criteria gate has passed, run validation once with the same frozen policy and credential configuration, then judge that completed validation run using its input path:

```bash
uv run --env-file .env cloudserve evaluate-quality \
  --run-dir outputs/validation-evaluation/<RUN_ID> \
  --input data/tickets/validation_tickets.json \
  --ground-truth data/evaluation/ground_truth_responses.json
```

The `.env.example` default intentionally leaves `OPENAI_API_KEY` empty. With that default, or whenever query embeddings are unavailable, the application safely records unavailable retrieval evidence and routes without a source. That credential-free offline mode is useful for smoke runs, but it cannot establish whether retrieval and citation-quality criteria pass. A configured key does not bypass packaged-index freshness checks: schema-3 validation compares the current canonical passage records with the packaged Chroma records before retrieval. The current packaged index passes the local retrieval check.

Run-observed volume, business, technical, and governance measures are grouped under `run_observed`; prior-support outcomes are grouped under `historical_baseline`. `drafts_for_agent_review` counts unsent drafts, not automatic replies. First-contact resolution and first-response time for the current run are marked unavailable because this CLI does not send replies or observe case resolution. Historical fixture resolution-time mean and median are labeled as prior-support baselines. The scored processing-latency median and p95 exclude draft-generation model wait; generation duration and full generation-inclusive p95 are reported separately. Human ratings and actual customer satisfaction remain unmeasured; the separate GPT-6 Luna scorecard is explicitly marked as a model-estimated proxy.

Set `OPENAI_API_KEY` in `.env` for KB ingestion and any explicitly configured provider-backed run. Keep `LLM_PROVIDER=OFFLINE` for deterministic classification. `DRAFT_GENERATION_PROVIDER` defaults to `LLM_PROVIDER` and can be set to `OPENAI` independently when model-generated drafts are intended. `OPENAI_EMBEDDING_MODEL` defaults to `text-embedding-3-small`; `OPENAI_LLM_MODEL` and `OPENAI_QUALITY_JUDGE_MODEL` default to `gpt-6-luna`.

Create or refresh the persistent `cloudserve_kb` vector store and ingest every article passage:

```bash
uv run --env-file .env python scripts/ingest_kb.py
```

The ingestion command is safe to rerun. It reads article text from `data/kb/documentation.json` and project-owned eligibility/citation metadata from `data/kb/metadata_manifest.json`, splits articles at level-two section headings, embeds passages with the configured embedding model, upserts stable passage IDs, removes records for deleted or changed passages, and writes `data/index/manifest.json` with source hashes and a canonical digest of indexed passage IDs, text, and metadata. Set `OPENAI_EMBEDDING_MODEL` in `.env` to select another embeddings model; rebuilding re-embeds the full corpus so stored vectors use one consistent model. The JSON corpus and its metadata overlay are the rebuildable source of truth. Rebuild after article text or indexed metadata changes; runtime also checks all stored passage records against the current derived corpus before returning a source.

The packaged corpus has 29 articles. The project metadata manifest names the capstone maintainer role, provenance, review procedure, update rule, `capstone-1.0` article versions, and ISO review dates. It marks 27 articles approved for this demonstration, one expired, and one unapproved. These decisions support reproducible capstone retrieval only; they do not claim CloudServe production approval, technical-policy validation, or production freshness. Retrieval checks status, the configured review interval, provenance, and a minimum similarity of `0.45`, and returns no source when evidence is weak. The relevance floor is an initial capstone heuristic; retrieval accuracy and later development-only threshold calibration are measured separately from routing.

Initialize the local runtime SQL tables with:

```bash
uv run python scripts/init_runtime_db.py
```

This creates `data/runtime/cloudserve.sqlite3` with `processing_runs`, `ticket_runs`, and `audit_events`, and safely adds the `label` and `output_dir` columns to an older database. The run table tracks each input batch, the ticket table tracks each ticket occurrence within a run, and the audit table records decisions and human-review events. It is safe to rerun and preserves existing records. Use `--database PATH` to initialize a different SQLite file. See [docs/sql-database-schema.md](docs/sql-database-schema.md) for the full schema, relationships, example lifecycle, and the distinction between SQLite and `outputs/`.

Structured ticket classification and draft generation can use `gpt-6-luna`; set `OPENAI_LLM_MODEL` to change the model, `LLM_PROVIDER` to change the classifier, and `DRAFT_GENERATION_PROVIDER` to change draft generation independently. The offline classifier and extractive draft adapter need no model API, while retrieval needs the embedding API. Every factual draft claim remains an exact excerpt tied to a returned passage ID and source reference; general guidance is explicitly framed as unconfirmed for the customer’s case. The result records citation, privacy, injection, source eligibility, and output checks in both the result and SQLite audit. Drafts always require agent approval and have `send_status: not_sent`. Escalation and degradation results contain a handoff with issue, operational context, eligible sources, attempted steps, uncertainty, reason, owner, and next action. Downstream dashboard or case-management tools own assignment, agent review, approval, and any later send action; this CLI does not assign or send cases. As of November 9, 2026, OpenAI lists `text-embedding-3-small` at $0.02 per million input tokens and `gpt-6-luna` at $0.10 input / $0.50 output per million tokens at standard pricing. Prices and model availability can change; see the [OpenAI API pricing page](https://developers.openai.com/api/docs/pricing).

To add or update the Chroma dependency as a maintainer, use `uv add` and commit both `pyproject.toml` and `uv.lock`:

```bash
uv add 'chromadb>=1.5.9,<2'
```

The vector record design is documented in [docs/vector-store-schema.md](docs/vector-store-schema.md).

The A1–A12 acceptance evidence map, including the exact tests and clean-checkout limitations, is in [docs/acceptance-evidence.md](docs/acceptance-evidence.md).
