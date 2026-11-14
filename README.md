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
- `scripts/` — initialization utilities.
- `.agents/skills/` — development-only Codex workflows; omitted from the recipient ZIP.
- `.workarea/` — development-only feature specs, queue state, implementation logs, and review packets; omitted from the recipient ZIP.
- `scripts/ingest_kb.py` — rebuilds the passage index from the packaged KB using Google Gemini embeddings.

See [Agent.md](Agent.md) for repository conventions, data handling rules, and scope.

## Python and Chroma setup

The project uses Python 3.13 and `uv`. The Chroma dependency is declared in `pyproject.toml` and pinned by `uv.lock`.
`jsonschema` is declared for output-contract validation; `pytest` is a development dependency for the test suite.

From a clean checkout, run the following from the repository root. Copy the example configuration to the ignored local `.env`; it contains no credential, so add an API key before processing tickets:

```bash
uv sync --frozen
cp .env.example .env
uv run python scripts/init_runtime_db.py
uv run python scripts/init_output_schemas.py
uv run cloudserve --help
```

On Windows PowerShell, use `Copy-Item .env.example .env`. Set `GEMINI_API_KEY` or `GOOGLE_API_KEY` in `.env` for ticket classification, retrieval, Gemini drafting, post-run judging, and KB ingestion. `.env` is ignored by Git; never commit it or include it in a distributable archive. Without a key, ticket classification returns a safe fallback and routes tickets for human review.

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

The single test command for the repository is `uv run pytest`. The suite includes a live Decisions API classification test using 201 synthetic tickets, which makes two API requests. It reads `GEMINI_API_KEY` or `GOOGLE_API_KEY` from the shell or ignored `.env` file and requires network access.

## Process a ticket batch

Routing policy is configuration, so keep the calibrated policy in the `conf/` folder rather than in per-run output. Calibration writes `conf/frozen_routing_policy.json` by default, and the CLI run command reads that same file by default. Calibrate once from development tickets, review and commit the resulting configuration, then run development, validation, or other ticket batches without specifying `--output` or `--policy`:

```bash
uv run cloudserve calibrate --development-input data/tickets/development_tickets.json
uv run cloudserve run --input data/tickets/development_tickets.json --output-root outputs/development --database outputs/development.sqlite3 --label "development batch"
```

The run command has no interactive prompts or retry steps. It prints progress for classification, retrieval, routing, ticket processing, and export finalization, then shows the output folder and the `metrics.md` content, including its Markdown metrics table. With `--label`, the output folder is named `<safe-label>_YYYY-MM-DD_HH-MM` using the UTC start time rounded to the minute (for example, `development-batch_2026-11-14_14-35`); spaces and other unsupported filename characters become hyphens. Without a usable label, the run ID is used as the folder name. Each invocation writes `results.jsonl`, `decision_log.jsonl`, `metrics.json`, `metrics.md`, `exceptions.jsonl`, and one `failed_records.json` listing each mismatched ticket ID and its failed label KPIs. `results.jsonl` emits one flat, diagnostic-ready row per input occurrence (`run_id` and `input_index`) with the original supplied `ticket_id`, `channel`, `subject`, and `body` (plus optional `received_at`, `customer_tier`, and `language_fluency` when present in the input), six flat `expected_*`/`predicted_*` label pairs (`intent`, `urgency`, `route`, `answerable_from_docs`, `doc_ids`, and `must_not_auto_respond`), one shared `classifier_confidence` with classifier metadata, routing threshold/policy/rule/retrieval diagnostics, per-source retrieval `similarity` (similarity, not calibrated confidence), `draft` (`draft.plain_text` when generated), non-duplicative `handoff`, guardrails, generation metadata, and timings. Expected labels are attached only after processing from the persisted comparison view (`null` when missing or invalid, and all six `predicted_*` fields are `null` when comparison-view predictions are unavailable); this breaking export contract removes `generated_labels`, `draft_response`, duplicate top-level `route`, nested `classification` predictions, routing `evidence_ready`/`protected_case` duplicates, and repeated handoff context/sources/reason. `metrics.md` summarizes ticket totals, persisted run wall time, and per-label match counts and rates. It also writes the SQLite run, ticket, and audit rows to `--database`. `--label` is optional metadata. Exit status `0` means the batch and exports completed, including when a ticket has a recorded degradation or malformed-record result; exit status `1` means startup, persistence, or export finalization failed. Calibration errors return `2`.

The routing calibration command uses Google Gemini to classify development ticket text after normalization, excludes invalid records, and never reads fixture `labels` or outcome `history`. It selects a confidence floor from the 20th percentile of non-protected development confidence scores, clamps the floor to `[0.65, 0.98]`, and writes the frozen policy with an operational-input hash and calibration counts. Calibration requires a valid API key and uses the same Decisions batch classifier as ticket runs. Do not calibrate with validation data. A validation run reads the same frozen policy and records its full provenance in every result and audit event.

The public route enum remains `draft_for_review`, `escalate`, and `degrade`. For evaluation documentation only, fixture label `auto_respond` maps to `draft_for_review`, and fixture label `escalate` maps to `escalate`. Stage 3 prompt term `draft_for_agent_review` maps to `draft_for_review`; `escalate_for_human_review` maps to `escalate`. The `must_not_auto_respond` fixture flag is a safety oracle, not a route value. None of these fixture fields are read by the operational classifier or router. The processing path excludes `labels` and evaluation-only `history` from route inputs and persisted ticket context.

Each input item receives exactly one result row. A failed Decisions batch degrades every eligible ticket in that chunk and is never retried ticket by ticket; later chunks and sequential downstream processing continue. Draft-provider failures and invalid draft output produce a failed `degrade` result plus an exception record for the affected ticket. A fully processed batch exits with status `0`; a startup or final-export failure exits with status `1`.

Before draft eligibility, a separate deterministic guardrail checks operational ticket subject, body, and descriptive context for protected intent and symptom-only requests. It can escalate a ticket even when the classifier reports an ordinary category. Evaluation labels and outcome history do not enter this check.

Ticket classification uses Google Gemini's Decisions API with `gemini-decision-1` (`v1beta`) with named typed choices for intent and urgency. The CLI submits the full set of valid, unblocked tickets to the batch classifier before downstream processing. While the Decisions API allows up to 400 questions per request (200 tickets), the application bounds classification chunks to at most 25 tickets (50 questions) per request, and uses smaller chunks when the shared input reaches the API size limit. Capping application chunks to at most 25 tickets splits larger workloads (such as the 80-ticket validation set into 25+25+25+5) to bound per-ticket batch wait time (`classification_request_duration_seconds`) and assist latency while preserving the Decisions API's 400-question ceiling; the full 80-ticket validation retest remains the gate for the 3.0-second scored p95 SLA. It maps answers back by ticket position; retrieval, drafting, and persistence then run sequentially. A failed chunk is not retried as individual ticket requests. The Decisions API uses `gemini-decision-1`; `GEMINI_LLM_MODEL` configures Gemini draft generation (`gemini-3.8-flash`). Classification results contain intent, urgency, confidence, and model provenance. Refusals, missing answers, malformed choices, and provider failures produce safe unclear fallbacks for the affected chunk. Retrieval uses `gemini-embedding-001` at 768 dimensions with manual L2 normalization, and drafting uses `gemini-3.8-flash` by default; `DRAFT_GENERATION_PROVIDER=OFFLINE` can explicitly select the extractive draft adapter. Classification, retrieval, and Gemini drafting require `GEMINI_API_KEY` or `GOOGLE_API_KEY` and network access. If embeddings or the packaged index are unavailable, the result records an evidence gap and eligible tickets route to human review without a fabricated source. The separate post-run judge uses `GEMINI_QUALITY_JUDGE_MODEL`, defaulting to `gemini-3.8-flash`. The generation prompt treats ticket and source text as untrusted data and accepts only verbatim source excerpts for factual claims; the composer labels general possibilities as unconfirmed. The model selects only an allow-listed missing-detail category for clarification; the application renders a fixed question and omits it when the requested detail is already present. Provider-authored free-form question text is rejected, so it cannot add unsupported assertions or request credentials. A frozen routing policy is tied to its classifier model; changing the classifier requires calibrating a new policy from development data.

Classifier intent values are validated against the public result schema. Gemini output is constrained to supported intent and urgency values, with confidence validated before routing. Urgency follows the owner-confirmed operational impact rubric: `high` is assigned to critical customer-facing outages, active security incidents, account compromise, confirmed data loss or exposure, suspected exposure or compromise of credentials in a breach at another service when credentials could access this service (even before confirmed misuse), or similar severe current impact; `medium` is the standard default for operational problems, partial failures, errors, build or deployment issues, bounded API key permission or access issues, webhook issues, or bounded service degradation; `low` covers general questions, how-to inquiries, routine invoice or billing clarifications, routine data-residency inquiries, feature requests, and staging/sandbox environments (routine invoice and data-residency questions are not high); `unknown` is an exceptional fallback reserved for blank or unreadable text. Provider failures or invalid output are marked as fallbacks and sent for manual review.

CloudServe CLI commands load values from the repository-root `.env` automatically. Values already exported in the shell take precedence. Direct Python commands that do not load `.env` themselves can use `uv run --env-file .env` or exported shell variables.

After Gemini calibration, run development evaluation and inspect its report, then run validation once with the same frozen policy and thresholds. Without an API key, classification returns a safe fallback and retrieval records unavailable evidence. If you choose to rebuild the KB index, set `GEMINI_API_KEY` or `GOOGLE_API_KEY` in `.env` first and run the ingestion command in the section below; this makes external embedding API requests and may incur provider charges.

### Reproducible development and validation gates

For a development evaluation run, configure a valid `GEMINI_API_KEY` or `GOOGLE_API_KEY` in the ignored local `.env` file. Ticket classification, retrieval embeddings, and drafting use Google Gemini. Do not put the key in a command, commit it, or include `.env` in an archive. The following check confirms the settings are loaded without printing the key or making a provider request:

```bash
uv run --env-file .env python -c 'from cloudserve_support.config import Settings; settings = Settings.from_env(); assert settings.gemini_api_key, "Set GEMINI_API_KEY or GOOGLE_API_KEY in the ignored .env file"; print("Gemini classification, embedding, and drafting credential configured")'
```

The development evaluation makes external Gemini API requests and may incur provider charges. Calibrate a policy from development tickets with the supported `gemini-decision-1` Decisions classifier before the development gate. Calibration writes the active policy to `conf/frozen_routing_policy.json`, which the CLI automatically reads for development and validation runs. Calibration classifies the development set in batches of up to 25 tickets per Decisions request. Do not recalibrate between the development and validation gates. Run the development command with `.env` loaded, so credentials are available without passing them on the command line. The CLI prints an end-of-run summary with predicted intent, urgency, and route counts; per-intent support, predictions, correct counts, precision/recall against the project-set 85% target; per-urgency precision by class against the project-set 70% target with recall and exact agreement diagnostics; answerability and protected-case summaries; and post-run label matches.

Create the Gemini-calibrated policy once:

```bash
uv run cloudserve calibrate \
  --development-input data/tickets/development_tickets.json
```

Run the development command first and reconcile every input, result, decision, and SQLite audit record. Label comparisons remain visible diagnostics and do not require exact equality. The criteria-specific development tasks evaluate the blocking thresholds before validation; comparable-group metrics and their under-five-point target are reported but do not gate progression under PRD v2.6.

```bash
uv run cloudserve run --input data/tickets/development_tickets.json --output-root outputs/development-evaluation --database outputs/development-evaluation.sqlite3 --label "development evaluation"
```

Only after all blocking development criteria have passed, run validation once with the same default policy and credential configuration. The group-quality target remains reported but does not block this transition:

```bash
uv run cloudserve run --input data/tickets/validation_tickets.json --output-root outputs/validation-evaluation --database outputs/validation-evaluation.sqlite3 --label "validation evaluation"
```

The `.env.example` template intentionally leaves `GEMINI_API_KEY` and `GOOGLE_API_KEY` empty. Without a key, classification returns a safe fallback and retrieval evidence is unavailable; this only checks the missing-credential path and cannot establish whether retrieval and citation-quality criteria pass. A configured key does not bypass packaged-index freshness checks: schema-3 validation compares the current canonical passage records with the packaged Chroma records before retrieval. The current packaged index passes the local retrieval check.

For each completed run, verify `metrics.json` reports `evaluation.integrity.status` as `complete` and `evaluation.run_observed.technical.decision_log_reconciliation.value` as `1`. Reconcile the input count with `results.jsonl`, `decision_log.jsonl`, SQLite `ticket_runs`, and SQLite `audit_events`; confirm `exceptions.jsonl` contains any per-ticket failures.

Each `run` writes mechanical post-processing metrics. These include per-class classification results, latency, private-data output checks, decision/audit coverage, confidence calibration, group-quality comparisons, citation resolution, source provenance, exact claim-to-passage matches, and coverage. Under PRD v2.6, the scored p95 processing-latency target excludes draft-generation model wait while including retrieval and durable outcome/audit persistence; generation-inclusive latency is reported separately. Group variation retains its under-five-point target but is report-only and non-gating. Supplied fixture labels and history are read only after operational decisions have been persisted; operational prompts and `decision_log.jsonl` never include supplied labels, and `results.jsonl` attaches `expected_*` values only after processing from the isolated SQLite comparison snapshot. `metrics.md` compares valid values by native type, maps `auto_respond` to `draft_for_review`, compares document IDs as unordered sets, and excludes missing or invalid values from each field's match-rate denominator. Its answerability and protected-case rows compare operational proxies, not semantic ground truth. Mechanical citation/source-integrity checks do not establish semantic correctness or hallucination.

After a full development or validation run, use the separate post-run judge command. It selects exactly 20 distinct generated drafts reproducibly, sends only the ticket request, draft, cited eligible passages, and a matching reference response/rubric when available, then writes `quality_judgments.jsonl` and `evaluation.criteria.llm_judge` in `metrics.json`:

```bash
uv run cloudserve evaluate-quality \
  --run-dir outputs/development-evaluation/<RUN_ID> \
  --input data/tickets/development_tickets.json \
  --ground-truth data/evaluation/ground_truth_responses.json
```

The judge uses `GEMINI_QUALITY_JUDGE_MODEL=gemini-3.8-flash`, independently of `GEMINI_LLM_MODEL`. Set `GEMINI_API_KEY` or `GOOGLE_API_KEY` in the ignored `.env`; judging ticket and response text makes an external Gemini request and may incur charges. The report records the fixed sample seed, input and reference hashes, model and prompt/rubric identity, selected ticket IDs, all numerators and denominators, and the before/after result and decision-log hashes. The three model-estimated criteria are mean response quality >=4.0/5, citation support >=95%, and unsupported factual claims in <=5% of judged responses. Results are estimates, not human ratings or actual customer satisfaction. A completed judge command returns status 0 even when a target is `fail`; fewer than 20 valid judgments are marked incomplete and return status 1. The ticket-processing run remains complete either way.

Run the development judge before the development criteria gate. Only after the final development criteria gate has passed, run validation once with the same frozen policy and credential configuration, then judge that completed validation run using its input path:

```bash
uv run cloudserve evaluate-quality \
  --run-dir outputs/validation-evaluation/<RUN_ID> \
  --input data/tickets/validation_tickets.json \
  --ground-truth data/evaluation/ground_truth_responses.json
```

The `.env.example` template intentionally leaves `GEMINI_API_KEY` and `GOOGLE_API_KEY` empty. Without a key, classification returns a safe fallback and retrieval evidence is unavailable; this only checks the missing-credential path and cannot establish whether retrieval and citation-quality criteria pass. A configured key does not bypass packaged-index freshness checks: schema-3 validation compares the current canonical passage records with the packaged Chroma records before retrieval. The current packaged index passes the local retrieval check.

Run-observed volume, business, technical, and governance measures are grouped under `run_observed`; prior-support outcomes are grouped under `historical_baseline`. `drafts_for_agent_review` counts unsent drafts, not automatic replies. First-contact resolution and first-response time for the current run are marked unavailable because this CLI does not send replies or observe case resolution. Historical fixture resolution-time mean and median are labeled as prior-support baselines. The scored processing-latency median and p95 exclude draft-generation model wait; generation duration and full generation-inclusive p95 are reported separately. Human ratings and actual customer satisfaction remain unmeasured; the separate Gemini Flash scorecard is explicitly marked as a model-estimated proxy.

Set `GEMINI_API_KEY` or `GOOGLE_API_KEY` in `.env` for classification, KB ingestion, retrieval, and Gemini draft generation. `DRAFT_GENERATION_PROVIDER` defaults to `GEMINI`; set it to `OFFLINE` only to use extractive drafts. `GEMINI_EMBEDDING_MODEL` defaults to `gemini-embedding-001` (768 dimensions); `GEMINI_LLM_MODEL` configures drafts and `GEMINI_QUALITY_JUDGE_MODEL` configures the separate judge, both defaulting to `gemini-3.8-flash`. The Decisions classifier is fixed to `gemini-decision-1` and is not configurable through `GEMINI_LLM_MODEL`.

Create or refresh the persistent `cloudserve_kb` vector store and ingest every article passage:

```bash
uv run --env-file .env python scripts/ingest_kb.py
```

The ingestion command is safe to rerun. It reads article text from `data/kb/documentation.json` and project-owned eligibility/citation metadata from `data/kb/metadata_manifest.json`, splits articles at level-two section headings, embeds passages with the configured embedding model, upserts stable passage IDs, removes records for deleted or changed passages, and writes `data/index/manifest.json` with source hashes and a canonical digest of indexed passage IDs, text, and metadata. Set `GEMINI_EMBEDDING_MODEL` in `.env` to select another embeddings model; rebuilding re-embeds the full corpus so stored vectors use one consistent model. The JSON corpus and its metadata overlay are the rebuildable source of truth. Rebuild after article text or indexed metadata changes; runtime also checks all stored passage records against the current derived corpus before returning a source.

The packaged corpus has 29 articles. The project metadata manifest names the capstone maintainer role, provenance, review procedure, update rule, `capstone-1.0` article versions, and ISO review dates. It marks 27 articles approved for this demonstration, one expired, and one unapproved. These decisions support reproducible capstone retrieval only; they do not claim CloudServe production approval, technical-policy validation, or production freshness. Retrieval checks status, the configured review interval, provenance, and a minimum similarity of `0.45`, and returns no source when evidence is weak. The relevance floor is an initial capstone heuristic; retrieval accuracy and later development-only threshold calibration are measured separately from routing.

Initialize the local runtime SQL tables with:

```bash
uv run python scripts/init_runtime_db.py
```

This creates `data/runtime/cloudserve.sqlite3` with `processing_runs`, `ticket_runs`, and `audit_events`, and safely adds the `label` and `output_dir` columns to an older database. The run table tracks each input batch, the ticket table tracks each ticket occurrence within a run, and the audit table records decisions and human-review events. It is safe to rerun and preserves existing records. Use `--database PATH` to initialize a different SQLite file. See [docs/sql-database-schema.md](docs/sql-database-schema.md) for the full schema, relationships, example lifecycle, and the distinction between SQLite and `outputs/`.

Ticket classification uses the Decisions API with `gemini-decision-1`. Set `GEMINI_LLM_MODEL` to change the Gemini draft model, and `DRAFT_GENERATION_PROVIDER` to select Gemini or the extractive draft adapter. Classification, Gemini drafting, and retrieval use the model API.

To add or update the Chroma dependency as a maintainer, use `uv add` and commit both `pyproject.toml` and `uv.lock`:

```bash
uv add 'chromadb>=1.5.9,<2'
```

The vector record design is documented in [docs/vector-store-schema.md](docs/vector-store-schema.md).

The A1–A12 acceptance evidence map, including the exact tests and clean-checkout limitations, is in [docs/acceptance-evidence.md](docs/acceptance-evidence.md).
