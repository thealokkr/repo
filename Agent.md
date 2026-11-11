# Agent Guide

## Repository purpose

This repository contains the self-contained CLI capstone for CloudServe Support Automation. It is intended to be zipped and run by another person without relying on the original assignment folder or machine-specific paths. Source code, packaged KB articles, sample datasets, the local vector index, and generated runtime artifacts belong under this repository.

## Current layout

- `data/kb/` — packaged source knowledge base. `documentation.json` holds article text; `metadata_manifest.json` is the capstone-owned overlay for version, review date, and demo eligibility.
- `data/index/` — persistent Chroma vector store built from `data/kb/`. It is derived data and must be rebuildable from the packaged corpus.
- `docs/` — implementation and design notes for the repo. See [`vector-store-schema.md`](docs/vector-store-schema.md) for the passage schema, retrieval design, requirement evidence, and storage trade-offs, and [`sql-database-schema.md`](docs/sql-database-schema.md) for the SQLite runtime schema, table purposes, relationships, and how it differs from output exports.
- `.agents/skills/` — Codex-discoverable repo workflows: `create-tasks`, `implement-task`, and `kiss`.
- `.workarea/` — the only location for development feature specs, queue state, implementation logs, and temporary review packets. See [`.workarea/README.md`](.workarea/README.md).
- `data/tickets/` — development and validation ticket sets.
- `data/evaluation/` — reference answers and evaluation-only material. Do not expose labels or reference answers to the ticket-processing path.
- `data/incoming/` — location for additional ticket files supplied for a run. The CLI must also accept an explicit input path.
- `data/runtime/cloudserve.sqlite3` — local SQLite decision and audit database, created with `uv run python scripts/init_runtime_db.py`.
- `outputs/` — per-run result exports, metrics, and exception reports. These are derived from persisted decisions; do not use them as a second source of truth.
- `src/cloudserve_support/` — application package, separated by domain, ingestion, classification, knowledge retrieval, routing, generation, guardrails, persistence, evaluation, and reporting responsibilities.
- `tests/` — unit, integration, and acceptance tests.
- `scripts/` — repeatable developer and demo helpers.
- `.agents/skills/` — Codex-discoverable repo workflows: `create-tasks`, `implement-task`, and `kiss`.

## Data and portability rules

- Use repository-relative paths resolved from configuration or the repository root. Never hard-code the current user's absolute path.
- New ticket files must be accepted by input path and processed when they match the documented ticket schema and supported channels. Validate each record; a malformed record must produce a visible error result and must not silently disappear from a batch.
- Operational inference must not use evaluation labels or ground-truth responses. Use those only after processing to calculate evaluation metrics.
- Keep the source KB and Chroma index together in the package. Provide a repeatable index rebuild operation so a recipient can recreate the index from `data/kb/documentation.json`.
- To ingest or refresh KB articles, run `uv run --env-file .env python scripts/ingest_kb.py` from the repo root after setting `OPENAI_API_KEY` in the ignored `.env` file. The script reads `data/kb/documentation.json` and `data/kb/metadata_manifest.json`, creates passage embeddings with `OPENAI_EMBEDDING_MODEL` (default `text-embedding-3-small`), writes `data/index/chroma/` and a source-hash-linked `data/index/manifest.json`, and preserves project-owned eligibility and citation provenance. The manifest's authority is limited to the capstone demo and does not represent CloudServe production governance. Never place API keys in the repository or index metadata.
- Use `OPENAI_LLM_MODEL` (recommended default `gpt-6-luna`) for LLM drafting. `DRAFT_GENERATION_PROVIDER` defaults to `OPENAI`; set it to `OFFLINE` only to select the optional extractive draft adapter. Ticket classification uses the Decisions API with its fixed supported model, `gpt-6-luna`. Clarification prompts use an allow-listed missing-detail category and fixed application-rendered text; never emit arbitrary model-authored follow-up questions. The ingestion command calls only the embedding endpoint.
- Put all development task plans and progress artifacts under `.workarea/`; do not create a second task folder or queue. Keep Codex skill instructions under `.agents/skills/`.
- SQLite is the durable, queryable audit record for ticket state transitions and final outcomes. Output files are per-run exports generated from those records; some summary fields may intentionally appear in both.
- The SQLite initializer creates `processing_runs`, `ticket_runs`, and `audit_events` without deleting existing data. `processing_runs` tracks one batch invocation, `ticket_runs` tracks each ticket occurrence per run, and `audit_events` stores its chronological decision and human-action history. Store cited passage snapshots in `audit_events.sources_json`. See [`docs/sql-database-schema.md`](docs/sql-database-schema.md) for column meanings, relationships, and a flow example.
- Never package API keys or other secrets. Document any model-provider credential and network requirements for recipients.
- Keep `.venv`, caches, secrets, local SQLite runtime state, and generated outputs out of a distributable ZIP unless a task explicitly calls for them. The packaged KB, ticket fixtures, and Chroma index are intentional project data.

## Scope and behavior

- The interface is CLI-first; a customer-facing UI is not required for Release 1.
- Release 1 produces drafts for agent review and escalates protected, unclear, unsupported, or low-confidence cases. It does not automatically send replies.
- Preserve channel, ticket context/history, source identity and freshness metadata, route reason, validation results, agent edits/approval, outcome, timestamps, and processing errors in the audit trail as the implementation supports them.
- Keep retrieval, routing, model-provider, and persistence boundaries replaceable and independently testable.
- Runtime retrieval uses only normalized operational ticket subject and body. An unavailable index/provider, a stale or unapproved article, missing provenance, or weak relevance yields no source and routes safely for human review. Fixture `labels` and `history` remain evaluation-only.

## Project requirements

### Source of truth

The assignment source documents are in the sibling `../Capstone_Pack/Docs/` directory; they are development references, not runtime dependencies or files to package into this repository. Start with [`../Capstone_Pack/agent.md`](../Capstone_Pack/agent.md) for the pack index and [`../Capstone_Pack/Docs/01_Read_First/README.md`](../Capstone_Pack/Docs/01_Read_First/README.md) for the intended reading order. Prefer editable `.md`, `.docx`, and `.json` files when available; PDFs are rendered copies.

Consult the documents that apply to the task:

- **Assignment and acceptance criteria:** [`00_PROJECT_INSTRUCTIONS.docx`](../Capstone_Pack/Docs/00_PROJECT_INSTRUCTIONS.docx), [`01_Read_First/00_Start_Here.docx`](../Capstone_Pack/Docs/01_Read_First/00_Start_Here.docx), [`01_Read_First/01_Project_Brief.docx`](../Capstone_Pack/Docs/01_Read_First/01_Project_Brief.docx), and especially [`01_Read_First/02_Build_Specification.docx`](../Capstone_Pack/Docs/01_Read_First/02_Build_Specification.docx).
- **Product and implementation decisions:** [`02_Stage_Workbooks/Stage_2_PRD_filled.docx`](../Capstone_Pack/Docs/02_Stage_Workbooks/Stage_2_PRD_filled.docx), [`02_Stage_Workbooks/Stage_3_Prompt_Library_filled.docx`](../Capstone_Pack/Docs/02_Stage_Workbooks/Stage_3_Prompt_Library_filled.docx), [`02_Stage_Workbooks/Stage_4_Sprint_Plan_filled.docx`](../Capstone_Pack/Docs/02_Stage_Workbooks/Stage_4_Sprint_Plan_filled.docx), and [`02_Stage_Workbooks/Stage_5_PRD_Revision_Log.docx`](../Capstone_Pack/Docs/02_Stage_Workbooks/Stage_5_PRD_Revision_Log.docx). Check the revision log for changes to the original PRD.
- **Evaluation and governance:** [`03_Reference/Evaluation_Framework.docx`](../Capstone_Pack/Docs/03_Reference/Evaluation_Framework.docx), [`03_Reference/Governance_Framework.docx`](../Capstone_Pack/Docs/03_Reference/Governance_Framework.docx), and [`03_Reference/Setup_Guide.docx`](../Capstone_Pack/Docs/03_Reference/Setup_Guide.docx).
- **Data and submission:** [`05_Datasets/Dataset_Guide.docx`](../Capstone_Pack/Docs/05_Datasets/Dataset_Guide.docx), [`05_Datasets/Stakeholder_Interviews.docx`](../Capstone_Pack/Docs/05_Datasets/Stakeholder_Interviews.docx), and the development, validation, documentation, and ground-truth JSON files in `05_Datasets/`; see also [`04_Submission/Submission_Guide.docx`](../Capstone_Pack/Docs/04_Submission/Submission_Guide.docx).

If requirements appear to conflict, verify the relevant source documents and their revisions before changing behavior; do not infer requirements from generated outputs or sample data alone. Do not modify the source pack unless explicitly asked.

In particular, maintain support for all four channels, deterministic routing, source-grounded citations, guardrail blocking, persistent decision logs, unattended evaluation with arbitrary input/output paths, graceful failure, and metrics reporting.
