# Runtime SQL Database Design

## Purpose

The runtime SQLite database is the durable, queryable record of ticket processing. It stores which input was processed, the status of each ticket within a run, and the sequence of automated and human decisions made for that ticket. Result files under `outputs/` are export artifacts; they can be regenerated from persisted run and audit data and are not a second system of record.

The default database file is `data/runtime/cloudserve.sqlite3`. Initialize or verify its tables with:

```bash
uv run python scripts/init_runtime_db.py
```

The initializer creates the schema only. It does not ingest tickets or write processing results. Use `--database PATH` to initialize another SQLite file.

Schema version 2 adds nullable `label` and `output_dir` columns to `processing_runs`. Initializing an existing version 1 database adds these columns in place and preserves its run, ticket, and audit rows.

## Table responsibilities

| Table | One row represents | Purpose |
| --- | --- | --- |
| `processing_runs` | One invocation processing an input file or batch | Records run-level input reference, optional label and output directory, start and completion times, overall status, system version, and any run-level error. |
| `ticket_runs` | One ticket as handled in one run | Associates the ticket with its run and records ticket context, processing status, and ticket-level errors. The composite key allows the same ticket to be processed again in a later run. |
| `audit_events` | One processing, decision, or human-review event for a ticket in a run | Preserves the chronological evidence for classification, retrieval, draft generation, guardrail checks, escalation, agent edits/approval, and outcome as those stages are implemented. Includes model/prompt identifiers, decisions, reasons, checks, cited source snapshots, changes, and outcome fields. |

## Relationships

```text
processing_runs (1) ──── (many) ticket_runs (1) ──── (many) audit_events
                         PK: run_id, ticket_id
```

- `ticket_runs.run_id` references `processing_runs.run_id`.
- `audit_events.(run_id, ticket_id)` references `ticket_runs.(run_id, ticket_id)`.
- Both relationships cascade updates and deletes. Deleting a run therefore removes its ticket rows and their audit events; normal run initialization does not delete anything.
- `idx_ticket_runs_status` supports finding tickets by run and processing status.
- `idx_audit_events_ticket_time` supports reading a ticket's event history in time order.

## Stored fields and boundaries

### `processing_runs`

| Column | Meaning |
| --- | --- |
| `run_id` | Unique identifier for the processing invocation; primary key. |
| `input_ref` | Path or identifier for the ticket input used by the run. |
| `started_at`, `completed_at` | Run start and optional completion timestamps. |
| `status` | `running`, `completed`, `completed_with_errors`, or `failed`. |
| `system_version` | Version of the application or processing configuration used. |
| `error_message` | Run-level failure details, when applicable. |
| `label` | Optional human-readable label supplied to `cloudserve run --label`. |
| `output_dir` | Per-run export directory, recorded before ticket processing starts. |

### `ticket_runs`

| Column | Meaning |
| --- | --- |
| `run_id`, `ticket_id` | Composite primary key and association to the run. |
| `channel`, `received_at` | Ticket channel and received timestamp. |
| `customer_tier`, `language_fluency` | Ticket attributes relevant to routing and evaluation. |
| `context_json` | Additional ticket context and conversation history as JSON text. |
| `processing_status` | `pending`, `running`, `completed`, or `failed`. |
| `error_code`, `error_message` | Ticket-level processing error, when applicable. |

### `audit_events`

| Column | Meaning |
| --- | --- |
| `event_id` | Unique identifier for this event; primary key. |
| `run_id`, `ticket_id` | Identifies the ticket occurrence this event belongs to. |
| `occurred_at`, `event_type` | Event timestamp and stage/action category. |
| `actor_type`, `actor_id` | Whether the action came from the system, model, or a human, and optional actor identifier. |
| `input_summary` | Brief record of the relevant input at this event. |
| `model_name`, `model_version`, `prompt_version` | Model and prompt provenance when a model was used. |
| `requirement_ids_json` | Requirement identifiers applied at this stage. |
| `prediction_json`, `alternatives_json`, `threshold_applied` | Predicted classification and applicable confidence threshold. `alternatives_json` remains an empty audit field; the classifier exports only its selected intent and urgency. |
| `action_taken`, `reason` | Resulting action and explanation, including escalation or review rationale. |
| `guardrail_results_json` | Guardrail checks and their results. |
| `sources_json` | Snapshot of cited KB article/passage identity and supporting source details used for the decision. This preserves what was cited even if the index is later rebuilt. |
| `changes_json` | Agent or system changes made to a draft or decision. |
| `approval_action` | Human approval, rejection, or other review action when present. |
| `result_payload_json`, `outcome` | Event result details and eventual ticket outcome. |

Fields ending in `_json` are SQLite `TEXT` columns containing JSON-encoded values. They keep the schema small while allowing the event payloads and ticket context to evolve without a table for every intermediate object. Application code should validate these payloads when writing and reading them.

## Example lifecycle

For a batch containing tickets `T-104` and `T-105`:

1. Insert one `processing_runs` row for the batch.
2. Insert one `ticket_runs` row for each ticket using the same `run_id`.
3. Append `audit_events` rows as each ticket is classified, searched, drafted, validated, or sent for agent review. A source-grounded decision stores the cited passage snapshot in `sources_json`.
4. Update each ticket's `processing_status`, and update the run's final status and completion time.
5. Generate a human-readable or machine-readable export under `outputs/` from the persisted run, ticket, and event records.

If `T-104` is submitted again later, it gets a new `ticket_runs` row because its `run_id` differs. Its history from earlier runs remains queryable under the previous run.

## Why these three tables

- The run table avoids repeating batch-level input and status on every event.
- The ticket occurrence table keeps each ticket's context and processing state attached to the run in which it was processed, including repeat submissions.
- The event table preserves multiple automated and human decisions without adding a table for each pipeline stage. This supports explainability, review, and later audit queries.
- The KB remains in Chroma; SQLite stores the source references actually used in each decision, not vector embeddings or a duplicate copy of the KB.
- `outputs/` is for portable per-run deliverables and reports, while SQLite remains the structured operational record used to inspect and derive them.

The schema is intentionally limited to runtime execution and audit needs. Evaluation ground-truth labels remain in `data/evaluation/` and must not enter the ticket-processing path.

The runtime audit reader returns a run's events ordered by `occurred_at`, with `event_id` as a stable tie-breaker. Each ticket's final event preserves the recommended route and reason, eligible source snapshots, guardrail results, and the complete result payload so a reviewer can replay the decision evidence from SQLite.

## Current implementation status

`scripts/init_runtime_db.py` creates or verifies these three tables, indexes, foreign keys, and schema version 2. It preserves existing records and initializes an empty database when none exists. The `cloudserve run` path writes one ticket and audit row for each input item, then exports results from those audit records. Classification, retrieval, source-bound drafting, guardrail checks, and complete handoffs are recorded in each ticket's result payload and audit event. Drafts remain unapproved and unsent; downstream dashboard or case-management tools own assignment, approval, and any later send action.
