# Runtime SQL Database Design

## Purpose

The runtime SQLite database is the durable, queryable record of ticket processing. It stores which input was processed, the status of each ticket within a run, and the sequence of automated and human decisions made for that ticket. Result files under `outputs/` are export artifacts; they can be regenerated from persisted run and audit data and are not a second system of record.

The default database file is `data/runtime/cloudserve.sqlite3`. Initialize or verify its tables with:

```bash
uv run python scripts/init_runtime_db.py
```

The initializer creates the schema only. It does not ingest tickets or write processing results. Use `--database PATH` to initialize another SQLite file.

Schema version 2 added nullable `label` and `output_dir` columns to `processing_runs`. Schema version 3 renamed `audit_events.result_payload_json` to `event_payload_json`. Schema version 4 adds nullable `ticket_runs.input_labels_json` and the `ticket_label_comparison` view. Initializing an existing database applies the required additive migration transactionally; prior run, ticket, and audit rows and their JSON payloads are preserved. Conflicting or incompatible audit payload columns fail initialization rather than discarding data.

## Table responsibilities

| Table | One row represents | Purpose |
| --- | --- | --- |
| `processing_runs` | One invocation processing an input file or batch | Records run-level input reference, optional label and output directory, start and completion times, overall status, system version, and any run-level error. |
| `ticket_runs` | One ticket as handled in one run | Associates the ticket with its run and records operational context, an isolated input-label snapshot, processing status, and ticket-level errors. The composite key allows the same ticket to be processed again in a later run. |
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
| `context_json` | Additional operational ticket context as JSON text; fixture labels and evaluation history are excluded. |
| `input_labels_json` | The input record's `labels` value serialized as JSON for later comparison only. Missing labels are SQL `NULL`; this snapshot is not part of the normalized ticket context or processing prompts. |
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
| `event_payload_json`, `outcome` | Event result details and eventual ticket outcome. |

Fields ending in `_json` are SQLite `TEXT` columns containing JSON-encoded values. They keep the schema small while allowing the event payloads and ticket context to evolve without a table for every intermediate object. Application code should validate these payloads when writing and reading them.

## Example lifecycle

For a batch containing tickets `T-104` and `T-105`:

1. Insert one `processing_runs` row for the batch.
2. Insert one `ticket_runs` row for each ticket using the same `run_id`.
3. Commit one `classification` event per occurrence after the full classification phase resolves.
4. Resolve and commit one `retrieval` event per occurrence. Retrieval event payloads record status and separate document/passage IDs with scores, but no passage text. The full batch retrieval phase is persisted before the first routing event.
5. Commit one `routing` event per occurrence, then a `generation` event after drafting or a safe skip resolves. Every stage event stores `{input_index, ticket_id, stage, status, value, reason}` in `event_payload_json`.
6. Append one terminal `ticket_completed` or `ticket_failed` event with the complete result and eligible source snapshots, and update the ticket's status in the same transaction.
7. Generate exports under `outputs/` from persisted events. `results.jsonl` projects one flat, diagnostic-ready row per input occurrence after processing from the original input fields, the persisted terminal outcome, and `ticket_label_comparison`; `decision_log.jsonl` contains every stage and terminal event in chronological order and retains the original terminal payload.

If `T-104` is submitted again later, it gets a new `ticket_runs` row because its `run_id` differs. Its history from earlier runs remains queryable under the previous run.

## Why these three tables

- The run table avoids repeating batch-level input and status on every event.
- The ticket occurrence table keeps each ticket's context and processing state attached to the run in which it was processed, including repeat submissions.
- The event table preserves multiple automated and human decisions without adding a table for each pipeline stage. This supports explainability, review, and later audit queries.
- The KB remains in Chroma; SQLite stores the source references actually used in each decision, not vector embeddings or a duplicate copy of the KB.
- `outputs/` is for portable per-run deliverables and reports, while SQLite remains the structured operational record used to inspect and derive them.

The schema is intentionally limited to runtime execution and audit needs. Input labels are captured only as an isolated post-processing comparison snapshot; normalized operational context, classification, routing, retrieval, and generation never receive them. Evaluation ground-truth labels in `data/evaluation/` likewise remain outside the ticket-processing path.

The runtime audit reader returns a run's events ordered by `occurred_at` and SQLite insertion order for timestamp ties. A terminal event preserves the recommended route and reason, eligible source snapshots, guardrail results, and complete result payload. Stage events preserve typed outcomes as they resolve, so a reviewer can inspect durable progress even if a later stage or run export fails.

## Ticket label comparison view

`ticket_label_comparison` returns one row for every `ticket_runs` occurrence, including tickets with missing labels or no usable terminal result. `input_labels_json` preserves the supplied `labels` value and all of its keys as JSON; it is SQL `NULL` when the input omitted `labels`. Historical rows are not backfilled. The view does not modify audit events or per-run exports.

For a completed and determinate outcome, `generated_labels_json` is JSON with exactly the six input-label keys and their native JSON types: `intent` (supported classifier-vocabulary string), `urgency` (supported classifier-vocabulary string), `expected_route` (string), `answerable_from_docs` (boolean), `expected_doc_ids` (array of strings), and `must_not_auto_respond` (boolean). The comparison view uses the same canonical intent and urgency vocabularies as classification and post-run evaluation. The values are generated only from the terminal result:

- `intent` and `urgency` come from the final classification.
- `expected_route` is the actual application route (`draft_for_review`, `escalate`, or `degrade`). Evaluation input `auto_respond` labels map to `draft_for_review` for comparison only; a draft is not sent automatically.
- `answerable_from_docs` is the final `routing.evidence_ready` signal. It indicates eligible evidence availability, not semantic answerability.
- `expected_doc_ids` is the sorted, unique set of document IDs in final eligible `sources`; source presence does not prove that the documents semantically answer the ticket.
- `must_not_auto_respond` reflects the router's `routing.protected_case` signal. This is a protected-case proxy, not a complete semantic safety judgement; all drafts remain unsent for review.

`generated_labels_status` is `available` only when exactly one `ticket_completed` event has a completed result, classified intent and urgency from the supported vocabularies, a supported actual route, boolean readiness/protected-case signals, and an array of sources with nonempty string document IDs. Otherwise it is `unavailable`, `generated_labels_json` is SQL `NULL`, and `generated_labels_reason` identifies missing/duplicate terminal events, a failed or incomplete result, invalid JSON, unsupported intent/urgency, or other missing/invalid signals. No field is guessed for failed, partial, or historical results.

Example read-only comparison query:

```sql
SELECT run_id, ticket_id, input_labels_json, generated_labels_json,
       generated_labels_status, generated_labels_reason
FROM ticket_label_comparison
ORDER BY run_id, ticket_id;
```

`generated_labels_json` is text in SQLite; when parsed as JSON, booleans remain booleans and `expected_doc_ids` remains an array (not a JSON-encoded string). The runtime initializer recreates this derived view transactionally; it does not rewrite historical `ticket_runs` or `audit_events`.

## Result labels and Markdown metrics

`results.jsonl` projects one flat, diagnostic-ready record per input occurrence (`run_id` and `input_index`) after processing completes. It preserves non-empty source `ticket_id` and `channel` strings verbatim (`null` when absent, empty, or non-string), preserves source `subject` and `body` strings verbatim including empty or whitespace strings (`null` when absent or non-string), includes optional `received_at`, `customer_tier`, and `language_fluency` when supplied, and JSON-escapes unpaired surrogates so output bytes remain valid UTF-8. Each row exposes six flat comparison pairs—`expected_intent`/`predicted_intent`, `expected_urgency`/`predicted_urgency`, `expected_route`/`predicted_route`, `expected_answerable_from_docs`/`predicted_answerable_from_docs`, `expected_doc_ids`/`predicted_doc_ids`, and `expected_must_not_auto_respond`/`predicted_must_not_auto_respond`—derived from `ticket_label_comparison` after inference. Missing or invalid expected fields are `null` (valid empty document-ID lists remain `[]`, and `expected_route` preserves raw values such as `auto_respond`), and when `generated_labels_status` is `unavailable` all six `predicted_*` fields are `null`. Each row includes one shared `classifier_confidence` (`null` on fallback) with `classifier_provider`, `classifier_model`, and `classifier_status`; `routing` threshold, source, calibration, matched-rule, policy-provenance, and retrieval diagnostics without duplicate `evidence_ready` or `protected_case` flags; detailed `sources` with per-source retrieval `similarity` (not a calibrated confidence score); `draft` (`draft.plain_text` as the sole draft text, without `draft_response`); non-duplicative `handoff` (`summary`, `issue`, `attempted_steps`, `uncertainties`, `owner`, and `next_action`); `guardrails`; `generation`; optional `error`; and timing fields. This breaking export contract replaces the earlier nested `generated_labels` wrapper while leaving persisted terminal audit payloads and `decision_log.jsonl` unchanged.

Each run also writes `metrics.md` beside the JSON exports. It summarizes total, successfully processed, failed, and labeled ticket occurrences; elapsed run wall time is calculated only from persisted `processing_runs.started_at` through `completed_at` (missing or invalid timestamps display as unavailable, never an estimated duration). Its six label rows show `Matched`, `Different`, `Not compared`, and `Match rate`. Invalid, missing, or unavailable values count as not compared; those values are excluded from the match-rate denominator, and a zero denominator displays `N/A`. `auto_respond` labels compare to actual `draft_for_review` without indicating a message was sent. Expected document IDs compare as unordered sets. Answerability and protected-case metrics compare to evidence-readiness and protected-case routing proxies, not semantic ground truth. Both inputs and generated values are read after processing from the occurrence-matched SQLite comparison view; label snapshots remain evaluation-only.

## Current implementation status

`scripts/init_runtime_db.py` creates or verifies the three runtime tables, indexes, foreign keys, the schema-version-4 nullable input-label column, and the comparison view. It migrates the v2 audit payload column without rewriting event JSON and preserves existing records. The `cloudserve run` path creates one running ticket row per input before stage work, commits classification, retrieval, routing, and generation checkpoints, and appends one terminal event per occurrence. Exports are regenerated from those persisted events after processing and remain derived from the audit record. Retrieval checkpoints omit source text; terminal source snapshots contain only passages that passed current eligibility and guardrail filtering. Drafts remain unapproved and unsent; downstream dashboard or case-management tools own assignment, approval, and any later send action.
