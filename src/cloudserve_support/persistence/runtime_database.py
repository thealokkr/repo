"""Initialize and write the durable SQLite ticket-run and audit records."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from cloudserve_support.domain.ticket_classification_vocabulary import (
    TICKET_INTENT_VALUES,
    TICKET_URGENCY_VALUES,
)


DATABASE_SCHEMA_VERSION = 4
RUNTIME_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS processing_runs (
        run_id TEXT PRIMARY KEY,
        input_ref TEXT NOT NULL,
        started_at TEXT NOT NULL,
        completed_at TEXT,
        status TEXT NOT NULL CHECK (
            status IN ('running', 'completed', 'completed_with_errors', 'failed')
        ),
        system_version TEXT,
        error_message TEXT,
        label TEXT,
        output_dir TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS ticket_runs (
        run_id TEXT NOT NULL,
        ticket_id TEXT NOT NULL,
        channel TEXT NOT NULL,
        received_at TEXT,
        customer_tier TEXT,
        language_fluency TEXT,
        context_json TEXT NOT NULL DEFAULT '{}',
        input_labels_json TEXT,
        processing_status TEXT NOT NULL CHECK (
            processing_status IN ('pending', 'running', 'completed', 'failed')
        ),
        error_code TEXT,
        error_message TEXT,
        PRIMARY KEY (run_id, ticket_id),
        FOREIGN KEY (run_id) REFERENCES processing_runs(run_id)
            ON UPDATE CASCADE ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS audit_events (
        event_id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL,
        ticket_id TEXT NOT NULL,
        occurred_at TEXT NOT NULL,
        event_type TEXT NOT NULL,
        actor_type TEXT NOT NULL,
        actor_id TEXT,
        input_summary TEXT,
        model_name TEXT,
        model_version TEXT,
        prompt_version TEXT,
        requirement_ids_json TEXT NOT NULL DEFAULT '[]',
        prediction_json TEXT,
        alternatives_json TEXT NOT NULL DEFAULT '[]',
        threshold_applied REAL,
        action_taken TEXT,
        reason TEXT,
        guardrail_results_json TEXT NOT NULL DEFAULT '{}',
        sources_json TEXT NOT NULL DEFAULT '[]',
        changes_json TEXT NOT NULL DEFAULT '{}',
        approval_action TEXT,
        event_payload_json TEXT,
        outcome TEXT,
        FOREIGN KEY (run_id, ticket_id)
            REFERENCES ticket_runs(run_id, ticket_id)
            ON UPDATE CASCADE ON DELETE CASCADE
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_ticket_runs_status ON ticket_runs(run_id, processing_status)",
    "CREATE INDEX IF NOT EXISTS idx_audit_events_ticket_time ON audit_events(run_id, ticket_id, occurred_at)",
)
# Keep the comparison view's classification checks on the canonical supported vocabularies.
TICKET_INTENT_SQL_VALUES = ", ".join(
    "'" + value.replace("'", "''") + "'" for value in sorted(TICKET_INTENT_VALUES)
)
TICKET_URGENCY_SQL_VALUES = ", ".join(
    "'" + value.replace("'", "''") + "'" for value in sorted(TICKET_URGENCY_VALUES)
)
TICKET_LABEL_COMPARISON_VIEW_SQL = f"""
CREATE VIEW ticket_label_comparison AS
WITH terminal_event_counts AS (
    SELECT run_id, ticket_id, COUNT(*) AS terminal_event_count
    FROM audit_events
    WHERE event_type IN ('ticket_completed', 'ticket_failed')
    GROUP BY run_id, ticket_id
),
single_terminal_event AS (
    SELECT event.run_id, event.ticket_id, event.event_type,
           event.event_payload_json, counts.terminal_event_count
    FROM audit_events AS event
    JOIN terminal_event_counts AS counts
      ON counts.run_id = event.run_id
     AND counts.ticket_id = event.ticket_id
    WHERE counts.terminal_event_count = 1
      AND event.event_type IN ('ticket_completed', 'ticket_failed')
),
terminal_payload AS (
    SELECT run_id, ticket_id, event_type, terminal_event_count,
           CASE WHEN json_valid(event_payload_json) = 1
                THEN event_payload_json ELSE '{{}}' END AS payload_json,
           CASE WHEN json_valid(event_payload_json) = 1 THEN 1 ELSE 0 END AS payload_is_valid
    FROM single_terminal_event
),
terminal_signals AS (
    SELECT payload.*,
           CASE WHEN
               json_type(payload_json, '$.result.status') = 'text'
               AND json_extract(payload_json, '$.result.status') = 'completed'
               AND json_type(payload_json, '$.result.classification.provider_status') = 'text'
               AND json_extract(payload_json, '$.result.classification.provider_status') = 'classified'
               AND json_type(payload_json, '$.result.classification.intent') = 'text'
               AND json_extract(payload_json, '$.result.classification.intent') IN ({TICKET_INTENT_SQL_VALUES})
               AND json_type(payload_json, '$.result.classification.urgency') = 'text'
               AND json_extract(payload_json, '$.result.classification.urgency') IN ({TICKET_URGENCY_SQL_VALUES})
               AND json_type(payload_json, '$.result.route') = 'text'
               AND json_extract(payload_json, '$.result.route') IN ('draft_for_review', 'escalate', 'degrade')
               AND json_type(payload_json, '$.result.routing.evidence_ready') IN ('true', 'false')
               AND json_type(payload_json, '$.result.routing.protected_case') IN ('true', 'false')
               AND json_type(payload_json, '$.result.sources') = 'array'
               AND NOT EXISTS (
                   SELECT 1
                   FROM json_each(payload.payload_json, '$.result.sources') AS source
                   WHERE CASE WHEN source.type = 'object'
                       THEN json_type(source.value, '$.doc_id') = 'text'
                            AND length(trim(json_extract(source.value, '$.doc_id'))) > 0
                       ELSE 0 END = 0
               )
               THEN 1 ELSE 0 END AS generated_signals_available
    FROM terminal_payload AS payload
)
SELECT ticket.run_id,
       ticket.ticket_id,
       ticket.input_labels_json,
       CASE WHEN signal.event_type = 'ticket_completed'
                  AND signal.generated_signals_available = 1
            THEN json_object(
                'intent', json_extract(signal.payload_json, '$.result.classification.intent'),
                'urgency', json_extract(signal.payload_json, '$.result.classification.urgency'),
                'expected_route', json_extract(signal.payload_json, '$.result.route'),
                'answerable_from_docs', json(CASE
                    WHEN json_extract(signal.payload_json, '$.result.routing.evidence_ready') = 1
                    THEN 'true' ELSE 'false' END),
                'expected_doc_ids', CASE
                    WHEN json_extract(signal.payload_json, '$.result.routing.evidence_ready') = 1
                    THEN json(COALESCE((
                        SELECT json_group_array(doc_id)
                        FROM (
                            SELECT DISTINCT json_extract(source.value, '$.doc_id') AS doc_id
                            FROM json_each(signal.payload_json, '$.result.sources') AS source
                            WHERE COALESCE(json_extract(source.value, '$.source_kind'), 'primary') = 'primary'
                            ORDER BY doc_id
                        )
                    ), '[]'))
                    ELSE json('[]')
                END,
                'must_not_auto_respond', json(CASE
                    WHEN json_extract(signal.payload_json, '$.result.routing.protected_case') = 1
                    THEN 'true' ELSE 'false' END)
            )
            ELSE NULL END AS generated_labels_json,
       CASE WHEN signal.event_type = 'ticket_completed'
                  AND signal.generated_signals_available = 1
            THEN 'available' ELSE 'unavailable' END AS generated_labels_status,
       CASE
           WHEN counts.terminal_event_count IS NULL THEN 'terminal_event_missing'
           WHEN counts.terminal_event_count <> 1 THEN 'multiple_terminal_events'
           WHEN signal.event_type = 'ticket_failed' THEN 'ticket_failed'
           WHEN signal.payload_is_valid = 0 THEN 'terminal_payload_invalid_json'
           WHEN json_type(signal.payload_json, '$.result.status') IS NOT 'text'
                OR json_extract(signal.payload_json, '$.result.status') <> 'completed'
                THEN 'completed_result_unavailable'
           WHEN json_type(signal.payload_json, '$.result.classification.intent') = 'text'
                AND json_extract(signal.payload_json, '$.result.classification.intent') NOT IN ({TICKET_INTENT_SQL_VALUES})
                THEN 'unsupported_intent'
           WHEN json_type(signal.payload_json, '$.result.classification.urgency') = 'text'
                AND json_extract(signal.payload_json, '$.result.classification.urgency') NOT IN ({TICKET_URGENCY_SQL_VALUES})
                THEN 'unsupported_urgency'
           WHEN signal.generated_signals_available <> 1 THEN 'required_generated_signal_unavailable'
           ELSE NULL
       END AS generated_labels_reason
FROM ticket_runs AS ticket
LEFT JOIN terminal_event_counts AS counts
  ON counts.run_id = ticket.run_id
 AND counts.ticket_id = ticket.ticket_id
LEFT JOIN terminal_signals AS signal
  ON signal.run_id = ticket.run_id
 AND signal.ticket_id = ticket.ticket_id;
"""
REQUIRED_RUNTIME_COLUMNS = {
    "processing_runs": {
        "run_id", "input_ref", "started_at", "completed_at", "status",
        "system_version", "error_message", "label", "output_dir",
    },
    "ticket_runs": {
        "run_id", "ticket_id", "channel", "received_at", "customer_tier",
        "language_fluency", "context_json", "input_labels_json", "processing_status",
        "error_code", "error_message",
    },
    "audit_events": {
        "event_id", "run_id", "ticket_id", "occurred_at", "event_type",
        "actor_type", "actor_id", "input_summary", "model_name", "model_version",
        "prompt_version", "requirement_ids_json", "prediction_json",
        "alternatives_json", "threshold_applied", "action_taken", "reason",
        "guardrail_results_json", "sources_json", "changes_json", "approval_action",
        "event_payload_json", "outcome",
    },
}


def validate_runtime_schema(connection: sqlite3.Connection) -> None:
    """Check each SQLite runtime table for the columns required by this version."""
    for table_name, required_columns in REQUIRED_RUNTIME_COLUMNS.items():
        actual_columns = {
            row["name"] for row in connection.execute(f"PRAGMA table_info({table_name})")
        }
        missing_columns = required_columns - actual_columns
        if missing_columns:
            raise RuntimeError(
                f"Runtime SQLite schema mismatch in {table_name}: missing columns "
                + ", ".join(sorted(missing_columns))
            )


def initialize_runtime_database(database_path: Path) -> None:
    """Create runtime tables and add run metadata without deleting existing records."""
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path, timeout=5.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        connection.execute("BEGIN IMMEDIATE")
        current_version = connection.execute("PRAGMA user_version").fetchone()[0]
        if current_version > DATABASE_SCHEMA_VERSION:
            raise RuntimeError(
                f"Runtime SQLite schema version {current_version} is newer than this "
                f"initializer supports ({DATABASE_SCHEMA_VERSION})."
            )
        existing_tables = {
            row["name"]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        had_audit_events_table = "audit_events" in existing_tables
        for statement in RUNTIME_SCHEMA_STATEMENTS:
            connection.execute(statement)
        existing_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(processing_runs)")
        }
        if "label" not in existing_columns:
            connection.execute("ALTER TABLE processing_runs ADD COLUMN label TEXT")
        if "output_dir" not in existing_columns:
            connection.execute("ALTER TABLE processing_runs ADD COLUMN output_dir TEXT")
        ticket_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(ticket_runs)")
        }
        if "input_labels_json" not in ticket_columns:
            connection.execute("ALTER TABLE ticket_runs ADD COLUMN input_labels_json TEXT")
        audit_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(audit_events)")
        }
        has_legacy_payload = "result_payload_json" in audit_columns
        has_event_payload = "event_payload_json" in audit_columns
        if has_legacy_payload and has_event_payload:
            raise RuntimeError(
                "Runtime SQLite schema mismatch in audit_events: both legacy "
                "result_payload_json and event_payload_json are present."
            )
        if has_legacy_payload:
            if current_version >= 3:
                raise RuntimeError(
                    f"Runtime SQLite schema version {current_version} has the legacy "
                    "audit_events.result_payload_json column."
                )
            connection.execute(
                "ALTER TABLE audit_events RENAME COLUMN "
                "result_payload_json TO event_payload_json"
            )
        elif not has_event_payload or (
            had_audit_events_table and current_version in (1, 2)
        ):
            raise RuntimeError(
                "Runtime SQLite schema mismatch in audit_events: expected the legacy "
                "result_payload_json column for migration or event_payload_json for "
                f"schema version {DATABASE_SCHEMA_VERSION}; found schema version "
                f"{current_version}."
            )
        connection.execute("DROP VIEW IF EXISTS ticket_label_comparison")
        connection.execute(TICKET_LABEL_COMPARISON_VIEW_SQL)
        validate_runtime_schema(connection)
        connection.execute(f"PRAGMA user_version = {DATABASE_SCHEMA_VERSION}")
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def connect_runtime_database(database_path: Path) -> sqlite3.Connection:
    """Open the initialized SQLite runtime database with foreign keys enabled."""
    initialize_runtime_database(database_path)
    connection = sqlite3.connect(database_path, timeout=5.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def create_processing_run(
    connection: sqlite3.Connection,
    *,
    run_id: str,
    input_ref: str,
    started_at: str,
    label: str | None,
    output_dir: str,
) -> None:
    """Insert the immutable run identifier and its optional label/output directory."""
    connection.execute(
        """INSERT INTO processing_runs
           (run_id, input_ref, started_at, status, system_version, label, output_dir)
           VALUES (?, ?, ?, 'running', '0.1.0', ?, ?)""",
        (run_id, input_ref, started_at, label, output_dir),
    )


def create_ticket_audit_records(
    connection: sqlite3.Connection,
    *,
    run_id: str,
    storage_ticket_id: str,
    channel: str,
    received_at: str | None,
    customer_tier: str | None,
    language_fluency: str | None,
    context: dict[str, Any],
    processing_status: str,
    error_code: str | None,
    error_message: str | None,
    event_id: str,
    occurred_at: str,
    event_type: str,
    input_summary: str,
    action_taken: str,
    reason: str,
    result_payload: dict[str, Any],
    model_name: str | None = None,
    model_version: str | None = None,
    prompt_version: str | None = None,
    requirement_ids: list[str] | None = None,
    prediction: dict[str, Any] | None = None,
    alternatives: list[dict[str, Any]] | None = None,
    threshold_applied: float | None = None,
    guardrail_results: dict[str, Any] | None = None,
) -> None:
    """Persist one ticket occurrence and its classification and routing audit event."""
    import json

    connection.execute(
        """INSERT INTO ticket_runs
           (run_id, ticket_id, channel, received_at, customer_tier, language_fluency,
            context_json, processing_status, error_code, error_message)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            run_id, storage_ticket_id, channel, received_at, customer_tier,
            language_fluency, json.dumps(context, ensure_ascii=False),
            processing_status, error_code, error_message,
        ),
    )
    connection.execute(
        """INSERT INTO audit_events
           (event_id, run_id, ticket_id, occurred_at, event_type, actor_type,
            input_summary, model_name, model_version, prompt_version,
            requirement_ids_json, prediction_json, alternatives_json,
            threshold_applied, action_taken, reason, guardrail_results_json, sources_json,
            event_payload_json)
           VALUES (?, ?, ?, ?, ?, 'system', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            event_id, run_id, storage_ticket_id, occurred_at, event_type,
            input_summary, model_name, model_version, prompt_version,
            json.dumps(requirement_ids or [], ensure_ascii=False),
            json.dumps(prediction, ensure_ascii=False) if prediction is not None else None,
            json.dumps(alternatives or [], ensure_ascii=False),
            threshold_applied, action_taken, reason,
            json.dumps(guardrail_results or {}, ensure_ascii=False),
            json.dumps(result_payload.get("sources", []), ensure_ascii=False),
            json.dumps(result_payload, ensure_ascii=False),
        ),
    )


def create_ticket_run_record(
    connection: sqlite3.Connection,
    *,
    run_id: str,
    storage_ticket_id: str,
    channel: str,
    received_at: str | None,
    customer_tier: str | None,
    language_fluency: str | None,
    context: dict[str, Any],
    input_labels_json: str | None,
) -> None:
    """Insert one running ticket occurrence before its first processing stage."""
    import json

    connection.execute(
        """INSERT INTO ticket_runs
           (run_id, ticket_id, channel, received_at, customer_tier, language_fluency,
            context_json, input_labels_json, processing_status)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'running')""",
        (
            run_id, storage_ticket_id, channel, received_at, customer_tier,
            language_fluency, json.dumps(context, ensure_ascii=False), input_labels_json,
        ),
    )


def create_audit_event_record(
    connection: sqlite3.Connection,
    *,
    event_id: str,
    run_id: str,
    storage_ticket_id: str,
    occurred_at: str,
    event_type: str,
    result_payload: dict[str, Any],
    input_summary: str | None = None,
    model_name: str | None = None,
    prompt_version: str | None = None,
    prediction: dict[str, Any] | None = None,
    requirement_ids: list[str] | None = None,
    alternatives: list[dict[str, Any]] | None = None,
    threshold_applied: float | None = None,
    action_taken: str | None = None,
    reason: str | None = None,
    guardrail_results: dict[str, Any] | None = None,
    sources: list[dict[str, Any]] | None = None,
) -> None:
    """Append one stage or terminal event for an existing ticket occurrence."""
    import json

    connection.execute(
        """INSERT INTO audit_events
           (event_id, run_id, ticket_id, occurred_at, event_type, actor_type,
            input_summary, model_name, prompt_version, requirement_ids_json,
            prediction_json, alternatives_json, threshold_applied, action_taken,
            reason, guardrail_results_json, sources_json, event_payload_json)
           VALUES (?, ?, ?, ?, ?, 'system', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            event_id,
            run_id,
            storage_ticket_id,
            occurred_at,
            event_type,
            input_summary,
            model_name,
            prompt_version,
            json.dumps(requirement_ids or [], ensure_ascii=False),
            json.dumps(prediction, ensure_ascii=False) if prediction is not None else None,
            json.dumps(alternatives or [], ensure_ascii=False),
            threshold_applied,
            action_taken,
            reason,
            json.dumps(guardrail_results or {}, ensure_ascii=False),
            json.dumps(sources or [], ensure_ascii=False),
            json.dumps(result_payload, ensure_ascii=False),
        ),
    )


def update_ticket_run_status(
    connection: sqlite3.Connection,
    *,
    run_id: str,
    storage_ticket_id: str,
    processing_status: str,
    error_code: str | None,
    error_message: str | None,
) -> None:
    """Mark an occurrence terminal after its terminal audit event is inserted."""
    cursor = connection.execute(
        """UPDATE ticket_runs SET processing_status = ?, error_code = ?, error_message = ?
           WHERE run_id = ? AND ticket_id = ?""",
        (processing_status, error_code, error_message, run_id, storage_ticket_id),
    )
    if cursor.rowcount != 1:
        raise RuntimeError("Expected exactly one ticket occurrence when recording its status.")


def update_terminal_audit_event_payload(
    connection: sqlite3.Connection,
    *,
    event_id: str,
    result_payload: dict[str, Any],
) -> None:
    """Refresh one terminal event after measured processing time is known."""
    import json

    cursor = connection.execute(
        """UPDATE audit_events SET event_payload_json = ?, sources_json = ?
           WHERE event_id = ? AND event_type IN ('ticket_completed', 'ticket_failed')""",
        (
            json.dumps(result_payload, ensure_ascii=False),
            json.dumps(result_payload.get("result", {}).get("sources", []), ensure_ascii=False),
            event_id,
        ),
    )
    if cursor.rowcount != 1:
        raise RuntimeError("Expected exactly one terminal event when recording latency.")


def read_run_audit_events(
    connection: sqlite3.Connection,
    run_id: str,
) -> list[sqlite3.Row]:
    """Read replayable run evidence in chronological order."""
    return list(
        connection.execute(
            """SELECT event_id, run_id, ticket_id, occurred_at, event_type,
                      actor_type, action_taken, reason, sources_json,
                      guardrail_results_json, event_payload_json
               FROM audit_events WHERE run_id = ?
               ORDER BY occurred_at ASC, rowid ASC""",
            (run_id,),
        )
    )


def read_run_ticket_ids(
    connection: sqlite3.Connection,
    run_id: str,
) -> list[str]:
    """Read persisted ticket occurrence IDs for one run in stable order."""
    return [
        row["ticket_id"]
        for row in connection.execute(
            "SELECT ticket_id FROM ticket_runs WHERE run_id = ? ORDER BY ticket_id ASC",
            (run_id,),
        )
    ]


def update_processing_run_status(
    connection: sqlite3.Connection,
    *,
    run_id: str,
    completed_at: str,
    status: str,
    error_message: str | None = None,
) -> None:
    """Record completion only after all per-run exports have been written."""
    connection.execute(
        """UPDATE processing_runs
           SET completed_at = ?, status = ?, error_message = ? WHERE run_id = ?""",
        (completed_at, status, error_message, run_id),
    )
    connection.commit()
