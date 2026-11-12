"""Initialize and write the durable SQLite ticket-run and audit records."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


DATABASE_SCHEMA_VERSION = 2
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
        result_payload_json TEXT,
        outcome TEXT,
        FOREIGN KEY (run_id, ticket_id)
            REFERENCES ticket_runs(run_id, ticket_id)
            ON UPDATE CASCADE ON DELETE CASCADE
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_ticket_runs_status ON ticket_runs(run_id, processing_status)",
    "CREATE INDEX IF NOT EXISTS idx_audit_events_ticket_time ON audit_events(run_id, ticket_id, occurred_at)",
)
REQUIRED_RUNTIME_COLUMNS = {
    "processing_runs": {
        "run_id", "input_ref", "started_at", "completed_at", "status",
        "system_version", "error_message", "label", "output_dir",
    },
    "ticket_runs": {
        "run_id", "ticket_id", "channel", "received_at", "customer_tier",
        "language_fluency", "context_json", "processing_status", "error_code",
        "error_message",
    },
    "audit_events": {
        "event_id", "run_id", "ticket_id", "occurred_at", "event_type",
        "actor_type", "actor_id", "input_summary", "model_name", "model_version",
        "prompt_version", "requirement_ids_json", "prediction_json",
        "alternatives_json", "threshold_applied", "action_taken", "reason",
        "guardrail_results_json", "sources_json", "changes_json", "approval_action",
        "result_payload_json", "outcome",
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
    current_version = connection.execute("PRAGMA user_version").fetchone()[0]
    if current_version > DATABASE_SCHEMA_VERSION:
        connection.close()
        raise RuntimeError(
            f"Runtime SQLite schema version {current_version} is newer than this "
            f"initializer supports ({DATABASE_SCHEMA_VERSION})."
        )

    try:
        connection.execute("BEGIN IMMEDIATE")
        for statement in RUNTIME_SCHEMA_STATEMENTS:
            connection.execute(statement)
        existing_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(processing_runs)")
        }
        if "label" not in existing_columns:
            connection.execute("ALTER TABLE processing_runs ADD COLUMN label TEXT")
        if "output_dir" not in existing_columns:
            connection.execute("ALTER TABLE processing_runs ADD COLUMN output_dir TEXT")
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
            result_payload_json)
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
) -> None:
    """Insert one running ticket occurrence before its first processing stage."""
    import json

    connection.execute(
        """INSERT INTO ticket_runs
           (run_id, ticket_id, channel, received_at, customer_tier, language_fluency,
            context_json, processing_status)
           VALUES (?, ?, ?, ?, ?, ?, ?, 'running')""",
        (
            run_id, storage_ticket_id, channel, received_at, customer_tier,
            language_fluency, json.dumps(context, ensure_ascii=False),
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
            reason, guardrail_results_json, sources_json, result_payload_json)
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
        """UPDATE audit_events SET result_payload_json = ?, sources_json = ?
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
                      guardrail_results_json, result_payload_json
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
