"""Build validated per-run JSONL and metrics exports from SQLite audit events."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from cloudserve_support.persistence.runtime_database import (
    read_run_audit_events,
    read_run_ticket_ids,
)
from cloudserve_support.evaluation.run_metrics import evaluate_completed_run
from cloudserve_support.reporting.output_schemas import (
    load_output_schemas,
    validate_output_record,
)


ROUTE_VALUES = ("draft_for_review", "escalate", "degrade")


def write_run_exports(
    connection: Any,
    *,
    run_id: str,
    output_directory: Path,
    input_records: list[Any],
    input_path: Path,
    source_ref: str,
) -> dict[str, Any]:
    """Export database-backed decisions and exceptions into one fresh run directory."""
    schemas = load_output_schemas()
    persisted_ticket_ids = read_run_ticket_ids(connection, run_id)
    events = read_run_audit_events(connection, run_id)
    if len(persisted_ticket_ids) != len(input_records):
        raise RuntimeError(
            "Export reconciliation failed: input record count does not match "
            "persisted ticket occurrences."
        )
    event_ticket_ids = [event["ticket_id"] for event in events]
    if (
        len(event_ticket_ids) != len(persisted_ticket_ids)
        or len(set(event_ticket_ids)) != len(event_ticket_ids)
        or set(event_ticket_ids) != set(persisted_ticket_ids)
    ):
        raise RuntimeError(
            "Export reconciliation failed: each persisted ticket occurrence must "
            "have exactly one audit result."
        )

    decisions: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for event in events:
        result = json.loads(event["result_payload_json"])
        results.append(result)
        decisions.append(
            {
                "event_id": event["event_id"],
                "run_id": event["run_id"],
                "input_index": result["input_index"],
                "ticket_id": result["ticket_id"],
                "occurred_at": event["occurred_at"],
                "event_type": event["event_type"],
                "actor_type": event["actor_type"],
                "action_taken": event["action_taken"],
                "reason": event["reason"],
                "result": result,
            }
        )

    results.sort(key=lambda record: record["input_index"])
    decisions.sort(key=lambda record: record["input_index"])
    if [result["input_index"] for result in results] != list(range(len(input_records))):
        raise RuntimeError(
            "Export reconciliation failed: result rows do not cover each input index once."
        )

    exceptions = [
        {
            "run_id": result["run_id"],
            "input_index": result["input_index"],
            "ticket_id": result["ticket_id"],
            "occurred_at": decision["occurred_at"],
            "error_code": result["error"]["code"],
            "message": result["error"]["message"],
        }
        for result, decision in zip(results, decisions, strict=True)
        if "error" in result
    ]
    route_counts = {route: 0 for route in ROUTE_VALUES}
    for result in results:
        route_counts[result["route"]] += 1
        validate_output_record(
            result,
            schema_filename="result.schema.json",
            schemas=schemas,
        )
    for decision in decisions:
        validate_output_record(
            decision,
            schema_filename="decision-log.schema.json",
            schemas=schemas,
        )
    for exception in exceptions:
        validate_output_record(
            exception,
            schema_filename="exception.schema.json",
            schemas=schemas,
        )

    evaluation = evaluate_completed_run(
        run_id=run_id,
        results=results,
        input_records=input_records,
        input_path=input_path,
        source_ref=source_ref,
        decision_events=decisions,
    )
    metrics = {
        "run_id": run_id,
        "total_count": len(results),
        "completed_count": sum(result["status"] == "completed" for result in results),
        "failed_count": sum(result["status"] == "failed" for result in results),
        "exception_count": len(exceptions),
        "route_counts": route_counts,
        "batch_export_duration_seconds": 0.0,
        "evaluation": evaluation,
    }
    validate_output_record(
        metrics,
        schema_filename="metrics.schema.json",
        schemas=schemas,
    )

    _write_jsonl_atomic(output_directory / "results.jsonl", results)
    _write_jsonl_atomic(output_directory / "decision_log.jsonl", decisions)
    _write_json_atomic(output_directory / "metrics.json", metrics)
    _write_jsonl_atomic(output_directory / "exceptions.jsonl", exceptions)
    return metrics


def record_batch_export_duration(
    *,
    metrics: dict[str, Any],
    output_directory: Path,
    duration_seconds: float,
) -> dict[str, Any]:
    """Add separately measured export and run-finalization time to metrics.json."""
    metrics["batch_export_duration_seconds"] = max(float(duration_seconds), 0.0)
    schemas = load_output_schemas()
    validate_output_record(
        metrics,
        schema_filename="metrics.schema.json",
        schemas=schemas,
    )
    _write_json_atomic(output_directory / "metrics.json", metrics)
    return metrics


def _write_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    """Write JSON Lines through a same-directory temporary file before replacing."""
    serialized = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        for record in records
    )
    _replace_file_atomically(path, serialized)


def _write_json_atomic(path: Path, record: dict[str, Any]) -> None:
    """Write one formatted JSON export through a same-directory temporary file."""
    _replace_file_atomically(path, json.dumps(record, ensure_ascii=False, indent=2) + "\n")


def _replace_file_atomically(path: Path, contents: str) -> None:
    """Write complete export contents before replacing its fresh run file."""
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as temporary_file:
            temporary_file.write(contents)
            temporary_path = temporary_file.name
        os.replace(temporary_path, path)
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)
