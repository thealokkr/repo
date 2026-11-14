"""Build validated per-run JSONL and metrics exports from SQLite audit events."""

from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from cloudserve_support.domain.ticket_classification_vocabulary import (
    TICKET_INTENT_VALUES,
    TICKET_URGENCY_VALUES,
)
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
LABEL_FIELDS = (
    "intent",
    "urgency",
    "expected_route",
    "answerable_from_docs",
    "expected_doc_ids",
    "must_not_auto_respond",
)
EXPECTED_ROUTE_ALIASES = {"auto_respond": "draft_for_review"}
EXPECTED_ROUTE_LABEL_VALUES = frozenset(
    {"auto_respond", "draft_for_review", "escalate", "degrade"}
)
OPTIONAL_INPUT_METADATA_FIELDS = (
    "received_at",
    "customer_tier",
    "language_fluency",
)
LABEL_PROJECTION_FIELDS = (
    ("intent", "intent"),
    ("urgency", "urgency"),
    ("expected_route", "route"),
    ("answerable_from_docs", "answerable_from_docs"),
    ("expected_doc_ids", "doc_ids"),
    ("must_not_auto_respond", "must_not_auto_respond"),
)
FAILED_KPI_NAMES = {
    "intent": "intent",
    "urgency": "urgency",
    "expected_route": "route",
    "answerable_from_docs": "answerability",
    "expected_doc_ids": "expected_doc_ids",
    "must_not_auto_respond": "must_not_auto_respond",
}
_UNPAIRED_SURROGATE_PATTERN = re.compile(r"[\ud800-\udfff]")


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
    if not set(event_ticket_ids).issubset(set(persisted_ticket_ids)):
        raise RuntimeError(
            "Export reconciliation failed: an audit event has an unknown ticket occurrence."
        )

    decisions: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    terminal_events: list[dict[str, Any]] = []
    terminal_ticket_ids: set[str] = set()
    identity_by_occurrence: dict[str, tuple[int, str]] = {}
    for event in events:
        payload = json.loads(event["event_payload_json"])
        input_index = payload.get("input_index")
        ticket_id = payload.get("ticket_id")
        if (
            not isinstance(input_index, int)
            or isinstance(input_index, bool)
            or not 0 <= input_index < len(input_records)
            or not isinstance(ticket_id, str)
        ):
            raise RuntimeError("Export reconciliation failed: audit event identity is invalid.")
        occurrence_id = event["ticket_id"]
        identity = (input_index, ticket_id)
        if (
            occurrence_id in identity_by_occurrence
            and identity_by_occurrence[occurrence_id] != identity
        ):
            raise RuntimeError(
                "Export reconciliation failed: events disagree on ticket occurrence identity."
            )
        identity_by_occurrence[occurrence_id] = identity
        decision = {
            "event_id": event["event_id"],
            "run_id": event["run_id"],
            "ticket_occurrence_id": event["ticket_id"],
            "input_index": input_index,
            "ticket_id": ticket_id,
            "occurred_at": event["occurred_at"],
            "event_type": event["event_type"],
            "actor_type": event["actor_type"],
            "terminal": event["event_type"] in {"ticket_completed", "ticket_failed"},
        }
        if decision["terminal"]:
            if occurrence_id in terminal_ticket_ids:
                raise RuntimeError("Export reconciliation failed: duplicate terminal occurrence event.")
            result = payload.get("result")
            if not isinstance(result, dict):
                raise RuntimeError("Export reconciliation failed: terminal event has no result.")
            if input_index in {record["input_index"] for record in results}:
                raise RuntimeError("Export reconciliation failed: duplicate terminal input index.")
            if result.get("input_index") != input_index or result.get("ticket_id") != ticket_id:
                raise RuntimeError("Export reconciliation failed: terminal result identity differs from its event.")
            results.append(result)
            terminal_ticket_ids.add(occurrence_id)
            terminal_events.append(decision)
            decision.update(
                {
                    "action_taken": event["action_taken"],
                    "reason": event["reason"],
                    "result": result,
                }
            )
        else:
            stage = payload.get("stage")
            if not isinstance(stage, str) or stage != event["event_type"]:
                raise RuntimeError("Export reconciliation failed: stage event identity is invalid.")
            decision["stage_result"] = {
                "stage": stage,
                "status": payload.get("status"),
                "value": payload.get("value"),
                "reason": payload.get("reason"),
            }
        decisions.append(decision)

    if (
        terminal_ticket_ids != set(persisted_ticket_ids)
        or len(terminal_events) != len(persisted_ticket_ids)
    ):
        raise RuntimeError(
            "Export reconciliation failed: each persisted ticket occurrence must have exactly one terminal event."
        )

    results.sort(key=lambda record: record["input_index"])
    terminal_events.sort(key=lambda record: record["input_index"])
    if [result["input_index"] for result in results] != list(range(len(input_records))):
        raise RuntimeError(
            "Export reconciliation failed: result rows do not cover each input index once."
        )

    label_comparisons = _read_run_label_comparisons(
        connection,
        run_id=run_id,
        identity_by_occurrence=identity_by_occurrence,
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
        for result, decision in zip(results, terminal_events, strict=True)
        if "error" in result
    ]
    route_counts = {route: 0 for route in ROUTE_VALUES}
    for result in results:
        route_counts[result["route"]] += 1
    exported_results = [
        _project_diagnostic_result(
            result,
            input_record=input_records[result["input_index"]],
            comparison=label_comparisons[result["input_index"]],
        )
        for result in results
    ]
    for result in exported_results:
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
        decision_events=terminal_events,
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

    _write_jsonl_atomic(output_directory / "results.jsonl", exported_results)
    _write_jsonl_atomic(output_directory / "decision_log.jsonl", decisions)
    _write_json_atomic(output_directory / "metrics.json", metrics)
    _write_jsonl_atomic(output_directory / "exceptions.jsonl", exceptions)
    _write_json_atomic(
        output_directory / "failed_records.json",
        _build_failed_records_report(run_id=run_id, results=exported_results),
    )
    return metrics


def _project_diagnostic_result(
    result: dict[str, Any],
    *,
    input_record: Any,
    comparison: dict[str, Any],
) -> dict[str, Any]:
    """Project persisted outcomes and post-processing labels into one diagnostic row."""
    raw = input_record if isinstance(input_record, dict) else {}
    row: dict[str, Any] = {
        "run_id": result["run_id"],
        "input_index": result["input_index"],
        "ticket_id": _preserve_nonempty_input_string(raw.get("ticket_id")),
        "channel": _preserve_nonempty_input_string(raw.get("channel")),
        "subject": raw["subject"] if isinstance(raw.get("subject"), str) else None,
        "body": raw["body"] if isinstance(raw.get("body"), str) else None,
    }
    for field in OPTIONAL_INPUT_METADATA_FIELDS:
        if field in raw:
            row[field] = _preserve_nonempty_input_string(raw.get(field))

    provided = comparison["provided_labels"] or {}
    generated = comparison["generated_labels"]
    for label_field, output_suffix in LABEL_PROJECTION_FIELDS:
        expected_value = provided.get(label_field)
        row[f"expected_{output_suffix}"] = (
            list(expected_value)
            if label_field == "expected_doc_ids"
            and _valid_label_value(label_field, expected_value, generated=False)
            else expected_value
            if _valid_label_value(label_field, expected_value, generated=False)
            else None
        )
        predicted_value = generated.get(label_field) if generated is not None else None
        row[f"predicted_{output_suffix}"] = (
            list(predicted_value)
            if label_field == "expected_doc_ids"
            and _valid_label_value(label_field, predicted_value, generated=True)
            else predicted_value
            if _valid_label_value(label_field, predicted_value, generated=True)
            else None
        )

    row["status"] = result["status"]
    row["reason"] = result["reason"]
    if "error" in result:
        row["error"] = result["error"]

    classification = result["classification"]
    row["classifier_confidence"] = (
        classification["confidence"]
        if classification["provider_status"] == "classified"
        else None
    )
    row["classifier_provider"] = classification["provider"]
    row["classifier_model"] = classification["model_name"]
    row["classifier_status"] = classification["provider_status"]
    row["routing"] = {
        key: value
        for key, value in result["routing"].items()
        if key not in {"evidence_ready", "protected_case"}
    }
    row["sources"] = result["sources"]
    row["draft"] = result["draft"]
    handoff = result["handoff"]
    row["handoff"] = (
        {
            key: value
            for key, value in handoff.items()
            if key not in {"context", "sources", "reason"}
        }
        if handoff is not None
        else None
    )
    row["guardrails"] = result["guardrails"]
    row["generation"] = result["generation"]
    row["processing_duration_seconds"] = result["processing_duration_seconds"]
    row["draft_generation_request_duration_seconds"] = result[
        "draft_generation_request_duration_seconds"
    ]
    row["processing_latency_seconds"] = result["processing_latency_seconds"]
    return row


def _build_failed_records_report(
    *,
    run_id: str,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """List ticket IDs and label KPIs that failed their post-run comparison."""
    failed_records: list[dict[str, Any]] = []
    for result in results:
        failed_kpis: list[str] = []
        for label_field, output_suffix in LABEL_PROJECTION_FIELDS:
            expected = result.get(f"expected_{output_suffix}")
            predicted = result.get(f"predicted_{output_suffix}")
            if not _valid_label_value(label_field, expected, generated=False):
                continue
            if not _valid_label_value(label_field, predicted, generated=True):
                failed_kpis.append(FAILED_KPI_NAMES[label_field])
                continue
            if label_field == "expected_doc_ids":
                is_match = set(expected) == set(predicted)
            else:
                expected_value = (
                    EXPECTED_ROUTE_ALIASES.get(expected, expected)
                    if label_field == "expected_route"
                    else expected
                )
                is_match = expected_value == predicted
            if not is_match:
                failed_kpis.append(FAILED_KPI_NAMES[label_field])

        if failed_kpis:
            failed_records.append(
                {
                    "input_index": result["input_index"],
                    "ticket_id": result["ticket_id"],
                    "failed_kpis": failed_kpis,
                }
            )

    return {
        "run_id": run_id,
        "failed_record_count": len(failed_records),
        "failed_records": failed_records,
    }


def _preserve_nonempty_input_string(value: Any) -> str | None:
    """Keep a supplied non-empty string verbatim without coercing missing or non-string input."""
    return value if isinstance(value, str) and value != "" else None



def write_markdown_run_metrics(
    connection: Any,
    *,
    run_id: str,
    output_directory: Path,
) -> None:
    """Write a human-readable label comparison report from persisted run data."""
    events = read_run_audit_events(connection, run_id)
    identity_by_occurrence: dict[str, tuple[int, str]] = {}
    for event in events:
        if event["event_type"] not in {"ticket_completed", "ticket_failed"}:
            continue
        try:
            payload = json.loads(event["event_payload_json"])
        except (TypeError, json.JSONDecodeError) as error:
            raise RuntimeError(
                "Run metrics export failed: terminal event identity is invalid JSON."
            ) from error
        input_index = payload.get("input_index")
        ticket_id = payload.get("ticket_id")
        if (
            not isinstance(input_index, int)
            or isinstance(input_index, bool)
            or not isinstance(ticket_id, str)
        ):
            raise RuntimeError("Run metrics export failed: terminal event identity is invalid.")
        identity_by_occurrence[event["ticket_id"]] = (input_index, ticket_id)

    label_comparisons = _read_run_label_comparisons(
        connection,
        run_id=run_id,
        identity_by_occurrence=identity_by_occurrence,
    )
    run = connection.execute(
        "SELECT started_at, completed_at FROM processing_runs WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    elapsed_seconds = _elapsed_run_seconds(run)
    matched_counts = dict.fromkeys(LABEL_FIELDS, 0)
    different_counts = dict.fromkeys(LABEL_FIELDS, 0)
    tickets_with_labels = sum(
        comparison["provided_labels_present"]
        for comparison in label_comparisons.values()
    )

    for comparison in label_comparisons.values():
        labels = comparison["provided_labels"]
        generated_labels = comparison["generated_labels"]
        for field in LABEL_FIELDS:
            if labels is None or generated_labels is None or field not in labels:
                continue
            provided_value = labels[field]
            generated_value = generated_labels.get(field)
            if not _valid_label_value(field, provided_value, generated=False):
                continue
            if not _valid_label_value(field, generated_value, generated=True):
                continue
            if field == "expected_doc_ids":
                is_match = set(provided_value) == set(generated_value)
            else:
                expected_value = EXPECTED_ROUTE_ALIASES.get(
                    provided_value, provided_value
                ) if field == "expected_route" else provided_value
                is_match = expected_value == generated_value
            if is_match:
                matched_counts[field] += 1
            else:
                different_counts[field] += 1

    lines = [
        "# Run metrics",
        "",
        f"- Run ID: `{run_id}`",
        f"- Total tickets: {len(label_comparisons)}",
        f"- Processed successfully: {sum(comparison['processing_status'] == 'completed' for comparison in label_comparisons.values())}",
        f"- Processing failures: {sum(comparison['processing_status'] == 'failed' for comparison in label_comparisons.values())}",
        f"- Tickets with provided labels: {tickets_with_labels}",
        f"- Elapsed run time: {_format_elapsed_run_time(elapsed_seconds)}",
        "  (from `processing_runs.started_at` through `completed_at`; not summed ticket latency)",
        "",
        "| Label | Matched | Different | Not compared | Match rate |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for field in LABEL_FIELDS:
        comparable_count = matched_counts[field] + different_counts[field]
        not_compared_count = len(label_comparisons) - comparable_count
        match_rate = (
            f"{matched_counts[field] / comparable_count * 100:.1f}%"
            if comparable_count
            else "N/A"
        )
        lines.append(
            f"| {field} | {matched_counts[field]} | {different_counts[field]} | "
            f"{not_compared_count} | {match_rate} |"
        )
    _replace_file_atomically(output_directory / "metrics.md", "\n".join(lines))


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


def _read_run_label_comparisons(
    connection: Any,
    *,
    run_id: str,
    identity_by_occurrence: dict[str, tuple[int, str]],
) -> dict[int, dict[str, Any]]:
    """Read labels and generated predictions by their persisted ticket occurrence."""
    rows = connection.execute(
        """SELECT comparison.ticket_id, comparison.input_labels_json,
                  comparison.generated_labels_json, comparison.generated_labels_status,
                  ticket.processing_status,
                  comparison.input_labels_json IS NOT NULL AS provided_labels_present
           FROM ticket_label_comparison AS comparison
           JOIN ticket_runs AS ticket
             ON ticket.run_id = comparison.run_id
            AND ticket.ticket_id = comparison.ticket_id
           WHERE comparison.run_id = ?""",
        (run_id,),
    ).fetchall()
    if len(rows) != len(identity_by_occurrence):
        raise RuntimeError(
            "Run metrics export failed: comparison view does not cover every ticket occurrence."
        )

    comparisons_by_input_index: dict[int, dict[str, Any]] = {}
    for row in rows:
        occurrence_id = row["ticket_id"]
        identity = identity_by_occurrence.get(occurrence_id)
        if identity is None:
            raise RuntimeError(
                "Run metrics export failed: comparison view has an unknown ticket occurrence."
            )
        input_index, _ticket_id = identity
        if input_index in comparisons_by_input_index:
            raise RuntimeError(
                "Run metrics export failed: comparison view repeats an input occurrence."
            )
        provided_labels = _decode_json_object(row["input_labels_json"])
        generated_labels = None
        if row["generated_labels_status"] == "available":
            candidate = _decode_json_object(row["generated_labels_json"])
            if (
                candidate is not None
                and set(candidate) == set(LABEL_FIELDS)
                and all(
                    _valid_label_value(field, candidate[field], generated=True)
                    for field in LABEL_FIELDS
                )
            ):
                generated_labels = candidate
        comparisons_by_input_index[input_index] = {
            "provided_labels": provided_labels,
            "provided_labels_present": bool(row["provided_labels_present"]),
            "generated_labels": generated_labels,
            "processing_status": row["processing_status"],
        }
    return comparisons_by_input_index


def _decode_json_object(value: Any) -> dict[str, Any] | None:
    """Decode one persisted JSON object without coercing invalid or non-object labels."""
    if not isinstance(value, str):
        return None
    try:
        decoded = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return None
    return decoded if isinstance(decoded, dict) else None


def _valid_label_value(field: str, value: Any, *, generated: bool) -> bool:
    """Check native label types before comparing evaluation fields."""
    if field == "intent":
        return isinstance(value, str) and value in TICKET_INTENT_VALUES
    if field == "urgency":
        return isinstance(value, str) and value in TICKET_URGENCY_VALUES
    if field == "expected_route":
        valid_routes = ROUTE_VALUES if generated else EXPECTED_ROUTE_LABEL_VALUES
        return isinstance(value, str) and value in valid_routes
    if field in {"answerable_from_docs", "must_not_auto_respond"}:
        return type(value) is bool
    if field == "expected_doc_ids":
        return isinstance(value, list) and all(
            isinstance(item, str) and bool(item.strip()) for item in value
        )
    return False


def _elapsed_run_seconds(run: Any) -> float | None:
    """Calculate run wall time only from its persisted UTC start and completion times."""
    if run is None or not run["started_at"] or not run["completed_at"]:
        return None
    try:
        started_at = datetime.fromisoformat(run["started_at"].replace("Z", "+00:00"))
        completed_at = datetime.fromisoformat(run["completed_at"].replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if started_at.tzinfo is None or completed_at.tzinfo is None:
        return None
    elapsed_seconds = (completed_at - started_at).total_seconds()
    return elapsed_seconds if elapsed_seconds >= 0 else None


def _format_elapsed_run_time(elapsed_seconds: float | None) -> str:
    """Render persisted wall time or an explicit unavailable value."""
    return f"{elapsed_seconds:.3f} seconds" if elapsed_seconds is not None else "Unavailable"


def _write_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    """Write JSON Lines through a same-directory temporary file before replacing."""
    serialized = "".join(_serialize_jsonl_record(record) for record in records)
    _replace_file_atomically(path, serialized)


def _serialize_jsonl_record(record: dict[str, Any]) -> str:
    """Serialize one JSONL record as UTF-8-safe JSON while escaping unpaired surrogates."""
    encoded = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
    safe_json = _UNPAIRED_SURROGATE_PATTERN.sub(
        lambda match: f"\\u{ord(match.group(0)):04x}",
        encoded,
    )
    return safe_json + "\n"


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
