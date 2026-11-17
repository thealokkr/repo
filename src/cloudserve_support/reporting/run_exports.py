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
_CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x1f\x7f]")
PRE_DRAFT_SAFETY_CHECK_NAMES = frozenset(
    {
        "source_eligibility",
        "prompt_injection",
        "private_data_input",
        "private_data_source",
    }
)
POST_DRAFT_CHECK_ORDER = (
    "output_structure",
    "citation_grounding",
    "private_data_output",
    "instruction_integrity",
)


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
    persisted_input_ref = _read_run_input_ref(connection, run_id=run_id)
    failure_report = _build_failure_report(
        run_id=run_id,
        persisted_input_ref=persisted_input_ref,
        results=results,
        terminal_events=terminal_events,
        exceptions=exceptions,
        label_comparisons=label_comparisons,
        route_counts=route_counts,
    )
    _write_json_atomic(output_directory / "failure_report.json", failure_report)
    _replace_file_atomically(
        output_directory / "failure_report.md",
        _render_failure_report_markdown(failure_report),
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


def _read_run_input_ref(connection: Any, *, run_id: str) -> str:
    """Read the persisted input_ref for one run from SQLite."""
    row = connection.execute(
        "SELECT input_ref FROM processing_runs WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    if row is not None and isinstance(row["input_ref"], str) and row["input_ref"]:
        return row["input_ref"]
    return "unavailable"


def _build_failure_report(
    *,
    run_id: str,
    persisted_input_ref: str,
    results: list[dict[str, Any]],
    terminal_events: list[dict[str, Any]],
    exceptions: list[dict[str, Any]],
    label_comparisons: dict[int, dict[str, Any]],
    route_counts: dict[str, int],
) -> dict[str, Any]:
    """Build consolidated failure_report.json strictly from persisted SQLite records."""
    metrics_summary = _build_persisted_failure_report_metrics(
        run_id=run_id,
        persisted_input_ref=persisted_input_ref,
        results=results,
        terminal_events=terminal_events,
        label_comparisons=label_comparisons,
    )
    failed_records: list[dict[str, Any]] = []
    tickets: list[dict[str, Any]] = []
    operational_failure_count = 0

    for result in results:
        comparison = label_comparisons[result["input_index"]]
        evaluation = _build_ticket_evaluation_summary(result, comparison=comparison)
        if evaluation["failed_kpis"]:
            failed_records.append(
                {
                    "input_index": result["input_index"],
                    "ticket_id": result["ticket_id"],
                    "failed_kpis": evaluation["failed_kpis"],
                }
            )
        ticket_entry, is_operational_failure = _build_failure_report_ticket(
            result,
            evaluation=evaluation,
        )
        if is_operational_failure:
            operational_failure_count += 1
        if ticket_entry is not None:
            tickets.append(ticket_entry)

    blocked_by_guardrails = sum(
        any(
            item.get("blocked") is True
            for item in (result.get("guardrails") or {}).get("checks", [])
            if isinstance(item, dict)
        )
        for result in results
    )
    return {
        "schema_version": 1,
        "run": {
            "run_id": run_id,
            "input_ref": persisted_input_ref,
            "tickets_processed": len(results),
            "completed": sum(result.get("status") == "completed" for result in results),
            "processing_failures": sum(result.get("status") == "failed" for result in results),
            "exceptions": len(exceptions),
            "drafts_for_review": route_counts["draft_for_review"],
            "escalated": route_counts["escalate"],
            "degraded": route_counts["degrade"],
            "blocked_by_guardrails": blocked_by_guardrails,
            "operational_failure_count": operational_failure_count,
            "kpi_failed_record_count": len(failed_records),
            "reported_ticket_count": len(tickets),
        },
        "metrics": metrics_summary,
        "failed_records": failed_records,
        "tickets": tickets,
    }


def _build_persisted_failure_report_metrics(
    *,
    run_id: str,
    persisted_input_ref: str,
    results: list[dict[str, Any]],
    terminal_events: list[dict[str, Any]],
    label_comparisons: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """Compute summary and calibration metrics from SQLite-persisted results and labels only."""
    # Reconstruct evaluation inputs exclusively from SQLite records so caller input_records are never read.
    persisted_records = []
    for result in results:
        provided = label_comparisons[result["input_index"]]["provided_labels"]
        record: dict[str, Any] = {"ticket_id": result["ticket_id"]}
        if provided is not None:
            record["labels"] = provided
        persisted_records.append(record)

    persisted_path = Path(persisted_input_ref) if persisted_input_ref != "unavailable" else Path("unavailable")
    evaluation = evaluate_completed_run(
        run_id=run_id,
        results=results,
        input_records=persisted_records,
        input_path=persisted_path,
        source_ref=persisted_input_ref,
        decision_events=terminal_events,
    )
    technical = evaluation["run_observed"]["technical"]
    criteria = evaluation["criteria"]
    calibration = criteria["confidence_calibration"]
    return {
        "intent_accuracy": technical["classification"]["accuracy"],
        "urgency_exact_match": technical["urgency"]["exact_match_rate"],
        "route_accuracy": technical["route_accuracy"],
        "answerability_accuracy": technical["answerability"]["accuracy"],
        "retrieval_hit_rate": technical["retrieval_hit_rate"],
        "decision_audit_coverage": criteria["decision_audit_coverage"],
        "p95_runtime_latency_seconds": technical["latency"]["p95_seconds"],
        "confidence_calibration": {
            "status": calibration["status"],
            "target": calibration["target"],
            "value": calibration["value"],
            "numerator": calibration.get("numerator"),
            "denominator": calibration.get("denominator"),
            "sample_size": calibration.get("sample_size"),
            "missing_count": calibration.get("missing_count"),
            "note": calibration.get("note"),
            "populated_bands": list(calibration.get("populated_bands", [])),
        },
    }


def _build_ticket_evaluation_summary(
    result: dict[str, Any],
    *,
    comparison: dict[str, Any],
) -> dict[str, Any]:
    """Compare persisted expected labels against generated predictions for one ticket."""
    provided = comparison["provided_labels"] or {}
    generated = comparison["generated_labels"]
    expected: dict[str, Any] = {}
    predicted: dict[str, Any] = {}
    failed_kpis: list[str] = []

    for label_field, _ in LABEL_PROJECTION_FIELDS:
        expected_raw = provided.get(label_field)
        expected_valid = _valid_label_value(label_field, expected_raw, generated=False)
        expected[label_field] = (
            list(expected_raw) if label_field == "expected_doc_ids" and expected_valid
            else expected_raw if expected_valid
            else None
        )
        predicted_key = "route" if label_field == "expected_route" else label_field
        predicted_raw = generated.get(label_field) if generated is not None else None
        predicted_valid = _valid_label_value(label_field, predicted_raw, generated=True)
        if label_field != "expected_doc_ids":
            predicted[predicted_key] = predicted_raw if predicted_valid else None

        if not expected_valid:
            continue
        if not predicted_valid:
            failed_kpis.append(FAILED_KPI_NAMES[label_field])
            continue
        if label_field == "expected_doc_ids":
            is_match = set(expected_raw) == set(predicted_raw)
        elif label_field == "expected_route":
            is_match = EXPECTED_ROUTE_ALIASES.get(expected_raw, expected_raw) == predicted_raw
        else:
            is_match = expected_raw == predicted_raw
        if not is_match:
            failed_kpis.append(FAILED_KPI_NAMES[label_field])

    sources = [
        item for item in (result.get("sources") or []) if isinstance(item, dict) and item.get("doc_id")
    ]
    retrieved_doc_ids = list(
        dict.fromkeys(
            item["doc_id"]
            for item in sources
            if item.get("source_kind", "primary") == "primary"
        )
    )
    related_doc_ids = list(
        dict.fromkeys(
            item["doc_id"]
            for item in sources
            if item.get("source_kind") == "related"
        )
    )
    predicted["retrieved_doc_ids"] = retrieved_doc_ids
    predicted["related_doc_ids"] = related_doc_ids
    cited_doc_ids = _extract_cited_doc_ids(result, sources=sources)
    predicted["cited_doc_ids"] = cited_doc_ids

    expected_doc_note = _expected_doc_note(
        expected_docs=expected.get("expected_doc_ids"),
        retrieved_doc_ids=retrieved_doc_ids,
        cited_doc_ids=cited_doc_ids,
        draft_emitted=isinstance(result.get("draft"), dict),
        failed_kpis=failed_kpis,
    )
    return {
        "expected": expected,
        "predicted": predicted,
        "failed_kpis": failed_kpis,
        "expected_doc_note": expected_doc_note,
    }


def _extract_cited_doc_ids(
    result: dict[str, Any],
    *,
    sources: list[dict[str, Any]],
) -> list[str]:
    """Resolve unique sorted cited doc IDs from emitted draft claims or withheld diagnostics."""
    draft = result.get("draft")
    diagnostics = result.get("post_draft_diagnostics")
    if isinstance(draft, dict) and isinstance(draft.get("claims"), list):
        claims = draft["claims"]
    elif isinstance(diagnostics, dict) and isinstance(diagnostics.get("withheld_claims"), list):
        claims = diagnostics["withheld_claims"]
    else:
        return []

    doc_id_by_source_id: dict[str, str] = {}
    for item in sources:
        doc_id = item.get("doc_id")
        if not isinstance(doc_id, str) or not doc_id:
            continue
        doc_id_by_source_id[doc_id] = doc_id
        passage_id = item.get("passage_id")
        if isinstance(passage_id, str) and passage_id:
            doc_id_by_source_id[passage_id] = doc_id

    cited_docs: set[str] = set()
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        for source_id in claim.get("source_ids") or []:
            if not isinstance(source_id, str) or not source_id:
                continue
            resolved = doc_id_by_source_id.get(source_id)
            if resolved is None:
                resolved = source_id.split(":", 1)[0] if ":" in source_id else source_id
            if resolved:
                cited_docs.add(resolved)
    return sorted(cited_docs)


def _expected_doc_note(
    *,
    expected_docs: list[str] | None,
    retrieved_doc_ids: list[str],
    cited_doc_ids: list[str],
    draft_emitted: bool,
    failed_kpis: list[str],
) -> str | None:
    """Note when expected docs were retrieved despite an exact-set or cited-doc mismatch."""
    if "expected_doc_ids" not in failed_kpis or not expected_docs:
        return None
    retrieved_set = set(retrieved_doc_ids)
    cited_differs = bool(cited_doc_ids) and set(cited_doc_ids) != set(expected_docs)
    citation_actor = "the emitted draft" if draft_emitted else "generated draft claims"
    if set(expected_docs).issubset(retrieved_set):
        if cited_differs:
            return (
                f"Expected doc(s) {', '.join(expected_docs)} were retrieved in "
                f"retrieved_doc_ids, while {citation_actor} cited {', '.join(cited_doc_ids)}."
            )
        return (
            f"Expected doc(s) {', '.join(expected_docs)} were retrieved in "
            "retrieved_doc_ids despite the exact-set expected_doc_ids mismatch."
        )
    partial = [doc_id for doc_id in expected_docs if doc_id in retrieved_set]
    if partial:
        if cited_differs:
            return (
                f"Expected doc(s) {', '.join(partial)} were partially retrieved in "
                f"retrieved_doc_ids, while {citation_actor} cited {', '.join(cited_doc_ids)}."
            )
        return (
            f"Expected doc(s) {', '.join(partial)} were partially retrieved in "
            "retrieved_doc_ids despite the exact-set expected_doc_ids mismatch."
        )
    return None


def _build_failure_report_ticket(
    result: dict[str, Any],
    *,
    evaluation: dict[str, Any],
) -> tuple[dict[str, Any] | None, bool]:
    """Assemble one failure_report.json ticket entry when operational or KPI issues exist."""
    routing = result.get("routing") if isinstance(result.get("routing"), dict) else {}
    routing_checks = [
        dict(check) for check in (routing.get("checks") or []) if isinstance(check, dict)
    ]
    final_decision = (
        dict(routing["final_decision"])
        if isinstance(routing.get("final_decision"), dict)
        else None
    )
    guardrails = result.get("guardrails") if isinstance(result.get("guardrails"), dict) else {}
    all_guardrail_checks = [
        dict(check) for check in (guardrails.get("checks") or []) if isinstance(check, dict)
    ]
    safety_checks = [
        check for check in all_guardrail_checks if check.get("check") in PRE_DRAFT_SAFETY_CHECK_NAMES
    ]
    post_draft_checks = [
        check for check in all_guardrail_checks if check.get("check") not in PRE_DRAFT_SAFETY_CHECK_NAMES
    ]
    generation = result.get("generation") if isinstance(result.get("generation"), dict) else {}
    classification = (
        result.get("classification") if isinstance(result.get("classification"), dict) else {}
    )
    error = dict(result["error"]) if isinstance(result.get("error"), dict) else None
    error_code = error.get("code") if error else None

    failed_routing_checks = [c for c in routing_checks if c.get("status") == "failed"]
    blocked_non_eligibility_safety = [
        c
        for c in safety_checks
        if c.get("check") != "source_eligibility"
        and (not c.get("passed", True) or c.get("blocked", False))
    ]
    failed_post_draft = [
        c for c in post_draft_checks if not c.get("passed", True) or c.get("blocked", False)
    ]
    has_policy_failure = bool(failed_routing_checks) or (
        final_decision is not None and final_decision.get("failed_check") is not None
    )
    has_classification_failure = (
        classification.get("provider_status") not in {None, "classified"}
        or error_code == "classification_unavailable"
    )
    has_operational_failure = (
        result.get("status") == "failed"
        or error is not None
        or generation.get("status") == "failed"
        or has_policy_failure
        or has_classification_failure
        or bool(blocked_non_eligibility_safety)
        or bool(failed_post_draft)
    )
    if not has_operational_failure and not evaluation["failed_kpis"]:
        return None, False

    first_blocking_check = _first_pre_draft_blocking_check(
        error_code=error_code,
        final_decision=final_decision,
        failed_routing_checks=failed_routing_checks,
        blocked_non_eligibility_safety=blocked_non_eligibility_safety,
        has_classification_failure=has_classification_failure,
    )
    gen_status = generation.get("status") or "not_attempted"
    if gen_status == "not_attempted":
        draft_disposition = "not_attempted"
    elif gen_status == "failed":
        draft_disposition = "failed"
    elif result.get("draft") is None:
        draft_disposition = "withheld"
    else:
        draft_disposition = "emitted"

    after_drafting = _build_after_drafting_section(
        gen_status=gen_status,
        draft_disposition=draft_disposition,
        post_draft_checks=post_draft_checks,
        routing_checks=routing_checks,
        post_draft_diagnostics=(
            result.get("post_draft_diagnostics")
            if isinstance(result.get("post_draft_diagnostics"), dict)
            else None
        ),
    )

    stopped_at, failure_stage, failed_check = _classify_report_stop_and_stage(
        has_operational_failure=has_operational_failure,
        error_code=error_code,
        first_blocking_check=first_blocking_check,
        failed_routing_checks=failed_routing_checks,
        final_decision=final_decision,
        blocked_non_eligibility_safety=blocked_non_eligibility_safety,
        has_classification_failure=has_classification_failure,
        generation=generation,
        after_drafting=after_drafting,
    )

    return (
        {
            "input_index": result["input_index"],
            "ticket_id": result["ticket_id"],
            "channel": result.get("channel"),
            "processing_status": result["status"],
            "final_route": result["route"],
            "final_reason": result["reason"],
            "error": error,
            "stopped_at": stopped_at,
            "failure_stage": failure_stage,
            "failed_check": failed_check,
            "before_drafting": {
                "safety_checks": safety_checks,
                "routing_checks": routing_checks,
                "final_policy_decision": final_decision,
                "first_blocking_check": first_blocking_check,
            },
            "drafting": {
                "attempted": gen_status != "not_attempted",
                "status": gen_status,
                "provider": generation.get("provider"),
                "model": generation.get("model_name"),
                "error_code": generation.get("error_code"),
                "draft_disposition": draft_disposition,
            },
            "after_drafting": after_drafting,
            "evaluation": evaluation,
        },
        has_operational_failure,
    )


def _build_after_drafting_section(
    *,
    gen_status: str,
    draft_disposition: str,
    post_draft_checks: list[dict[str, Any]],
    routing_checks: list[dict[str, Any]],
    post_draft_diagnostics: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build ordered after_drafting checks and clause diagnostics from persisted SQLite data."""
    if gen_status not in {"generated", "failed"} or not post_draft_checks:
        return {
            "status": "not_run",
            "checks": [],
            "blocking_check": None,
            "distinct_from_pre_draft_claim_grounding_note": None,
            "withheld_claims": [],
            "unsupported_points": [],
            "conflicting_points": [],
            "failed_claims": [],
            "diagnostics_note": None,
        }

    order_index = {name: idx for idx, name in enumerate(POST_DRAFT_CHECK_ORDER)}
    sorted_raw = sorted(
        post_draft_checks,
        key=lambda item: order_index.get(str(item.get("check")), len(POST_DRAFT_CHECK_ORDER)),
    )
    minimum_ratio = (
        post_draft_diagnostics.get("minimum_ratio")
        if post_draft_diagnostics is not None
        else None
    )
    ordered_checks: list[dict[str, Any]] = []
    for raw_check in sorted_raw:
        name = raw_check.get("check")
        passed = bool(raw_check.get("passed", True)) and not bool(raw_check.get("blocked", False))
        entry: dict[str, Any] = {
            "name": name,
            "status": "passed" if passed else "failed",
            "passed": bool(raw_check.get("passed", True)),
            "blocked": bool(raw_check.get("blocked", False)),
            "evidence": raw_check.get("evidence"),
            "action": raw_check.get("action"),
        }
        check_ratio = raw_check.get("minimum_ratio", minimum_ratio)
        if name == "citation_grounding" and check_ratio is not None:
            entry["minimum_ratio"] = check_ratio
        ordered_checks.append(entry)

    blocking_check = next(
        (check["name"] for check in ordered_checks if check["status"] == "failed"),
        None,
    )
    pre_draft_claim_grounding_passed = any(
        check.get("name") == "claim_grounding_check" and check.get("status") == "passed"
        for check in routing_checks
    )
    distinct_note = (
        "Pre-draft routing claim_grounding_check passed on policy-verified claims, "
        "but post-draft deterministic citation_grounding blocked the generated draft "
        "because its clauses did not meet grounding requirements."
        if gen_status == "generated"
        and blocking_check == "citation_grounding"
        and pre_draft_claim_grounding_passed
        else None
    )

    if gen_status == "generated" and post_draft_diagnostics is not None:
        withheld_claims = list(post_draft_diagnostics.get("withheld_claims") or [])
        unsupported_points = list(post_draft_diagnostics.get("unsupported_points") or [])
        conflicting_points = list(post_draft_diagnostics.get("conflicting_points") or [])
        failed_claims = list(post_draft_diagnostics.get("failed_claims") or [])
        diagnostics_note = None
    elif gen_status == "generated" and draft_disposition == "withheld":
        withheld_claims = []
        unsupported_points = []
        conflicting_points = []
        failed_claims = []
        diagnostics_note = (
            "Post-draft claim and clause diagnostics were not recorded in this persisted SQLite event."
        )
    else:
        withheld_claims = []
        unsupported_points = []
        conflicting_points = []
        failed_claims = []
        diagnostics_note = None

    return {
        "status": "failed" if blocking_check is not None else "passed",
        "checks": ordered_checks,
        "blocking_check": blocking_check,
        "distinct_from_pre_draft_claim_grounding_note": distinct_note,
        "withheld_claims": withheld_claims,
        "unsupported_points": unsupported_points,
        "conflicting_points": conflicting_points,
        "failed_claims": failed_claims,
        "diagnostics_note": diagnostics_note,
    }


def _first_pre_draft_blocking_check(
    *,
    error_code: str | None,
    final_decision: dict[str, Any] | None,
    failed_routing_checks: list[dict[str, Any]],
    blocked_non_eligibility_safety: list[dict[str, Any]],
    has_classification_failure: bool,
) -> str | None:
    """Identify the first pre-draft check or validation error that blocked drafting."""
    if error_code in {"invalid_ticket_record", "duplicate_ticket_id"}:
        return error_code
    policy_check = (
        (final_decision.get("failed_check") if final_decision else None)
        or (failed_routing_checks[0].get("name") if failed_routing_checks else None)
    )
    if policy_check:
        return policy_check
    if blocked_non_eligibility_safety:
        return blocked_non_eligibility_safety[0].get("check")
    if has_classification_failure:
        return "classification_unavailable"
    return None


def _classify_report_stop_and_stage(
    *,
    has_operational_failure: bool,
    error_code: str | None,
    first_blocking_check: str | None,
    failed_routing_checks: list[dict[str, Any]],
    final_decision: dict[str, Any] | None,
    blocked_non_eligibility_safety: list[dict[str, Any]],
    has_classification_failure: bool,
    generation: dict[str, Any],
    after_drafting: dict[str, Any],
) -> tuple[str, str, str | None]:
    """Determine stopped_at, failure_stage, and failed_check for failure_report.json."""
    if not has_operational_failure:
        return "evaluation_only", "evaluation", None
    gen_status = generation.get("status") or "not_attempted"
    if gen_status == "not_attempted" or first_blocking_check is not None:
        if error_code in {"invalid_ticket_record", "duplicate_ticket_id"}:
            return "before_drafting", "input_validation", error_code
        policy_check = (
            (final_decision.get("failed_check") if final_decision else None)
            or (failed_routing_checks[0].get("name") if failed_routing_checks else None)
        )
        if policy_check:
            return "before_drafting", "routing_policy", policy_check
        if blocked_non_eligibility_safety:
            return "before_drafting", "safety_guardrail", blocked_non_eligibility_safety[0].get("check")
        if has_classification_failure:
            return "before_drafting", "classification", "classification_unavailable"
        return "before_drafting", "processing", error_code or "processing_failed"
    if gen_status == "failed":
        return (
            "during_drafting",
            "generation",
            error_code or generation.get("error_code") or "generation_failed",
        )
    if after_drafting["blocking_check"] is not None:
        return "after_drafting", "guardrail", after_drafting["blocking_check"]
    return "after_drafting", "processing", error_code or "processing_failed"


_STOPPED_AT_LABELS = {
    "before_drafting": "Before drafting",
    "during_drafting": "During drafting",
    "after_drafting": "After drafting",
    "evaluation_only": "Evaluation only",
}
_EVALUATION_HEADER_METRICS = (
    ("intent", "intent_accuracy"),
    ("urgency", "urgency_exact_match"),
    ("route", "route_accuracy"),
    ("answerability", "answerability_accuracy"),
    ("retrieval", "retrieval_hit_rate"),
)


def _render_failure_report_markdown(failure_report: dict[str, Any]) -> str:
    """Render quick-read failure_report.md directly from the canonical failure_report dict."""
    run = failure_report["run"]
    metrics = failure_report.get("metrics") or {}
    tickets = failure_report.get("tickets") or []

    lines = [
        "# Failure Report",
        "",
        f"- Run: `{_sanitize_markdown_text(run['run_id'])}` (input: `{_sanitize_markdown_text(run['input_ref'])}`)",
        (
            f"- Processing: {run['completed']}/{run['tickets_processed']} completed "
            f"({run['processing_failures']} processing failures, {run['exceptions']} exceptions)"
        ),
        (
            f"- Outcome: {run['drafts_for_review']} drafts for review, "
            f"{run['escalated']} escalated, {run['degraded']} degraded, "
            f"{run['blocked_by_guardrails']} drafts blocked by guardrails"
        ),
    ]
    eval_parts = _format_evaluation_header_parts(metrics)
    if eval_parts:
        lines.append(f"- Evaluation: {', '.join(eval_parts)}")
    lines.append("")

    if not tickets:
        lines.append("No operational failures or evaluation KPI mismatches occurred in this run.")
        lines.append("")
    else:
        lines.extend(
            [
                "| Ticket | Stopped at | Main reason | Result |",
                "| --- | --- | --- | --- |",
            ]
        )
        for ticket in tickets:
            lines.append(_render_summary_table_row(ticket))
        lines.append("")
        for ticket in tickets:
            lines.extend(_render_failure_report_ticket_lines(ticket))

    lines.extend(
        [
            "---",
            "",
            "See `failure_report.json` for full check statuses, thresholds, claim/clause details, source IDs, and metrics.",
        ]
    )
    return "\n".join(lines) + "\n"


def _format_evaluation_header_parts(metrics: dict[str, Any]) -> list[str]:
    """Format numerator/denominator match counts for labeled evaluation metrics."""
    parts: list[str] = []
    for label, metric_key in _EVALUATION_HEADER_METRICS:
        entry = metrics.get(metric_key)
        if not isinstance(entry, dict):
            continue
        denominator = entry.get("denominator")
        numerator = entry.get("numerator")
        if isinstance(denominator, int) and denominator > 0 and isinstance(numerator, int):
            parts.append(f"{label} {numerator}/{denominator}")
    return parts


def _render_summary_table_row(ticket: dict[str, Any]) -> str:
    """Format one escaped single-line row for the failure_report.md summary table."""
    stopped_label = _STOPPED_AT_LABELS.get(ticket.get("stopped_at", ""), ticket.get("stopped_at", ""))
    main_reason = _format_ticket_main_reason(ticket)
    result_text = f"{ticket.get('final_route', '')} ({ticket.get('processing_status', '')})"
    return (
        f"| {_sanitize_markdown_table_cell(ticket.get('ticket_id'))} "
        f"| {_sanitize_markdown_table_cell(stopped_label)} "
        f"| {_sanitize_markdown_table_cell(main_reason)} "
        f"| {_sanitize_markdown_table_cell(result_text)} |"
    )


def _format_ticket_main_reason(ticket: dict[str, Any]) -> str:
    """Summarize the primary blocker or KPI mismatch for one summary table row."""
    stopped_at = ticket.get("stopped_at")
    if stopped_at == "evaluation_only":
        failed_kpis = ticket.get("evaluation", {}).get("failed_kpis") or []
        return f"KPI mismatch: {', '.join(failed_kpis)}"
    if stopped_at == "before_drafting":
        blocker = ticket.get("before_drafting", {}).get("first_blocking_check")
        if blocker == "classification_confidence_check":
            check = _find_named_check(
                ticket.get("before_drafting", {}).get("routing_checks") or [],
                "classification_confidence_check",
            )
            if check and isinstance(check.get("confidence"), (int, float)) and check.get("threshold") is not None:
                return f"classification_confidence_check ({check['confidence']:.3f} < {check['threshold']})"
        return blocker or ticket.get("final_reason") or "pre-draft failure"
    if stopped_at == "during_drafting":
        error_code = ticket.get("drafting", {}).get("error_code") or ticket.get("failed_check")
        return error_code or "draft generation failed"
    if stopped_at == "after_drafting":
        blocking_check = ticket.get("after_drafting", {}).get("blocking_check")
        if blocking_check == "citation_grounding":
            rejected = len(ticket.get("after_drafting", {}).get("failed_claims") or [])
            return f"citation_grounding ({rejected} rejected clauses)" if rejected else "citation_grounding"
        return blocking_check or ticket.get("final_reason") or "post-draft block"
    return ticket.get("final_reason") or "unknown"


def _render_failure_report_ticket_lines(ticket: dict[str, Any]) -> list[str]:
    """Render concise per-ticket diagnostic lines from failure_report.json."""
    stopped_at = ticket.get("stopped_at", "")
    stopped_label = _STOPPED_AT_LABELS.get(stopped_at, stopped_at)
    ticket_id = _sanitize_markdown_text(ticket.get("ticket_id"))
    lines = [f"## {ticket_id} — {stopped_label}", ""]

    if stopped_at == "before_drafting":
        lines.extend(_render_before_drafting_lines(ticket))
    elif stopped_at == "during_drafting":
        lines.extend(_render_during_drafting_lines(ticket))
    elif stopped_at == "after_drafting":
        lines.extend(_render_after_drafting_lines(ticket))
    elif stopped_at == "evaluation_only":
        lines.append(
            f"- Before drafting: passed; routed to `{_sanitize_markdown_text(ticket.get('final_route'))}`."
        )

    evaluation = ticket.get("evaluation") or {}
    mismatches = _format_kpi_mismatches(evaluation)
    if mismatches:
        lines.append(f"- Other mismatches: {_sanitize_markdown_text('; '.join(mismatches))}")
    doc_note = evaluation.get("expected_doc_note")
    if doc_note:
        lines.append(f"- Document note: {_sanitize_markdown_text(doc_note)}")

    lines.append("")
    return lines


def _render_before_drafting_lines(ticket: dict[str, Any]) -> list[str]:
    """Render Blocker and Then lines for a ticket stopped before drafting."""
    before = ticket.get("before_drafting") or {}
    routing_checks = before.get("routing_checks") or []
    blocker_name = before.get("first_blocking_check") or ticket.get("failed_check")
    blocker_text = _describe_pre_draft_blocker(ticket, before, blocker_name)
    skipped = [c["name"] for c in routing_checks if isinstance(c, dict) and c.get("status") == "skipped"]
    final_route = _sanitize_markdown_text(ticket.get("final_route"))
    if skipped:
        then_text = (
            f"Skipped {', '.join(skipped)}; drafting not attempted; "
            f"routed to `{final_route}` (no reply sent)."
        )
    else:
        then_text = f"Drafting not attempted; routed to `{final_route}` (no reply sent)."
    return [f"- Blocker: {blocker_text}", f"- Then: {then_text}"]


def _describe_pre_draft_blocker(
    ticket: dict[str, Any],
    before: dict[str, Any],
    blocker_name: str | None,
) -> str:
    """Describe the first pre-draft check or validation error that blocked the ticket."""
    routing_checks = before.get("routing_checks") or []
    check = _find_named_check(routing_checks, blocker_name)
    if blocker_name == "classification_confidence_check" and check is not None:
        conf = check.get("confidence")
        threshold = check.get("threshold")
        if isinstance(conf, (int, float)) and threshold is not None:
            return (
                f"`classification_confidence_check`: confidence {conf:.3f} < threshold {threshold} "
                f"({_sanitize_markdown_text(check.get('reason'))})"
            )
    if check is not None and check.get("reason"):
        return f"`{_sanitize_markdown_text(blocker_name)}`: {_sanitize_markdown_text(check.get('reason'))}"
    safety_check = _find_named_check(before.get("safety_checks") or [], blocker_name, key="check")
    if safety_check is not None:
        return (
            f"`{_sanitize_markdown_text(blocker_name)}`: "
            f"{_sanitize_markdown_text(safety_check.get('evidence') or ticket.get('final_reason'))}"
        )
    error = ticket.get("error")
    if isinstance(error, dict) and error.get("message"):
        return f"`{_sanitize_markdown_text(error.get('code') or blocker_name)}`: {_sanitize_markdown_text(error.get('message'))}"
    return _sanitize_markdown_text(ticket.get("final_reason") or blocker_name or "pre-draft block")


def _render_during_drafting_lines(ticket: dict[str, Any]) -> list[str]:
    """Render Before drafting, Blocker, and Then lines for a generation failure."""
    before_summary = _format_before_drafting_pass_summary(ticket)
    drafting = ticket.get("drafting") or {}
    error = ticket.get("error") if isinstance(ticket.get("error"), dict) else {}
    error_code = drafting.get("error_code") or error.get("code") or ticket.get("failed_check") or "generation_failed"
    detail = error.get("message") or ticket.get("final_reason") or "Draft generation failed."
    final_route = _sanitize_markdown_text(ticket.get("final_route"))
    return [
        f"- Before drafting: {before_summary}",
        f"- Blocker: `{_sanitize_markdown_text(error_code)}`: {_sanitize_markdown_text(detail)}",
        f"- Then: Drafting failed; after-drafting checks not run; routed to `{final_route}` (no reply sent).",
    ]


def _render_after_drafting_lines(ticket: dict[str, Any]) -> list[str]:
    """Render Before drafting, Blocker, and Then lines for a post-draft block."""
    before_summary = _format_before_drafting_pass_summary(ticket)
    after = ticket.get("after_drafting") or {}
    blocking_check = after.get("blocking_check") or ticket.get("failed_check")
    checks = after.get("checks") or []
    check = _find_named_check(checks, blocking_check)
    if blocking_check == "citation_grounding":
        failed_claims = after.get("failed_claims") or []
        cited_docs = (ticket.get("evaluation") or {}).get("predicted", {}).get("cited_doc_ids") or []
        parts = [f"{len(failed_claims)} rejected clause(s)"]
        if check and check.get("minimum_ratio") is not None:
            parts.append(f"minimum_ratio={check['minimum_ratio']}")
        if cited_docs:
            parts.append(f"cited docs: {', '.join(cited_docs)}")
        if after.get("diagnostics_note"):
            parts.append(str(after["diagnostics_note"]))
        blocker_text = f"`citation_grounding`: {_sanitize_markdown_text('; '.join(parts))}"
    else:
        evidence = (
            (check.get("evidence") if check else None)
            or after.get("diagnostics_note")
            or ticket.get("final_reason")
        )
        blocker_text = f"`{_sanitize_markdown_text(blocking_check)}`: {_sanitize_markdown_text(evidence)}"

    disposition = (ticket.get("drafting") or {}).get("draft_disposition") or "withheld"
    final_route = _sanitize_markdown_text(ticket.get("final_route"))
    return [
        f"- Before drafting: {before_summary}",
        f"- Blocker: {blocker_text}",
        f"- Then: Draft { _sanitize_markdown_text(disposition) }; routed to `{final_route}` (no reply sent).",
    ]


def _format_before_drafting_pass_summary(ticket: dict[str, Any]) -> str:
    """Summarize passed pre-draft checks when a ticket failed during or after drafting."""
    routing_checks = (ticket.get("before_drafting") or {}).get("routing_checks") or []
    passed_count = sum(1 for c in routing_checks if isinstance(c, dict) and c.get("status") == "passed")
    return f"passed ({passed_count} routing checks passed)." if passed_count else "passed."


def _format_kpi_mismatches(evaluation: dict[str, Any]) -> list[str]:
    """Format expected -> predicted strings for each failed KPI on a ticket."""
    failed_kpis = evaluation.get("failed_kpis") or []
    expected = evaluation.get("expected") or {}
    predicted = evaluation.get("predicted") or {}
    kpi_pairs = {
        "intent": (expected.get("intent"), predicted.get("intent")),
        "urgency": (expected.get("urgency"), predicted.get("urgency")),
        "route": (expected.get("expected_route"), predicted.get("route")),
        "answerability": (expected.get("answerable_from_docs"), predicted.get("answerable_from_docs")),
        "expected_doc_ids": (expected.get("expected_doc_ids"), predicted.get("retrieved_doc_ids")),
        "must_not_auto_respond": (
            expected.get("must_not_auto_respond"),
            predicted.get("must_not_auto_respond"),
        ),
    }
    formatted: list[str] = []
    for kpi in failed_kpis:
        exp_val, pred_val = kpi_pairs.get(kpi, (None, None))
        formatted.append(f"{kpi}: {exp_val} -> {pred_val}")
    return formatted


def _find_named_check(
    checks: list[dict[str, Any]],
    name: str | None,
    *,
    key: str = "name",
) -> dict[str, Any] | None:
    """Return the first check dictionary matching the given name."""
    if not name:
        return None
    return next((c for c in checks if isinstance(c, dict) and c.get(key) == name), None)


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


def _escape_unpaired_surrogates(text: str) -> str:
    """Replace unpaired UTF-16 surrogates with visible \\uXXXX escapes for UTF-8 safety."""
    return _UNPAIRED_SURROGATE_PATTERN.sub(
        lambda match: f"\\u{ord(match.group(0)):04x}",
        text,
    )


def _sanitize_markdown_text(value: Any) -> str:
    """Escape unpaired surrogates and control characters for single-line UTF-8 Markdown."""
    if not isinstance(value, str):
        return str(value)
    escaped = _escape_unpaired_surrogates(value)
    return _CONTROL_CHAR_PATTERN.sub(
        lambda match: " " if match.group(0) in {"\n", "\r", "\t"} else f"\\u{ord(match.group(0)):04x}",
        escaped,
    )


def _sanitize_markdown_table_cell(value: Any) -> str:
    """Escape surrogates, control characters, and pipe characters for single-line table cells."""
    return re.sub(
        r"(\\*)\|",
        lambda match: (match.group(1) * 2) + "\\|",
        _sanitize_markdown_text(value),
    )


def _serialize_jsonl_record(record: dict[str, Any]) -> str:
    """Serialize one JSONL record as UTF-8-safe JSON while escaping unpaired surrogates."""
    encoded = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
    return _escape_unpaired_surrogates(encoded) + "\n"


def _write_json_atomic(path: Path, record: dict[str, Any]) -> None:
    """Write one formatted JSON export through a same-directory temporary file."""
    encoded = json.dumps(record, ensure_ascii=False, indent=2)
    _replace_file_atomically(path, _escape_unpaired_surrogates(encoded) + "\n")


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
