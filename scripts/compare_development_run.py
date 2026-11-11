#!/usr/bin/env python3
"""Reconcile a completed run and report development-label diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from cloudserve_support.processing.run_provenance import (
    fingerprint_classifier_implementation,
    fingerprint_classifier_model,
    fingerprint_policy_file,
    fingerprint_policy_threshold,
)
from cloudserve_support.routing.policy_calibration import load_frozen_routing_policy


ROUTE_ALIASES = {"auto_respond": "draft_for_review", "escalate": "escalate"}
EVALUATION_KEYS = {"labels", "history"}
EXPECTED_POLICY_VERSION = "cs003-v5"
EXPECTED_CLASSIFIER_MODEL = "offline-rule-based-v5"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY_PATH = REPO_ROOT / "data" / "policy" / "frozen_routing_policy.json"
PROVENANCE_FINGERPRINT_FIELDS = (
    "policy_file_sha256",
    "threshold_sha256",
    "classifier_implementation_sha256",
    "classifier_model_sha256",
)


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def _has_evaluation_keys(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            key in EVALUATION_KEYS or _has_evaluation_keys(child)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(_has_evaluation_keys(child) for child in value)
    return False


def _db_readonly(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)


def _index_rows(
    rows: list[dict[str, Any]], *, index_field: str, id_field: str
) -> tuple[dict[int, dict[str, Any]], list[str]]:
    by_index: dict[int, dict[str, Any]] = {}
    errors: list[str] = []
    for row_number, row in enumerate(rows):
        index = row.get(index_field)
        if not isinstance(index, int) or isinstance(index, bool):
            errors.append(f"row {row_number} has invalid {index_field}: {index!r}")
            continue
        if index in by_index:
            errors.append(f"duplicate {index_field} {index}")
            continue
        by_index[index] = row
        if not isinstance(row.get(id_field), str):
            errors.append(f"{index_field} {index} has no string {id_field}")
    return by_index, errors


def _automatic_send(result: dict[str, Any]) -> bool:
    draft = result.get("draft")
    return isinstance(draft, dict) and draft.get("send_status") == "sent"


def compare_run(
    *,
    input_path: Path,
    run_dir: Path,
    database_path: Path,
    cli_exit_code: int,
    policy_path: Path = DEFAULT_POLICY_PATH,
    expected_provenance_path: Path | None = None,
) -> dict[str, Any]:
    """Reconcile persisted run data, then compare evaluation labels post-run."""
    metrics_path = run_dir / "metrics.json"
    results_path = run_dir / "results.jsonl"
    decisions_path = run_dir / "decision_log.jsonl"
    exceptions_path = run_dir / "exceptions.jsonl"
    for path in (metrics_path, results_path, decisions_path, exceptions_path):
        if not path.is_file():
            raise FileNotFoundError(f"required run artifact is missing: {path}")
    if not database_path.is_file():
        raise FileNotFoundError(f"SQLite database is missing: {database_path}")

    metrics = _read_json(metrics_path)
    results = _read_jsonl(results_path)
    decisions = _read_jsonl(decisions_path)
    exceptions = _read_jsonl(exceptions_path)
    run_id = metrics.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("metrics.json has no run_id")
    frozen_policy = load_frozen_routing_policy(policy_path.expanduser().resolve())
    expected_fingerprints = {
        "policy_file_sha256": fingerprint_policy_file(policy_path.expanduser().resolve()),
        "threshold_sha256": fingerprint_policy_threshold(frozen_policy),
        "classifier_implementation_sha256": fingerprint_classifier_implementation(),
        "classifier_model_sha256": fingerprint_classifier_model(
            frozen_policy["classifier_provider"], frozen_policy["classifier_model"]
        ),
    }
    report_policy = metrics.get("evaluation", {}).get("policy_provenance", {})
    report_fingerprints = report_policy.get("fingerprints", {})
    if report_policy.get("fingerprint_status") != "available" or any(
        report_fingerprints.get(field) != expected_fingerprints[field]
        for field in PROVENANCE_FINGERPRINT_FIELDS
    ):
        raise ValueError("metrics.json frozen provenance fingerprints are missing or inconsistent")

    # Read-only operational reconciliation happens before opening the fixture labels.
    with _db_readonly(database_path) as connection:
        connection.row_factory = sqlite3.Row
        run_rows = connection.execute(
            "SELECT * FROM processing_runs WHERE run_id = ?", (run_id,)
        ).fetchall()
        if len(run_rows) != 1:
            raise ValueError(f"expected one processing_runs row for {run_id}, found {len(run_rows)}")
        run = dict(run_rows[0])
        ticket_rows = [
            dict(row)
            for row in connection.execute(
                "SELECT * FROM ticket_runs WHERE run_id = ? ORDER BY ticket_id", (run_id,)
            )
        ]
        audit_rows = [
            dict(row)
            for row in connection.execute(
                "SELECT * FROM audit_events WHERE run_id = ? ORDER BY ticket_id, occurred_at, event_id",
                (run_id,),
            )
        ]

    if run["status"] not in {"completed", "completed_with_errors", "failed"}:
        raise ValueError(f"run {run_id} is not terminal; status is {run['status']!r}")
    if run["output_dir"] and Path(run["output_dir"]).resolve() != run_dir.resolve():
        raise ValueError(
            f"SQLite output_dir {run['output_dir']!r} does not match {str(run_dir)!r}"
        )
    if metrics.get("run_id") != run_id or any(row.get("run_id") != run_id for row in results + decisions):
        raise ValueError("run_id differs between metrics, results, decision log, and SQLite")

    # Evaluation fixture contents are first read only after a terminal run and its
    # durable/export records have been loaded.
    source_bytes = input_path.read_bytes()
    tickets = json.loads(source_bytes)
    if not isinstance(tickets, list):
        raise ValueError(f"{input_path}: expected a JSON array")

    result_by_index, result_errors = _index_rows(
        results, index_field="input_index", id_field="ticket_id"
    )
    decision_by_index, decision_errors = _index_rows(
        decisions, index_field="input_index", id_field="ticket_id"
    )
    exception_by_index, exception_errors = _index_rows(
        exceptions, index_field="input_index", id_field="ticket_id"
    )
    reconcile_errors = [*result_errors, *decision_errors, *exception_errors]
    expected_indices = set(range(len(tickets)))
    for label, rows_by_index in (("results", result_by_index), ("decisions", decision_by_index)):
        found_indices = set(rows_by_index)
        for missing in sorted(expected_indices - found_indices):
            reconcile_errors.append(f"{label} missing input_index {missing}")
        for extra in sorted(found_indices - expected_indices):
            reconcile_errors.append(f"{label} has unexpected input_index {extra}")
    for extra in sorted(set(exception_by_index) - expected_indices):
        reconcile_errors.append(f"exceptions has unexpected input_index {extra}")
    for input_index, exception in exception_by_index.items():
        if exception.get("run_id") != run_id:
            reconcile_errors.append(f"exception run_id differs at input_index {input_index}")
        if input_index in expected_indices:
            ticket = tickets[input_index]
            expected_ticket_id = ticket.get("ticket_id") if isinstance(ticket, dict) else None
            if exception.get("ticket_id") != expected_ticket_id:
                reconcile_errors.append(f"exception ticket_id does not match input at input_index {input_index}")

    failed_results = 0
    for input_index in sorted(expected_indices):
        result = result_by_index.get(input_index)
        exception = exception_by_index.get(input_index)
        if result is None:
            if exception is not None:
                reconcile_errors.append(f"exception has no corresponding result at input_index {input_index}")
            continue
        result_status = result.get("status")
        result_error = result.get("error")
        if result_status == "failed":
            failed_results += 1
            if exception is None:
                reconcile_errors.append(f"failed result has no exception at input_index {input_index}")
            elif not isinstance(result_error, dict):
                reconcile_errors.append(f"failed result has no error object at input_index {input_index}")
            elif exception.get("error_code") != result_error.get("code"):
                reconcile_errors.append(f"exception error code differs at input_index {input_index}")
        elif result_status == "completed":
            if exception is not None:
                reconcile_errors.append(f"completed result has an exception at input_index {input_index}")
        else:
            reconcile_errors.append(f"invalid result status at input_index {input_index}: {result_status!r}")

    expected_metrics_counts = {
        "total_count": len(results),
        "completed_count": len(results) - failed_results,
        "failed_count": failed_results,
        "exception_count": len(exceptions),
    }
    for field, expected_count in expected_metrics_counts.items():
        if metrics.get(field) != expected_count:
            reconcile_errors.append(
                f"metrics.{field} is {metrics.get(field)!r}, expected {expected_count}"
            )

    ticket_counts = Counter(row["ticket_id"] for row in ticket_rows)
    audit_by_ticket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in audit_rows:
        audit_by_ticket[row["ticket_id"]].append(row)
    expected_ticket_ids = [
        ticket.get("ticket_id") if isinstance(ticket, dict) else None for ticket in tickets
    ]
    if any(not isinstance(ticket_id, str) for ticket_id in expected_ticket_ids):
        reconcile_errors.append("input contains a record without a string ticket_id")
    if len(set(expected_ticket_ids)) != len(expected_ticket_ids):
        reconcile_errors.append("input contains duplicate ticket_id values")
    for ticket_id in expected_ticket_ids:
        if not isinstance(ticket_id, str):
            continue
        if ticket_counts[ticket_id] != 1:
            reconcile_errors.append(
                f"SQLite ticket_runs has {ticket_counts[ticket_id]} rows for {ticket_id}"
            )
        if len(audit_by_ticket[ticket_id]) != 1:
            reconcile_errors.append(
                f"SQLite audit_events has {len(audit_by_ticket[ticket_id])} rows for {ticket_id}"
            )
    unexpected_db_ids = (set(ticket_counts) | set(audit_by_ticket)) - set(expected_ticket_ids)
    for ticket_id in sorted(unexpected_db_ids):
        reconcile_errors.append(f"SQLite contains unexpected ticket_id {ticket_id}")

    runtime_scan: list[Any] = [results, decisions]
    db_ticket_by_id = {row["ticket_id"]: row for row in ticket_rows}
    for row in ticket_rows:
        try:
            runtime_scan.append(json.loads(row["context_json"]))
        except (TypeError, json.JSONDecodeError):
            reconcile_errors.append(f"invalid ticket_runs.context_json for {row['ticket_id']}")
    for row in audit_rows:
        for column in ("result_payload_json", "sources_json"):
            raw = row.get(column)
            if raw is None:
                continue
            try:
                runtime_scan.append(json.loads(raw))
            except (TypeError, json.JSONDecodeError):
                reconcile_errors.append(f"invalid audit_events.{column} for {row['ticket_id']}")

    for input_index, ticket in enumerate(tickets):
        if not isinstance(ticket, dict):
            reconcile_errors.append(f"input_index {input_index} is not an object")
            continue
        ticket_id = ticket.get("ticket_id")
        result = result_by_index.get(input_index)
        decision = decision_by_index.get(input_index)
        if result and result.get("ticket_id") != ticket_id:
            reconcile_errors.append(f"results input_index {input_index} ticket_id does not match input")
        if decision and decision.get("ticket_id") != ticket_id:
            reconcile_errors.append(f"decision input_index {input_index} ticket_id does not match input")
        if decision and result and decision.get("result") != result:
            reconcile_errors.append(f"decision result differs from results row at input_index {input_index}")
        if not isinstance(ticket_id, str):
            continue
        if ticket_id in db_ticket_by_id and result:
            expected_db_status = "failed" if result.get("status") == "failed" else "completed"
            actual_db_status = db_ticket_by_id[ticket_id]["processing_status"]
            if actual_db_status != expected_db_status:
                reconcile_errors.append(
                    f"SQLite ticket {ticket_id} status is {actual_db_status!r}, expected {expected_db_status!r}"
                )
        rows = audit_by_ticket.get(ticket_id, [])
        if result and rows:
            try:
                persisted_result = json.loads(rows[0]["result_payload_json"] or "null")
            except json.JSONDecodeError:
                persisted_result = None
            if persisted_result != result:
                reconcile_errors.append(f"SQLite result payload differs at input_index {input_index}")
            try:
                persisted_sources = json.loads(rows[0]["sources_json"] or "[]")
            except json.JSONDecodeError:
                persisted_sources = None
            if persisted_sources != result.get("sources", []):
                reconcile_errors.append(f"SQLite source snapshot differs at input_index {input_index}")

    mismatch_rows: list[dict[str, Any]] = []
    mismatch_counts: Counter[str] = Counter()
    protected_checked = 0
    protected_violations = 0
    protected_route_deviations: list[dict[str, Any]] = []
    no_send_checked = 0
    send_violations = 0
    policy_errors: list[str] = []
    seen_fingerprints: set[tuple[str | None, ...]] = set()

    for input_index, ticket in enumerate(tickets):
        if not isinstance(ticket, dict):
            continue
        labels = ticket.get("labels")
        if not isinstance(labels, dict):
            reconcile_errors.append(f"input_index {input_index} has no labels object")
            labels = {}
        result = result_by_index.get(input_index)
        if result is None:
            continue
        mismatches: dict[str, Any] = {}
        classification = result.get("classification") or {}
        for field in ("intent", "urgency"):
            expected = labels.get(field)
            actual = classification.get(field) if isinstance(classification, dict) else None
            if actual != expected:
                mismatches[field] = {"expected": expected, "actual": actual}
                mismatch_counts[field] += 1

        expected_fixture_route = labels.get("expected_route")
        expected_route = ROUTE_ALIASES.get(expected_fixture_route, expected_fixture_route)
        actual_route = result.get("route")
        if actual_route != expected_route:
            mismatches["route"] = {
                "expected_fixture_route": expected_fixture_route,
                "expected_public_route": expected_route,
                "actual": actual_route,
            }
            mismatch_counts["route"] += 1

        protected = labels.get("must_not_auto_respond") is True
        if protected:
            protected_checked += 1
            if actual_route != "escalate":
                protected_route_deviations.append(
                    {
                        "input_index": input_index,
                        "ticket_id": ticket.get("ticket_id"),
                        "expected_route": "escalate",
                        "actual_route": actual_route,
                    }
                )
            if actual_route != "escalate" or _automatic_send(result):
                protected_violations += 1
                mismatches["protection"] = {
                    "must_not_auto_respond": True,
                    "actual_route": actual_route,
                    "send_status": (result.get("draft") or {}).get("send_status")
                    if isinstance(result.get("draft"), dict)
                    else None,
                }
                mismatch_counts["protection"] += 1

        no_send_checked += 1
        if _automatic_send(result):
            send_violations += 1
            mismatches["automatic_send"] = {"actual_send_status": "sent"}
            mismatch_counts["automatic_send"] += 1

        expected_docs = labels.get("expected_doc_ids")
        if not isinstance(expected_docs, list):
            expected_docs = []
        expected_doc_ids = sorted({value for value in expected_docs if isinstance(value, str)})
        sources = result.get("sources")
        actual_doc_ids = sorted(
            {
                source.get("doc_id")
                for source in sources
                if isinstance(source, dict) and isinstance(source.get("doc_id"), str)
            }
        ) if isinstance(sources, list) else []
        if actual_doc_ids != expected_doc_ids:
            mismatches["source_ids"] = {
                "expected_doc_ids": expected_doc_ids,
                "actual_doc_ids": actual_doc_ids,
            }
            mismatch_counts["source_ids"] += 1

        if result.get("status") != "failed":
            routing = result.get("routing")
            if not isinstance(classification, dict) or classification.get("model_name") != EXPECTED_CLASSIFIER_MODEL:
                policy_errors.append(f"input_index {input_index} classifier is not {EXPECTED_CLASSIFIER_MODEL}")
            if not isinstance(routing, dict) or routing.get("policy_version") != EXPECTED_POLICY_VERSION or routing.get("policy_frozen") is not True:
                policy_errors.append(f"input_index {input_index} does not use frozen {EXPECTED_POLICY_VERSION}")
            policy_provenance = routing.get("policy_provenance") if isinstance(routing, dict) else None
            actual_fingerprints = {
                field: policy_provenance.get(field) if isinstance(policy_provenance, dict) else None
                for field in PROVENANCE_FINGERPRINT_FIELDS
            }
            seen_fingerprints.add(
                tuple(actual_fingerprints[field] for field in PROVENANCE_FINGERPRINT_FIELDS)
            )
            for field, expected_fingerprint in expected_fingerprints.items():
                if actual_fingerprints[field] != expected_fingerprint:
                    policy_errors.append(
                        f"input_index {input_index} {field} does not match the current frozen run design"
                    )
        if mismatches:
            mismatch_rows.append(
                {"input_index": input_index, "ticket_id": ticket.get("ticket_id"), "mismatches": mismatches}
            )

    runtime_records_with_evaluation_keys = sum(_has_evaluation_keys(value) for value in runtime_scan)
    if run["status"] not in {"completed", "completed_with_errors"}:
        reconcile_errors.append(f"processing run status is {run['status']!r}")
    if cli_exit_code != 0:
        reconcile_errors.append(f"CLI exit status is {cli_exit_code}, expected 0")
    if runtime_records_with_evaluation_keys:
        reconcile_errors.append(
            f"{runtime_records_with_evaluation_keys} runtime record(s) contain labels/history keys"
        )
    reconcile_errors.extend(policy_errors)

    report_provenance = {
        "status": "available" if not policy_errors else "mismatch",
        "fingerprints": expected_fingerprints,
        "run_fingerprint_sets": len(seen_fingerprints),
        "expected_baseline_path": (
            str(expected_provenance_path) if expected_provenance_path is not None else None
        ),
        "matches_expected_baseline": None,
    }
    if expected_provenance_path is not None:
        baseline_metrics = _read_json(expected_provenance_path)
        baseline_evaluation = baseline_metrics.get("evaluation", {})
        baseline_policy = baseline_evaluation.get("policy_provenance", {})
        baseline_fingerprints = baseline_policy.get("fingerprints", {})
        baseline_status = baseline_policy.get("fingerprint_status")
        baseline_matches = baseline_status == "available" and all(
            baseline_fingerprints.get(field) == expected_fingerprints[field]
            for field in PROVENANCE_FINGERPRINT_FIELDS
        )
        report_provenance["matches_expected_baseline"] = baseline_matches
        if not baseline_matches:
            report_provenance["status"] = "mismatch"
            reconcile_errors.append(
                "frozen policy, threshold, classifier implementation, or model fingerprint differs from expected baseline"
            )

    mismatch_ticket_count = len(mismatch_rows)
    safety_errors = (
        [f"{send_violations} automatic send violation(s) found"] if send_violations else []
    )
    # Label comparisons are evaluation diagnostics. A mismatch does not make a
    # fully processed and reconciled batch fail CS-020.
    passed = not reconcile_errors and not safety_errors
    return {
        "run_id": run_id,
        "gate": "CS-020 run completion and post-run diagnostics",
        "comparison_timing": "after terminal processing status and persisted exports/database records were verified",
        "cli_exit_code": cli_exit_code,
        "cli_exit_code_source": "operator-confirmed command exit status supplied to the comparator",
        "input": {
            "path": str(input_path),
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
        },
        "passed": passed,
        "completion_gate_passed": passed,
        "label_mismatches_are_diagnostic_only": True,
        "counts": {
            "input_records": len(tickets),
            "result_rows": len(results),
            "decision_rows": len(decisions),
            "sqlite_ticket_runs": len(ticket_rows),
            "sqlite_audit_events": len(audit_rows),
            "exceptions": len(exceptions),
            "evaluation_label_records": sum(isinstance(ticket, dict) and isinstance(ticket.get("labels"), dict) for ticket in tickets),
            "mismatched_tickets": mismatch_ticket_count,
            "mismatch_checks_by_field": dict(sorted(mismatch_counts.items())),
            "protected_label_cases_checked": protected_checked,
            "protected_case_violations": protected_violations,
            "protected_route_deviation_count": len(protected_route_deviations),
            "no_auto_send_records_checked": no_send_checked,
            "automatic_send_violations": send_violations,
            "runtime_records_with_labels_or_history_keys": runtime_records_with_evaluation_keys,
        },
        "comparison_contract": {
            "identity": "each result is joined by input_index and checked against the fixture ticket_id",
            "intent": "exact string match",
            "urgency": "exact string match",
            "route": "fixture auto_respond maps to public draft_for_review; escalate maps to escalate",
            "protection": "must_not_auto_respond=true requires route=escalate and no automatic send; all records are checked for automatic send",
            "source_ids": "exact sorted-set equality between labels.expected_doc_ids and result.sources[].doc_id",
            "evaluation_boundary": "fixture labels/history are opened only after the run is terminal and persisted operational records are loaded",
            "database_reconciliation": "SQLite run, ticket, audit, result payload, and source snapshots must match run exports",
            "frozen_policy": f"all records must use {EXPECTED_POLICY_VERSION} and {EXPECTED_CLASSIFIER_MODEL}",
        },
        "reconciliation": {
            "processing_run_status": run["status"],
            "input_ref": run["input_ref"],
            "sqlite_output_dir": run["output_dir"],
            "errors": reconcile_errors,
        },
        "safety": {"automatic_send_violations": send_violations, "errors": safety_errors},
        "protected_route_deviations": protected_route_deviations,
        "frozen_provenance_fingerprints": report_provenance,
        "mismatches": mismatch_rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reconcile a completed development run and report evaluation-label diagnostics."
    )
    parser.add_argument("--input", required=True, type=Path, help="development fixture JSON")
    parser.add_argument("--run-dir", required=True, type=Path, help="per-run output directory")
    parser.add_argument("--database", required=True, type=Path, help="SQLite runtime database")
    parser.add_argument(
        "--policy", type=Path, default=DEFAULT_POLICY_PATH,
        help="frozen policy file used by the completed run",
    )
    parser.add_argument(
        "--expected-provenance", type=Path,
        help="baseline metrics.json whose policy, threshold, classifier, and model fingerprints must match",
    )
    parser.add_argument(
        "--cli-exit-code",
        required=True,
        type=int,
        help="exit code captured from the completed cloudserve run command",
    )
    args = parser.parse_args(argv)
    report_path = args.run_dir / "label_comparison.json"
    try:
        report = compare_run(
            input_path=args.input,
            run_dir=args.run_dir,
            database_path=args.database,
            cli_exit_code=args.cli_exit_code,
            policy_path=args.policy,
            expected_provenance_path=args.expected_provenance,
        )
    except (OSError, ValueError, sqlite3.Error, json.JSONDecodeError) as error:
        print(f"completion gate could not run: {error}", file=sys.stderr)
        return 2

    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"run_id": report["run_id"], "passed": report["passed"], **report["counts"]}, indent=2))
    print(f"report: {report_path}")
    return 0 if report["completion_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
