import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from compare_development_run import compare_run
from cloudserve_support.processing.run_provenance import (
    fingerprint_classifier_implementation,
    fingerprint_classifier_model,
    fingerprint_policy_file,
    fingerprint_policy_threshold,
)
from cloudserve_support.routing.policy_calibration import load_frozen_routing_policy


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _make_run(tmp_path, *, labels=None, route="draft_for_review"):
    run_id = "run-1"
    input_path = tmp_path / "tickets.json"
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    database_path = tmp_path / "runtime.sqlite3"
    policy_path = Path("data/policy/frozen_routing_policy.json").resolve()
    frozen_policy = load_frozen_routing_policy(policy_path)
    fingerprints = {
        "policy_file_sha256": fingerprint_policy_file(policy_path),
        "threshold_sha256": fingerprint_policy_threshold(frozen_policy),
        "classifier_implementation_sha256": fingerprint_classifier_implementation(),
        "classifier_model_sha256": fingerprint_classifier_model(
            frozen_policy["classifier_provider"], frozen_policy["classifier_model"]
        ),
    }
    policy_provenance = {
        "policy_version": frozen_policy["policy_version"],
        "classification_confidence_threshold": frozen_policy[
            "classification_confidence_threshold"
        ],
        "threshold_source": frozen_policy["threshold_source"],
        "calibration_method": frozen_policy["calibration_method"],
        "calibration_source": frozen_policy["calibration_source"],
        "frozen": frozen_policy["frozen"],
        **fingerprints,
    }
    ticket = {
        "ticket_id": "T-1",
        "labels": labels or {
            "intent": "billing_query",
            "urgency": "medium",
            "expected_route": "auto_respond",
            "expected_doc_ids": ["DOC-1"],
            "must_not_auto_respond": False,
        },
        "history": {"escalated": False},
    }
    result = {
        "run_id": run_id,
        "input_index": 0,
        "ticket_id": "T-1",
        "route": route,
        "status": "completed",
        "classification": {
            "intent": "billing_query",
            "urgency": "medium",
            "model_name": "offline-rule-based-v5",
            "provider": "offline",
        },
        "routing": {
            "policy_version": "cs003-v5",
            "policy_frozen": True,
            "policy_provenance": policy_provenance,
        },
        "sources": [{"doc_id": "DOC-1"}],
        "draft": {"send_status": "not_sent"},
    }
    _write_json(input_path, [ticket])
    _write_json(
        run_dir / "metrics.json",
        {
            "run_id": run_id,
            "total_count": 1,
            "completed_count": 1,
            "failed_count": 0,
            "exception_count": 0,
            "evaluation": {
                "policy_provenance": {
                    "fingerprint_status": "available",
                    "fingerprints": fingerprints,
                }
            },
        },
    )
    _write_jsonl(run_dir / "results.jsonl", [result])
    _write_jsonl(
        run_dir / "decision_log.jsonl",
        [{"run_id": run_id, "input_index": 0, "ticket_id": "T-1", "result": result}],
    )
    _write_jsonl(run_dir / "exceptions.jsonl", [])

    with sqlite3.connect(database_path) as connection:
        connection.executescript(
            """
            CREATE TABLE processing_runs (
                run_id TEXT, input_ref TEXT, status TEXT, output_dir TEXT
            );
            CREATE TABLE ticket_runs (
                run_id TEXT, ticket_id TEXT, context_json TEXT, processing_status TEXT
            );
            CREATE TABLE audit_events (
                event_id TEXT, run_id TEXT, ticket_id TEXT, occurred_at TEXT,
                result_payload_json TEXT, sources_json TEXT
            );
            """
        )
        connection.execute(
            "INSERT INTO processing_runs VALUES (?, ?, ?, ?)",
            (run_id, str(input_path), "completed", str(run_dir)),
        )
        connection.execute(
            "INSERT INTO ticket_runs VALUES (?, ?, ?, ?)",
            (run_id, "T-1", json.dumps({"ticket_id": "T-1"}), "completed"),
        )
        connection.execute(
            "INSERT INTO audit_events VALUES (?, ?, ?, ?, ?, ?)",
            ("E-1", run_id, "T-1", "2026-01-01T00:00:00Z", json.dumps(result), json.dumps(result["sources"])),
        )
    return input_path, run_dir, database_path


def test_completion_gate_passes_when_exports_and_database_match(tmp_path):
    input_path, run_dir, database_path = _make_run(tmp_path)

    report = compare_run(
        input_path=input_path,
        run_dir=run_dir,
        database_path=database_path,
        cli_exit_code=0,
    )

    assert report["passed"] is True
    assert report["counts"]["runtime_records_with_labels_or_history_keys"] == 0
    assert report["reconciliation"]["errors"] == []
    assert report["frozen_provenance_fingerprints"]["status"] == "available"
    assert report["frozen_provenance_fingerprints"]["fingerprints"]["policy_file_sha256"]


def test_completion_gate_compares_frozen_provenance_with_baseline_metrics(tmp_path):
    input_path, run_dir, database_path = _make_run(tmp_path)

    report = compare_run(
        input_path=input_path,
        run_dir=run_dir,
        database_path=database_path,
        cli_exit_code=0,
        expected_provenance_path=run_dir / "metrics.json",
    )

    assert report["passed"] is True
    assert report["frozen_provenance_fingerprints"]["matches_expected_baseline"] is True


def test_completion_gate_rejects_changed_frozen_provenance_baseline(tmp_path):
    input_path, run_dir, database_path = _make_run(tmp_path)
    baseline_path = tmp_path / "baseline-metrics.json"
    baseline_metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    baseline_metrics["evaluation"]["policy_provenance"]["fingerprints"][
        "policy_file_sha256"
    ] = "f" * 64
    _write_json(baseline_path, baseline_metrics)

    report = compare_run(
        input_path=input_path,
        run_dir=run_dir,
        database_path=database_path,
        cli_exit_code=0,
        expected_provenance_path=baseline_path,
    )

    assert report["passed"] is False
    assert report["frozen_provenance_fingerprints"]["status"] == "mismatch"
    assert report["frozen_provenance_fingerprints"]["matches_expected_baseline"] is False
    assert any("fingerprint differs from expected baseline" in error for error in report["reconciliation"]["errors"])


def test_label_and_protection_mismatches_are_diagnostic_not_completion_failures(tmp_path):
    input_path, run_dir, database_path = _make_run(
        tmp_path,
        labels={
            "intent": "security_incident",
            "urgency": "high",
            "expected_route": "escalate",
            "expected_doc_ids": ["DOC-2"],
            "must_not_auto_respond": True,
        },
    )

    report = compare_run(
        input_path=input_path,
        run_dir=run_dir,
        database_path=database_path,
        cli_exit_code=0,
    )

    assert report["passed"] is True
    assert report["label_mismatches_are_diagnostic_only"] is True
    assert report["counts"]["mismatched_tickets"] == 1
    assert report["counts"]["protected_case_violations"] == 1
    assert report["counts"]["mismatch_checks_by_field"] == {
        "intent": 1,
        "protection": 1,
        "route": 1,
        "source_ids": 1,
        "urgency": 1,
    }


def test_ticket_level_failure_with_matching_exception_does_not_fail_completed_batch(tmp_path):
    input_path, run_dir, database_path = _make_run(tmp_path)
    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics.update(total_count=1, completed_count=0, failed_count=1, exception_count=1)
    _write_json(metrics_path, metrics)

    result = json.loads((run_dir / "results.jsonl").read_text(encoding="utf-8"))
    result.update(status="failed", error={"code": "provider_error", "message": "unavailable"})
    _write_jsonl(run_dir / "results.jsonl", [result])
    _write_jsonl(
        run_dir / "decision_log.jsonl",
        [{"run_id": "run-1", "input_index": 0, "ticket_id": "T-1", "result": result}],
    )
    _write_jsonl(
        run_dir / "exceptions.jsonl",
        [{
            "run_id": "run-1",
            "input_index": 0,
            "ticket_id": "T-1",
            "error_code": "provider_error",
            "message": "unavailable",
        }],
    )
    with sqlite3.connect(database_path) as connection:
        connection.execute("UPDATE processing_runs SET status = 'completed_with_errors'")
        connection.execute("UPDATE ticket_runs SET processing_status = 'failed'")
        connection.execute(
            "UPDATE audit_events SET result_payload_json = ?",
            (json.dumps(result),),
        )

    report = compare_run(
        input_path=input_path,
        run_dir=run_dir,
        database_path=database_path,
        cli_exit_code=0,
    )

    assert report["completion_gate_passed"] is True
    assert report["counts"]["exceptions"] == 1
    assert report["reconciliation"]["errors"] == []


def test_exception_ticket_identity_mismatch_fails_run_reconciliation(tmp_path):
    input_path, run_dir, database_path = _make_run(tmp_path)
    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics.update(total_count=1, completed_count=0, failed_count=1, exception_count=1)
    _write_json(metrics_path, metrics)

    result = json.loads((run_dir / "results.jsonl").read_text(encoding="utf-8"))
    result.update(status="failed", error={"code": "provider_error", "message": "unavailable"})
    _write_jsonl(run_dir / "results.jsonl", [result])
    _write_jsonl(
        run_dir / "decision_log.jsonl",
        [{"run_id": "run-1", "input_index": 0, "ticket_id": "T-1", "result": result}],
    )
    _write_jsonl(
        run_dir / "exceptions.jsonl",
        [{
            "run_id": "run-1",
            "input_index": 0,
            "ticket_id": "T-OTHER",
            "error_code": "provider_error",
            "message": "unavailable",
        }],
    )
    with sqlite3.connect(database_path) as connection:
        connection.execute("UPDATE processing_runs SET status = 'completed_with_errors'")
        connection.execute("UPDATE ticket_runs SET processing_status = 'failed'")
        connection.execute(
            "UPDATE audit_events SET result_payload_json = ?",
            (json.dumps(result),),
        )

    report = compare_run(
        input_path=input_path,
        run_dir=run_dir,
        database_path=database_path,
        cli_exit_code=0,
    )

    assert report["completion_gate_passed"] is False
    assert "exception ticket_id does not match input at input_index 0" in report["reconciliation"]["errors"]
