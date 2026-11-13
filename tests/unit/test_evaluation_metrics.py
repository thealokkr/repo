"""Check post-run metrics, evaluation boundaries, and explicit missingness."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from cloudserve_support.evaluation.run_metrics import evaluate_completed_run
from cloudserve_support.reporting.terminal_summary import build_terminal_summary
from cloudserve_support.reporting.output_schemas import EVALUATION_REPORT_SCHEMA
from jsonschema import Draft202012Validator


def _result(*, input_index: int, ticket_id: str) -> dict[str, object]:
    """Build a persisted outcome with one mechanically grounded cited claim."""
    source = {
        "passage_id": "DOC-1:section:0",
        "doc_id": "DOC-1",
        "source_ref": "data/kb/documentation.json#DOC-1#Section",
        "passage_text": "Rotate the key from the account security page.",
        "approval_status": "approved",
        "approval_status_source": "test project metadata manifest",
        "metadata_owner": "test maintainer",
        "metadata_source": "test project metadata manifest",
        "article_version": "test-1.0",
        "content_sha256": "1" * 64,
        "metadata_manifest_sha256": "2" * 64,
        "reviewed_at": (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat(),
        "review_interval_days": 365,
    }
    return {
        "run_id": "a" * 32,
        "input_index": input_index,
        "ticket_id": ticket_id,
        "route": "draft_for_review",
        "status": "completed",
        "generation": {"status": "generated"},
        "classification": {
            "intent": "billing",
            "urgency": "medium",
            "confidence": 0.8,
            "provider": "gemini",
            "model_name": "gemini-decision-1",
        },
        "sources": [source],
        "draft": {
            "claims": [
                {
                    "text": "Rotate the key from the account security page.",
                    "source_ids": [source["passage_id"]],
                    "citations": [
                        {
                            "passage_id": source["passage_id"],
                            "source_ref": source["source_ref"],
                        }
                    ],
                }
            ]
        },
        "guardrails": {
            "checks": [
                {
                    "check": "private_data_output",
                    "blocked": False,
                }
            ]
        },
        "routing": {
            "policy_provenance": {
                "policy_version": "cs003-v1",
                "classification_confidence_threshold": 0.8,
                "frozen": True,
                "policy_file_sha256": "a" * 64,
                "threshold_sha256": "b" * 64,
                "classifier_implementation_sha256": "c" * 64,
                "classifier_model_sha256": "d" * 64,
            }
        },
    }


def _decision(result: dict[str, object]) -> dict[str, object]:
    """Return the corresponding minimal decision event."""
    return {
        "input_index": result["input_index"],
        "ticket_id": result["ticket_id"],
        "action_taken": result["route"],
    }


def test_evaluate_completed_run_reports_reproducible_metrics_and_segments(
    tmp_path: Path,
) -> None:
    record = {
        "ticket_id": "EVAL-1",
        "language_fluency": "non_fluent",
        "customer_tier": "business",
        "labels": {
            "intent": "billing",
            "urgency": "medium",
            "expected_route": "auto_respond",
            "answerable_from_docs": True,
            "must_not_auto_respond": False,
            "expected_doc_ids": ["DOC-1"],
        },
        "history": {
            "first_contact_resolution": True,
            "escalated": False,
            "repeat_contact": False,
            "csat_rating": 4,
            "resolution_time_minutes": 40,
        },
    }
    input_path = tmp_path / "development_evaluation.json"
    input_path.write_text(json.dumps([record]), encoding="utf-8")
    result = _result(input_index=0, ticket_id="EVAL-1")

    report = evaluate_completed_run(
        run_id="a" * 32,
        results=[result],
        input_records=[record],
        input_path=input_path,
        source_ref="data/tickets/development_tickets.json",
        decision_events=[
            {
                "input_index": 0,
                "ticket_id": "EVAL-1",
                "terminal": False,
                "event_type": "classification",
            },
            {**_decision(result), "terminal": True},
        ],
    )

    assert report["source_set"] == "development"
    assert report["integrity"]["status"] == "complete"
    assert report["input_sha256"]
    assert report["policy_provenance"]["policy"]["policy_version"] == "cs003-v1"
    assert report["policy_provenance"]["classifier_model_names"] == ["gemini-decision-1"]
    assert report["policy_provenance"]["fingerprint_status"] == "available"
    assert report["policy_provenance"]["fingerprints"]["policy_file_sha256"] == "a" * 64
    observed = report["run_observed"]
    technical = observed["technical"]
    assert technical["classification"]["accuracy"]["value"] == 1
    assert technical["classification"]["per_class"]["billing"]["precision"]["denominator"] == 1
    assert technical["classification"]["precision_target"] == 0.85
    assert technical["classification"]["precision_target_status"] == "pass"
    assert technical["classification"]["per_class"]["billing"]["precision_target_status"] == "met"
    assert technical["classification"]["confusion_matrix"]["billing"]["billing"] == 1
    assert technical["urgency"]["precision_target"] == 0.70
    assert technical["urgency"]["target_status"] == "pass"
    assert technical["urgency"]["per_class"]["medium"]["precision_target_status"] == "met"
    assert technical["urgency"]["confusion_matrix"]["medium"]["medium"] == 1
    assert technical["route_accuracy"]["value"] == 1
    assert technical["retrieval_hit_rate"]["value"] == 1
    assert "claim_grounding" not in technical
    assert technical["citation_resolution"]["denominator"] == 1
    citation_integrity = technical["citation_source_integrity"]
    assert citation_integrity["generated_draft_count"] == 1
    assert citation_integrity["claim_count"] == 1
    assert citation_integrity["citation_resolution"]["value"] == 1
    assert citation_integrity["eligible_source_provenance"]["value"] == 1
    assert citation_integrity["claim_excerpt_match"]["value"] == 1
    assert citation_integrity["claim_traceability"]["value"] == 1
    assert citation_integrity["expected_document_coverage"]["value"] == 1
    assert citation_integrity["semantic_quality"]["hallucination"]["status"] == "not_measured"
    assert "model-estimated" in citation_integrity["semantic_quality"]["human_rated_citation_accuracy"]["reason"]
    assert technical["coverage"]["denominator"] == 1
    assert technical["decision_log_reconciliation"]["value"] == 1
    assert technical["latency"]["p95_seconds"]["status"] == "unavailable"
    assert technical["generation_inclusive_processing_duration"]["p95_seconds"]["status"] == "unavailable"
    assert technical["draft_generation_request_duration"]["p95_seconds"]["status"] == "unavailable"
    assert observed["governance"]["logged_decisions"]["value"] == 1
    assert report["historical_baseline"]["causal_claim_about_this_run"] is False
    assert report["historical_baseline"]["overall"]["csat_mean"]["value"] == 4
    assert report["historical_baseline"]["segments"]["language_fluency"]["non_fluent"]["sample_size"] == 1
    assert report["historical_baseline"]["segments"]["customer_tier"]["business"]["metrics"]["first_contact_resolution"]["value"] == 1


def test_latency_distributions_use_scored_processing_and_nearest_rank_p95(
    tmp_path: Path,
) -> None:
    records = []
    results = []
    for index in range(20):
        ticket_id = f"LATENCY-{index:02d}"
        records.append(
            {
                "ticket_id": ticket_id,
                "labels": {
                    "intent": "billing",
                    "urgency": "medium",
                    "expected_route": "auto_respond",
                    "answerable_from_docs": True,
                    "must_not_auto_respond": False,
                    "expected_doc_ids": ["DOC-1"],
                },
            }
        )
        result = _result(input_index=index, ticket_id=ticket_id)
        result["processing_latency_seconds"] = float(index + 1)
        result["draft_generation_request_duration_seconds"] = 100.0
        result["processing_duration_seconds"] = float(index + 101)
        results.append(result)
    input_path = tmp_path / "latency_evaluation.json"
    input_path.write_text(json.dumps(records), encoding="utf-8")

    report = evaluate_completed_run(
        run_id="a" * 32,
        results=results,
        input_records=records,
        input_path=input_path,
        source_ref="data/tickets/development_tickets.json",
        decision_events=[_decision(result) for result in results],
    )

    technical = report["run_observed"]["technical"]
    assert technical["latency"]["median_seconds"]["value"] == 10.5
    assert technical["latency"]["p95_seconds"]["value"] == 19.0
    assert technical["generation_inclusive_processing_duration"]["p95_seconds"]["value"] == 119.0
    assert technical["draft_generation_request_duration"]["p95_seconds"]["value"] == 100.0


def test_classification_target_marks_zero_predictions_not_applicable_and_reports_urgency_errors(
    tmp_path: Path,
) -> None:
    records = [
        {"ticket_id": "PRIORITY-HIGH", "labels": {"intent": "account_access", "urgency": "high"}},
        {"ticket_id": "PRIORITY-LOW", "labels": {"intent": "billing_query", "urgency": "low"}},
    ]
    input_path = tmp_path / "development_priority_evaluation.json"
    input_path.write_text(json.dumps(records), encoding="utf-8")
    results = [
        _result(input_index=index, ticket_id=record["ticket_id"])
        for index, record in enumerate(records)
    ]

    report = evaluate_completed_run(
        run_id="a" * 32,
        results=results,
        input_records=records,
        input_path=input_path,
        source_ref="external-priority-evaluation.json",
        decision_events=[_decision(result) for result in results],
    )

    technical = report["run_observed"]["technical"]
    classification = technical["classification"]
    assert classification["precision_target_status"] == "below_target"
    assert classification["per_class"]["account_access"]["precision"]["status"] == "not_applicable"
    assert classification["per_class"]["account_access"]["precision_target_status"] == "not_applicable"
    assert classification["per_class"]["account_access"]["recall"]["value"] == 0
    urgency = technical["urgency"]
    assert urgency["precision_target"] == 0.70
    assert urgency["target_status"] == "below_target"
    assert urgency["confusion_matrix"]["high"]["medium"] == 1
    assert urgency["confusion_matrix"]["low"]["medium"] == 1
    assert urgency["prioritization_error_counts"] == {
        "over_prioritized": 1,
        "under_prioritized": 1,
    }
    assert [error["ticket_id"] for error in urgency["prioritization_errors"]] == [
        "PRIORITY-HIGH",
        "PRIORITY-LOW",
    ]


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("intent", "bogus_intent"),
        ("urgency", "urgent"),
        ("expected_route", "bad_route"),
    ],
)
def test_malformed_development_labels_fail_reconciliation_and_block_precision_pass(
    tmp_path: Path,
    field: str,
    invalid_value: str,
) -> None:
    labels = {
        "intent": "billing",
        "urgency": "high",
        "expected_route": "escalate",
        "expected_doc_ids": [],
        "answerable_from_docs": False,
        "must_not_auto_respond": True,
    }
    labels[field] = invalid_value
    record = {
        "ticket_id": "MALFORMED-LABEL",
        "language_fluency": "fluent",
        "customer_tier": "standard",
        "labels": labels,
    }
    input_path = tmp_path / "malformed-development-label.json"
    input_path.write_text(json.dumps([record]), encoding="utf-8")
    result = _result(input_index=0, ticket_id="MALFORMED-LABEL")
    result["classification"][field] = invalid_value

    report = evaluate_completed_run(
        run_id="a" * 32,
        results=[result],
        input_records=[record],
        input_path=input_path,
        source_ref="data/tickets/development_tickets.json",
        decision_events=[
            {
                "input_index": 0,
                "ticket_id": "EVAL-1",
                "terminal": False,
                "event_type": "classification",
            },
            {**_decision(result), "terminal": True},
        ],
    )

    assert report["integrity"]["status"] == "mismatch"
    assert any(
        f"Evaluation label '{field}' has an invalid value" in error
        for error in report["integrity"]["errors"]
    )
    classification = report["run_observed"]["technical"]["classification"]
    assert classification["precision_target_status"] == "invalid_evaluation_data"
    assert report["run_observed"]["technical"]["urgency"]["target_status"] == "invalid_evaluation_data"
    assert "bogus_intent" not in classification["per_class"]
    assert report["criteria"]["confidence_calibration"]["status"] == "not_measurable"
    group_quality = report["criteria"]["group_quality_variation"]
    assert group_quality["status"] == "not_measurable"
    metric_name = {
        "intent": "intent_accuracy",
        "urgency": "urgency_accuracy",
        "expected_route": "route_accuracy",
    }[field]
    invalid_label_segment = group_quality["dimensions"]["language_fluency"]["measures"][metric_name]["segments"][0]
    assert invalid_label_segment["denominator"] == 0
    assert invalid_label_segment["missing_count"] == 1
    summary = build_terminal_summary(
        input_records=[record],
        results=[result],
        metrics={
            "total_count": 1,
            "completed_count": 1,
            "exception_count": 0,
            "route_counts": {"draft_for_review": 0, "escalate": 1, "degrade": 0},
            "evaluation": report,
        },
    )
    assert summary["classification_accuracy"]["status"] == "invalid_evaluation_data"


def test_invalid_predictions_are_counted_as_unavailable_not_as_new_classes(
    tmp_path: Path,
) -> None:
    record = {
        "ticket_id": "INVALID-PREDICTION",
        "labels": {"intent": "billing", "urgency": "high"},
    }
    input_path = tmp_path / "prediction-evaluation.json"
    input_path.write_text(json.dumps([record]), encoding="utf-8")
    result = _result(input_index=0, ticket_id="INVALID-PREDICTION")
    result["classification"]["intent"] = "bogus_intent"
    result["classification"]["urgency"] = "critical"

    report = evaluate_completed_run(
        run_id="a" * 32,
        results=[result],
        input_records=[record],
        input_path=input_path,
        source_ref=str(input_path),
        decision_events=[_decision(result)],
    )

    technical = report["run_observed"]["technical"]
    classification = technical["classification"]
    assert report["integrity"]["status"] == "complete"
    assert classification["missing_prediction_count"] == 1
    assert classification["invalid_prediction_count"] == 1
    assert classification["per_class"]["billing"]["predicted_count"] == 0
    assert classification["per_class"]["billing"]["missing_prediction_count"] == 1
    assert "bogus_intent" not in classification["per_class"]
    assert classification["confusion_matrix"]["billing"]["__unavailable_prediction__"] == 1

    urgency = technical["urgency"]
    assert urgency["missing_prediction_count"] == 1
    assert urgency["invalid_prediction_count"] == 1
    assert "critical" not in urgency["per_class"]
    assert urgency["confusion_matrix"]["high"]["__unavailable_prediction__"] == 1
    assert urgency["prioritization_error_counts"] == {"different_or_unavailable": 1}
    assert urgency["prioritization_errors"][0]["predicted"] is None


def test_historical_baseline_segments_include_sample_missingness_and_scope(
    tmp_path: Path,
) -> None:
    records = [
        {
            "ticket_id": "HISTORY-1",
            "language_fluency": "non_fluent",
            "customer_tier": "business",
            "history": {
                "first_contact_resolution": True,
                "escalated": False,
                "repeat_contact": False,
                "csat_rating": 4,
                "resolution_time_minutes": 40,
            },
        },
        {
            "ticket_id": "HISTORY-2",
            "language_fluency": "non_fluent",
            "customer_tier": "business",
            "history": None,
        },
        {
            "ticket_id": "HISTORY-3",
            "language_fluency": "fluent",
            "customer_tier": "standard",
            "history": {
                "first_contact_resolution": False,
                "escalated": True,
                "repeat_contact": True,
                "csat_rating": 2,
                "resolution_time_minutes": 80,
            },
        },
    ]
    input_path = tmp_path / "segmented-history.json"
    input_path.write_text(json.dumps(records), encoding="utf-8")
    results = [
        _result(input_index=index, ticket_id=record["ticket_id"])
        for index, record in enumerate(records)
    ]

    report = evaluate_completed_run(
        run_id="a" * 32,
        results=results,
        input_records=records,
        input_path=input_path,
        source_ref=str(input_path),
        decision_events=[_decision(result) for result in results],
    )

    baseline = report["historical_baseline"]
    non_fluent_business = baseline["segments"]["language_fluency"]["non_fluent"]
    assert non_fluent_business["sample_size"] == 2
    assert non_fluent_business["metrics"]["first_contact_resolution"] == {
        "value": 1,
        "status": "available",
        "numerator": 1,
        "denominator": 1,
        "sample_size": 1,
        "missing_count": 1,
        "note": None,
    }
    assert non_fluent_business["metrics"]["escalation_rate"]["value"] == 0
    assert non_fluent_business["metrics"]["repeat_contact_rate"]["value"] == 0
    assert non_fluent_business["metrics"]["csat_mean"]["value"] == 4
    assert non_fluent_business["metrics"]["resolution_time_minutes_mean"]["value"] == 40
    assert non_fluent_business["metrics"]["resolution_time_minutes_median"]["value"] == 40
    first_response = non_fluent_business["metrics"]["first_response_time_minutes"]
    assert first_response["status"] == "unavailable"
    assert first_response["value"] is None
    assert first_response["denominator"] == 0
    assert first_response["sample_size"] == first_response["missing_count"] == 2
    business_tier = baseline["segments"]["customer_tier"]["business"]
    assert business_tier["sample_size"] == 2
    assert business_tier["metrics"]["first_contact_resolution"]["missing_count"] == 1
    assert business_tier["metrics"]["csat_mean"]["missing_count"] == 1

    fluent_standard = baseline["segments"]["customer_tier"]["standard"]
    assert fluent_standard["sample_size"] == 1
    assert fluent_standard["metrics"]["first_contact_resolution"]["value"] == 0
    assert fluent_standard["metrics"]["escalation_rate"]["value"] == 1
    assert fluent_standard["metrics"]["repeat_contact_rate"]["value"] == 1
    assert fluent_standard["metrics"]["csat_mean"]["value"] == 2
    assert fluent_standard["metrics"]["resolution_time_minutes_mean"]["value"] == 80
    assert baseline["source"] == "input.history"
    assert baseline["sample_size"] == 2
    assert baseline["input_sample_size"] == 3
    assert baseline["causal_claim_about_this_run"] is False
    Draft202012Validator(EVALUATION_REPORT_SCHEMA).validate(report)


def test_evaluator_reports_duplicate_ids_and_unavailable_labels(tmp_path: Path) -> None:
    records = [
        {"ticket_id": "DUPLICATE"},
        {"ticket_id": "DUPLICATE"},
    ]
    input_path = tmp_path / "unlabeled.json"
    input_path.write_text(json.dumps(records), encoding="utf-8")
    results = [
        _result(input_index=0, ticket_id="DUPLICATE"),
        _result(input_index=1, ticket_id="DUPLICATE"),
    ]
    report = evaluate_completed_run(
        run_id="a" * 32,
        results=results,
        input_records=records,
        input_path=input_path,
        source_ref=str(input_path),
        decision_events=[_decision(item) for item in results],
    )

    assert report["integrity"]["status"] == "mismatch"
    assert report["integrity"]["duplicate_ticket_ids"] == ["DUPLICATE"]
    assert "Input records contain duplicate ticket IDs." in report["integrity"]["errors"]
    assert report["run_observed"]["technical"]["classification"]["status"] == "unavailable"
    classification_accuracy = report["run_observed"]["technical"]["classification"]["accuracy"]
    assert classification_accuracy["status"] == "unavailable"
    assert classification_accuracy["missing_count"] == 2
    route_accuracy = report["run_observed"]["technical"]["route_accuracy"]
    assert route_accuracy["status"] == "unavailable"
    assert route_accuracy["missing_count"] == 2
    retrieval_hit_rate = report["run_observed"]["technical"]["retrieval_hit_rate"]
    assert retrieval_hit_rate["status"] == "unavailable"
    assert retrieval_hit_rate["missing_count"] == 2
    assert report["historical_baseline"]["overall"]["first_contact_resolution"]["status"] == "unavailable"


def test_verified_zero_expected_documents_is_not_applicable(tmp_path: Path) -> None:
    record = {
        "ticket_id": "NO-EXPECTED-DOCS",
        "labels": {
            "intent": "billing",
            "expected_route": "escalate",
            "answerable_from_docs": False,
            "expected_doc_ids": [],
        },
    }
    input_path = tmp_path / "verified-no-docs.json"
    input_path.write_text(json.dumps([record]), encoding="utf-8")
    result = _result(input_index=0, ticket_id="NO-EXPECTED-DOCS")

    report = evaluate_completed_run(
        run_id="a" * 32,
        results=[result],
        input_records=[record],
        input_path=input_path,
        source_ref=str(input_path),
        decision_events=[_decision(result)],
    )
    retrieval_hit_rate = report["run_observed"]["technical"]["retrieval_hit_rate"]

    assert retrieval_hit_rate["status"] == "not_applicable"
    assert retrieval_hit_rate["denominator"] == 0
    assert retrieval_hit_rate["missing_count"] == 0


def test_route_tradeoffs_answerability_and_protected_safety_are_separate(
    tmp_path: Path,
) -> None:
    records = [
        {
            "ticket_id": "FALSE-ESCALATION",
            "labels": {
                "expected_route": "auto_respond",
                "answerable_from_docs": True,
                "must_not_auto_respond": False,
                "expected_doc_ids": ["DOC-1"],
            },
        },
        {
            "ticket_id": "PROTECTED-ROUTE-DEV",
            "labels": {
                "expected_route": "escalate",
                "answerable_from_docs": False,
                "must_not_auto_respond": True,
                "expected_doc_ids": [],
            },
        },
        {
            "ticket_id": "EVIDENCE-READY",
            "labels": {
                "expected_route": "auto_respond",
                "answerable_from_docs": True,
                "must_not_auto_respond": False,
                "expected_doc_ids": ["DOC-1"],
            },
        },
        {
            "ticket_id": "MISSING-READINESS",
            "labels": {
                "expected_route": "degrade",
                "answerable_from_docs": False,
                "must_not_auto_respond": False,
                "expected_doc_ids": [],
            },
        },
    ]
    input_path = tmp_path / "external-route-evaluation.json"
    input_path.write_text(json.dumps(records), encoding="utf-8")
    results = [
        _result(input_index=0, ticket_id="FALSE-ESCALATION"),
        _result(input_index=1, ticket_id="PROTECTED-ROUTE-DEV"),
        _result(input_index=2, ticket_id="EVIDENCE-READY"),
        _result(input_index=3, ticket_id="MISSING-READINESS"),
    ]
    results[0].update({"route": "escalate", "sources": [], "routing": {"evidence_ready": False, "retrieval": {"status": "unavailable"}}})
    results[1].update({"route": "draft_for_review", "draft": {"send_status": "not_sent"}, "routing": {"evidence_ready": True, "retrieval": {"status": "available"}}})
    results[2].update({"route": "draft_for_review", "routing": {"evidence_ready": True, "retrieval": {"status": "available"}}})
    results[3].update({"route": "invalid-route", "routing": {"evidence_ready": None}})

    report = evaluate_completed_run(
        run_id="a" * 32,
        results=results,
        input_records=records,
        input_path=input_path,
        source_ref=str(input_path),
        decision_events=[_decision(result) for result in results],
    )
    technical = report["run_observed"]["technical"]
    route_quality = technical["route_quality"]
    answerability = technical["answerability"]
    protected = report["run_observed"]["governance"]["protected_case_safety"]

    assert route_quality["confusion_matrix"]["draft_for_review"]["escalate"] == 1
    assert route_quality["confusion_matrix"]["escalate"]["draft_for_review"] == 1
    assert route_quality["confusion_matrix"]["degrade"]["__unavailable_prediction__"] == 1
    assert route_quality["false_escalation"] == {"count": 1, "denominator": 2, "rate": 0.5}
    assert route_quality["unsafe_automation_route"] == {"count": 1, "denominator": 1, "rate": 1.0}
    assert "target" not in route_quality
    assert answerability["accuracy"]["numerator"] == 1
    assert answerability["accuracy"]["denominator"] == 4
    assert answerability["accuracy"]["missing_count"] == 1
    assert answerability["missing_evidence_readiness_count"] == 1
    assert answerability["missing_count"] == 1
    assert answerability["confusion_matrix"]["true"]["false"] == 1
    assert answerability["confusion_matrix"]["false"]["__unavailable_prediction__"] == 1
    assert technical["retrieval_hit_rate"]["denominator"] == 2
    assert protected["zero_tolerance_status"] == "pass"
    assert protected["protected_automatic_send_violation_count"] == 0
    assert protected["protected_route_deviation_count"] == 1
    assert protected["all_ticket_automatic_send_check_denominator"] == 4


def test_empty_run_marks_classification_not_applicable(tmp_path: Path) -> None:
    input_path = tmp_path / "empty-evaluation.json"
    input_path.write_text("[]", encoding="utf-8")

    report = evaluate_completed_run(
        run_id="a" * 32,
        results=[],
        input_records=[],
        input_path=input_path,
        source_ref=str(input_path),
        decision_events=[],
    )
    classification = report["run_observed"]["technical"]["classification"]

    assert classification["status"] == "not_applicable"
    assert classification["sample_size"] == 0
    assert classification["accuracy"]["status"] == "not_applicable"
    assert classification["accuracy"]["denominator"] == 0
    assert classification["accuracy"]["missing_count"] == 0


def test_missing_private_output_guardrail_is_unavailable(tmp_path: Path) -> None:
    record = {"ticket_id": "MISSING-GUARDRAIL"}
    input_path = tmp_path / "missing-guardrail.json"
    input_path.write_text(json.dumps([record]), encoding="utf-8")
    result = _result(input_index=0, ticket_id="MISSING-GUARDRAIL")
    result["guardrails"] = {"checks": []}

    report = evaluate_completed_run(
        run_id="a" * 32,
        results=[result],
        input_records=[record],
        input_path=input_path,
        source_ref=str(input_path),
        decision_events=[_decision(result)],
    )
    private_output_guardrail = report["run_observed"]["governance"][
        "private_output_guardrail"
    ]

    assert private_output_guardrail["status"] == "unavailable"
    assert private_output_guardrail["denominator"] == 0
    assert private_output_guardrail["sample_size"] == 1
    assert private_output_guardrail["missing_count"] == 1


def test_per_class_metrics_count_partial_labels_and_preserve_verified_zero(
    tmp_path: Path,
) -> None:
    records = [
        {
            "ticket_id": "LABELED",
            "labels": {"intent": "billing"},
        },
        {"ticket_id": "UNLABELED"},
    ]
    input_path = tmp_path / "partial-labels.json"
    input_path.write_text(json.dumps(records), encoding="utf-8")
    results = [
        _result(input_index=0, ticket_id="LABELED"),
        _result(input_index=1, ticket_id="UNLABELED"),
    ]

    partial_report = evaluate_completed_run(
        run_id="a" * 32,
        results=results,
        input_records=records,
        input_path=input_path,
        source_ref=str(input_path),
        decision_events=[_decision(item) for item in results],
    )
    partial_precision = partial_report["run_observed"]["technical"]["classification"]["per_class"]["billing"]["precision"]
    partial_recall = partial_report["run_observed"]["technical"]["classification"]["per_class"]["billing"]["recall"]
    assert partial_precision["missing_count"] == 1
    assert partial_recall["missing_count"] == 1

    no_prediction = _result(input_index=0, ticket_id="LABELED")
    no_prediction["classification"]["intent"] = "api"
    single_input_path = tmp_path / "single-label.json"
    single_input_path.write_text(json.dumps([records[0]]), encoding="utf-8")
    zero_report = evaluate_completed_run(
        run_id="a" * 32,
        results=[no_prediction],
        input_records=[records[0]],
        input_path=single_input_path,
        source_ref=str(single_input_path),
        decision_events=[_decision(no_prediction)],
    )
    zero_precision = zero_report["run_observed"]["technical"]["classification"]["per_class"]["billing"]["precision"]
    assert zero_precision["denominator"] == 0
    assert zero_precision["missing_count"] == 0
    assert zero_precision["status"] == "not_applicable"
