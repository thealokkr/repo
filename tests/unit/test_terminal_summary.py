from cloudserve_support.reporting.terminal_summary import (
    build_terminal_summary,
    format_terminal_summary,
)


def test_terminal_summary_reports_prediction_metrics_and_post_run_label_matches():
    inputs = [
        {
            "labels": {
                "intent": "billing_query",
                "urgency": "high",
                "expected_route": "auto_respond",
                "answerable_from_docs": True,
                "must_not_auto_respond": False,
            }
        },
        {
            "labels": {
                "intent": "api_key_issue",
                "urgency": "medium",
                "expected_route": "escalate",
                "answerable_from_docs": False,
                "must_not_auto_respond": True,
            }
        },
        {
            "labels": {
                "intent": "account_access",
                "urgency": "low",
                "expected_route": "escalate",
                "answerable_from_docs": True,
                "must_not_auto_respond": True,
            }
        },
    ]
    results = [
        {
            "input_index": 0,
            "ticket_id": "T-1",
            "classification": {"intent": "billing_query", "urgency": "high"},
            "route": "draft_for_review",
            "sources": [{"doc_id": "DOC-1"}],
            "draft": {"send_status": "not_sent"},
        },
        {
            "input_index": 1,
            "ticket_id": "T-2",
            "classification": {"intent": "billing_query", "urgency": "low"},
            "route": "escalate",
            "sources": [],
            "draft": {"send_status": "not_sent"},
        },
        {
            "input_index": 2,
            "ticket_id": "T-3",
            "classification": {"intent": "account_access", "urgency": "low"},
            "route": "draft_for_review",
            "sources": [{"doc_id": "DOC-2"}],
            "draft": {"send_status": "sent"},
        },
    ]
    metrics = {
        "completed_count": 3,
        "exception_count": 0,
        "route_counts": {"draft_for_review": 2, "escalate": 1, "degrade": 0},
    }

    summary = build_terminal_summary(
        input_records=inputs,
        results=results,
        metrics=metrics,
    )

    assert summary["processed_count"] == 3
    assert summary["exception_count"] == 0
    assert summary["predicted_counts"]["intent"] == {
        "account_access": 1,
        "billing_query": 2,
    }
    assert summary["intent_quality"]["billing_query"] == {
        "support": 1,
        "predicted_count": 2,
        "correct_count": 1,
        "label_match_count": 1,
        "precision": 0.5,
        "recall": 1.0,
        "precision_target_status": "below_target",
    }
    assert summary["intent_quality"]["api_key_issue"]["precision"] is None
    assert summary["intent_precision_target"] == {
        "threshold": 0.85,
        "status": "below_target",
        "supported_class_count": 3,
        "meeting_class_count": 1,
        "note": "Development-set criterion; unsupported or unpredicted classes are not applicable and do not count as meeting the target.",
    }
    assert summary["urgency_quality"]["target"] is None
    assert summary["urgency_quality"]["target_status"] == "measured_no_numeric_target"
    assert summary["urgency_quality"]["exact_match_count"] == 2
    assert summary["urgency_quality"]["sample_count"] == 3
    assert summary["label_match_counts"]["urgency"] == {
        "matched": 2,
        "mismatched": 1,
        "denominator": 3,
        "label_counts": {"high": 1, "low": 1, "medium": 1},
    }
    assert summary["answerability"]["evidence_availability_proxy_matches"] == 3
    assert summary["protected_cases"] == {
        "label_count": 2,
        "escalated_count": 1,
        "route_deviation_count": 1,
        "automatic_send_count": 1,
    }
    assert "Development evaluation summary:" in format_terminal_summary(summary)


def test_terminal_summary_includes_unlabeled_predictions_and_undefined_label_metrics():
    summary = build_terminal_summary(
        input_records=[{}],
        results=[
            {
                "input_index": 0,
                "classification": {"intent": "unknown", "urgency": "unknown"},
                "route": "degrade",
                "sources": [],
            }
        ],
        metrics={
            "completed_count": 0,
            "exception_count": 1,
            "route_counts": {"draft_for_review": 0, "escalate": 0, "degrade": 1},
        },
    )

    assert summary["exception_count"] == 1
    assert summary["predicted_counts"]["intent"] == {"unknown": 1}
    assert summary["intent_quality"] == {"unknown": {
        "support": 0,
        "predicted_count": 1,
        "correct_count": 0,
        "label_match_count": 0,
        "precision": None,
        "recall": None,
        "precision_target_status": "not_applicable",
    }}
    assert summary["undefined_metrics"] == [
        {"intent": "unknown", "metric": "precision"},
        {"intent": "unknown", "metric": "recall"},
    ]
    assert summary["intent_precision_target"]["status"] == "not_applicable"


def test_terminal_summary_uses_canonical_route_answerability_and_safety_metrics():
    route_quality = {"route_deviation_count": 4, "false_escalation": {"count": 3}}
    answerability = {"accuracy": {"value": 0.75}, "missing_count": 2}
    protected_cases = {
        "protected_case_count": 6,
        "protected_route_deviation_count": 1,
        "protected_automatic_send_violation_count": 0,
        "zero_tolerance_status": "pass",
    }
    metrics = {
        "completed_count": 1,
        "exception_count": 0,
        "route_counts": {"draft_for_review": 0, "escalate": 1, "degrade": 0},
        "batch_export_duration_seconds": 2.5,
        "evaluation": {
            "run_observed": {
                "technical": {
                    "route_quality": route_quality,
                    "answerability": answerability,
                    "latency": {"p95_seconds": {"value": 2.0}},
                    "generation_inclusive_processing_duration": {
                        "p95_seconds": {"value": 4.0}
                    },
                    "draft_generation_request_duration": {
                        "p95_seconds": {"value": 2.0}
                    },
                },
                "governance": {"protected_case_safety": protected_cases},
            },
            "criteria": {"runtime_latency": {"status": "pass", "value": 2.0}},
        },
    }

    summary = build_terminal_summary(
        input_records=[{"labels": {"answerable_from_docs": True, "must_not_auto_respond": True}}],
        results=[{"input_index": 0, "ticket_id": "SUMMARY-1", "route": "escalate"}],
        metrics=metrics,
    )

    assert summary["route_quality"] == route_quality
    assert summary["answerability"] == answerability
    assert summary["protected_cases"] == protected_cases
    assert summary["latency"] == {
        "processing_latency_seconds": {"p95_seconds": {"value": 2.0}},
        "generation_inclusive_processing_duration": {"p95_seconds": {"value": 4.0}},
        "draft_generation_request_duration": {"p95_seconds": {"value": 2.0}},
        "runtime_latency_criterion": {"status": "pass", "value": 2.0},
        "batch_export_duration_seconds": 2.5,
    }
