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
    assert summary["classification_accuracy"] == {
        "correct_count": 2,
        "sample_count": 3,
        "percentage": 66.7,
        "target_percentage": 85.0,
        "status": "below_target",
    }
    assert "intent_quality" not in summary
    assert summary["urgency_accuracy"] == {
        "correct_count": 2,
        "sample_count": 3,
        "percentage": 66.7,
    }
    assert "per_class" not in summary
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
    assert summary["classification_accuracy"] == {
        "correct_count": 0,
        "sample_count": 0,
        "percentage": None,
        "target_percentage": 85.0,
        "status": "not_applicable",
    }
    assert "intent_quality" not in summary


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
