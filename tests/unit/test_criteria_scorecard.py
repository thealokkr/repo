"""Check target statuses and denominators in the post-run criteria matrix."""

from __future__ import annotations

import pytest

from cloudserve_support.evaluation.criteria_scorecard import build_criteria_scorecard


def test_scorecard_surfaces_target_failures_and_scope_exclusions() -> None:
    input_records = []
    results = []
    events = []
    for index in range(12):
        group = "fluent" if index < 6 else "non_fluent"
        truth = "billing"
        prediction = "billing" if index < 11 else "api"
        ticket_id = f"T-{index}"
        input_records.append(
            {
                "ticket_id": ticket_id,
                "language_fluency": group,
                "customer_tier": "standard",
                "labels": {
                    "intent": truth,
                    "urgency": "medium",
                    "expected_route": "auto_respond",
                    "answerable_from_docs": True,
                },
            }
        )
        result = {
            "input_index": index,
            "ticket_id": ticket_id,
            "route": "draft_for_review",
            "processing_duration_seconds": 0.5,
            "generation": {"status": "generated"},
            "classification": {
                "intent": prediction,
                "urgency": "medium",
                "confidence": 0.8,
            },
            "routing": {"evidence_ready": True},
            "guardrails": {"checks": [{"check": "private_data_output", "blocked": False}]},
        }
        results.append(result)
        events.append(
            {"input_index": index, "ticket_id": ticket_id, "action_taken": "draft_for_review"}
        )

    technical = {
        "latency": {
            "p95_seconds": {
                "value": 0.5,
                "status": "available",
                "numerator": None,
                "denominator": 12,
                "sample_size": 12,
                "missing_count": 0,
            }
        },
        "decision_log_reconciliation": {"numerator": 12},
        "classification": {
            "confidence_calibration": [
                {
                    "lower_bound": 0.8,
                    "upper_bound": 1.0,
                    "mean_confidence": 0.8,
                    "observed_accuracy": 0.9,
                    "sample_size": 12,
                }
            ]
        },
    }
    scorecard = build_criteria_scorecard(
        results=results,
        input_records=input_records,
        decision_events=events,
        technical=technical,
        integrity_status="complete",
    )

    assert scorecard["runtime_latency"]["status"] == "pass"
    assert scorecard["private_data_output"]["status"] == "pass"
    assert scorecard["decision_audit_coverage"]["status"] == "pass"
    assert scorecard["confidence_calibration"]["status"] == "fail"
    assert scorecard["confidence_calibration"]["value"] == pytest.approx(0.1)
    assert scorecard["group_quality_variation"]["status"] == "fail"
    intent_variation = scorecard["group_quality_variation"]["dimensions"][
        "language_fluency"
    ]["measures"]["intent_accuracy"]
    assert intent_variation["value"] == pytest.approx(1 / 6)
    assert [row["denominator"] for row in intent_variation["segments"]] == [6, 6]
    assert scorecard["availability"]["status"] == "non_gating"
    assert "PRD v2.3" in scorecard["availability"]["target"]
    assert scorecard["business_outcomes"]["status"] == "excluded"


def test_scorecard_does_not_pass_with_missing_audit_or_privacy_checks() -> None:
    result = {
        "input_index": 0,
        "ticket_id": "T-1",
        "route": "draft_for_review",
        "generation": {"status": "generated"},
        "classification": {"intent": "billing", "confidence": 0.8},
        "routing": {"evidence_ready": True},
    }
    input_record = {
        "ticket_id": "T-1",
        "language_fluency": chr(0xD800),
        "labels": {},
    }
    scorecard = build_criteria_scorecard(
        results=[result],
        input_records=[input_record, {"ticket_id": "T-2", "labels": {}}],
        decision_events=[],
        technical={
            "latency": {
                "p95_seconds": {
                    "value": 0.2,
                    "status": "available",
                    "numerator": None,
                    "denominator": 1,
                    "sample_size": 1,
                    "missing_count": 0,
                }
            },
            "decision_log_reconciliation": {"numerator": 0},
            "classification": {
                "confidence_calibration": [
                    {
                        "lower_bound": 0.8,
                        "upper_bound": 1.0,
                        "mean_confidence": 0.8,
                        "observed_accuracy": 0.8,
                        "sample_size": 1,
                    }
                ]
            },
        },
        integrity_status="mismatch",
    )

    assert scorecard["private_data_output"]["status"] == "not_measurable"
    assert scorecard["private_data_output"]["missing_count"] == 1
    assert scorecard["decision_audit_coverage"]["status"] == "fail"
    assert scorecard["decision_audit_coverage"]["value"] == 0
    assert scorecard["confidence_calibration"]["status"] == "not_measurable"
    assert scorecard["confidence_calibration"]["missing_count"] == 1
    assert scorecard["group_quality_variation"]["dimensions"]["language_fluency"][
        "missing_segment_count"
    ] == 2
