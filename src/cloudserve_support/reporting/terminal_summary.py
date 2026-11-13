"""Build a concise, post-run terminal summary for labeled ticket batches."""

from __future__ import annotations

from collections import Counter
from typing import Any

from cloudserve_support.classification.ticket_classification import INTENT_VALUES, URGENCY_VALUES
from cloudserve_support.evaluation.run_metrics import (
    INTENT_ACCURACY_TARGET,
)

ROUTE_ALIASES = {"auto_respond": "draft_for_review", "escalate": "escalate"}
ROUTE_VALUES = frozenset({"draft_for_review", "escalate", "degrade"})
EXPECTED_ROUTE_LABEL_VALUES = frozenset(
    {"auto_respond", "draft_for_review", "escalate", "degrade"}
)


def build_terminal_summary(
    *,
    input_records: list[Any],
    results: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """Summarize predictions and post-run labels without changing decisions."""
    result_by_index = {
        result.get("input_index"): result
        for result in results
        if isinstance(result.get("input_index"), int)
        and not isinstance(result.get("input_index"), bool)
    }
    intent_correct_count = 0
    intent_sample_count = 0
    urgency_correct_count = 0
    urgency_sample_count = 0
    route_predicted: Counter[str] = Counter()
    label_counts: dict[str, Counter[str]] = {"route": Counter()}
    label_matches: Counter[str] = Counter()
    label_totals: Counter[str] = Counter()
    answerability_labels: Counter[str] = Counter()
    answerability_proxy: Counter[str] = Counter()
    answerability_agreements = 0
    answerability_compared = 0
    protected_count = 0
    protected_escalated = 0
    protected_route_deviations = 0
    protected_auto_sends = 0

    for input_index, record in enumerate(input_records):
        result = result_by_index.get(input_index)
        if not isinstance(record, dict) or result is None:
            continue
        classification = result.get("classification")
        classification = classification if isinstance(classification, dict) else {}
        if "predicted_intent" in result:
            raw_intent = result.get("predicted_intent")
            predicted_intent: str | None = (
                raw_intent
                if isinstance(raw_intent, str) and raw_intent in INTENT_VALUES
                else None
            )
        else:
            predicted_intent = _value(classification.get("intent"), "unknown")
            if predicted_intent not in INTENT_VALUES:
                predicted_intent = "unknown"

        if "predicted_urgency" in result:
            raw_urgency = result.get("predicted_urgency")
            predicted_urgency: str | None = (
                raw_urgency
                if isinstance(raw_urgency, str) and raw_urgency in URGENCY_VALUES
                else None
            )
        else:
            predicted_urgency = _value(classification.get("urgency"), "unknown")
            if predicted_urgency not in URGENCY_VALUES:
                predicted_urgency = "unknown"

        if "predicted_route" in result:
            raw_route = result.get("predicted_route")
            predicted_route: str | None = (
                raw_route
                if isinstance(raw_route, str) and raw_route in ROUTE_VALUES
                else None
            )
        else:
            predicted_route = _value(result.get("route"), "unknown")
        if predicted_route is not None:
            route_predicted[predicted_route] += 1

        labels = record.get("labels")
        if not isinstance(labels, dict):
            continue
        expected_intent = labels.get("intent")
        expected_urgency = labels.get("urgency")
        expected_route_raw = labels.get("expected_route")
        expected_route = ROUTE_ALIASES.get(expected_route_raw, expected_route_raw)

        if (
            predicted_intent is not None
            and isinstance(expected_intent, str)
            and expected_intent in INTENT_VALUES
        ):
            intent_sample_count += 1
            if predicted_intent == expected_intent:
                intent_correct_count += 1
        if (
            predicted_urgency is not None
            and isinstance(expected_urgency, str)
            and expected_urgency in URGENCY_VALUES
        ):
            urgency_sample_count += 1
            if predicted_urgency == expected_urgency:
                urgency_correct_count += 1
        if (
            predicted_route is not None
            and isinstance(expected_route_raw, str)
            and expected_route_raw in EXPECTED_ROUTE_LABEL_VALUES
        ):
            label_counts["route"][expected_route] += 1
            label_totals["route"] += 1
            if predicted_route == expected_route:
                label_matches["route"] += 1

        answerable = labels.get("answerable_from_docs")
        if type(answerable) is bool:
            if "predicted_answerable_from_docs" in result:
                raw_answerable = result.get("predicted_answerable_from_docs")
                evidence_ready: bool | None = (
                    raw_answerable if type(raw_answerable) is bool else None
                )
            else:
                sources = result.get("sources")
                evidence_ready = isinstance(sources, list) and any(
                    isinstance(source, dict) and isinstance(source.get("doc_id"), str)
                    for source in sources
                )
            if evidence_ready is not None:
                answerability_labels[str(answerable).lower()] += 1
                answerability_proxy[str(evidence_ready).lower()] += 1
                answerability_compared += 1
                if answerable == evidence_ready:
                    answerability_agreements += 1

        if labels.get("must_not_auto_respond") is True:
            protected_count += 1
            if _automatic_send(result):
                protected_auto_sends += 1
            if predicted_route is not None:
                if predicted_route == "escalate":
                    protected_escalated += 1
                else:
                    protected_route_deviations += 1

    evaluation = metrics.get("evaluation")
    technical_metrics = (
        evaluation.get("run_observed", {}).get("technical", {})
        if isinstance(evaluation, dict)
        and isinstance(evaluation.get("run_observed"), dict)
        else {}
    )
    classification_scorecard = (
        technical_metrics.get("classification") if isinstance(technical_metrics, dict) else None
    )
    route_scorecard = technical_metrics.get("route_quality") if isinstance(technical_metrics, dict) else None
    answerability_scorecard = technical_metrics.get("answerability") if isinstance(technical_metrics, dict) else None
    governance_metrics = (
        evaluation.get("run_observed", {}).get("governance", {})
        if isinstance(evaluation, dict)
        and isinstance(evaluation.get("run_observed"), dict)
        else {}
    )
    protected_scorecard = (
        governance_metrics.get("protected_case_safety")
        if isinstance(governance_metrics, dict)
        else None
    )
    accuracy_metric = (
        classification_scorecard.get("accuracy")
        if isinstance(classification_scorecard, dict)
        and isinstance(classification_scorecard.get("accuracy"), dict)
        else {}
    )
    has_authoritative_intent_metric = (
        isinstance(accuracy_metric, dict) and "status" in accuracy_metric
    )
    accuracy_value = accuracy_metric.get("value")
    if not isinstance(accuracy_value, (int, float)) or isinstance(accuracy_value, bool):
        accuracy_value = (
            None
            if has_authoritative_intent_metric
            else (intent_correct_count / intent_sample_count if intent_sample_count else None)
        )
    reported_correct_count = accuracy_metric.get("numerator")
    if not isinstance(reported_correct_count, int) or isinstance(reported_correct_count, bool):
        reported_correct_count = 0 if has_authoritative_intent_metric else intent_correct_count
    reported_sample_count = accuracy_metric.get("denominator")
    if not isinstance(reported_sample_count, int) or isinstance(reported_sample_count, bool):
        reported_sample_count = 0 if has_authoritative_intent_metric else intent_sample_count
    classification_target_status = (
        "invalid_evaluation_data"
        if isinstance(classification_scorecard, dict)
        and classification_scorecard.get("precision_target_status") == "invalid_evaluation_data"
        else "not_applicable"
        if not reported_sample_count or accuracy_value is None
        else "met"
        if accuracy_value >= INTENT_ACCURACY_TARGET
        else "below_target"
    )
    urgency_scorecard = technical_metrics.get("urgency") if isinstance(technical_metrics, dict) else None
    urgency_match_metric = (
        urgency_scorecard.get("exact_match_rate")
        if isinstance(urgency_scorecard, dict)
        else None
    )
    has_authoritative_urgency_metric = (
        isinstance(urgency_match_metric, dict) and "status" in urgency_match_metric
    )
    urgency_accuracy_value = (
        urgency_match_metric.get("value")
        if isinstance(urgency_match_metric, dict)
        else None
    )
    if not isinstance(urgency_accuracy_value, (int, float)) or isinstance(urgency_accuracy_value, bool):
        urgency_accuracy_value = (
            None
            if has_authoritative_urgency_metric
            else (urgency_correct_count / urgency_sample_count if urgency_sample_count else None)
        )
    reported_urgency_correct_count = (
        urgency_match_metric.get("numerator")
        if isinstance(urgency_match_metric, dict)
        else None
    )
    if not isinstance(reported_urgency_correct_count, int) or isinstance(reported_urgency_correct_count, bool):
        reported_urgency_correct_count = (
            0 if has_authoritative_urgency_metric else urgency_correct_count
        )
    reported_urgency_sample_count = (
        urgency_match_metric.get("denominator")
        if isinstance(urgency_match_metric, dict)
        else None
    )
    if not isinstance(reported_urgency_sample_count, int) or isinstance(reported_urgency_sample_count, bool):
        reported_urgency_sample_count = (
            0 if has_authoritative_urgency_metric else urgency_sample_count
        )

    routes_from_metrics = metrics.get("route_counts")
    if isinstance(routes_from_metrics, dict):
        route_predicted = Counter({str(k): int(v) for k, v in routes_from_metrics.items()})

    route_accuracy_metric = (
        technical_metrics.get("route_accuracy")
        if isinstance(technical_metrics, dict)
        and isinstance(technical_metrics.get("route_accuracy"), dict)
        else None
    )
    reported_route_matches = (
        route_accuracy_metric.get("numerator")
        if isinstance(route_accuracy_metric, dict)
        else None
    )
    reported_route_total = (
        route_accuracy_metric.get("denominator")
        if isinstance(route_accuracy_metric, dict)
        else None
    )
    reported_route_expected_counts = (
        route_scorecard.get("expected_counts")
        if isinstance(route_scorecard, dict)
        and isinstance(route_scorecard.get("expected_counts"), dict)
        else None
    )
    has_authoritative_route_metric = (
        isinstance(route_accuracy_metric, dict)
        and "status" in route_accuracy_metric
        and isinstance(reported_route_total, int)
        and not isinstance(reported_route_total, bool)
        and reported_route_expected_counts is not None
    )
    if has_authoritative_route_metric:
        authoritative_route_matches = (
            reported_route_matches
            if isinstance(reported_route_matches, int)
            and not isinstance(reported_route_matches, bool)
            else 0
        )
        comparison = {
            "route": {
                "matched": authoritative_route_matches,
                "mismatched": reported_route_total - authoritative_route_matches,
                "denominator": reported_route_total,
                "label_counts": dict(sorted(reported_route_expected_counts.items())),
            }
        }
    else:
        comparison = {
            field: {
                "matched": label_matches[field],
                "mismatched": label_totals[field] - label_matches[field],
                "denominator": label_totals[field],
                "label_counts": dict(sorted(label_counts[field].items())),
            }
            for field in ("route",)
        }
    return {
        "input_count": metrics.get("total_count", len(input_records)),
        "processed_count": len(results),
        "completed_count": metrics.get("completed_count"),
        "failed_count": metrics.get("failed_count"),
        "exception_count": metrics.get("exception_count", 0),
        "result_count": len(results),
        "latency": _latency_summary(metrics),
        "predicted_counts": {
            "route": dict(sorted(route_predicted.items())),
        },
        "classification_accuracy": {
            "correct_count": reported_correct_count,
            "sample_count": reported_sample_count,
            "percentage": (
                round(float(accuracy_value) * 100, 1)
                if accuracy_value is not None
                else None
            ),
            "target_percentage": INTENT_ACCURACY_TARGET * 100,
            "status": classification_target_status,
        },
        "urgency_accuracy": {
            "correct_count": reported_urgency_correct_count,
            "sample_count": reported_urgency_sample_count,
            "percentage": (
                round(float(urgency_accuracy_value) * 100, 1)
                if urgency_accuracy_value is not None
                else None
            ),
        },
        "label_match_counts": comparison,
        "route_quality": route_scorecard,
        "answerability": answerability_scorecard or {
            "label_counts": dict(sorted(answerability_labels.items())),
            "evidence_available_counts": dict(sorted(answerability_proxy.items())),
            "evidence_availability_proxy_matches": answerability_agreements,
            "denominator": answerability_compared,
            "note": "Source availability is a retrieval-readiness proxy, not a semantic answerability judgment.",
        },
        "protected_cases": protected_scorecard or {
            "label_count": protected_count,
            "escalated_count": protected_escalated,
            "route_deviation_count": protected_route_deviations,
            "automatic_send_count": protected_auto_sends,
        },
        "artifact_notes": {
            "ticket_level_mismatches": "label_comparison.json after the post-run comparator is executed",
        },
        "evaluation_labels_read_after_persistence": True,
    }


def _latency_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    """Expose scored, generation-inclusive, generation-wait, and export timings."""
    evaluation = metrics.get("evaluation")
    evaluation = evaluation if isinstance(evaluation, dict) else {}
    observed = evaluation.get("run_observed")
    observed = observed if isinstance(observed, dict) else {}
    technical = observed.get("technical")
    technical = technical if isinstance(technical, dict) else {}
    criteria = evaluation.get("criteria")
    criteria = criteria if isinstance(criteria, dict) else {}
    return {
        "processing_latency_seconds": technical.get("latency"),
        "generation_inclusive_processing_duration": technical.get(
            "generation_inclusive_processing_duration"
        ),
        "draft_generation_request_duration": technical.get(
            "draft_generation_request_duration"
        ),
        "runtime_latency_criterion": criteria.get("runtime_latency"),
        "batch_export_duration_seconds": metrics.get("batch_export_duration_seconds"),
    }


def format_terminal_summary(summary: dict[str, Any]) -> str:
    """Format the post-run summary as concise human-readable JSON."""
    import json

    return "Development evaluation summary:\n" + json.dumps(
        summary, indent=2, ensure_ascii=False, sort_keys=True
    )


def _value(value: Any, fallback: str) -> str:
    return value if isinstance(value, str) and value else fallback


def _automatic_send(result: dict[str, Any]) -> bool:
    draft = result.get("draft")
    return isinstance(draft, dict) and draft.get("send_status") == "sent"
