"""Build a concise, post-run terminal summary for labeled ticket batches."""

from __future__ import annotations

from collections import Counter
from typing import Any

from cloudserve_support.classification.ticket_classification import INTENT_VALUES, URGENCY_VALUES

ROUTE_ALIASES = {"auto_respond": "draft_for_review", "escalate": "escalate"}
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
    intent_support: Counter[str] = Counter()
    intent_predicted: Counter[str] = Counter()
    intent_correct: Counter[str] = Counter()
    urgency_predicted: Counter[str] = Counter()
    route_predicted: Counter[str] = Counter()
    label_counts: dict[str, Counter[str]] = {
        "intent": Counter(),
        "urgency": Counter(),
        "route": Counter(),
    }
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
    undefined_metrics: list[dict[str, str]] = []

    for input_index, record in enumerate(input_records):
        result = result_by_index.get(input_index)
        if not isinstance(record, dict) or result is None:
            continue
        classification = result.get("classification")
        classification = classification if isinstance(classification, dict) else {}
        predicted_intent = _value(classification.get("intent"), "unknown")
        if predicted_intent not in INTENT_VALUES:
            predicted_intent = "unknown"
        predicted_urgency = _value(classification.get("urgency"), "unknown")
        if predicted_urgency not in URGENCY_VALUES:
            predicted_urgency = "unknown"
        predicted_route = _value(result.get("route"), "unknown")
        intent_predicted[predicted_intent] += 1
        urgency_predicted[predicted_urgency] += 1
        route_predicted[predicted_route] += 1

        labels = record.get("labels")
        if not isinstance(labels, dict):
            continue
        expected_intent = labels.get("intent")
        expected_urgency = labels.get("urgency")
        expected_route_raw = labels.get("expected_route")
        expected_route = ROUTE_ALIASES.get(expected_route_raw, expected_route_raw)

        if isinstance(expected_intent, str) and expected_intent in INTENT_VALUES:
            intent_support[expected_intent] += 1
            label_counts["intent"][expected_intent] += 1
            label_totals["intent"] += 1
            if predicted_intent == expected_intent:
                label_matches["intent"] += 1
                intent_correct[expected_intent] += 1
        if isinstance(expected_urgency, str) and expected_urgency in URGENCY_VALUES:
            label_counts["urgency"][expected_urgency] += 1
            label_totals["urgency"] += 1
            if predicted_urgency == expected_urgency:
                label_matches["urgency"] += 1
        if isinstance(expected_route_raw, str) and expected_route_raw in EXPECTED_ROUTE_LABEL_VALUES:
            label_counts["route"][expected_route] += 1
            label_totals["route"] += 1
            if predicted_route == expected_route:
                label_matches["route"] += 1

        answerable = labels.get("answerable_from_docs")
        if isinstance(answerable, bool):
            answerability_labels[str(answerable).lower()] += 1
            sources = result.get("sources")
            evidence_ready = isinstance(sources, list) and any(
                isinstance(source, dict) and isinstance(source.get("doc_id"), str)
                for source in sources
            )
            answerability_proxy[str(evidence_ready).lower()] += 1
            answerability_compared += 1
            if answerable == evidence_ready:
                answerability_agreements += 1

        if labels.get("must_not_auto_respond") is True:
            protected_count += 1
            if predicted_route == "escalate":
                protected_escalated += 1
            else:
                protected_route_deviations += 1
            if _automatic_send(result):
                protected_auto_sends += 1

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
    intent_rows: dict[str, dict[str, Any]] = {}
    if (
        isinstance(classification_scorecard, dict)
        and isinstance(classification_scorecard.get("per_class"), dict)
    ):
        for intent, row in classification_scorecard["per_class"].items():
            precision_metric = row.get("precision", {})
            recall_metric = row.get("recall", {})
            correct_count = precision_metric.get("numerator") or 0
            intent_rows[intent] = {
                "support": row.get("support", 0),
                "predicted_count": row.get("predicted_count", 0),
                "correct_count": correct_count,
                "label_match_count": correct_count,
                "precision": precision_metric.get("value"),
                "recall": recall_metric.get("value"),
                "precision_target_status": row.get("precision_target_status", "not_applicable"),
            }
            if precision_metric.get("status") != "available":
                undefined_metrics.append({"intent": intent, "metric": "precision"})
            if recall_metric.get("status") != "available":
                undefined_metrics.append({"intent": intent, "metric": "recall"})
        intent_target_status = classification_scorecard.get(
            "precision_target_status", "not_applicable"
        )
        supported_class_count = classification_scorecard.get(
            "precision_classes_with_support", 0
        )
        intent_target_met_count = classification_scorecard.get(
            "precision_classes_meeting_target", 0
        )
    else:
        predicted_classes = set(intent_predicted) & INTENT_VALUES
        if intent_predicted["unknown"]:
            predicted_classes.add("unknown")
        all_intents = sorted(set(intent_support) | predicted_classes)
        for intent in all_intents:
            support = intent_support[intent]
            predicted = intent_predicted[intent]
            correct = intent_correct[intent]
            precision = correct / predicted if predicted and support else None
            recall = correct / support if support else None
            precision_target_status = (
                "not_applicable"
                if not support or not predicted
                else "met"
                if precision is not None and precision >= 0.85
                else "below_target"
            )
            intent_rows[intent] = {
                "support": support,
                "predicted_count": predicted,
                "correct_count": correct,
                "label_match_count": correct,
                "precision": precision,
                "recall": recall,
                "precision_target_status": precision_target_status,
            }
            if not support or not predicted:
                undefined_metrics.append({"intent": intent, "metric": "precision"})
            if recall is None:
                undefined_metrics.append({"intent": intent, "metric": "recall"})
        supported_intent_rows = [row for row in intent_rows.values() if row["support"] > 0]
        intent_target_met_count = sum(
            row["precision_target_status"] == "met" for row in supported_intent_rows
        )
        supported_class_count = len(supported_intent_rows)
        intent_target_status = (
            "pass"
            if supported_intent_rows and intent_target_met_count == len(supported_intent_rows)
            else "below_target"
            if supported_intent_rows
            else "not_applicable"
        )
    urgency_scorecard = technical_metrics.get("urgency") if isinstance(technical_metrics, dict) else None
    urgency_match_metric = (
        urgency_scorecard.get("exact_match_rate")
        if isinstance(urgency_scorecard, dict)
        else None
    )
    urgency_match_count = (
        urgency_match_metric.get("numerator")
        if isinstance(urgency_match_metric, dict)
        else label_matches["urgency"]
    )
    urgency_sample_count = (
        urgency_match_metric.get("denominator")
        if isinstance(urgency_match_metric, dict)
        else label_totals["urgency"]
    )
    urgency_match_rate = (
        urgency_match_metric.get("value")
        if isinstance(urgency_match_metric, dict)
        else (
            label_matches["urgency"] / label_totals["urgency"]
            if label_totals["urgency"]
            else None
        )
    )

    routes_from_metrics = metrics.get("route_counts")
    if isinstance(routes_from_metrics, dict):
        route_predicted = Counter({str(k): int(v) for k, v in routes_from_metrics.items()})

    comparison = {
        field: {
            "matched": label_matches[field],
            "mismatched": label_totals[field] - label_matches[field],
            "denominator": label_totals[field],
            "label_counts": dict(sorted(label_counts[field].items())),
        }
        for field in ("intent", "urgency", "route")
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
            "intent": dict(sorted(intent_predicted.items())),
            "urgency": dict(sorted(urgency_predicted.items())),
            "route": dict(sorted(route_predicted.items())),
        },
        "intent_quality": intent_rows,
        "intent_precision_target": {
            "threshold": 0.85,
            "status": intent_target_status,
            "supported_class_count": supported_class_count,
            "meeting_class_count": intent_target_met_count,
            "note": "Development-set criterion; unsupported or unpredicted classes are not applicable and do not count as meeting the target.",
        },
        "urgency_quality": {
            "target": None,
            "target_status": (
                urgency_scorecard.get("target_status", "measured_no_numeric_target")
                if isinstance(urgency_scorecard, dict)
                else "measured_no_numeric_target"
            ),
            "exact_match_count": urgency_match_count,
            "sample_count": urgency_sample_count,
            "exact_match_rate": urgency_match_rate,
            "confusion_matrix": (
                urgency_scorecard.get("confusion_matrix")
                if isinstance(urgency_scorecard, dict)
                else None
            ),
            "prioritization_error_counts": (
                urgency_scorecard.get("prioritization_error_counts")
                if isinstance(urgency_scorecard, dict)
                else None
            ),
            "prioritization_errors": (
                urgency_scorecard.get("prioritization_errors")
                if isinstance(urgency_scorecard, dict)
                else []
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
        "undefined_metrics": undefined_metrics,
        "artifact_notes": {
            "confusion_matrix": "metrics.json evaluation classification confusion_matrix",
            "urgency_confusion_matrix": "metrics.json evaluation run_observed technical urgency confusion_matrix",
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
