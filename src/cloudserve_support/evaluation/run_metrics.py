"""Calculate post-run technical and historical metrics without changing decisions."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cloudserve_support.classification.ticket_classification import INTENT_VALUES, URGENCY_VALUES
from cloudserve_support.evaluation.citation_source_integrity import (
    evaluate_citation_source_integrity,
)
from cloudserve_support.evaluation.criteria_scorecard import build_criteria_scorecard

INVALID_OPERATIONAL_ERRORS = frozenset({"invalid_ticket_record", "duplicate_ticket_id"})
EXPECTED_ROUTE_ALIASES = {"auto_respond": "draft_for_review", "escalate": "escalate"}
EXPECTED_ROUTE_LABEL_VALUES = frozenset(
    {"auto_respond", "draft_for_review", "escalate", "degrade"}
)
PUBLIC_ROUTE_VALUES = frozenset({"draft_for_review", "escalate", "degrade"})
INTENT_PRECISION_TARGET = 0.85
# The terminal report applies the same 85% threshold to total labeled intent accuracy.
INTENT_ACCURACY_TARGET = INTENT_PRECISION_TARGET
def evaluate_completed_run(
    *,
    run_id: str,
    results: list[dict[str, Any]],
    input_records: list[Any],
    input_path: Path,
    source_ref: str,
    decision_events: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare persisted run results with evaluation-only labels and historical outcomes."""
    decision_events = [
        event for event in decision_events if event.get("terminal", True) is True
    ]
    source_set = _source_set(source_ref)
    record_by_index = {index: record for index, record in enumerate(input_records)}
    result_by_index: dict[int, dict[str, Any]] = {}
    duplicate_ids: set[str] = set()
    seen_ids: set[str] = set()
    mismatch_indices: list[int] = []
    evaluation_errors: list[str] = []

    for result in results:
        input_index = result.get("input_index")
        if not isinstance(input_index, int) or isinstance(input_index, bool):
            evaluation_errors.append("A persisted result has an invalid input_index.")
            continue
        if input_index in result_by_index:
            mismatch_indices.append(input_index)
            evaluation_errors.append("Persisted results contain a duplicate input_index.")
            continue
        result_by_index[input_index] = result
        source_record = record_by_index.get(input_index)
        if source_record is None:
            mismatch_indices.append(input_index)
            evaluation_errors.append("A persisted result has no matching input record.")
            continue
        expected_id = _source_ticket_id(source_record, input_index)
        if result.get("ticket_id") != expected_id:
            mismatch_indices.append(input_index)
            evaluation_errors.append("A persisted ticket ID does not match its input index.")
        if expected_id in seen_ids:
            duplicate_ids.add(expected_id)
        seen_ids.add(expected_id)

    missing_result_indices = sorted(set(record_by_index) - set(result_by_index))
    if missing_result_indices:
        evaluation_errors.append("One or more input records have no persisted result.")
    if duplicate_ids:
        evaluation_errors.append("Input records contain duplicate ticket IDs.")
    decision_count = len(decision_events)
    if len(results) != decision_count:
        evaluation_errors.append("Decision-log event count does not match persisted results.")
    if _count_matching_decisions(results, decision_events) != decision_count:
        evaluation_errors.append("Decision-log event content does not match persisted results.")
    policy = _single_policy_provenance(results)
    if policy.get("status") == "mismatch":
        evaluation_errors.append("Persisted results contain inconsistent frozen-policy provenance.")

    labels_by_index: dict[int, dict[str, Any]] = {}
    history_records = 0
    for index, record in record_by_index.items():
        if not isinstance(record, dict):
            continue
        labels = record.get("labels")
        history = record.get("history")
        if isinstance(labels, dict):
            labels_by_index[index] = labels
        if isinstance(history, dict):
            history_records += 1

    evaluation_label_errors = _evaluation_label_errors(
        input_records=input_records,
        labels_by_index=labels_by_index,
        source_set=source_set,
    )
    evaluation_errors.extend(evaluation_label_errors)

    labeled_pairs = [
        (result_by_index[index], labels_by_index[index])
        for index in sorted(set(result_by_index) & set(labels_by_index))
    ]
    technical = _technical_metrics(
        run_id=run_id,
        results=results,
        result_by_index=result_by_index,
        input_count=len(input_records),
        labels=labels_by_index,
        labeled_pairs=labeled_pairs,
        decision_events=decision_events,
        evaluation_errors=evaluation_errors,
        policy_provenance=policy,
    )
    if evaluation_label_errors:
        technical["classification"]["precision_target_status"] = "invalid_evaluation_data"
        technical["classification"]["precision_target_note"] = (
            "Development intent precision cannot pass because one or more evaluation labels are invalid."
        )
        technical["urgency"]["target_status"] = "invalid_evaluation_data"
    governance = _governance_metrics(results, decision_events, input_records)
    governance["protected_case_safety"] = _protected_case_safety(
        results=results,
        result_by_index=result_by_index,
        input_records=input_records,
        labels=labels_by_index,
    )
    volume = _volume_metrics(results)
    business = _business_metrics(results, input_records)
    historical = _historical_baseline(input_records)
    source_sha256 = _hash_input_file(input_path)
    integrity_status = "complete" if not evaluation_errors else "mismatch"
    criteria = build_criteria_scorecard(
        results=results,
        input_records=input_records,
        decision_events=decision_events,
        technical=technical,
        integrity_status=integrity_status,
        evaluation_labels_valid=not evaluation_label_errors,
    )
    return {
        "report_version": 1,
        "run_id": run_id,
        "source_set": source_set,
        "input_ref": _safe_unicode_text(source_ref) or "unavailable",
        "input_sha256": source_sha256,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "sample_count": len(input_records),
        "evaluation_data": {
            "labels_record_count": len(labels_by_index),
            "history_record_count": history_records,
            "labels_used_after_processing": True,
            "history_used_after_processing": True,
        },
        "integrity": {
            "status": integrity_status,
            "expected_count": len(input_records),
            "result_count": len(results),
            "decision_count": decision_count,
            "missing_result_indices": missing_result_indices,
            "mismatched_indices": sorted(set(mismatch_indices)),
            "duplicate_ticket_ids": sorted(duplicate_ids),
            "errors": evaluation_errors,
        },
        "policy_provenance": policy,
        "run_observed": {
            "volume": volume,
            "business": business,
            "technical": technical,
            "governance": governance,
        },
        "criteria": criteria,
        "historical_baseline": historical,
        "limitations": [
            "Historical outcomes describe the fixture's prior support process and are not caused by this CLI run.",
            "Scored per-ticket processing latency excludes only draft-generation request wait; generation-inclusive processing duration and request wait are reported separately. First-response time and first-contact resolution are unavailable because this CLI drafts or escalates and does not send or resolve tickets.",
            "Historical resolution-time measures are fixture baselines, not response times caused by this CLI run.",
            "Citation resolution and exact excerpt grounding are mechanical checks; semantic model-judged estimates are recorded separately in evaluation.criteria.llm_judge.",
            "Customer satisfaction is historical fixture data; this run did not collect live customer feedback or human ratings.",
        ],
    }


def _technical_metrics(
    *,
    run_id: str,
    results: list[dict[str, Any]],
    result_by_index: dict[int, dict[str, Any]],
    input_count: int,
    labels: dict[int, dict[str, Any]],
    labeled_pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    decision_events: list[dict[str, Any]],
    evaluation_errors: list[str],
    policy_provenance: dict[str, Any],
) -> dict[str, Any]:
    intent_pairs = [
        (
            label["intent"],
            _validated_classification_prediction(result, "intent", INTENT_VALUES),
        )
        for result, label in labeled_pairs
        if isinstance(label.get("intent"), str)
        and label["intent"] in INTENT_VALUES
        and _safe_unicode_text(label["intent"]) is not None
    ]
    urgency_pairs = [
        (
            label["urgency"],
            _validated_classification_prediction(result, "urgency", URGENCY_VALUES),
            result,
        )
        for result, label in labeled_pairs
        if isinstance(label.get("urgency"), str)
        and label["urgency"] in URGENCY_VALUES
        and _safe_unicode_text(label["urgency"]) is not None
    ]
    route_pairs = [
        (
            EXPECTED_ROUTE_ALIASES.get(label.get("expected_route"), label.get("expected_route")),
            _validated_route_prediction(result),
            result,
        )
        for result, label in labeled_pairs
        if isinstance(label.get("expected_route"), str)
        and label["expected_route"] in EXPECTED_ROUTE_LABEL_VALUES
        and _safe_unicode_text(label.get("expected_route")) is not None
    ]
    invalid_intent_prediction_count = _invalid_classification_prediction_count(
        labeled_pairs, "intent", INTENT_VALUES
    )
    invalid_urgency_prediction_count = _invalid_classification_prediction_count(
        labeled_pairs, "urgency", URGENCY_VALUES
    )
    true_classes = sorted({truth for truth, _ in intent_pairs})
    predicted_classes = sorted(
        {prediction for _, prediction in intent_pairs if isinstance(prediction, str)}
    )
    class_names = sorted(set(true_classes) | set(predicted_classes))
    missing_intent_count = input_count - len(intent_pairs)
    per_class: dict[str, Any] = {}
    confusion: dict[str, dict[str, int]] = {}
    for class_name in class_names:
        true_positive = sum(
            truth == class_name and prediction == class_name
            for truth, prediction in intent_pairs
        )
        support = sum(truth == class_name for truth, _ in intent_pairs)
        predicted_count = sum(prediction == class_name for _, prediction in intent_pairs)
        class_missing_prediction_count = sum(
            truth == class_name and not isinstance(prediction, str)
            for truth, prediction in intent_pairs
        )
        if support == 0 or predicted_count == 0:
            precision = _unavailable_metric(
                0,
                "No labeled support or no predictions for this class.",
            )
        else:
            precision = _metric(
                true_positive,
                predicted_count,
                missing_intent_count,
                "No predictions for this class.",
            )
        if support == 0:
            recall = _unavailable_metric(0, "No labeled support for this class.")
        else:
            recall = _metric(
                true_positive,
                support,
                missing_intent_count,
                "No labeled support for this class.",
            )
        if support == 0 or predicted_count == 0:
            precision_target_status = "not_applicable"
        elif precision["value"] >= INTENT_PRECISION_TARGET:
            precision_target_status = "met"
        else:
            precision_target_status = "below_target"
        per_class[class_name] = {
            "precision": precision,
            "recall": recall,
            "support": support,
            "predicted_count": predicted_count,
            "missing_prediction_count": class_missing_prediction_count,
            "precision_target_status": precision_target_status,
        }
    confusion_predictions = set(predicted_classes)
    if any(not isinstance(prediction, str) for _, prediction in intent_pairs):
        confusion_predictions.add("__unavailable_prediction__")
    for truth in true_classes:
        confusion[truth] = {
            prediction: sum(
                actual == truth
                and (predicted if isinstance(predicted, str) else "__unavailable_prediction__") == prediction
                for actual, predicted in intent_pairs
            )
            for prediction in sorted(confusion_predictions)
        }
    intent_target_classes = [
        row for row in per_class.values() if row["support"] > 0
    ]
    intent_target_met_count = sum(
        row["precision_target_status"] == "met" for row in intent_target_classes
    )
    intent_target_status = (
        "pass"
        if intent_target_classes
        and intent_target_met_count == len(intent_target_classes)
        else "below_target"
        if intent_target_classes
        else "not_applicable"
    )

    urgency_true_classes = sorted({truth for truth, _, _ in urgency_pairs})
    urgency_predicted_classes = sorted(
        {prediction for _, prediction, _ in urgency_pairs if isinstance(prediction, str)}
    )
    urgency_missing_count = sum(
        not isinstance(prediction, str) for _, prediction, _ in urgency_pairs
    )
    urgency_per_class: dict[str, Any] = {}
    urgency_confusion: dict[str, dict[str, int]] = {}
    urgency_errors: list[dict[str, Any]] = []
    urgency_direction_counts: Counter[str] = Counter()
    urgency_rank = {"low": 1, "medium": 2, "high": 3, "unknown": 0}
    for urgency in sorted(set(urgency_true_classes) | set(urgency_predicted_classes)):
        true_positive = sum(
            truth == urgency and prediction == urgency
            for truth, prediction, _ in urgency_pairs
        )
        support = sum(truth == urgency for truth, _, _ in urgency_pairs)
        predicted_count = sum(
            prediction == urgency for _, prediction, _ in urgency_pairs
        )
        precision = _metric(
            true_positive,
            predicted_count,
            urgency_missing_count,
            "No predictions for this urgency class.",
        )
        if support == 0 or predicted_count == 0:
            precision_target_status = "not_applicable"
        else:
            precision_target_status = "diagnostic"
        urgency_per_class[urgency] = {
            "precision": precision,
            "recall": _metric(
                true_positive, support, urgency_missing_count,
                "No labeled support for this urgency class.",
            ),
            "support": support,
            "predicted_count": predicted_count,
            "precision_target_status": precision_target_status,
        }
    urgency_target_classes = [
        row for row in urgency_per_class.values() if row["support"] > 0
    ]
    urgency_target_status = "diagnostic" if urgency_pairs else "not_applicable"
    urgency_confusion_predictions = set(urgency_predicted_classes)
    if urgency_missing_count:
        urgency_confusion_predictions.add("__unavailable_prediction__")
    for truth in urgency_true_classes:
        urgency_confusion[truth] = {
            prediction: sum(
                actual == truth
                and (predicted if isinstance(predicted, str) else "__unavailable_prediction__") == prediction
                for actual, predicted, _ in urgency_pairs
            )
            for prediction in sorted(urgency_confusion_predictions)
        }
    for index, (expected, predicted, result) in enumerate(urgency_pairs):
        if expected == predicted:
            continue
        if not isinstance(predicted, str):
            direction = "different_or_unavailable"
        elif urgency_rank.get(predicted, -1) < urgency_rank.get(expected, -1):
            direction = "under_prioritized"
        elif urgency_rank.get(predicted, -1) > urgency_rank.get(expected, -1):
            direction = "over_prioritized"
        else:
            direction = "different_or_unavailable"
        urgency_direction_counts[direction] += 1
        urgency_errors.append(
            {
                "input_index": result.get("input_index", index),
                "ticket_id": result.get("ticket_id"),
                "expected": expected,
                "predicted": predicted,
                "direction": direction,
            }
        )
    correct_intents = sum(truth == prediction for truth, prediction in intent_pairs)
    if intent_pairs:
        classification_status = "available"
    elif input_count:
        classification_status = "unavailable"
    else:
        classification_status = "not_applicable"

    expected_doc_pairs: list[tuple[dict[str, Any], list[str]]] = []
    missing_expected_docs = 0
    for input_index in range(input_count):
        result = result_by_index.get(input_index)
        label = labels.get(input_index)
        if result is None:
            missing_expected_docs += 1
            continue
        if label is None:
            missing_expected_docs += 1
            continue
        expected_docs = label.get("expected_doc_ids")
        if isinstance(expected_docs, list) and all(
            isinstance(item, str) for item in expected_docs
        ):
            if expected_docs:
                expected_doc_pairs.append((result, expected_docs))
            elif label.get("answerable_from_docs") is True:
                missing_expected_docs += 1
        elif label.get("answerable_from_docs") is not False:
            missing_expected_docs += 1
    retrieval_hits = sum(
        bool(
            {
                source.get("doc_id")
                for source in result.get("sources", [])
                if isinstance(source, dict)
            }
            & set(expected_docs)
        )
        for result, expected_docs in expected_doc_pairs
    )

    route_confusion: dict[str, dict[str, int]] = {}
    route_predictions = set(PUBLIC_ROUTE_VALUES) | {"__unavailable_prediction__"}
    for expected in sorted({expected for expected, _, _ in route_pairs}):
        route_confusion[expected] = {
            predicted: sum(
                truth == expected
                and (actual if actual is not None else "__unavailable_prediction__") == predicted
                for truth, actual, _ in route_pairs
            )
            for predicted in sorted(route_predictions)
        }
    route_label_counts = Counter(expected for expected, _, _ in route_pairs)
    route_predicted_counts = Counter(
        actual if actual is not None else "__unavailable_prediction__"
        for _, actual, _ in route_pairs
    )
    route_deviation_count = sum(expected != actual for expected, actual, _ in route_pairs)
    false_escalations = sum(
        expected == "draft_for_review" and actual == "escalate"
        for expected, actual, _ in route_pairs
    )
    unsafe_automation_routes = sum(
        expected == "escalate" and actual == "draft_for_review"
        for expected, actual, _ in route_pairs
    )
    other_route_deviations = route_deviation_count - false_escalations - unsafe_automation_routes

    answerability_pairs: list[tuple[bool, bool | None, dict[str, Any] | None]] = []
    answerability_labeled_count = 0
    answerability_missing_label_count = 0
    answerability_missing_evidence_count = 0
    answerability_evidence_counts: Counter[str] = Counter()
    retrieval_status_counts: Counter[str] = Counter()
    for input_index in range(input_count):
        label = labels.get(input_index)
        if label is None or not isinstance(label.get("answerable_from_docs"), bool):
            answerability_missing_label_count += 1
            continue
        answerability_labeled_count += 1
        result = result_by_index.get(input_index)
        routing = result.get("routing") if isinstance(result, dict) else None
        evidence_ready = routing.get("evidence_ready") if isinstance(routing, dict) else None
        if not isinstance(evidence_ready, bool):
            answerability_missing_evidence_count += 1
            answerability_evidence_counts["unavailable"] += 1
        else:
            answerability_evidence_counts[str(evidence_ready).lower()] += 1
        answerability_pairs.append((label["answerable_from_docs"], evidence_ready, result))
        retrieval = routing.get("retrieval") if isinstance(routing, dict) else None
        retrieval_status = retrieval.get("status") if isinstance(retrieval, dict) else None
        retrieval_status_counts[
            retrieval_status if isinstance(retrieval_status, str) and retrieval_status else "unavailable"
        ] += 1
    answerability_confusion = {
        answerable: {
            prediction: sum(
                truth == (answerable == "true")
                and (
                    str(ready).lower()
                    if isinstance(ready, bool)
                    else "__unavailable_prediction__"
                ) == prediction
                for truth, ready, _ in answerability_pairs
            )
            for prediction in ("false", "true", "__unavailable_prediction__")
        }
        for answerable in ("false", "true")
    }
    answerability_correct = sum(truth == evidence_ready for truth, evidence_ready, _ in answerability_pairs)

    citation_source_integrity = evaluate_citation_source_integrity(
        run_id=run_id,
        results=results,
        labels_by_index=labels,
    )

    valid_operational_count = sum(
        1
        for index in range(input_count)
        if not _is_invalid_operational_result(result_by_index.get(index))
    )
    covered_count = sum(
        1
        for index in range(input_count)
        if (result := result_by_index.get(index)) is not None
        and not _is_invalid_operational_result(result)
        and result.get("status") == "completed"
        and result.get("route") in {"draft_for_review", "escalate"}
    )
    matched_decisions = _count_matching_decisions(results, decision_events)
    decision_denominator = max(len(results), len(decision_events), 1 if results else 0)
    confidences: list[tuple[float, bool]] = []
    for index, result in result_by_index.items():
        label = labels.get(index, {})
        prediction = result.get("classification", {})
        confidence = prediction.get("confidence") if isinstance(prediction, dict) else None
        if (
            isinstance(label.get("intent"), str)
            and label["intent"] in INTENT_VALUES
            and isinstance(confidence, (int, float))
            and not isinstance(confidence, bool)
        ):
            predicted_intent = prediction.get("intent")
            confidences.append((float(confidence), predicted_intent == label["intent"]))

    return {
        "classification": {
            "status": classification_status,
            "sample_size": len(intent_pairs),
            "missing_count": input_count - len(intent_pairs),
            "missing_prediction_count": sum(
                not isinstance(prediction, str) for _, prediction in intent_pairs
            ),
            "invalid_prediction_count": invalid_intent_prediction_count,
            "accuracy": _metric(
                correct_intents,
                len(intent_pairs),
                input_count - len(intent_pairs),
                "Intent labels or predictions are missing.",
            ),
            "precision_target": INTENT_PRECISION_TARGET,
            "precision_target_status": intent_target_status,
            "precision_classes_with_support": len(intent_target_classes),
            "precision_classes_meeting_target": intent_target_met_count,
            "precision_target_note": (
                "Development-set calibration criterion only; classes without labeled support "
                "or predictions are not applicable and do not count as meeting the target."
            ),
            "per_class": per_class,
            "confusion_matrix": confusion,
            "confidence_calibration": _confidence_calibration_bins(confidences),
        },
        "urgency": {
            "status": "available" if urgency_pairs else "not_applicable",
            "precision_target": None,
            "target_status": urgency_target_status,
            "precision_classes_with_support": len(urgency_target_classes),
            "precision_classes_meeting_target": None,
            "precision_target_note": (
                "Evaluation_Framework.docx specifies no per-class urgency precision cutoff; "
                "urgency metrics are reported as diagnostic only. Classes without labeled "
                "support or predictions are not applicable."
            ),
            "exact_match_rate": _metric(
                sum(expected == predicted for expected, predicted, _ in urgency_pairs),
                len(urgency_pairs),
                input_count - len(urgency_pairs),
                "Urgency labels or predictions are missing.",
            ),
            "sample_size": len(urgency_pairs),
            "missing_prediction_count": urgency_missing_count,
            "invalid_prediction_count": invalid_urgency_prediction_count,
            "per_class": urgency_per_class,
            "confusion_matrix": urgency_confusion,
            "prioritization_error_counts": dict(sorted(urgency_direction_counts.items())),
            "prioritization_errors": urgency_errors,
            "note": (
                "Evaluation_Framework.docx specifies no per-class urgency precision cutoff; "
                "urgency exact-match rate, per-class precision/recall, confusion matrix, and "
                "prioritization trade-offs are reported as diagnostic."
            ),
        },
        "route_accuracy": _metric(
            sum(expected == actual for expected, actual, _ in route_pairs),
            len(route_pairs),
            input_count - len(route_pairs),
            "Expected route labels are unavailable.",
        ),
        "route_quality": {
            "sample_size": len(route_pairs),
            "missing_count": input_count - len(route_pairs),
            "unavailable_prediction_count": sum(actual is None for _, actual, _ in route_pairs),
            "confusion_matrix": route_confusion,
            "expected_counts": dict(sorted(route_label_counts.items())),
            "predicted_counts": dict(sorted(route_predicted_counts.items())),
            "false_escalation": {
                "count": false_escalations,
                "denominator": route_label_counts["draft_for_review"],
                "rate": _ratio_or_none(false_escalations, route_label_counts["draft_for_review"]),
            },
            "unsafe_automation_route": {
                "count": unsafe_automation_routes,
                "denominator": route_label_counts["escalate"],
                "rate": _ratio_or_none(unsafe_automation_routes, route_label_counts["escalate"]),
            },
            "route_deviation_count": route_deviation_count,
            "other_deviation_count": other_route_deviations,
            "threshold_provenance": policy_provenance,
            "note": (
                "Trade-offs are measured at the frozen policy threshold. The unsafe-automation route "
                "count means a draft route where the label expects escalation; drafts remain unsent, "
                "and automatic sends are checked separately. No route-accuracy cutoff is specified."
            ),
        },
        "answerability": {
            "status": "available" if answerability_pairs else "unavailable" if input_count else "not_applicable",
            "accuracy": _metric(
                answerability_correct,
                answerability_labeled_count,
                answerability_missing_evidence_count,
                "Answerability labels are unavailable.",
            ),
            "sample_size": answerability_labeled_count,
            "missing_label_count": answerability_missing_label_count,
            "missing_evidence_readiness_count": answerability_missing_evidence_count,
            "missing_count": answerability_missing_label_count + answerability_missing_evidence_count,
            "confusion_matrix": answerability_confusion,
            "evidence_readiness_counts": dict(sorted(answerability_evidence_counts.items())),
            "retrieval_status_counts": dict(sorted(retrieval_status_counts.items())),
            "note": (
                "This compares answerable_from_docs labels with the persisted routing evidence_ready signal. "
                "It measures evidence availability, not semantic answerability; retrieval_hit_rate separately "
                "checks expected document IDs. Missing readiness remains in the labeled denominator as an "
                "unavailable prediction and is counted above."
            ),
        },
        "retrieval_hit_rate": _metric(
            retrieval_hits,
            len(expected_doc_pairs),
            missing_expected_docs,
            "No labeled expected document IDs are available.",
        ),
        "citation_resolution": citation_source_integrity["citation_resolution"],
        "citation_source_integrity": citation_source_integrity,
        "coverage": _metric(
            covered_count,
            valid_operational_count,
            input_count - valid_operational_count,
            "No valid operational tickets are available.",
        ),
        "availability": _metric(
            sum(result.get("status") == "completed" for result in results),
            len(results),
            0,
            "No persisted results are available.",
        ),
        "latency": {
            "median_seconds": _latency_metric(results, statistic="median"),
            "p95_seconds": _latency_metric(results, statistic="p95"),
        },
        "generation_inclusive_processing_duration": {
            "median_seconds": _duration_metric(
                results,
                field="processing_duration_seconds",
                statistic="median",
            ),
            "p95_seconds": _duration_metric(
                results,
                field="processing_duration_seconds",
                statistic="p95",
            ),
        },
        "draft_generation_request_duration": {
            "median_seconds": _duration_metric(
                results,
                field="draft_generation_request_duration_seconds",
                statistic="median",
            ),
            "p95_seconds": _duration_metric(
                results,
                field="draft_generation_request_duration_seconds",
                statistic="p95",
            ),
        },
        "decision_log_reconciliation": _metric(
            matched_decisions,
            decision_denominator,
            0,
            "Decision event count or content does not reconcile with results."
            if evaluation_errors
            else "Every decision event reconciles by input index and ticket ID.",
        ),
    }


def _volume_metrics(results: list[dict[str, Any]]) -> dict[str, int]:
    """Summarize each ticket outcome and ticket-level guardrail blocks."""
    blocked_tickets = sum(
        any(
            item.get("blocked") is True
            for item in result.get("guardrails", {}).get("checks", [])
            if isinstance(item, dict)
        )
        for result in results
    )
    return {
        "tickets_processed": len(results),
        "drafts_for_agent_review": sum(
            result.get("route") == "draft_for_review" for result in results
        ),
        "escalated": sum(result.get("route") == "escalate" for result in results),
        "degraded": sum(result.get("route") == "degrade" for result in results),
        "blocked_by_guardrails": blocked_tickets,
    }


def _business_metrics(
    results: list[dict[str, Any]], input_records: list[Any]
) -> dict[str, Any]:
    """Report observed escalation and explicit gaps for downstream outcomes."""
    escalated = sum(result.get("route") == "escalate" for result in results)
    return {
        "first_contact_resolution": _unavailable_metric(
            len(results), "This CLI does not send replies or observe ticket resolution."
        ),
        "first_response_time_minutes_mean": _unavailable_metric(
            len(results), "This CLI does not send replies or record first-response timestamps."
        ),
        "first_response_time_minutes_median": _unavailable_metric(
            len(results), "This CLI does not send replies or record first-response timestamps."
        ),
        "escalation_rate": _metric(
            escalated,
            len(results),
            0,
            "No ticket outcomes were recorded.",
        ),
        "historical_resolution_time_minutes_mean": _numeric_metric(
            [
                record.get("history", {})
                if isinstance(record, dict) and isinstance(record.get("history"), dict)
                else {}
                for record in input_records
            ],
            "resolution_time_minutes",
            len(input_records),
            minimum=0,
        ),
        "historical_resolution_time_minutes_median": _numeric_metric(
            [
                record.get("history", {})
                if isinstance(record, dict) and isinstance(record.get("history"), dict)
                else {}
                for record in input_records
            ],
            "resolution_time_minutes",
            len(input_records),
            minimum=0,
            statistic="median",
        ),
    }


def _latency_metric(
    results: list[dict[str, Any]], *, statistic: str
) -> dict[str, Any]:
    """Summarize scored processing latency after generation request wait."""
    return _duration_metric(
        results,
        field="processing_latency_seconds",
        statistic=statistic,
    )


def _duration_metric(
    results: list[dict[str, Any]],
    *,
    field: str,
    statistic: str,
) -> dict[str, Any]:
    """Summarize one per-ticket duration field using the requested statistic."""
    values = sorted(
        float(result[field])
        for result in results
        if isinstance(result.get(field), (int, float))
        and not isinstance(result.get(field), bool)
        and math.isfinite(float(result[field]))
        and float(result[field]) >= 0
    )
    if not values:
        return _unavailable_metric(
            len(results),
            f"Per-ticket {field} timings are unavailable.",
        )
    if statistic == "median":
        value = statistics.median(values)
    else:
        rank = math.ceil(0.95 * len(values))
        value = values[max(rank - 1, 0)]
    return _metric_value(value, None, len(values), len(results) - len(values), None)


def _governance_metrics(
    results: list[dict[str, Any]],
    decision_events: list[dict[str, Any]],
    input_records: list[Any],
) -> dict[str, Any]:
    checks: Counter[str] = Counter()
    blocked_checks: Counter[str] = Counter()
    for result in results:
        guardrails = result.get("guardrails", {})
        items = guardrails.get("checks", []) if isinstance(guardrails, dict) else []
        for item in items if isinstance(items, list) else []:
            if isinstance(item, dict) and isinstance(item.get("check"), str):
                checks[item["check"]] += 1
                if item.get("blocked") is True:
                    blocked_checks[item["check"]] += 1
    guardrail_activations = {
        name: _metric(
            blocked_checks[name],
            checks[name],
            len(results) - checks[name],
            "Some results did not record this guardrail check.",
        )
        for name in sorted(
            checks
            | Counter(
                {
                    "source_eligibility": 0,
                    "prompt_injection": 0,
                    "private_data_input": 0,
                    "private_data_source": 0,
                    "output_structure": 0,
                    "citation_grounding": 0,
                    "private_data_output": 0,
                    "instruction_integrity": 0,
                }
            )
        )
    }
    private_checks = checks["private_data_output"]
    private_blocks = blocked_checks["private_data_output"]
    return {
        "logged_decisions": _metric(
            _count_matching_decisions(results, decision_events),
            max(len(decision_events), len(input_records)),
            max(len(input_records) - len(decision_events), 0),
            "Decision event count does not match input record count.",
        ),
        "guardrail_activations": guardrail_activations,
        "private_output_guardrail": _metric(
            private_blocks,
            private_checks,
            len(results) - private_checks,
            "The private-data output check was not recorded.",
        ),
        "frozen_policy": _single_policy_provenance(results),
    }


def _protected_case_safety(
    *,
    results: list[dict[str, Any]],
    result_by_index: dict[int, dict[str, Any]],
    input_records: list[Any],
    labels: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """Measure protected route deviations and send violations after processing."""
    protected_indices = [
        index
        for index, label in labels.items()
        if label.get("must_not_auto_respond") is True
    ]
    route_deviations = []
    protected_auto_sends = 0
    for index in protected_indices:
        result = result_by_index.get(index)
        route = result.get("route") if isinstance(result, dict) else None
        sent = _automatic_send(result)
        protected_auto_sends += int(sent)
        if route != "escalate":
            record = input_records[index] if index < len(input_records) else None
            route_deviations.append(
                {
                    "input_index": index,
                    "ticket_id": (
                        result.get("ticket_id")
                        if isinstance(result, dict)
                        else _source_ticket_id(record, index)
                    ),
                    "predicted_route": route if route is not None else "__unavailable_prediction__",
                    "automatic_send": sent,
                }
            )

    automatic_send_count = sum(_automatic_send(result) for result in results)
    protected_missing_result_count = len(protected_indices) - sum(
        index in result_by_index for index in protected_indices
    )
    if protected_auto_sends:
        zero_tolerance_status = "violation"
    elif protected_missing_result_count:
        zero_tolerance_status = "unavailable"
    else:
        zero_tolerance_status = "pass"
    return {
        "protected_case_count": len(protected_indices),
        "checked_protected_case_count": len(protected_indices) - protected_missing_result_count,
        "missing_protected_result_count": protected_missing_result_count,
        "protected_automatic_send_violation_count": protected_auto_sends,
        "protected_route_deviation_count": len(route_deviations),
        "protected_route_deviations": route_deviations,
        "all_ticket_automatic_send_count": automatic_send_count,
        "all_ticket_automatic_send_checked_count": len(results),
        "all_ticket_automatic_send_check_denominator": len(input_records),
        "zero_tolerance_status": zero_tolerance_status,
        "note": (
            "Every protected case must escalate and no ticket is automatically sent. Route deviations "
            "and automatic sends are reported separately; draft_for_review remains an unsent draft."
        ),
    }


def _historical_baseline(input_records: list[Any]) -> dict[str, Any]:
    groups: dict[str, dict[str, list[dict[str, Any]]]] = {
        "language_fluency": defaultdict(list),
        "customer_tier": defaultdict(list),
    }
    histories: list[dict[str, Any]] = []
    for record in input_records:
        source_record = record if isinstance(record, dict) else {}
        history = source_record.get("history")
        observed_history = history if isinstance(history, dict) else {}
        if isinstance(history, dict):
            histories.append(history)
        for dimension in groups:
            group_name = source_record.get(dimension)
            normalized_group = (
                _safe_unicode_text(group_name.strip())
                if isinstance(group_name, str)
                and group_name.strip()
                and _safe_unicode_text(group_name) is not None
                else "unknown"
            )
            groups[dimension][normalized_group].append(observed_history)
    return {
        "source": "input.history",
        "sample_size": len(histories),
        "input_sample_size": len(input_records),
        "causal_claim_about_this_run": False,
        "overall": _history_group_metrics(
            [
                record.get("history", {})
                if isinstance(record, dict) and isinstance(record.get("history"), dict)
                else {}
                for record in input_records
            ],
            len(input_records),
        ),
        "segments": {
            dimension: {
                group_name: {
                    "sample_size": len(group_histories),
                    "metrics": _history_group_metrics(group_histories, len(group_histories)),
                }
                for group_name, group_histories in sorted(dimension_groups.items())
            }
            for dimension, dimension_groups in groups.items()
        },
    }


def _history_group_metrics(
    histories: list[dict[str, Any]],
    eligible_count: int,
) -> dict[str, Any]:
    return {
        "first_contact_resolution": _boolean_metric(
            histories, "first_contact_resolution", eligible_count
        ),
        "escalation_rate": _boolean_metric(histories, "escalated", eligible_count),
        "repeat_contact_rate": _boolean_metric(histories, "repeat_contact", eligible_count),
        "csat_mean": _numeric_metric(histories, "csat_rating", eligible_count, minimum=1, maximum=5),
        "resolution_time_minutes_mean": _numeric_metric(
            histories, "resolution_time_minutes", eligible_count, minimum=0
        ),
        "resolution_time_minutes_median": _numeric_metric(
            histories,
            "resolution_time_minutes",
            eligible_count,
            minimum=0,
            statistic="median",
        ),
        "first_response_time_minutes": _unavailable_metric(
            eligible_count, "Historical records do not contain first-response timestamps."
        ),
    }


def _boolean_metric(
    records: list[dict[str, Any]],
    field_name: str,
    eligible_count: int,
) -> dict[str, Any]:
    observed = [
        record[field_name]
        for record in records
        if isinstance(record.get(field_name), bool)
    ]
    return _metric(
        sum(observed),
        len(observed),
        eligible_count - len(observed),
        f"History field {field_name} is missing or invalid.",
    )


def _numeric_metric(
    records: list[dict[str, Any]],
    field_name: str,
    eligible_count: int,
    *,
    minimum: float,
    maximum: float | None = None,
    statistic: str = "mean",
) -> dict[str, Any]:
    values = [
        float(record[field_name])
        for record in records
        if isinstance(record.get(field_name), (int, float))
        and not isinstance(record.get(field_name), bool)
        and math.isfinite(float(record[field_name]))
        and float(record[field_name]) >= minimum
        and (maximum is None or float(record[field_name]) <= maximum)
    ]
    if values:
        if statistic == "median":
            value = statistics.median(values)
            numerator = None
        else:
            numerator = sum(values)
            value = numerator / len(values)
        return _metric_value(value, numerator, len(values), eligible_count - len(values), None)
    return _unavailable_metric(
        eligible_count, f"History field {field_name} is missing or invalid."
    )


def _confidence_calibration_bins(
    confidences: list[tuple[float, bool]],
    *,
    bins: int = 5,
) -> list[dict[str, Any]]:
    rows = []
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        members = [
            (confidence, correct)
            for confidence, correct in confidences
            if lower <= confidence < upper or (index == bins - 1 and confidence == 1)
        ]
        if not members:
            continue
        rows.append(
            {
                "lower_bound": lower,
                "upper_bound": upper,
                "sample_size": len(members),
                "mean_confidence": sum(confidence for confidence, _ in members) / len(members),
                "observed_accuracy": sum(correct for _, correct in members) / len(members),
            }
        )
    return rows


def _single_policy_provenance(results: list[dict[str, Any]]) -> dict[str, Any]:
    provenances = []
    for result in results:
        routing = result.get("routing", {})
        provenance = routing.get("policy_provenance") if isinstance(routing, dict) else None
        if isinstance(provenance, dict):
            provenances.append(provenance)
    if not provenances:
        return {"status": "unavailable", "policy": None, "consistent": None}
    canonical = json.dumps(provenances[0], sort_keys=True, separators=(",", ":"))
    consistent = all(
        json.dumps(item, sort_keys=True, separators=(",", ":")) == canonical
        for item in provenances
    )
    fingerprint_fields = (
        "policy_file_sha256",
        "threshold_sha256",
        "classifier_implementation_sha256",
        "classifier_model_sha256",
    )
    fingerprints = {field: provenances[0].get(field) for field in fingerprint_fields}
    fingerprints_present = all(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
        for value in fingerprints.values()
    )
    fingerprint_status = (
        "available"
        if fingerprints_present and consistent
        else "mismatch"
        if any(value is not None for value in fingerprints.values())
        else "unavailable"
    )
    return {
        "status": "available" if consistent else "mismatch",
        "policy": provenances[0],
        "consistent": consistent,
        "fingerprint_status": fingerprint_status,
        "fingerprints": fingerprints,
        "sample_size": len(provenances),
        "classifier_providers": sorted(
            {
                provider
                for result in results
                if (
                    provider := (
                        result.get("classifier_provider")
                        if isinstance(result.get("classifier_provider"), str)
                        else (
                            result.get("classification", {}).get("provider")
                            if isinstance(result.get("classification"), dict)
                            else None
                        )
                    )
                )
                and provider == "gemini"
            }
        ),
        "classifier_model_names": sorted(
            {
                model_name
                for result in results
                if (
                    model_name := (
                        result.get("classifier_model")
                        if isinstance(result.get("classifier_model"), str)
                        else (
                            result.get("classification", {}).get("model_name")
                            if isinstance(result.get("classification"), dict)
                            else None
                        )
                    )
                )
                and isinstance(model_name, str)
                and model_name
            }
        ),
    }


def _count_matching_decisions(
    results: list[dict[str, Any]],
    decision_events: list[dict[str, Any]],
) -> int:
    result_by_index = {result.get("input_index"): result for result in results}
    matched = 0
    seen: set[int] = set()
    for event in decision_events:
        if event.get("terminal", True) is not True:
            continue
        result = result_by_index.get(event.get("input_index"))
        input_index = event.get("input_index")
        if (
            result is not None
            and isinstance(input_index, int)
            and input_index not in seen
            and event.get("ticket_id") == result.get("ticket_id")
            and event.get("action_taken") == result.get("route")
        ):
            matched += 1
            seen.add(input_index)
    return matched


def _is_invalid_operational_result(result: dict[str, Any] | None) -> bool:
    if result is None:
        return True
    error = result.get("error")
    return isinstance(error, dict) and error.get("code") in INVALID_OPERATIONAL_ERRORS


def _source_ticket_id(record: Any, input_index: int) -> str:
    if (
        isinstance(record, dict)
        and isinstance(record.get("ticket_id"), str)
        and record["ticket_id"].strip()
        and _safe_unicode_text(record["ticket_id"]) is not None
    ):
        return record["ticket_id"].strip()
    return f"invalid-{input_index + 1:06d}"


def _safe_unicode_text(value: str) -> str | None:
    """Return UTF-8-safe evaluation text or omit it from aggregate report keys."""
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return None
    return value


def _metric(
    numerator: int,
    denominator: int,
    missing_count: int,
    note: str,
) -> dict[str, Any]:
    if denominator <= 0:
        return _unavailable_metric(missing_count, note)
    return _metric_value(numerator / denominator, numerator, denominator, missing_count, None)


def _metric_value(
    value: float,
    numerator: int | float | None,
    denominator: int,
    missing_count: int,
    note: str | None,
) -> dict[str, Any]:
    return {
        "value": value,
        "status": "available",
        "numerator": numerator,
        "denominator": denominator,
        "sample_size": denominator,
        "missing_count": missing_count,
        "note": note,
    }


def _unavailable_metric(sample_size: int, note: str) -> dict[str, Any]:
    return {
        "value": None,
        "status": "unavailable" if sample_size else "not_applicable",
        "numerator": None,
        "denominator": 0,
        "sample_size": sample_size,
        "missing_count": sample_size,
        "note": note,
    }


def _hash_input_file(input_path: Path) -> str:
    try:
        return hashlib.sha256(input_path.read_bytes()).hexdigest()
    except OSError:
        return ""


def _source_set(source_ref: str) -> str:
    name = Path(source_ref).name.casefold()
    if "development" in name:
        return "development"
    if "validation" in name:
        return "validation"
    return "external"


def _validated_classification_prediction(
    result: dict[str, Any], field: str, allowed_values: frozenset[str]
) -> str | None:
    """Return a recognized classification prediction or mark it unavailable."""
    classification = result.get("classification")
    value = classification.get(field) if isinstance(classification, dict) else None
    return value if isinstance(value, str) and value in allowed_values else None


def _validated_route_prediction(result: dict[str, Any]) -> str | None:
    """Return a public route value or mark its persisted prediction unavailable."""
    route = result.get("route")
    return route if isinstance(route, str) and route in PUBLIC_ROUTE_VALUES else None


def _automatic_send(result: dict[str, Any] | None) -> bool:
    """Detect an actual automatic send from the persisted draft outcome."""
    draft = result.get("draft") if isinstance(result, dict) else None
    return isinstance(draft, dict) and draft.get("send_status") == "sent"


def _ratio_or_none(numerator: int, denominator: int) -> float | None:
    """Return a trade-off rate only when its labeled denominator is non-empty."""
    return numerator / denominator if denominator else None


def _invalid_classification_prediction_count(
    labeled_pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    field: str,
    allowed_values: frozenset[str],
) -> int:
    """Count non-empty out-of-vocabulary intent or urgency predictions."""
    invalid_count = 0
    for result, label in labeled_pairs:
        if not isinstance(label.get(field), str) or label[field] not in allowed_values:
            continue
        classification = result.get("classification")
        raw_prediction = classification.get(field) if isinstance(classification, dict) else None
        if raw_prediction is not None and not (
            isinstance(raw_prediction, str) and raw_prediction in allowed_values
        ):
            invalid_count += 1
    return invalid_count


def _evaluation_label_errors(
    *,
    input_records: list[Any],
    labels_by_index: dict[int, dict[str, Any]],
    source_set: str,
) -> list[str]:
    """Reject malformed post-run labels before they can affect evaluation scores."""
    errors: list[str] = []
    required_development_fields = {
        "intent", "urgency", "expected_route", "expected_doc_ids",
        "answerable_from_docs", "must_not_auto_respond",
    }
    allowed_values = {
        "intent": INTENT_VALUES,
        "urgency": URGENCY_VALUES,
        "expected_route": EXPECTED_ROUTE_LABEL_VALUES,
    }
    boolean_fields = {"answerable_from_docs", "must_not_auto_respond"}

    for input_index, record in enumerate(input_records):
        if not isinstance(record, dict):
            continue
        labels = labels_by_index.get(input_index)
        if labels is None:
            if source_set == "development":
                errors.append(f"Development evaluation labels are missing at input_index {input_index}.")
            continue
        if source_set == "development":
            for field in sorted(required_development_fields - labels.keys()):
                errors.append(
                    f"Development evaluation label '{field}' is missing at input_index {input_index}."
                )
        for field, allowed in allowed_values.items():
            if field in labels and (
                not isinstance(labels[field], str) or labels[field] not in allowed
            ):
                errors.append(
                    f"Evaluation label '{field}' has an invalid value at input_index {input_index}."
                )
        for field in boolean_fields:
            if field in labels and not isinstance(labels[field], bool):
                errors.append(
                    f"Evaluation label '{field}' must be boolean at input_index {input_index}."
                )
        if "expected_doc_ids" in labels and (
            not isinstance(labels["expected_doc_ids"], list)
            or not all(isinstance(value, str) for value in labels["expected_doc_ids"])
        ):
            errors.append(
                f"Evaluation label 'expected_doc_ids' must be a string array at input_index {input_index}."
            )
    return errors
