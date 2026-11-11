"""Summarize release criteria that can be observed after a completed run."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from cloudserve_support.classification.ticket_classification import (
    INTENT_VALUES,
    URGENCY_VALUES,
)

MINIMUM_SEGMENT_SAMPLE_SIZE = 5
QUALITY_VARIATION_TARGET = 0.05
CALIBRATION_ERROR_TARGET = 0.05
LATENCY_TARGET_SECONDS = 3.0
EXPECTED_ROUTE_VALUES = frozenset(
    {"auto_respond", "draft_for_review", "escalate", "degrade"}
)


def build_criteria_scorecard(
    *,
    results: list[dict[str, Any]],
    input_records: list[Any],
    decision_events: list[dict[str, Any]],
    technical: dict[str, Any],
    integrity_status: str,
    evaluation_labels_valid: bool = True,
) -> dict[str, Any]:
    """Return target-aware statuses with denominators and explicit exclusions."""
    latency = technical["latency"]["p95_seconds"]
    latency_status = _measured_status(latency)
    if latency.get("denominator") != len(results):
        latency_status = "not_measurable"
    if latency_status == "measurable":
        latency_status = "pass" if latency["value"] < LATENCY_TARGET_SECONDS else "fail"

    generated_draft_rows = [
        result
        for result in results
        if isinstance(result.get("generation"), dict)
        and result["generation"].get("status") == "generated"
    ]
    generated_drafts = len(generated_draft_rows)
    privacy_checks = 0
    private_occurrences = 0
    for result in generated_draft_rows:
        guardrails = result.get("guardrails")
        checks = guardrails.get("checks") if isinstance(guardrails, dict) else None
        draft_checks = [
            check
            for check in (checks if isinstance(checks, list) else [])
            if isinstance(check, dict) and check.get("check") == "private_data_output"
        ]
        privacy_checks += len(draft_checks)
        private_occurrences += sum(check.get("blocked") is True for check in draft_checks)
    privacy_missing = max(generated_drafts - privacy_checks, 0)
    if generated_drafts == 0:
        privacy_status = "not_measurable"
    elif privacy_checks != generated_drafts:
        privacy_status = "not_measurable"
    else:
        privacy_status = "pass" if private_occurrences == 0 else "fail"

    audit_count = len(input_records)
    matched_audits = technical["decision_log_reconciliation"]["numerator"] or 0
    exact_audit_coverage = (
        len(results) == audit_count
        and len(decision_events) == audit_count
        and matched_audits == audit_count
        and integrity_status == "complete"
    )
    audit_status = "not_measurable" if audit_count == 0 else (
        "pass" if exact_audit_coverage else "fail"
    )

    confidence_bins = technical["classification"]["confidence_calibration"]
    calibration_bands = []
    for band in confidence_bins:
        error = abs(band["mean_confidence"] - band["observed_accuracy"])
        calibration_bands.append(
            {
                "lower_bound": band["lower_bound"],
                "upper_bound": band["upper_bound"],
                "mean_confidence": band["mean_confidence"],
                "observed_accuracy": band["observed_accuracy"],
                "absolute_error": error,
                "sample_size": band["sample_size"],
            }
        )
    maximum_calibration_error = max(
        (band["absolute_error"] for band in calibration_bands), default=None
    )
    calibration_sample_size = sum(band["sample_size"] for band in calibration_bands)
    calibration_missing_count = max(len(input_records) - calibration_sample_size, 0)
    calibration_status = (
        "not_measurable"
        if maximum_calibration_error is None
        else "fail"
        if maximum_calibration_error > CALIBRATION_ERROR_TARGET
        else "not_measurable"
        if calibration_missing_count > 0
        else "pass"
    )

    group_quality = _group_quality(results, input_records)
    if not evaluation_labels_valid:
        calibration_status = "not_measurable"
        for dimension in group_quality.values():
            for measure in dimension["measures"].values():
                measure["status"] = "not_measurable"
    measured_differences = [
        measure["value"]
        for dimension in group_quality.values()
        for measure in dimension["measures"].values()
        if measure["status"] in {"pass", "fail"}
    ]
    quality_status = (
        "not_measurable"
        if not measured_differences
        else "fail"
        if "fail" in {
            measure["status"]
            for dimension in group_quality.values()
            for measure in dimension["measures"].values()
        }
        else "pass"
    )

    return {
        "runtime_latency": _criterion(
            status=latency_status,
            target=f"p95 < {LATENCY_TARGET_SECONDS} seconds",
            value=latency.get("value"),
            numerator=latency.get("numerator"),
            denominator=latency.get("denominator", 0),
            sample_size=latency.get("sample_size", 0),
            missing_count=latency.get("missing_count", 0),
            note=(
                "Nearest-rank p95 of processing_latency_seconds, calculated as full per-ticket "
                "processing_duration_seconds through the matching SQLite result/audit commit "
                "minus only draft_generation_request_duration_seconds. Batch export and the "
                "post-run quality judge are reported separately; this is not customer response time."
            ),
        ),
        "availability": _criterion(
            status="non_gating",
            target="99.5% availability target excluded from the Release 1 CLI gate under PRD v2.3",
            value=None,
            numerator=None,
            denominator=0,
            sample_size=0,
            missing_count=0,
            note=(
                "A one-off batch has no uptime observation window or availability-event "
                "denominator. Batch completion is not availability."
            ),
        ),
        "private_data_output": _criterion(
            status=privacy_status,
            target="0 detected private-data occurrences in every generated draft",
            value=private_occurrences,
            numerator=private_occurrences,
            denominator=privacy_checks,
            sample_size=generated_drafts,
            missing_count=privacy_missing,
            note=(
                "The output guardrail's blocked check count is used as the detected-occurrence "
                "count; every generated draft must have a recorded check."
            ),
        ),
        "decision_audit_coverage": _criterion(
            status=audit_status,
            target="exactly one matching persisted decision/audit record per processed ticket",
            value=(matched_audits / audit_count) if audit_count else None,
            numerator=matched_audits,
            denominator=audit_count,
            sample_size=len(decision_events),
            missing_count=max(audit_count - min(len(decision_events), audit_count), 0),
            note=(
                "Counts SQLite-backed decision events that match one result by input index, "
                "ticket ID, and action; requires complete result/input/audit reconciliation."
            ),
        ),
        "confidence_calibration": _criterion(
            status=calibration_status,
            target=f"absolute confidence-to-accuracy error <= {CALIBRATION_ERROR_TARGET:.0%} per populated band",
            value=maximum_calibration_error,
            numerator=None,
            denominator=calibration_sample_size,
            sample_size=calibration_sample_size,
            missing_count=calibration_missing_count,
            note=(
                "Compares mean confidence with observed intent accuracy separately in each populated confidence band."
                + (
                    " Evaluation-label validation failed, so target status is not measurable."
                    if not evaluation_labels_valid
                    else ""
                )
            ),
            details={"populated_bands": calibration_bands},
        ),
        "group_quality_variation": {
            "status": quality_status,
            "target": f"maximum comparable group gap < {QUALITY_VARIATION_TARGET:.0%}",
            "value": max(measured_differences, default=None),
            "numerator": None,
            "denominator": sum(
                measure["denominator"]
                for dimension in group_quality.values()
                for measure in dimension["measures"].values()
            ),
            "sample_size": sum(
                dimension["sample_size"] for dimension in group_quality.values()
            ),
            "missing_count": sum(
                dimension["missing_segment_count"] for dimension in group_quality.values()
            ),
            "note": (
                "Only intent, urgency, route, and evidence-readiness agreement label comparisons "
                "are used. A dimension is comparable when each named group has at least "
                f"{MINIMUM_SEGMENT_SAMPLE_SIZE} labeled examples and no missing labels or evidence. "
                "These model measures do not represent realized customer outcomes."
                + (
                    " Evaluation-label validation failed, so target status is not measurable."
                    if not evaluation_labels_valid
                    else ""
                )
            ),
            "dimensions": group_quality,
        },
        "business_outcomes": _criterion(
            status="excluded",
            target=None,
            value=None,
            numerator=None,
            denominator=0,
            sample_size=0,
            missing_count=0,
            note=(
                "First-response time, first-contact resolution, satisfaction, actual escalation "
                "outcomes, repeat contacts, and resolution outcomes require downstream agent or "
                "customer actions this not-sent CLI does not perform. Historical fixtures remain "
                "labeled as baselines and are not scored as current-run performance."
            ),
        ),
    }


def _group_quality(
    results: list[dict[str, Any]], input_records: list[Any]
) -> dict[str, Any]:
    result_by_index = {
        result.get("input_index"): result
        for result in results
        if isinstance(result.get("input_index"), int)
        and not isinstance(result.get("input_index"), bool)
    }
    dimensions: dict[str, dict[str, list[tuple[int, dict[str, Any], dict[str, Any]]]]] = {
        "language_fluency": defaultdict(list),
        "customer_tier": defaultdict(list),
    }
    for index, record in enumerate(input_records):
        source_record = record if isinstance(record, dict) else {}
        result = result_by_index.get(index, {})
        labels = source_record.get("labels")
        if not isinstance(labels, dict):
            labels = {}
        for dimension, groups in dimensions.items():
            group_name = _safe_segment_name(source_record.get(dimension)) or "unknown"
            groups[group_name].append((index, result, labels))

    quality_metrics = {
        "intent_accuracy": "intent",
        "urgency_accuracy": "urgency",
        "route_accuracy": "expected_route",
        "answerability_evidence_readiness_agreement": "answerable_from_docs",
    }
    report: dict[str, Any] = {}
    for dimension, groups in dimensions.items():
        named_groups = {name: rows for name, rows in groups.items() if name != "unknown"}
        measures: dict[str, Any] = {}
        for measure_name, label_field in quality_metrics.items():
            group_values = []
            for group_name, rows in sorted(named_groups.items()):
                eligible: list[tuple[Any, Any]] = []
                missing_labels = 0
                missing_predictions = 0
                for _, result, labels in rows:
                    truth = labels.get(label_field)
                    if label_field == "expected_route" and truth == "auto_respond":
                        truth = "draft_for_review"
                    prediction = _quality_prediction(result, measure_name)
                    if not _valid_quality_truth(measure_name, truth):
                        missing_labels += 1
                        continue
                    if prediction is None:
                        missing_predictions += 1
                    eligible.append((truth, prediction))
                denominator = len(eligible)
                numerator = sum(truth == prediction for truth, prediction in eligible)
                value = numerator / denominator if denominator else None
                missing_count = missing_labels + missing_predictions
                group_values.append(
                    {
                        "group": group_name,
                        "value": value,
                        "numerator": numerator if denominator else None,
                        "denominator": denominator,
                        "sample_size": denominator,
                        "missing_count": missing_count,
                    }
                )
            adequate = all(
                group["denominator"] >= MINIMUM_SEGMENT_SAMPLE_SIZE
                and group["missing_count"] == 0
                and group["value"] is not None
                for group in group_values
            ) and len(group_values) >= 2
            values = [group["value"] for group in group_values if group["value"] is not None]
            gap = max(values) - min(values) if adequate and values else None
            measures[measure_name] = {
                "status": (
                    "not_measurable"
                    if gap is None
                    else "pass"
                    if gap < QUALITY_VARIATION_TARGET
                    else "fail"
                ),
                "value": gap,
                "denominator": sum(group["denominator"] for group in group_values),
                "sample_size": sum(len(groups[name]) for name in named_groups),
                "missing_count": sum(group["missing_count"] for group in group_values),
                "segments": group_values,
            }
        report[dimension] = {
            "sample_size": sum(len(rows) for rows in groups.values()),
            "missing_segment_count": len(groups.get("unknown", [])),
            "measures": measures,
        }
    return report


def _quality_prediction(result: dict[str, Any], metric_name: str) -> Any:
    classification = result.get("classification")
    routing = result.get("routing")
    if not isinstance(classification, dict):
        classification = {}
    if not isinstance(routing, dict):
        routing = {}
    if metric_name == "intent_accuracy":
        return classification.get("intent")
    if metric_name == "urgency_accuracy":
        return classification.get("urgency")
    if metric_name == "route_accuracy":
        route = result.get("route")
        return route if isinstance(route, str) else None
    if metric_name == "answerability_evidence_readiness_agreement":
        ready = routing.get("evidence_ready")
        return ready if isinstance(ready, bool) else None
    return None


def _valid_quality_truth(metric_name: str, truth: Any) -> bool:
    if metric_name == "intent_accuracy":
        return isinstance(truth, str) and truth in INTENT_VALUES
    if metric_name == "urgency_accuracy":
        return isinstance(truth, str) and truth in URGENCY_VALUES
    if metric_name == "route_accuracy":
        return isinstance(truth, str) and truth in EXPECTED_ROUTE_VALUES
    return isinstance(truth, bool)


def _safe_segment_name(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip()
    try:
        normalized.encode("utf-8")
    except UnicodeEncodeError:
        return None
    return normalized


def _measured_status(metric: dict[str, Any]) -> str:
    return "measurable" if metric.get("status") == "available" and metric.get("value") is not None else "not_measurable"


def _criterion(
    *,
    status: str,
    target: str | None,
    value: float | int | None,
    numerator: float | int | None,
    denominator: int,
    sample_size: int,
    missing_count: int,
    note: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    criterion = {
        "status": status,
        "target": target,
        "value": value,
        "numerator": numerator,
        "denominator": denominator,
        "sample_size": sample_size,
        "missing_count": missing_count,
        "note": note,
    }
    if details is not None:
        criterion.update(details)
    return criterion
