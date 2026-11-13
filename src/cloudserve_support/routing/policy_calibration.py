"""Calibrate and freeze confidence routing policy using operational development inputs only."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from cloudserve_support.classification.ticket_classification import (
    DECISIONS_CLASSIFIER_MODEL,
    classify_tickets,
)
from cloudserve_support.config import Settings
from cloudserve_support.domain.ticket import CanonicalTicket
from cloudserve_support.ingestion.ticket_input import read_ticket_records
from cloudserve_support.ingestion.ticket_normalization import normalize_ticket_record
from cloudserve_support.routing.routing_policy import PROTECTED_INTENTS


ROUTING_POLICY_VERSION = "cs003-v5"
ROUTING_POLICY_SCHEMA_VERSION = 1
MINIMUM_CONFIDENCE_THRESHOLD = 0.65
MAXIMUM_CONFIDENCE_THRESHOLD = 0.98
CONFIDENCE_TAIL_PERCENTILE = 0.20


def calibrate_development_routing_policy(
    development_input_path: Path,
    *,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Use only normalized development ticket text to select a frozen confidence floor."""
    settings = settings or Settings.from_env()
    settings.require_gemini_api_key()
    classifier_provider = "gemini"
    classifier_model = DECISIONS_CLASSIFIER_MODEL
    records = read_ticket_records(development_input_path)
    tickets: list[CanonicalTicket] = [
        normalize_ticket_record(record, input_index=index)
        for index, record in enumerate(records)
    ]
    valid_tickets = [ticket for ticket in tickets if not ticket.validation_errors]
    if not valid_tickets:
        raise ValueError("Development calibration requires valid operational ticket records.")

    operational_records = [ticket_operational_projection(ticket) for ticket in valid_tickets]
    operational_bytes = json.dumps(
        operational_records,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    classifications = classify_tickets(valid_tickets, settings)
    failed_classifications = sum(
        result.provider_status != "classified" for result in classifications
    )
    if failed_classifications:
        raise RuntimeError(
            "Development calibration stopped because one or more model classifications failed."
        )
    confidence_scores = [
        result.confidence
        for result in classifications
        if result.intent not in PROTECTED_INTENTS and result.confidence > 0
    ]
    if not confidence_scores:
        raise ValueError("Development calibration found no non-protected confidence scores.")

    sorted_scores = sorted(confidence_scores)
    nearest_rank = max(1, math.ceil(CONFIDENCE_TAIL_PERCENTILE * len(sorted_scores)))
    raw_threshold = sorted_scores[nearest_rank - 1]
    threshold = round(
        min(MAXIMUM_CONFIDENCE_THRESHOLD, max(MINIMUM_CONFIDENCE_THRESHOLD, raw_threshold)),
        2,
    )
    return {
        "schema_version": ROUTING_POLICY_SCHEMA_VERSION,
        "policy_version": ROUTING_POLICY_VERSION,
        "classification_confidence_threshold": threshold,
        "threshold_source": "development operational confidence distribution",
        "calibration_method": "20th percentile nearest rank, clamped to [0.65, 0.98]",
        "classifier_provider": classifier_provider,
        "classifier_model": classifier_model,
        "frozen": True,
        "calibration_source": {
            "source_set": "development",
            "input_ref": development_input_path.as_posix(),
            "operational_sha256": hashlib.sha256(operational_bytes).hexdigest(),
            "total_records": len(records),
            "valid_operational_records": len(valid_tickets),
            "confidence_sample_count": len(sorted_scores),
            "failed_classifications": failed_classifications,
            "excluded_invalid_records": len(records) - len(valid_tickets),
        "evaluation_fields_used": False,
        },
    }


def ticket_operational_projection(ticket: CanonicalTicket) -> dict[str, Any]:
    """Serialize classifier-visible ticket context while excluding labels and outcomes."""
    return {
        "ticket_id": ticket.ticket_id,
        "channel": ticket.channel,
        "subject": ticket.subject,
        "body": ticket.body,
        "customer_tier": ticket.customer_tier,
        "language_fluency": ticket.language_fluency,
        "context": ticket.context,
    }


def load_frozen_routing_policy(policy_path: Path) -> dict[str, Any]:
    """Load a frozen routing policy and reject unsupported or unfrozen artifacts."""
    try:
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Frozen routing policy could not be read: {policy_path}") from error
    validate_frozen_routing_policy(policy)
    return policy


def write_frozen_routing_policy(policy: dict[str, Any], policy_path: Path) -> None:
    """Persist one validated development-calibrated routing policy as formatted JSON."""
    validate_frozen_routing_policy(policy)
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    policy_path.write_text(
        json.dumps(policy, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def validate_frozen_routing_policy(policy: dict[str, Any]) -> None:
    """Check the policy schema, development provenance, and threshold bounds."""
    required_fields = {
        "schema_version", "policy_version", "classification_confidence_threshold",
        "threshold_source", "calibration_method", "classifier_provider", "classifier_model",
        "frozen", "calibration_source",
    }
    if not isinstance(policy, dict) or set(policy) != required_fields:
        raise ValueError("Frozen routing policy fields do not match the CS-003 contract.")
    threshold = policy["classification_confidence_threshold"]
    if (
        not isinstance(threshold, (int, float))
        or isinstance(threshold, bool)
        or not 0 <= threshold <= 1
    ):
        raise ValueError("Frozen routing policy threshold must be a number from 0 to 1.")
    source = policy["calibration_source"]
    source_fields = {
        "source_set", "input_ref", "operational_sha256", "total_records",
        "valid_operational_records", "confidence_sample_count", "excluded_invalid_records",
        "failed_classifications", "evaluation_fields_used",
    }
    if not isinstance(source, dict) or set(source) != source_fields:
        raise ValueError("Frozen routing policy calibration source is invalid.")
    if (
        policy["schema_version"] != ROUTING_POLICY_SCHEMA_VERSION
        or policy["policy_version"] != ROUTING_POLICY_VERSION
        or policy["frozen"] is not True
        or source["source_set"] != "development"
        or source["evaluation_fields_used"] is not False
        or not isinstance(policy["classifier_provider"], str)
        or not policy["classifier_provider"]
        or not isinstance(policy["classifier_model"], str)
        or not policy["classifier_model"]
    ):
        raise ValueError("Frozen routing policy must come from development operational data.")
    if (
        policy["classifier_provider"] == "gemini"
        and policy["classifier_model"] != DECISIONS_CLASSIFIER_MODEL
    ):
        raise ValueError(
            f"Frozen routing policy classifier must use the Decisions API model {DECISIONS_CLASSIFIER_MODEL}."
        )
