"""Apply deterministic protected, uncertainty, confidence, and evidence routing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cloudserve_support.classification.ticket_classification import INTENT_VALUES
from cloudserve_support.domain.ticket import CanonicalTicket
from cloudserve_support.guardrails.operational_safety import (
    detect_protected_intent,
    is_unclear_operational_request,
    operational_ticket_text,
)


PROTECTED_INTENTS = frozenset(
    {
        "account_compromise",
        "billing_dispute",
        "compliance",
        "compliance_request",
        "data_location",
        "data_residency",
        "feature_request",
        "roadmap_request",
        "security",
        "security_incident",
        "unclear_request",
    }
)


@dataclass(frozen=True)
class RoutingDecision:
    """One route plus its matched rule and policy threshold provenance."""

    route: str
    reason: str
    matched_rule: str
    protected_case: bool
    evidence_ready: bool | None


def route_ticket_classification(
    classification: dict[str, Any],
    policy: dict[str, Any],
    *,
    evidence_ready: bool | None = None,
    ticket: CanonicalTicket | None = None,
) -> RoutingDecision:
    """Route classification deterministically without consulting fixture labels."""
    if not isinstance(classification, dict):
        return _degrade(
            "Classification signals are unavailable; manual review is required.",
            "classification_signals_unavailable",
            evidence_ready,
        )
    required_classification_fields = {
        "intent",
        "provider_status",
        "provider",
        "model_name",
        "confidence",
        "uncertainties",
    }
    if not required_classification_fields.issubset(classification):
        return _degrade(
            "Required classification signals are missing; manual review is required.",
            "classification_signals_unavailable",
            evidence_ready,
        )

    if not isinstance(policy, dict):
        return _degrade(
            "The frozen routing policy is unavailable; manual review is required.",
            "routing_policy_unavailable",
            evidence_ready,
        )
    required_policy_fields = {
        "classifier_provider", "classifier_model", "classification_confidence_threshold"
    }
    if not required_policy_fields.issubset(policy):
        return _degrade(
            "Required frozen policy signals are missing; manual review is required.",
            "routing_policy_unavailable",
            evidence_ready,
        )

    intent = classification["intent"]
    provider_status = classification["provider_status"]
    confidence = classification["confidence"]
    uncertainties = classification["uncertainties"]
    threshold = policy["classification_confidence_threshold"]
    if not isinstance(intent, str) or intent not in INTENT_VALUES:
        return _degrade(
            "Classification contains an unknown intent; manual review is required.",
            "classification_signal_invalid",
            evidence_ready,
        )
    if (
        not isinstance(confidence, (int, float))
        or isinstance(confidence, bool)
        or not 0 <= confidence <= 1
    ):
        return _degrade(
            "Classification confidence is missing or invalid; manual review is required.",
            "classification_signal_invalid",
            evidence_ready,
        )
    if not isinstance(uncertainties, (list, tuple)) or not all(
        isinstance(item, str) for item in uncertainties
    ):
        return _degrade(
            "Classification uncertainty signals are invalid; manual review is required.",
            "classification_signal_invalid",
            evidence_ready,
        )
    if (
        not isinstance(threshold, (int, float))
        or isinstance(threshold, bool)
        or not 0 <= threshold <= 1
        or not isinstance(policy["classifier_provider"], str)
        or not policy["classifier_provider"]
        or not isinstance(policy["classifier_model"], str)
        or not policy["classifier_model"]
    ):
        return _degrade(
            "Frozen policy signals are invalid; manual review is required.",
            "routing_policy_invalid",
            evidence_ready,
        )
    if evidence_ready is not None and not isinstance(evidence_ready, bool):
        return _degrade(
            "Evidence readiness is unknown; manual review is required.",
            "evidence_signal_invalid",
            None,
        )

    if not isinstance(provider_status, str) or provider_status not in {
        "classified",
        "fallback",
    }:
        return _degrade(
            "Classification status is unknown; manual review is required.",
            "classification_signal_invalid",
            evidence_ready,
        )
    model_matches_policy = (
        classification["provider"] == policy["classifier_provider"]
        and classification["model_name"] == policy["classifier_model"]
    )
    if provider_status != "classified" or not model_matches_policy:
        return RoutingDecision(
            route="degrade",
            reason=(
                "Classification is unavailable or does not match the frozen policy model; "
                "manual review is required."
            ),
            matched_rule="classification_unavailable_or_model_mismatch",
            protected_case=intent in PROTECTED_INTENTS,
            evidence_ready=evidence_ready,
        )
    if ticket is not None:
        operational_text = operational_ticket_text(ticket)
        operational_protected_intent = detect_protected_intent(operational_text)
        if operational_protected_intent:
            return RoutingDecision(
                route="escalate",
                reason=(
                    "Operational ticket text indicates a protected "
                    f"{operational_protected_intent.replace('_', ' ')} case."
                ),
                matched_rule=f"operational_protected_{operational_protected_intent}",
                protected_case=True,
                evidence_ready=evidence_ready,
            )
        if is_unclear_operational_request(operational_text):
            return RoutingDecision(
                route="escalate",
                reason=(
                    "Operational ticket text describes a symptom without naming the "
                    "affected feature; human review is required."
                ),
                matched_rule="operational_request_unclear",
                protected_case=True,
                evidence_ready=evidence_ready,
            )
    if intent in PROTECTED_INTENTS:
        return RoutingDecision(
            route="escalate",
            reason=f"The {intent.replace('_', ' ')} category requires human review.",
            matched_rule=f"protected_{intent}",
            protected_case=True,
            evidence_ready=evidence_ready,
        )
    if classification["uncertainties"]:
        return RoutingDecision(
            route="escalate",
            reason="Classification has unresolved uncertainty; human review is required.",
            matched_rule="classification_uncertain",
            protected_case=False,
            evidence_ready=evidence_ready,
        )
    if confidence < threshold:
        return RoutingDecision(
            route="escalate",
            reason=(
                f"Classification confidence {confidence:.2f} is below "
                f"the frozen threshold {threshold:.2f}."
            ),
            matched_rule="below_classification_confidence_threshold",
            protected_case=False,
            evidence_ready=evidence_ready,
        )
    if evidence_ready is False:
        return RoutingDecision(
            route="escalate",
            reason="No approved source evidence is ready for this ticket.",
            matched_rule="evidence_not_ready",
            protected_case=False,
            evidence_ready=False,
        )

    evidence_rule = (
        "classification_confident_evidence_ready"
        if evidence_ready is True
        else "classification_confident_pending_retrieval"
    )
    evidence_reason = (
        " Approved source evidence is ready for the next stage."
        if evidence_ready is True
        else " Retrieval is not part of this classification stage yet."
    )
    return RoutingDecision(
        route="draft_for_review",
        reason=f"Classification meets the frozen policy threshold.{evidence_reason}",
        matched_rule=evidence_rule,
        protected_case=False,
        evidence_ready=evidence_ready,
    )


def _degrade(reason: str, matched_rule: str, evidence_ready: bool | None) -> RoutingDecision:
    """Return a fail-closed route for missing or invalid policy signals."""
    return RoutingDecision(
        route="degrade",
        reason=reason,
        matched_rule=matched_rule,
        protected_case=False,
        evidence_ready=(
            evidence_ready
            if evidence_ready is None or isinstance(evidence_ready, bool)
            else None
        ),
    )
