"""Checks for strict classification parsing and credential-free ticket understanding."""

from __future__ import annotations

import json

import pytest

from cloudserve_support.classification.ticket_classification import (
    ClassificationOutputError,
    INTENT_VALUES,
    OFFLINE_CLASSIFIER_MODEL,
    classify_offline_ticket,
    classify_ticket,
    parse_ticket_classification,
)
from cloudserve_support.config import Settings
from cloudserve_support.ingestion.ticket_normalization import normalize_ticket_record


def _ticket(subject: str, body: str):
    """Build one evaluation-free normalized ticket for classification tests."""
    return normalize_ticket_record(
        {
            "ticket_id": "T-CLASSIFY",
            "channel": "email",
            "subject": subject,
            "body": body,
            "labels": {"expected_route": "must-not-be-read"},
            "history": {"csat_rating": 1},
        },
        input_index=0,
    )


class _InvalidClassificationProvider:
    """Return invalid JSON to exercise the safe provider fallback."""

    name = "openai"

    def classify_ticket(self, ticket, *, model_name: str) -> str:
        return "not json"


def test_offline_classification_is_repeatable_without_credentials() -> None:
    ticket = _ticket("Previous release", "How do I roll back a failing deployment?")

    first = classify_offline_ticket(ticket)
    second = classify_offline_ticket(ticket)

    assert first == second
    assert first.intent == "rollback_request"
    assert first.urgency == "medium"
    assert first.provider == "offline"
    assert first.model_name == OFFLINE_CLASSIFIER_MODEL
    assert first.provider_status == "classified"
    assert 0 <= first.confidence <= 1
    assert "labels" not in ticket.context
    assert "history" not in ticket.context


def test_offline_confidence_uses_development_calibration_and_safe_unclear_score() -> None:
    supported_ticket = _ticket("Single sign-on", "Our SAML identity provider is not provisioning users.")
    unclear_ticket = _ticket("Question", "It is not working. Can someone look into this?")

    supported = classify_offline_ticket(supported_ticket)
    unclear = classify_offline_ticket(unclear_ticket)

    assert supported.model_name == OFFLINE_CLASSIFIER_MODEL
    assert supported.confidence == 0.99
    assert unclear.intent == "unclear_request"
    assert unclear.confidence == 0.95


@pytest.mark.parametrize(
    ("subject", "body", "expected_intent"),
    [
        (
            "Retention period question",
            "Our policy requires two years of retention. What are our options given what the platform retains by default?",
            "compliance_request",
        ),
        (
            "Retention period question",
            "We need to demonstrate that read access to customer records is logged. I cannot find these events in the audit log.",
            "compliance_request",
        ),
        (
            "It is not working",
            "nothing is loading properly today. Can someone look into this? kindly check and revert.",
            "unclear_request",
        ),
    ],
)
def test_offline_classification_detects_compliance_and_vague_revert_requests(
    subject: str,
    body: str,
    expected_intent: str,
) -> None:
    result = classify_offline_ticket(_ticket(subject, body))

    assert result.intent == expected_intent


def test_structured_model_output_parses_all_classification_fields() -> None:
    payload = {
        "intent": "authentication",
        "urgency": "high",
        "confidence": 0.91,
        "alternatives": [{"intent": "account", "confidence": 0.18}],
        "uncertainties": [],
        "reason": "The ticket reports a failed SSO login.",
    }

    result = parse_ticket_classification(
        json.dumps(payload),
        provider="openai",
        model_name="gpt-6-luna",
    )

    assert result.as_dict() == {
        **payload,
        "confidence": 0.91,
        "alternatives": [{"intent": "account", "confidence": 0.18}],
        "provider": "openai",
        "model_name": "gpt-6-luna",
        "provider_status": "classified",
    }


@pytest.mark.parametrize(
    "payload",
    [
        "not json",
        json.dumps({"intent": "invented", "urgency": "low", "confidence": 0.9, "alternatives": [], "uncertainties": [], "reason": "unknown"}),
        json.dumps({"intent": "api", "urgency": "urgent", "confidence": 0.9, "alternatives": [], "uncertainties": [], "reason": "unknown urgency"}),
        json.dumps({"intent": "api", "urgency": "low", "confidence": 1.1, "alternatives": [], "uncertainties": [], "reason": "out of range"}),
        json.dumps({"intent": [], "urgency": "low", "confidence": 0.9, "alternatives": [], "uncertainties": [], "reason": "non-string enum"}),
    ],
)
def test_invalid_model_output_is_rejected(payload: str) -> None:
    with pytest.raises(ClassificationOutputError):
        parse_ticket_classification(payload, provider="openai", model_name="gpt-6-luna")


def test_provider_parse_failure_returns_unclear_safe_fallback() -> None:
    result = classify_ticket(
        _ticket("Question", "Please help."),
        Settings(provider="OFFLINE"),
        provider=_InvalidClassificationProvider(),
    )

    assert result.intent == "unclear_request"
    assert result.urgency == "unknown"
    assert result.confidence == 0
    assert result.provider_status == "fallback"
    assert result.alternatives == ()
    assert result.uncertainties


def test_openai_provider_without_credentials_uses_failure_fallback() -> None:
    result = classify_ticket(
        _ticket("API pagination", "How can I fetch the next page?"),
        Settings(provider="OPENAI", llm_model="gpt-6-luna"),
    )

    assert result.provider_status == "fallback"
    assert result.provider == "openai"
    assert result.model_name == "gpt-6-luna"
    assert "OPENAI_API_KEY" in result.uncertainties[0]


@pytest.mark.parametrize(
    ("subject", "body", "expected_intent"),
    [
        ("Limit reached", "We receive a 429 after too many requests per minute.", "rate_limit"),
        ("Export", "The queued export download link expired.", "data_export"),
        ("Database timeout", "Our SQL database connection pool is timing out.", "database_issue"),
        ("Webhook", "Webhook deliveries fail signature verification.", "webhook_issue"),
        ("Single sign-on", "Our SAML identity provider assertion is rejected.", "sso_configuration"),
        ("Project access", "A user role permission prevents access to the workspace.", "account_access"),
        ("Billing", "Why is this month's invoice higher?", "billing_query"),
        ("Getting started", "Where should I start using the platform?", "onboarding"),
        ("Configuration", "The environment variable is not being read.", "configuration_help"),
        ("API pagination", "How do I set the page size for this endpoint?", "api_usage_question"),
    ],
)
def test_offline_classifier_supports_development_intent_vocabulary(
    subject: str,
    body: str,
    expected_intent: str,
) -> None:
    result = classify_offline_ticket(_ticket(subject, body))

    assert expected_intent in INTENT_VALUES
    assert result.intent == expected_intent


@pytest.mark.parametrize(
    ("subject", "body", "expected_intent"),
    [
        (
            "Key committed to a public repo",
            "One of our engineers pushed a key to a public repository by mistake. We need to replace it without taking the service down.",
            "api_key_issue",
        ),
        (
            "Suspicious activity in audit log",
            "We have found activity in our audit log from an address none of us recognise. Resources were created that nobody on the team claims.",
            "security_incident",
        ),
        (
            "Question about backup regions",
            "Are backups replicated outside our primary region? A compliance review has raised this and I need a definite answer.",
            "data_residency",
        ),
        (
            "Would like custom retention",
            "We would like to configure our own retention period rather than taking the plan default. Is that something you are considering?",
            "feature_request",
        ),
        (
            "SAML login returns an error page",
            "New starters are not being created automatically when they log in through SSO for the first time. We expected them to be provisioned.",
            "sso_configuration",
        ),
        (
            "Duplicate webhook deliveries",
            "We are receiving the same event several times. Is this expected behaviour or is something wrong with our configuration?",
            "webhook_issue",
        ),
        (
            "It is not working",
            "Nothing is loading properly today. Can someone look into this? Kindly check and revert.",
            "unclear_request",
        ),
        (
            "Possible account compromise",
            "A former employee appears to still have access three weeks after leaving. We have found API calls made under their account. This is urgent.",
            "security_incident",
        ),
        (
            "Please add per-project spend caps",
            "Are staged rollouts on the roadmap? We would like to move traffic gradually rather than all at once.",
            "feature_request",
        ),
        (
            "Roadmap question",
            "Are staged rollouts on the roadmap? We want to move traffic gradually rather than all at once.",
            "feature_request",
        ),
    ],
)
def test_offline_classifier_resolves_development_error_collisions(
    subject: str,
    body: str,
    expected_intent: str,
) -> None:
    result = classify_offline_ticket(_ticket(subject, body))

    assert result.intent == expected_intent
    assert result.model_name == "offline-rule-based-v5"


@pytest.mark.parametrize(
    ("subject", "body", "expected_urgency"),
    [
        ("Production outage", "Production is down and all jobs are failing.", "high"),
        ("Invoice question", "Our invoice amount is unexpected.", "medium"),
        ("API guidance", "How do I set a query parameter?", "low"),
        ("Question", "Could you explain this setting?", "low"),
        ("Checkout broken", "A release has broken checkout for customers.", "high"),
    ],
)
def test_offline_classifier_predicts_a_supported_urgency(
    subject: str,
    body: str,
    expected_urgency: str,
) -> None:
    result = classify_offline_ticket(_ticket(subject, body))

    assert result.urgency == expected_urgency
