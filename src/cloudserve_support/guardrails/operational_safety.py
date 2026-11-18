"""Detect protected or underspecified requests from operational ticket content."""

from __future__ import annotations

import re
from collections.abc import Mapping

from cloudserve_support.domain.ticket import CanonicalTicket


_IGNORED_CONTEXT_KEYS = frozenset(
    {
        "ticket_id",
        "channel",
        "received_at",
        "customer_id",
        "customer_name",
        "customer_tier",
        "customer_region",
        "language_fluency",
        "labels",
        "history",
    }
)

_PROTECTED_PHRASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "compliance",
        (
            "compliance",
            "auditor",
            "audit review",
            "gdpr",
            "legal review",
            "regulatory requirement",
            "policy requires",
            "our policy",
        ),
    ),
    (
        "account_compromise",
        (
            "account compromise",
            "compromised account",
            "stolen account",
            "former employee",
            "unauthorized access",
            "someone accessed my account",
            "account was taken over",
            "took over my account",
        ),
    ),
    (
        "billing_dispute",
        (
            "billing dispute",
            "dispute this charge",
            "incorrect charge",
            "wrong charge",
            "double charged",
            "unauthorized charge",
            "refund",
        ),
    ),
    (
        "roadmap_request",
        (
            "roadmap",
            "when will you add",
            "planned feature",
            "feature timeline",
            "release date for",
        ),
    ),
    (
        "feature_request",
        (
            "feature request",
            "would be very useful",
            "would be useful to",
            "would be useful to add",
            "please add",
            "can you add",
            "could you add",
            "request a feature",
        ),
    ),
    (
        "security",
        (
            "security",
            "security incident",
            "suspicious activity",
            "address none of us recognise",
            "address none of us recognize",
            "unrecognized address",
            "unrecognised address",
            "credentials may have been exposed",
            "exposed in a breach",
            "exposed in breach",
            "breach at another service",
            "secrets in the logs",
            "secrets in logs",
            "credentials leaked",
        ),
    ),
)

_UNCLEAR_PHRASES = (
    "it is not working",
    "it isn't working",
    "it's not working",
    "nothing is loading",
    "nothing loads",
    "not loading properly",
    "can someone look into this",
    "can someone look into it",
    "please check and revert",
    "kindly check and revert",
    "something is wrong",
    "something is broken",
)

_CONCRETE_TOPIC_TERMS = (
    "api",
    "endpoint",
    "pagination",
    "webhook",
    "login",
    "sso",
    "password",
    "mfa",
    "invoice",
    "billing",
    "charge",
    "deployment",
    "deploy",
    "release",
    "rollback",
    "roll back",
    "build",
    "environment",
    "configuration",
    "integration",
    "terraform",
    "database",
    "query",
    "report",
    "dashboard",
    "workspace",
    "permission",
    "user role",
    "account",
    "project",
    "upload",
    "download",
    "export",
    "timeout",
    "error code",
    "500 error",
    "401 error",
)


def operational_ticket_text(ticket: CanonicalTicket) -> str:
    """Collect subject, body, and descriptive context while omitting identity/evaluation data."""
    fragments = [ticket.subject, ticket.body]
    _append_context_text(ticket.context, fragments)
    return " ".join(fragment for fragment in fragments if fragment).casefold()


def detect_protected_intent(text: str) -> str | None:
    """Return the first conservative protected intent indicated by ticket language."""
    normalized = text.casefold()
    for intent, phrases in _PROTECTED_PHRASES:
        if any(_has_phrase(normalized, phrase) for phrase in phrases):
            return intent
    if (
        ("audit log" in normalized or "audit trail" in normalized)
        and any(term in normalized for term in ("access", "read", "record", "logged"))
    ):
        return "compliance"
    return None


def is_unclear_operational_request(text: str) -> bool:
    """Identify generic symptom-only requests that contain no concrete topic anchor."""
    normalized = text.casefold()
    has_generic_symptom = any(phrase in normalized for phrase in _UNCLEAR_PHRASES)
    has_concrete_topic = any(_has_phrase(normalized, term) for term in _CONCRETE_TOPIC_TERMS)
    return has_generic_symptom and not has_concrete_topic


def _append_context_text(value: object, fragments: list[str], *, key: str | None = None) -> None:
    """Collect descriptive context values recursively, excluding identity and eval fields."""
    if key is not None and key.casefold() in _IGNORED_CONTEXT_KEYS:
        return
    if isinstance(value, str):
        fragments.append(value)
    elif isinstance(value, Mapping):
        for nested_key, nested_value in value.items():
            _append_context_text(nested_value, fragments, key=str(nested_key))
    elif isinstance(value, list):
        for nested_value in value:
            _append_context_text(nested_value, fragments)


def _has_phrase(text: str, phrase: str) -> bool:
    """Match a phrase at word boundaries so short terms do not match substrings."""
    return re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text) is not None
