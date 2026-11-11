"""Checks for the shared operational ticket projection across input channels."""

import json
from collections import Counter
from pathlib import Path

import pytest

from cloudserve_support.ingestion.ticket_normalization import normalize_ticket_record


@pytest.mark.parametrize("channel", ["chat", "email", "docs_comment", "forum"])
def test_each_supported_channel_uses_the_same_canonical_ticket_contract(channel: str) -> None:
    ticket = normalize_ticket_record(
        {
            "ticket_id": "T-1",
            "channel": channel,
            "subject": "Question",
            "body": "Please help.",
            "received_at": "2026-05-01T10:00:00Z",
            "customer_tier": "business",
            "language_fluency": "fluent",
            "customer_name": "Zoë 😃",
            "customer_region": "asia_pacific",
            "labels": {"expected_route": "must-not-be-operational"},
            "history": {"csat_rating": 5},
            "conversation": [{"message": "Earlier message"}],
        },
        input_index=0,
    )

    assert ticket.validation_errors == ()
    assert (ticket.ticket_id, ticket.channel, ticket.subject, ticket.body) == (
        "T-1", channel, "Question", "Please help."
    )
    assert ticket.received_at == "2026-05-01T10:00:00Z"
    assert ticket.customer_tier == "business"
    assert ticket.language_fluency == "fluent"
    assert ticket.context["customer_name"] == "Zoë 😃"
    assert ticket.context["conversation"] == [{"message": "Earlier message"}]
    assert "labels" not in ticket.context
    assert "history" not in ticket.context


@pytest.mark.parametrize(
    "fixture_name,expected_count",
    [("development_tickets.json", 500), ("validation_tickets.json", 80)],
)
def test_ticket_fixtures_normalize_without_evaluation_fields(
    fixture_name: str,
    expected_count: int,
) -> None:
    fixture_path = Path(__file__).resolve().parents[2] / "data" / "tickets" / fixture_name
    records = json.loads(fixture_path.read_text(encoding="utf-8"))

    normalized = [
        normalize_ticket_record(record, input_index=index)
        for index, record in enumerate(records)
    ]

    assert len(normalized) == expected_count
    assert all(ticket.validation_errors == () for ticket in normalized)
    assert Counter(ticket.channel for ticket in normalized) == {
        "chat": sum(record["channel"] == "chat" for record in records),
        "email": sum(record["channel"] == "email" for record in records),
        "docs_comment": sum(record["channel"] == "docs_comment" for record in records),
        "forum": sum(record["channel"] == "forum" for record in records),
    }
    assert all("labels" not in ticket.context and "history" not in ticket.context for ticket in normalized)


def test_malformed_record_has_location_surrogate_and_structured_errors() -> None:
    ticket = normalize_ticket_record(
        {"ticket_id": " ", "channel": "sms", "body": None},
        input_index=4,
    )

    assert ticket.ticket_id == "invalid-000005"
    assert not ticket.has_ticket_id
    assert ticket.validation_errors == (
        "ticket_id must be a non-empty string",
        "channel must be chat, docs_comment, email, or forum",
        "body must be a non-empty string",
    )
