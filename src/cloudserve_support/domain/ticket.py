"""Canonical ticket data passed from ingestion to operational processing."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CanonicalTicket:
    """One channel-independent ticket and its evaluation-free operational context."""

    ticket_id: str
    has_ticket_id: bool
    channel: str
    subject: str
    body: str
    received_at: str | None
    customer_tier: str | None
    language_fluency: str | None
    context: dict[str, object]
    validation_errors: tuple[str, ...] = ()
