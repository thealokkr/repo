"""Deterministic coverage for multi-request Decisions classifier chunking."""

from __future__ import annotations

import json
from types import SimpleNamespace

from google import genai
import pytest

from cloudserve_support.classification.ticket_classification import (
    CLASSIFICATION_DECISION_QUESTIONS,
    CLASSIFICATION_QUESTIONS_PER_TICKET,
    CLASSIFICATION_TICKETS_PER_BATCH,
    DECISIONS_API_MAX_TICKETS,
    DECISIONS_CLASSIFIER_MODEL,
    DECISIONS_MAX_INPUT_BYTES,
    DECISIONS_MAX_QUESTIONS,
    INTENT_CRITERIA,
    INTENT_VALUES,
    URGENCY_CRITERIA,
    URGENCY_VALUES,
    _ticket_batch_input,
    _ticket_classification_chunks,
    classify_ticket_batches,
)
from cloudserve_support.config import Settings
from cloudserve_support.ingestion.ticket_normalization import normalize_ticket_record


def test_local_classifier_chunks_at_most_25_tickets_and_maps_failed_chunk_positions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify application chunks are capped at 25 tickets (80 tickets split 25+25+25+5) and failed chunks map correctly."""
    assert CLASSIFICATION_TICKETS_PER_BATCH == 25
    assert DECISIONS_MAX_QUESTIONS == 400
    assert DECISIONS_API_MAX_TICKETS == 200
    assert CLASSIFICATION_QUESTIONS_PER_TICKET == 2

    # Verify 80-ticket workload splits into 25 + 25 + 25 + 5
    tickets = tuple(
        normalize_ticket_record(
            {
                "ticket_id": f"LOCAL-BATCH-{index:04d}",
                "channel": "email",
                "subject": f"API pagination {index}",
                "body": f"Use the cursor to fetch page {index}.",
            },
            input_index=index,
        )
        for index in range(80)
    )

    chunks = _ticket_classification_chunks(tickets)
    assert [start_index for start_index, _ in chunks] == [0, 25, 50, 75]
    assert [len(chunk_tickets) for _, chunk_tickets in chunks] == [25, 25, 25, 5]
    serialized_sizes = [
        len(_ticket_batch_input(chunk_tickets).encode("utf-8"))
        for _, chunk_tickets in chunks
    ]
    assert all(size <= DECISIONS_MAX_INPUT_BYTES for size in serialized_sizes)

    # Also verify 26 tickets split into 25 + 1 boundary
    chunks_26 = _ticket_classification_chunks(tickets[:26])
    assert [start_index for start_index, _ in chunks_26] == [0, 25]
    assert [len(chunk_tickets) for _, chunk_tickets in chunks_26] == [25, 1]

    recorded_requests: list[dict[str, object]] = []

    class LocalDecisionsApi:
        """Return predictable choices and fail the first chunk only."""

        def create(self, **request: object) -> object:
            ticket_fields = json.loads(str(request["input"]))["tickets"]
            ticket_indexes = [int(ticket["subject"].rsplit(" ", 1)[1]) for ticket in ticket_fields]
            recorded_requests.append(
                {
                    "ticket_indexes": ticket_indexes,
                    "question_count": len(request["questions"]),
                    "input_bytes": len(str(request["input"]).encode("utf-8")),
                }
            )
            if ticket_indexes[0] == 0:
                raise RuntimeError("synthetic local chunk failure")

            answers = []
            for local_index, ticket_index in enumerate(ticket_indexes):
                intent = (
                    "api_usage_question"
                    if ticket_index % 2 == 0
                    else "authentication_failure"
                )
                urgency = "low" if ticket_index % 3 == 0 else "medium"
                answers.extend(
                    [
                        SimpleNamespace(
                            type="choice",
                            name=f"ticket_{local_index}_intent",
                            choice=intent,
                            confidence=0.99,
                        ),
                        SimpleNamespace(
                            type="choice",
                            name=f"ticket_{local_index}_urgency",
                            choice=urgency,
                            confidence=0.99,
                        ),
                    ]
                )
            return SimpleNamespace(
                model=DECISIONS_CLASSIFIER_MODEL,
                answers=answers,
            )

    class LocalClient:
        """Supply a deterministic Decisions adapter without a network client."""

        def __init__(self, **options: object) -> None:
            self.decisions = LocalDecisionsApi()

    monkeypatch.setattr(genai, "Client", LocalClient)
    batches = classify_ticket_batches(
        tickets,
        Settings(gemini_api_key="local-test-key"),
    )

    assert [batch.start_index for batch in batches] == [0, 25, 50, 75]
    assert [len(batch.classifications) for batch in batches] == [25, 25, 25, 5]
    assert [len(request["ticket_indexes"]) for request in recorded_requests] == [25, 25, 25, 5]
    assert [request["question_count"] for request in recorded_requests] == [50, 50, 50, 10]
    assert all(request["question_count"] <= DECISIONS_MAX_QUESTIONS for request in recorded_requests)
    assert all(request["input_bytes"] <= DECISIONS_MAX_INPUT_BYTES for request in recorded_requests)

    mapped_classifications = [
        (batch.start_index + offset, classification)
        for batch in batches
        for offset, classification in enumerate(batch.classifications)
    ]
    assert [ticket_index for ticket_index, _ in mapped_classifications] == list(range(80))
    assert all(classification.intent in INTENT_VALUES for _, classification in mapped_classifications)
    assert all(classification.urgency in URGENCY_VALUES for _, classification in mapped_classifications)
    for ticket_index, classification in mapped_classifications:
        if 0 <= ticket_index < 25:
            assert classification.provider_status == "fallback"
            assert classification.intent == "unclear_request"
            assert classification.urgency == "unknown"
        else:
            assert classification.provider_status == "classified"
            expected_intent = (
                "api_usage_question"
                if ticket_index % 2 == 0
                else "authentication_failure"
            )
            expected_urgency = "low" if ticket_index % 3 == 0 else "medium"
            assert classification.intent == expected_intent
            assert classification.urgency == expected_urgency

    assert [request["ticket_indexes"] for request in recorded_requests] == [
        list(range(0, 25)),
        list(range(25, 50)),
        list(range(50, 75)),
        list(range(75, 80)),
    ]


def test_decisions_classifier_splits_before_serialized_input_byte_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Split by UTF-8 input size even when the 200-ticket bound is not reached."""
    tickets = tuple(
        normalize_ticket_record(
            {
                "ticket_id": f"LOCAL-BYTES-{index}",
                "channel": "email",
                "subject": f"Large input {index}",
                "body": "x" * 380_000,
            },
            input_index=index,
        )
        for index in range(3)
    )
    chunks = _ticket_classification_chunks(tickets)

    assert [start_index for start_index, _ in chunks] == [0, 2]
    assert [len(chunk_tickets) for _, chunk_tickets in chunks] == [2, 1]
    assert all(
        len(_ticket_batch_input(chunk_tickets).encode("utf-8"))
        <= DECISIONS_MAX_INPUT_BYTES
        for _, chunk_tickets in chunks
    )

    request_sizes: list[int] = []

    class ByteBoundaryDecisionsApi:
        def create(self, **request: object) -> object:
            ticket_fields = json.loads(str(request["input"]))["tickets"]
            request_sizes.append(len(ticket_fields))
            answers = [
                SimpleNamespace(
                    type="choice",
                    name=name,
                    choice=(
                        "api_usage_question"
                        if name.endswith("_intent")
                        else "low"
                    ),
                    confidence=0.99,
                )
                for name in (request["questions"].keys() if isinstance(request["questions"], dict) else [q["name"] for q in request["questions"]])
            ]
            return SimpleNamespace(model=DECISIONS_CLASSIFIER_MODEL, answers=answers)

    class ByteBoundaryClient:
        def __init__(self, **options: object) -> None:
            self.decisions = ByteBoundaryDecisionsApi()

    monkeypatch.setattr(genai, "Client", ByteBoundaryClient)
    batches = classify_ticket_batches(
        tickets,
        Settings(gemini_api_key="local-test-key"),
    )

    assert [batch.start_index for batch in batches] == [0, 2]
    assert [len(batch.classifications) for batch in batches] == [2, 1]
    assert request_sizes == [2, 1]
    assert all(
        classification.provider_status == "classified"
        for batch in batches
        for classification in batch.classifications
    )
    assert [
        (batch.start_index + offset, classification.intent)
        for batch in batches
        for offset, classification in enumerate(batch.classifications)
    ] == [
        (0, "api_usage_question"),
        (1, "api_usage_question"),
        (2, "api_usage_question"),
    ]


def test_classification_criteria_and_owner_urgency_rubric() -> None:
    """Verify criteria dictionary contracts and owner-confirmed urgency rubric boundaries."""
    assert set(INTENT_CRITERIA.keys()) == INTENT_VALUES
    assert set(URGENCY_CRITERIA.keys()) == URGENCY_VALUES

    # Owner-confirmed urgency rubric boundaries
    high_criterion = URGENCY_CRITERIA["high"]
    assert "critical customer-facing outage" in high_criterion.lower()
    assert "active security incident" in high_criterion.lower()
    assert "account compromise" in high_criterion.lower()
    assert "confirmed data loss or exposure" in high_criterion.lower()
    assert "suspected exposure or compromise of credentials in a breach at another service" in high_criterion.lower()

    medium_criterion = URGENCY_CRITERIA["medium"]
    assert "bounded api key permission or access issues" in medium_criterion.lower()

    low_criterion = URGENCY_CRITERIA["low"]
    assert "routine invoice or billing" in low_criterion.lower()
    assert "routine data-residency" in low_criterion.lower()

    urgency_instructions = CLASSIFICATION_DECISION_QUESTIONS[1]["instructions"]
    assert "routine invoice and data-residency questions are not high" in urgency_instructions.lower()
    assert "suspected exposure or compromise of credentials in a breach at another service" in urgency_instructions.lower()
    assert "even before confirmed misuse" in urgency_instructions.lower()

