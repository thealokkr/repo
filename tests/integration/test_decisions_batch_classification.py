"""Deterministic coverage for multi-request Decisions classifier chunking."""

from __future__ import annotations

import json
from types import SimpleNamespace

import openai
import pytest

from cloudserve_support.classification.ticket_classification import (
    DECISIONS_CLASSIFIER_MODEL,
    DECISIONS_MAX_INPUT_BYTES,
    INTENT_VALUES,
    URGENCY_VALUES,
    _ticket_batch_input,
    _ticket_classification_chunks,
    classify_ticket_batches,
)
from cloudserve_support.config import Settings
from cloudserve_support.ingestion.ticket_normalization import normalize_ticket_record


def test_local_classifier_chunks_201_tickets_and_maps_failed_chunk_positions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check the 200-ticket boundary and failed-chunk mapping without network calls."""
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
        for index in range(201)
    )

    chunks = _ticket_classification_chunks(tickets)
    assert [start_index for start_index, _ in chunks] == [0, 200]
    assert [len(chunk_tickets) for _, chunk_tickets in chunks] == [200, 1]
    serialized_sizes = [
        len(_ticket_batch_input(chunk_tickets).encode("utf-8"))
        for _, chunk_tickets in chunks
    ]
    assert all(size <= DECISIONS_MAX_INPUT_BYTES for size in serialized_sizes)

    recorded_requests: list[dict[str, object]] = []

    class LocalDecisionsApi:
        """Return predictable choices and fail the third local request only."""

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

    class LocalOpenAIClient:
        """Supply a deterministic Decisions adapter without a network client."""

        def __init__(self, **options: object) -> None:
            assert options["max_retries"] == 0
            self.decisions = LocalDecisionsApi()

    monkeypatch.setattr(openai, "OpenAI", LocalOpenAIClient)
    batches = classify_ticket_batches(
        tickets,
        Settings(openai_api_key="local-test-key"),
    )

    assert [batch.start_index for batch in batches] == [0, 200]
    assert [len(batch.classifications) for batch in batches] == [200, 1]
    assert [len(request["ticket_indexes"]) for request in recorded_requests] == [200, 1]
    assert [request["question_count"] for request in recorded_requests] == [400, 2]
    assert all(request["input_bytes"] <= DECISIONS_MAX_INPUT_BYTES for request in recorded_requests)

    mapped_classifications = [
        (batch.start_index + offset, classification)
        for batch in batches
        for offset, classification in enumerate(batch.classifications)
    ]
    assert [ticket_index for ticket_index, _ in mapped_classifications] == list(range(201))
    assert all(classification.intent in INTENT_VALUES for _, classification in mapped_classifications)
    assert all(classification.urgency in URGENCY_VALUES for _, classification in mapped_classifications)
    for ticket_index, classification in mapped_classifications:
        if 0 <= ticket_index < 200:
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
        list(range(0, 200)),
        [200],
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
                    name=question["name"],
                    choice=(
                        "api_usage_question"
                        if question["name"].endswith("_intent")
                        else "low"
                    ),
                    confidence=0.99,
                )
                for question in request["questions"]
            ]
            return SimpleNamespace(model=DECISIONS_CLASSIFIER_MODEL, answers=answers)

    class ByteBoundaryOpenAIClient:
        def __init__(self, **options: object) -> None:
            assert options["max_retries"] == 0
            self.decisions = ByteBoundaryDecisionsApi()

    monkeypatch.setattr(openai, "OpenAI", ByteBoundaryOpenAIClient)
    batches = classify_ticket_batches(
        tickets,
        Settings(openai_api_key="local-test-key"),
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
