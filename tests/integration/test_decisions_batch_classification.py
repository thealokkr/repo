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
    ClassificationOutputError,
    _ticket_batch_input,
    _ticket_batch_questions,
    _ticket_classification_chunks,
    classification_failure,
    classify_ticket_batches,
    parse_decision_classifications,
)
from cloudserve_support.domain.ticket_classification_vocabulary import (
    CLASSIFIER_INTENT_CHOICES,
    CLASSIFIER_URGENCY_CHOICES,
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
    """Verify 22-intent / 3-urgency vocabulary subsets, shared rubric input, and development-aligned criteria."""
    assert len(CLASSIFIER_INTENT_CHOICES) == 22
    assert CLASSIFIER_INTENT_CHOICES < INTENT_VALUES
    assert CLASSIFIER_URGENCY_CHOICES == frozenset({"high", "medium", "low"})
    assert CLASSIFIER_URGENCY_CHOICES < URGENCY_VALUES
    assert "unknown" in URGENCY_VALUES
    assert "unknown" not in CLASSIFIER_URGENCY_CHOICES

    assert set(INTENT_CRITERIA.keys()) == CLASSIFIER_INTENT_CHOICES
    assert set(URGENCY_CRITERIA.keys()) == CLASSIFIER_URGENCY_CHOICES
    assert [choice["value"] for choice in CLASSIFICATION_DECISION_QUESTIONS[0]["choices"]] == sorted(
        CLASSIFIER_INTENT_CHOICES
    )
    assert [choice["value"] for choice in CLASSIFICATION_DECISION_QUESTIONS[1]["choices"]] == sorted(
        CLASSIFIER_URGENCY_CHOICES
    )

    # Feature request vs quota_or_overage distinction
    assert "spend caps" in INTENT_CRITERIA["feature_request"].lower()
    assert "not unimplemented spend-cap features" in INTENT_CRITERIA["quota_or_overage"].lower()

    # Development-aligned urgency rubric boundaries
    high_criterion = URGENCY_CRITERIA["high"].lower()
    assert "rollback_request" in high_criterion
    assert "security_incident" in high_criterion
    assert "publicly exposed production api keys" in high_criterion
    assert "doubled p95" in high_criterion
    assert "blocked production deployments" in high_criterion

    medium_criterion = URGENCY_CRITERIA["medium"].lower()
    assert "unexpected or unfamiliar invoice or overage charges" in medium_criterion
    assert "data_residency" in medium_criterion
    assert "secrets in logs" in medium_criterion
    assert "unclear_request" in medium_criterion

    low_criterion = URGENCY_CRITERIA["low"].lower()
    assert "onboarding" in low_criterion
    assert "api pagination or rate-limit inquiries" in low_criterion
    assert "routine billing-period or payment-method" in low_criterion

    urgency_instructions = CLASSIFICATION_DECISION_QUESTIONS[1]["instructions"].lower()
    assert "routine invoice and data-residency questions are not high" in urgency_instructions
    assert "suspected exposure or compromise of credentials in a breach at another service" in urgency_instructions
    assert "even before confirmed misuse" in urgency_instructions


def test_25_ticket_batch_request_uses_22_intents_3_urgencies_and_shared_rubric() -> None:
    """AC-1: Verify 25-ticket request presents 22 intent and 3 urgency choices with shared rubric in input."""
    tickets = tuple(
        normalize_ticket_record(
            {
                "ticket_id": f"AC1-{index:02d}",
                "channel": "email",
                "subject": f"Subject {index}",
                "body": f"Body {index}",
            },
            input_index=index,
        )
        for index in range(25)
    )
    batch_input_raw = _ticket_batch_input(tickets)
    batch_input = json.loads(batch_input_raw)
    questions = _ticket_batch_questions(len(tickets))

    assert len(batch_input_raw.encode("utf-8")) <= DECISIONS_MAX_INPUT_BYTES
    assert len(questions) == 50
    assert len(questions) <= DECISIONS_MAX_QUESTIONS
    assert len(json.dumps(questions).encode("utf-8")) < 100_000

    assert set(batch_input["intent_rubric"].keys()) == CLASSIFIER_INTENT_CHOICES
    assert set(batch_input["urgency_rubric"].keys()) == CLASSIFIER_URGENCY_CHOICES
    assert len(batch_input["tickets"]) == 25

    for ticket_index in range(25):
        intent_q = questions[f"ticket_{ticket_index}_intent"]
        urgency_q = questions[f"ticket_{ticket_index}_urgency"]
        assert set(intent_q["criteria"].keys()) == CLASSIFIER_INTENT_CHOICES
        assert len(intent_q["criteria"]) == 22
        assert set(urgency_q["criteria"].keys()) == CLASSIFIER_URGENCY_CHOICES
        assert "unknown" not in urgency_q["criteria"]


def test_parse_and_batch_reject_unknown_urgency_and_umbrella_intents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC-2: Reject non-choice unknown urgency or umbrella intents while preserving fallback."""
    valid_decision = SimpleNamespace(
        model=DECISIONS_CLASSIFIER_MODEL,
        answers=[
            SimpleNamespace(type="choice", name="ticket_0_intent", choice="feature_request", confidence=0.91),
            SimpleNamespace(type="choice", name="ticket_0_urgency", choice="low", confidence=0.89),
        ],
    )
    parsed = parse_decision_classifications(valid_decision, ticket_count=1)
    assert len(parsed) == 1
    assert parsed[0].intent == "feature_request"
    assert parsed[0].urgency == "low"
    assert parsed[0].provider_status == "classified"

    unknown_urgency_decision = SimpleNamespace(
        model=DECISIONS_CLASSIFIER_MODEL,
        answers=[
            SimpleNamespace(type="choice", name="ticket_0_intent", choice="configuration_help", confidence=0.9),
            SimpleNamespace(type="choice", name="ticket_0_urgency", choice="unknown", confidence=0.9),
        ],
    )
    with pytest.raises(ClassificationOutputError):
        parse_decision_classifications(unknown_urgency_decision, ticket_count=1)

    umbrella_intent_decision = SimpleNamespace(
        model=DECISIONS_CLASSIFIER_MODEL,
        answers=[
            SimpleNamespace(type="choice", name="ticket_0_intent", choice="billing", confidence=0.9),
            SimpleNamespace(type="choice", name="ticket_0_urgency", choice="medium", confidence=0.9),
        ],
    )
    with pytest.raises(ClassificationOutputError):
        parse_decision_classifications(umbrella_intent_decision, ticket_count=1)

    class BadChoiceDecisionsApi:
        def create(self, **request: object) -> object:
            return umbrella_intent_decision

    class BadChoiceClient:
        def __init__(self, **options: object) -> None:
            self.decisions = BadChoiceDecisionsApi()

    monkeypatch.setattr(genai, "Client", BadChoiceClient)
    ticket = normalize_ticket_record(
        {
            "ticket_id": "AC2-FALLBACK",
            "channel": "email",
            "subject": "Invoice question",
            "body": "Why is this month's invoice higher?",
        },
        input_index=0,
    )
    batches = classify_ticket_batches((ticket,), Settings(gemini_api_key="local-test-key"))
    assert len(batches) == 1
    assert batches[0].classifications[0].provider_status == "fallback"
    assert batches[0].classifications[0].intent == "unclear_request"
    assert batches[0].classifications[0].urgency == "unknown"

    fallback = classification_failure(uncertainty="provider unavailable")
    assert fallback.intent == "unclear_request"
    assert fallback.urgency == "unknown"
    assert fallback.confidence == 0.0
    assert fallback.provider_status == "fallback"

