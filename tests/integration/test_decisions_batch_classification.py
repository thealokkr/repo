"""Live Decisions API test for the real batch classifier."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from dotenv import dotenv_values

from cloudserve_support.classification.ticket_classification import (
    CLASSIFICATION_TICKETS_PER_BATCH,
    DECISIONS_CLASSIFIER_MODEL,
    INTENT_VALUES,
    URGENCY_VALUES,
    classify_ticket_batches,
)
from cloudserve_support.config import Settings
from cloudserve_support.ingestion.ticket_normalization import normalize_ticket_record


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_real_decisions_classifier_chunks_and_returns_201_tickets() -> None:
    """Send 201 synthetic tickets through two real Decisions API requests."""
    api_key = os.environ.get("OPENAI_API_KEY") or dotenv_values(
        REPOSITORY_ROOT / ".env"
    ).get("OPENAI_API_KEY")
    if not isinstance(api_key, str) or not api_key.strip():
        pytest.fail("Set OPENAI_API_KEY in the environment or ignored .env file.")
    settings = Settings(openai_api_key=api_key.strip())

    tickets = tuple(
        normalize_ticket_record(
            {
                "ticket_id": f"LIVE-BATCH-{index:03d}",
                "channel": "email",
                "subject": (
                    f"API pagination example {index}"
                    if index != CLASSIFICATION_TICKETS_PER_BATCH
                    else "Unable to sign in example 200"
                ),
                "body": (
                    f"Example {index}: how can I use the API cursor to fetch the next page?"
                    if index != CLASSIFICATION_TICKETS_PER_BATCH
                    else "Example 200: login rejects valid credentials for every employee."
                ),
            },
            input_index=index,
        )
        for index in range(CLASSIFICATION_TICKETS_PER_BATCH + 1)
    )

    batches = classify_ticket_batches(tickets, settings)

    assert [batch.start_index for batch in batches] == [0, CLASSIFICATION_TICKETS_PER_BATCH]
    assert [len(batch.classifications) for batch in batches] == [
        CLASSIFICATION_TICKETS_PER_BATCH,
        1,
    ]
    classifications = tuple(
        result for batch in batches for result in batch.classifications
    )
    assert len(classifications) == len(tickets)
    assert all(result.provider_status == "classified" for result in classifications)
    assert all(result.model_name == DECISIONS_CLASSIFIER_MODEL for result in classifications)
    assert all(result.intent in INTENT_VALUES for result in classifications)
    assert all(result.urgency in URGENCY_VALUES for result in classifications)
    assert all(batch.request_duration_seconds > 0 for batch in batches)

    print(
        json.dumps(
            {
                "ticket_count": len(classifications),
                "batch_sizes": [len(batch.classifications) for batch in batches],
                "first_result": classifications[0].as_dict(),
                "last_result": classifications[-1].as_dict(),
            },
            sort_keys=True,
        )
    )
