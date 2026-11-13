"""Deterministically verify the CLI persists all bulk retrievals before routing."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from cloudserve_support.classification.ticket_classification import (
    DECISIONS_CLASSIFIER_MODEL,
    TicketClassification,
    TicketClassificationBatch,
)
from cloudserve_support.config import Settings
from cloudserve_support.domain.ticket import CanonicalTicket
from cloudserve_support.generation.draft_generation import OfflineDraftGenerationProvider
from cloudserve_support.knowledge.retrieval import (
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_QUERY_BATCH_SIZE,
    MINIMUM_SIMILARITY,
    RetrievalResult,
)
from cloudserve_support.processing import run_tickets
from cloudserve_support.processing.run_tickets import run_ticket_batch
from cloudserve_support.reporting.output_schemas import initialize_output_schemas


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = REPOSITORY_ROOT / "conf" / "frozen_routing_policy.json"


class DeterministicBulkRetriever:
    """Return one no-source result per query without contacting external services."""

    source_sha256 = "1" * 64
    metadata_manifest_sha256 = "2" * 64

    def __init__(self, events: list[tuple[str, str]]) -> None:
        self.events = events
        self.calls: list[tuple[list[str], int, int]] = []

    def retrieve_many(
        self,
        tickets: list[CanonicalTicket],
        *,
        embedding_batch_size: int,
        query_batch_size: int,
    ) -> tuple[RetrievalResult, ...]:
        self.calls.append(
            (
                [ticket.ticket_id for ticket in tickets],
                embedding_batch_size,
                query_batch_size,
            )
        )
        results = tuple(
            RetrievalResult(
                status="no_hit",
                reason_code="below_relevance_threshold",
                reason="No current eligible passage met the minimum relevance threshold.",
                sources=(),
                eligible_candidate_count=1,
                excluded_candidate_counts={"below_relevance_threshold": 1},
                minimum_similarity=MINIMUM_SIMILARITY,
                embedding_model="gemini-embedding-001",
                source_sha256=self.source_sha256,
                metadata_manifest_sha256=self.metadata_manifest_sha256,
            )
            for _ticket in tickets
        )
        self.events.extend(("retrieval_completed", ticket.ticket_id) for ticket in tickets)
        return results


def test_cli_commits_every_bulk_retrieval_before_first_route(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise bulk dispatch, committed retrieval checkpoints, then real routing."""
    assert DEFAULT_EMBEDDING_BATCH_SIZE == 5
    assert DEFAULT_QUERY_BATCH_SIZE == 5
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("DRAFT_GENERATION_PROVIDER", "OFFLINE")
    schema_directory = tmp_path / "schemas" / "outputs"
    initialize_output_schemas(schema_directory)
    monkeypatch.setattr(
        "cloudserve_support.reporting.output_schemas.OUTPUT_SCHEMA_DIRECTORY",
        schema_directory,
    )
    input_records = [
        {
            "ticket_id": f"BULK-CLI-{index}",
            "channel": "email",
            "subject": f"Subject {index}",
            "body": f"Question about service behavior {index}.",
            "labels": {"expected_doc_ids": ["must not be used"]},
            "history": {"note": "evaluation history must not be used"},
        }
        for index in range(3)
    ]
    input_path = tmp_path / "tickets.json"
    input_path.write_text(json.dumps(input_records), encoding="utf-8")
    database_path = tmp_path / "runtime.sqlite3"
    events: list[tuple[str, str]] = []
    retriever = DeterministicBulkRetriever(events)
    real_router = run_tickets.route_ticket_classification

    def deterministic_classifier(
        tickets: list[CanonicalTicket],
        _settings: Settings,
    ) -> tuple[TicketClassificationBatch, ...]:
        classifications = tuple(
            TicketClassification(
                intent="other",
                urgency="low",
                confidence=0.95,
                provider="gemini",
                model_name=DECISIONS_CLASSIFIER_MODEL,
                provider_status="classified",
            )
            for _ticket in tickets
        )
        return (TicketClassificationBatch(0, classifications, 0.0),)

    def recording_router(*args: Any, **kwargs: Any) -> Any:
        ticket = kwargs["ticket"]
        with sqlite3.connect(database_path) as connection:
            assert connection.execute(
                "SELECT COUNT(*) FROM audit_events WHERE event_type = 'retrieval'"
            ).fetchone()[0] == len(input_records)
        events.append(("route_started", ticket.ticket_id))
        return real_router(*args, **kwargs)

    monkeypatch.setattr(run_tickets, "classify_ticket_batches", deterministic_classifier)
    monkeypatch.setattr(run_tickets, "route_ticket_classification", recording_router)
    run_id, output_directory, _ = run_ticket_batch(
        input_path=input_path,
        output_root=tmp_path / "outputs",
        database_path=database_path,
        policy_path=POLICY_PATH,
        retriever=retriever,  # type: ignore[arg-type]
        generator=OfflineDraftGenerationProvider(),
    )

    assert retriever.calls == [
        ([record["ticket_id"] for record in input_records],
         DEFAULT_EMBEDDING_BATCH_SIZE,
         DEFAULT_QUERY_BATCH_SIZE)
    ]
    assert events[:3] == [
        ("retrieval_completed", record["ticket_id"]) for record in input_records
    ]
    assert events[3:] == [("route_started", record["ticket_id"]) for record in input_records]
    with sqlite3.connect(database_path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM audit_events WHERE run_id = ? AND event_type = 'retrieval'",
            (run_id,),
        ).fetchone()[0] == len(input_records)
    results = [
        json.loads(line)
        for line in (output_directory / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(results) == len(input_records)
    assert all(result["sources"] == [] for result in results)
