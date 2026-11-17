"""Live five-ticket proof that retrieval completes before routing begins."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest
from dotenv import dotenv_values

from cloudserve_support.config import Settings
from cloudserve_support.generation.draft_generation import create_draft_generation_provider
from cloudserve_support.guardrails.draft_safety import (
    contains_private_data,
    contains_prompt_injection,
    filter_evaluation_context,
    ticket_text_for_generation,
)
from cloudserve_support.ingestion.ticket_normalization import normalize_ticket_record
from cloudserve_support.knowledge.retrieval import KnowledgeRetriever
from cloudserve_support.processing import run_tickets
from cloudserve_support.processing.run_tickets import run_ticket_batch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEVELOPMENT_INPUT = REPOSITORY_ROOT / "data" / "tickets" / "development_tickets.json"


class RecordingLiveRetriever:
    """Record retrieval completion after delegating to the real retriever."""

    def __init__(self, delegate: KnowledgeRetriever, events: list[tuple[str, str]]) -> None:
        self.delegate = delegate
        self.events = events

    def __getattr__(self, name: str) -> object:
        return getattr(self.delegate, name)

    def retrieve(self, ticket: object) -> object:
        result = self.delegate.retrieve(ticket)
        self.events.append(("retrieval_completed", ticket.ticket_id))
        return result


@pytest.mark.parametrize("ticket_count", [5])
def test_live_run_persists_stage_checkpoints_before_routing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ticket_count: int,
) -> None:
    """Run the tracer through live Decisions, embeddings, Chroma, and drafting."""
    env = dotenv_values(REPOSITORY_ROOT / ".env")
    api_key = env.get("GEMINI_API_KEY") or env.get("GOOGLE_API_KEY")
    if not isinstance(api_key, str) or not api_key.strip():
        pytest.fail("Set GEMINI_API_KEY or GOOGLE_API_KEY in the environment or ignored .env file.")
    monkeypatch.setenv("GEMINI_API_KEY", api_key.strip())
    monkeypatch.setenv("DRAFT_GENERATION_PROVIDER", "GEMINI")
    settings = Settings.from_env()

    source_bytes = DEVELOPMENT_INPUT.read_bytes()
    source_records = json.loads(source_bytes)
    selected_records = source_records[:ticket_count]
    selected_ids = [record["ticket_id"] for record in selected_records]
    assert selected_ids == [f"DEV-{index:04d}" for index in range(1, ticket_count + 1)]

    operational_records = [
        {key: value for key, value in record.items() if key not in {"labels", "history"}}
        for record in selected_records
    ]
    prepared = [
        normalize_ticket_record(record, input_index=index)
        for index, record in enumerate(operational_records)
    ]
    assert all(not ticket.validation_errors for ticket in prepared)
    assert all(
        not contains_prompt_injection(ticket_text_for_generation(ticket))
        and not contains_private_data(ticket_text_for_generation(ticket))
        and not contains_private_data(
            json.dumps(filter_evaluation_context(ticket.context), ensure_ascii=False)
        )
        for ticket in prepared
    )

    sanitized_input_sha256 = hashlib.sha256(
        json.dumps(
            operational_records,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    input_path = tmp_path / "tickets-phase-order.json"
    input_path.write_text(json.dumps(operational_records), encoding="utf-8")

    real_retriever = KnowledgeRetriever(settings)
    events: list[tuple[str, str]] = []
    recording_retriever = RecordingLiveRetriever(real_retriever, events)
    real_router = run_tickets.route_ticket_classification
    database_path = tmp_path / "runtime.sqlite3"

    def recording_router(*args: object, **kwargs: object) -> object:
        ticket = kwargs.get("ticket")
        assert ticket is not None
        with sqlite3.connect(database_path) as connection:
            assert connection.execute(
                "SELECT COUNT(*) FROM audit_events WHERE event_type = 'retrieval'"
            ).fetchone()[0] == len(selected_ids)
        events.append(("route_started", ticket.ticket_id))
        return real_router(*args, **kwargs)

    monkeypatch.setattr(run_tickets, "route_ticket_classification", recording_router)
    run_id, output_directory, _ = run_ticket_batch(
        input_path=input_path,
        output_root=tmp_path / "outputs",
        database_path=database_path,
        retriever=recording_retriever,
        generator=create_draft_generation_provider(settings),
    )

    retrieval_events = [event for event in events if event[0] == "retrieval_completed"]
    route_events = [event for event in events if event[0] == "route_started"]
    assert [ticket_id for _, ticket_id in retrieval_events] == selected_ids
    assert [ticket_id for _, ticket_id in route_events] == selected_ids
    first_route_position = events.index(route_events[0])
    assert first_route_position == len(selected_ids)
    assert events[:first_route_position] == retrieval_events

    results = [
        json.loads(line)
        for line in (output_directory / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    decisions = [
        json.loads(line)
        for line in (output_directory / "decision_log.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    decision_by_index = {
        decision["input_index"]: decision["result"]
        for decision in decisions
        if decision["terminal"]
    }
    assert len(results) == ticket_count
    assert [result["ticket_id"] for result in results] == selected_ids
    assert all(result["classifier_status"] == "classified" for result in results)
    assert results[2]["predicted_route"] == "escalate"
    assert decision_by_index[2]["routing"]["protected_case"] is True
    expired_doc_id = "DOC-DEPLOY-002"
    assert real_retriever.article_metadata[expired_doc_id]["approval_status"] == "expired"
    expired_passage_count = sum(
        passage["metadata"]["doc_id"] == expired_doc_id
        for passage in real_retriever.expected_passages.values()
    )
    dev_0004 = results[3]
    assert all(
        source["doc_id"] != expired_doc_id for source in dev_0004["sources"]
    ), "DEV-0004 must not cite its expired source article."
    assert (
        dev_0004["routing"]["retrieval"]["excluded_candidate_counts"].get(
            "approval_status_expired", 0
        )
        == expired_passage_count
        > 0
    ), "All DOC-DEPLOY-002 passages must be excluded as expired."
    assert all(
        source["approval_status"] == "approved"
        and source["doc_id"] != expired_doc_id
        for result in results
        for source in result["sources"]
    ), "Returned sources must be approved and must exclude DOC-DEPLOY-002."
    assert any(result["sources"] for result in results if result["ticket_id"] != "DEV-0004"), (
        "At least one non-DEV-0004 ticket must provide eligible-source evidence."
    )
    assert len(decisions) == ticket_count * 5
    assert [decision["event_type"] for decision in decisions] == [
        *["classification"] * ticket_count,
        *["retrieval"] * ticket_count,
        *["routing"] * ticket_count,
        *[
            event_type
            for result in results
            for event_type in (
                "generation",
                "ticket_failed" if result["status"] == "failed" else "ticket_completed",
            )
        ],
    ]
    assert [decision["terminal"] for decision in decisions] == (
        [False] * (ticket_count * 3)
        + [value for _ in selected_ids for value in (False, True)]
    )
    assert all("action_taken" not in decision for decision in decisions if not decision["terminal"])

    result_by_index = {result["input_index"]: result for result in results}
    connection = sqlite3.connect(database_path)
    try:
        run_row = connection.execute(
            "SELECT status FROM processing_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        database_ticket_count = connection.execute(
            "SELECT COUNT(*) FROM ticket_runs WHERE run_id = ?", (run_id,)
        ).fetchone()[0]
        audit_rows = connection.execute(
            "SELECT ticket_id, event_type, sources_json, event_payload_json FROM audit_events "
            "WHERE run_id = ? ORDER BY rowid",
            (run_id,),
        ).fetchall()
    finally:
        connection.close()
    assert run_row == ("completed",)
    assert database_ticket_count == len(results)
    assert len(audit_rows) == len(results) * 5
    assert sum(event_type in {"ticket_completed", "ticket_failed"} for _, event_type, _, _ in audit_rows) == len(results)
    for _, event_type, sources_json, payload_json in audit_rows:
        persisted_sources = json.loads(sources_json)
        persisted_result = json.loads(payload_json)
        if event_type == "retrieval":
            assert "passage_text" not in payload_json
            assert persisted_sources == []
            continue
        if event_type not in {"ticket_completed", "ticket_failed"}:
            continue
        input_index = persisted_result["input_index"]
        result_sources = result_by_index[input_index]["sources"]
        assert persisted_result["result"]["sources"] == result_sources
        assert decision_by_index[input_index]["sources"] == result_sources
        assert persisted_sources == result_sources
        for source in result_sources:
            passage = real_retriever.expected_passages[source["passage_id"]]
            canonical_text = passage["document"].split("\n\n", 1)[1]
            assert source["passage_text"] == canonical_text

    print(
        json.dumps(
            {
                "run_id": run_id,
                "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
                "selected_ticket_ids": selected_ids,
                "sanitized_operational_input_sha256": sanitized_input_sha256,
                "retrieval_events_before_first_route": first_route_position,
                "ticket_count": len(results),
                "database_ticket_count": database_ticket_count,
                "database_audit_event_count": len(audit_rows),
            },
            sort_keys=True,
        )
    )
