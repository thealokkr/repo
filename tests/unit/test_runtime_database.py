"""Checks for ordered, replayable SQLite decision evidence."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from cloudserve_support.domain.ticket import CanonicalTicket
from cloudserve_support.ingestion.ticket_normalization import normalize_ticket_record
from cloudserve_support.persistence.runtime_database import (
    connect_runtime_database,
    create_processing_run,
    create_ticket_audit_records,
    read_run_audit_events,
)
from cloudserve_support.processing.run_tickets import run_ticket_batch


def test_run_audit_events_are_chronological_and_replay_decision_evidence(
    tmp_path: Path,
) -> None:
    connection = connect_runtime_database(tmp_path / "runtime.sqlite3")
    try:
        create_processing_run(
            connection,
            run_id="ordered-run",
            input_ref="tickets.json",
            started_at="2026-11-10T12:00:00Z",
            label=None,
            output_dir="outputs/ordered-run",
        )
        for event_id, ticket_id, occurred_at in (
            ("event-late", "T-LATE", "2026-11-10T12:00:02Z"),
            ("event-early", "T-EARLY", "2026-11-10T12:00:01Z"),
        ):
            source = {"passage_id": f"DOC:{ticket_id}", "passage_text": "Evidence."}
            check = {"check": "source_eligibility", "passed": True, "blocked": False}
            route = "draft_for_review"
            create_ticket_audit_records(
                connection,
                run_id="ordered-run",
                storage_ticket_id=ticket_id,
                channel="chat",
                received_at=None,
                customer_tier=None,
                language_fluency=None,
                context={},
                processing_status="completed",
                error_code=None,
                error_message=None,
                event_id=event_id,
                occurred_at=occurred_at,
                event_type="route_recommended",
                input_summary="A support question",
                action_taken=route,
                reason="A supported source was found.",
                result_payload={
                    "route": route,
                    "sources": [source],
                    "guardrails": {"checks": [check]},
                },
                guardrail_results={"checks": [check]},
            )
        connection.commit()

        events = read_run_audit_events(connection, "ordered-run")

        assert [event["event_id"] for event in events] == ["event-early", "event-late"]
        assert [event["action_taken"] for event in events] == ["draft_for_review"] * 2
        assert json.loads(events[0]["sources_json"]) == [
            {"passage_id": "DOC:T-EARLY", "passage_text": "Evidence."}
        ]
        assert json.loads(events[0]["guardrail_results_json"]) == {
            "checks": [{"check": "source_eligibility", "passed": True, "blocked": False}]
        }
        assert json.loads(events[0]["result_payload_json"])["route"] == "draft_for_review"
    finally:
        connection.close()


def test_audit_persistence_failure_rolls_back_and_does_not_return_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_ticket = {
        "ticket_id": "T-PERSISTENCE-FAILURE",
        "channel": "chat",
        "subject": "Question",
        "body": "Please help.",
    }
    ticket: CanonicalTicket = normalize_ticket_record(raw_ticket, input_index=0)
    result = {
        "status": "completed",
        "route": "escalate",
        "reason": "Held for review.",
        "classification": {"model_name": "test", "alternatives": []},
        "routing": {
            "confidence_threshold": 0.5,
            "protected_case": True,
            "matched_rule": "test",
            "evidence_ready": False,
            "retrieval": {},
        },
        "guardrails": {"checks": []},
        "sources": [],
    }
    monkeypatch.setattr(
        "cloudserve_support.processing.run_tickets._build_ticket_result",
        lambda *args, **kwargs: (result, ticket.ticket_id, ticket),
    )

    def fail_audit_insert(*args, **kwargs) -> None:
        raise sqlite3.OperationalError("simulated disk write failure")

    monkeypatch.setattr(
        "cloudserve_support.processing.run_tickets.create_ticket_audit_records",
        fail_audit_insert,
    )
    input_path = tmp_path / "ticket.json"
    input_path.write_text(json.dumps([raw_ticket]), encoding="utf-8")
    database_path = tmp_path / "runtime.sqlite3"

    with pytest.raises(sqlite3.OperationalError, match="simulated disk write failure"):
        run_ticket_batch(
            input_path=input_path,
            output_root=tmp_path / "outputs",
            database_path=database_path,
        )

    connection = sqlite3.connect(database_path)
    try:
        assert connection.execute("SELECT COUNT(*) FROM processing_runs").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM ticket_runs").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM audit_events").fetchone()[0] == 0
    finally:
        connection.close()
