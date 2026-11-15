"""Ensure evaluation-only labels affect reports without changing decisions."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from cloudserve_support.knowledge.retrieval import RetrievalResult
from cloudserve_support.processing.run_tickets import run_ticket_batch

TEST_ROUTING_POLICY_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "decisions-routing-policy.json"
)


class _NoEvidenceRetriever:
    """Return no KB evidence without contacting a provider."""

    def retrieve(self, ticket: object) -> RetrievalResult:
        del ticket
        return RetrievalResult(
            status="unavailable",
            reason_code="test_no_evidence",
            reason="No source evidence is available in this test.",
            sources=(),
            eligible_candidate_count=0,
            excluded_candidate_counts={},
            minimum_similarity=0.45,
            embedding_model="offline-test",
            source_sha256=None,
            metadata_manifest_sha256=None,
        )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_labels_and_history_change_only_post_run_evaluation(
    tmp_path: Path,
) -> None:
    base_record = {
        "ticket_id": "BOUNDARY-1",
        "channel": "chat",
        "subject": "API key stopped working",
        "body": "My API key started returning an authentication error.",
        "customer_tier": "business",
        "language_fluency": "fluent",
    }
    first_record = {
        **base_record,
        "labels": {"intent": "fixture-label-a", "expected_route": "escalate"},
        "history": {"first_contact_resolution": False, "csat_rating": 2},
    }
    changed_record = {
        **base_record,
        "labels": {"intent": "billing", "expected_route": "auto_respond"},
        "history": {"first_contact_resolution": True, "csat_rating": 5},
    }
    first_input = tmp_path / "first.json"
    changed_input = tmp_path / "changed.json"
    first_input.write_text(json.dumps([first_record]), encoding="utf-8")
    changed_input.write_text(json.dumps([changed_record]), encoding="utf-8")
    database_path = tmp_path / "runtime.sqlite3"
    output_root = tmp_path / "outputs"

    _, first_output, _ = run_ticket_batch(
        input_path=first_input,
        output_root=output_root,
        database_path=database_path,
        retriever=_NoEvidenceRetriever(),
        policy_path=TEST_ROUTING_POLICY_PATH,
    )
    _, changed_output, _ = run_ticket_batch(
        input_path=changed_input,
        output_root=output_root,
        database_path=database_path,
        retriever=_NoEvidenceRetriever(),
        policy_path=TEST_ROUTING_POLICY_PATH,
    )
    first_result = _read_jsonl(first_output / "results.jsonl")[0]
    changed_result = _read_jsonl(changed_output / "results.jsonl")[0]
    first_result.pop("run_id")
    changed_result.pop("run_id")
    first_result.pop("processing_duration_seconds")
    changed_result.pop("processing_duration_seconds")
    first_result.pop("draft_generation_request_duration_seconds")
    changed_result.pop("draft_generation_request_duration_seconds")
    first_result.pop("processing_latency_seconds")
    changed_result.pop("processing_latency_seconds")
    for result in (first_result, changed_result):
        for field in (
            "expected_intent",
            "expected_urgency",
            "expected_route",
            "expected_answerable_from_docs",
            "expected_doc_ids",
            "expected_must_not_auto_respond",
        ):
            result.pop(field)

    assert changed_result == first_result
    assert first_result["predicted_route"] == "escalate"
    first_metrics = json.loads((first_output / "metrics.json").read_text(encoding="utf-8"))
    changed_metrics = json.loads((changed_output / "metrics.json").read_text(encoding="utf-8"))
    assert first_metrics["evaluation"]["run_observed"]["technical"]["route_accuracy"]["value"] == 1
    assert changed_metrics["evaluation"]["run_observed"]["technical"]["route_accuracy"]["value"] == 0
    assert first_metrics["evaluation"]["historical_baseline"]["overall"]["csat_mean"]["value"] == 2
    assert changed_metrics["evaluation"]["historical_baseline"]["overall"]["csat_mean"]["value"] == 5
    for output in (first_output, changed_output):
        result_text = (output / "results.jsonl").read_text(encoding="utf-8")
        decision_text = (output / "decision_log.jsonl").read_text(encoding="utf-8")
        assert '"labels"' not in result_text + decision_text
        assert '"history"' not in result_text + decision_text
        assert "billing" not in decision_text

    with sqlite3.connect(database_path) as connection:
        persisted_rows = connection.execute(
            "SELECT context_json, event_payload_json FROM ticket_runs "
            "JOIN audit_events USING (run_id, ticket_id)"
        ).fetchall()
    for context_json, event_payload_json in persisted_rows:
        assert "labels" not in context_json
        assert "history" not in context_json
        assert "fixture-label-a" not in event_payload_json
        assert "billing" not in event_payload_json
