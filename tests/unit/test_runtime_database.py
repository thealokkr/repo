"""Checks for ordered, replayable SQLite decision evidence."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from jsonschema import ValidationError
import pytest

from cloudserve_support.classification.ticket_classification import (
    TicketClassification,
    classification_failure,
)
from cloudserve_support.config import Settings
from cloudserve_support.domain.ticket import CanonicalTicket
from cloudserve_support.evaluation.quality_judge import (
    QUALITY_JUDGE_SAMPLE_SEED,
    _build_quality_judge_scorecard,
)
from cloudserve_support.evaluation.run_metrics import evaluate_completed_run
from cloudserve_support.generation.draft_generation import OfflineDraftGenerationProvider
from cloudserve_support.ingestion.ticket_normalization import normalize_ticket_record
from cloudserve_support.knowledge.retrieval import MINIMUM_SIMILARITY, RetrievalResult
from cloudserve_support.persistence.runtime_database import (
    connect_runtime_database,
    create_processing_run,
    create_ticket_audit_records,
    initialize_runtime_database,
    read_run_audit_events,
)
from cloudserve_support.processing.run_provenance import (
    fingerprint_classifier_model,
    fingerprint_policy_file,
)
from cloudserve_support.processing.run_tickets import run_ticket_batch
from cloudserve_support.processing.ticket_outcomes import build_ticket_outcome
from cloudserve_support.reporting.output_schemas import (
    OUTPUT_SCHEMA_DIRECTORY,
    initialize_output_schemas,
    load_output_schemas,
    validate_output_record,
)
from cloudserve_support.reporting.run_exports import (
    write_markdown_run_metrics,
    write_run_exports,
)
from cloudserve_support.routing.routing_policy import RoutingPolicyEvaluator

TEST_ROUTING_POLICY_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "decisions-routing-policy.json"
)


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
        assert json.loads(events[0]["event_payload_json"])["route"] == "draft_for_review"
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
        "classification": {"model_name": "test"},
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
        "cloudserve_support.processing.run_tickets.create_audit_event_record",
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
            policy_path=TEST_ROUTING_POLICY_PATH,
        )

    connection = sqlite3.connect(database_path)
    try:
        run_status = connection.execute(
            "SELECT status FROM processing_runs"
        ).fetchone()
        assert run_status == ("failed",)
        assert connection.execute("SELECT COUNT(*) FROM ticket_runs").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM audit_events").fetchone()[0] == 0
        assert connection.execute(
            "SELECT processing_status FROM ticket_runs"
        ).fetchone() == ("running",)
    finally:
        connection.close()


def _sample_kb_source(doc_id: str, passage_id: str, title: str, text: str, **extra: object) -> dict[str, object]:
    """Build a minimal schema-valid KB passage snapshot for export and DB tests."""
    section = "Overview"
    snapshot: dict[str, object] = {
        "passage_id": passage_id,
        "doc_id": doc_id,
        "title": title,
        "section": section,
        "passage_text": text,
        "similarity": 0.88,
        "minimum_similarity": MINIMUM_SIMILARITY,
        "source_ref": f"data/kb/documentation.json#{doc_id}#{section}",
        "article_version": "1.0",
        "reviewed_at": "2026-11-01",
        "review_interval_days": 90,
        "approval_status": "approved",
        "approval_status_source": "data/kb/metadata_manifest.json",
        "metadata_owner": "capstone-kb-team",
        "metadata_source": "data/kb/metadata_manifest.json",
        "content_sha256": "1" * 64,
        "metadata_manifest_sha256": "2" * 64,
    }
    snapshot.update(extra)
    return snapshot


def _sample_valid_result_row() -> dict[str, Any]:
    policy = json.loads(TEST_ROUTING_POLICY_PATH.read_text(encoding="utf-8"))
    return {
        "run_id": "a" * 32,
        "input_index": 0,
        "ticket_id": "T-SCHEMA-1",
        "channel": "email",
        "subject": "API key rotation",
        "body": "How do I rotate my API key?",
        "expected_intent": "api_key_issue",
        "predicted_intent": "api_key_issue",
        "expected_urgency": "medium",
        "predicted_urgency": "medium",
        "expected_route": "draft_for_review",
        "predicted_route": "draft_for_review",
        "expected_answerable_from_docs": True,
        "predicted_answerable_from_docs": True,
        "expected_doc_ids": ["DOC-API-001"],
        "predicted_doc_ids": ["DOC-API-001"],
        "expected_must_not_auto_respond": False,
        "predicted_must_not_auto_respond": False,
        "status": "completed",
        "reason": "All checks passed.",
        "classifier_confidence": 0.91,
        "classifier_provider": "gemini",
        "classifier_model": "gemini-decision-1",
        "classifier_status": "classified",
        "routing": {
            "policy_version": policy["policy_version"],
            "matched_rule": "classification_confident_evidence_ready",
            "confidence_threshold": policy["classification_confidence_threshold"],
            "threshold_source": policy["threshold_source"],
            "calibration_method": policy["calibration_method"],
            "calibration_source": policy["calibration_source"],
            "policy_frozen": True,
            "policy_provenance": {
                "policy_version": policy["policy_version"],
                "classification_confidence_threshold": policy["classification_confidence_threshold"],
                "threshold_source": policy["threshold_source"],
                "calibration_method": policy["calibration_method"],
                "calibration_source": policy["calibration_source"],
                "frozen": True,
                "policy_file_sha256": "b" * 64,
                "threshold_sha256": "c" * 64,
                "classifier_implementation_sha256": "d" * 64,
                "classifier_model_sha256": "e" * 64,
            },
            "retrieval": {
                "status": "ready",
                "reason_code": "sources_ready",
                "reason": "Approved sources matched.",
                "eligible_candidate_count": 1,
                "excluded_candidate_counts": {},
                "minimum_similarity": MINIMUM_SIMILARITY,
                "embedding_model": "gemini-embedding-001",
                "source_sha256": "f" * 64,
                "metadata_manifest_sha256": "0" * 64,
            },
            "checks": [
                {
                    "name": "protected_rule_check",
                    "status": "passed",
                    "confidence": 1.0,
                    "matched_rule": None,
                    "protected_case": False,
                    "reason": "No protected rule matched.",
                },
                {
                    "name": "classification_confidence_check",
                    "status": "passed",
                    "confidence": 0.91,
                    "threshold": 0.65,
                    "matched_rule": None,
                    "reason": "Confidence meets threshold.",
                },
                {
                    "name": "source_presence_check",
                    "status": "passed",
                    "confidence": 1.0,
                    "retrieved_doc_ids": ["DOC-API-001"],
                    "related_doc_ids": ["DOC-SEC-001"],
                    "excluded_source_count": 0,
                    "matched_rule": None,
                    "reason": "Approved primary source present.",
                },
                {
                    "name": "claim_grounding_check",
                    "status": "passed",
                    "confidence": 0.95,
                    "claims": [{"text": "Rotate keys in Settings.", "source_ids": ["DOC-API-001:p1"]}],
                    "unsupported_points": [],
                    "conflicting_points": [],
                    "matched_rule": None,
                    "reason": "Claims grounded.",
                    "error_code": None,
                },
            ],
            "final_decision": {
                "route": "draft_for_review",
                "matched_rule": "classification_confident_evidence_ready",
                "reason": "All routing policy checks passed.",
                "failed_check": None,
                "protected_case": False,
                "evidence_ready": True,
            },
        },
        "sources": [
            _sample_kb_source("DOC-API-001", "DOC-API-001:p1", "API Keys", "Rotate keys in Settings.", source_kind="primary"),
            _sample_kb_source("DOC-SEC-001", "DOC-SEC-001:p1", "Security", "Use short-lived tokens.", source_kind="related"),
        ],
        "draft": {
            "plain_text": "Rotate keys in Settings.",
            "claims": [
                {
                    "text": "Rotate keys in Settings.",
                    "source_ids": ["DOC-API-001:p1"],
                    "citations": [{"passage_id": "DOC-API-001:p1", "source_ref": "data/kb/documentation.json#DOC-API-001"}],
                }
            ],
            "unsupported_points": [],
            "conflicting_points": [],
            "approval_required": True,
            "send_status": "not_sent",
        },
        "handoff": None,
        "guardrails": {"checks": []},
        "generation": {
            "provider": "offline",
            "model_name": "offline-extractive-v1",
            "reasoning_effort": None,
            "status": "generated",
            "error_code": None,
        },
        "processing_duration_seconds": 0.1,
        "draft_generation_request_duration_seconds": 0.0,
        "processing_latency_seconds": 0.1,
    }


def test_output_schemas_sync_and_validate_policy_checks_and_source_kind() -> None:
    """AC-1: Synchronized result.schema.json validates checks[], final_decision, and source_kind."""
    initialize_output_schemas(OUTPUT_SCHEMA_DIRECTORY)
    schemas = load_output_schemas()
    valid_row = _sample_valid_result_row()
    validate_output_record(valid_row, schema_filename="result.schema.json", schemas=schemas)

    extra_top_level = {**valid_row, "unexpected_field": True}
    with pytest.raises(ValidationError):
        validate_output_record(extra_top_level, schema_filename="result.schema.json", schemas=schemas)

    extra_routing = {**valid_row, "routing": {**valid_row["routing"], "extra_routing_key": 1}}
    with pytest.raises(ValidationError):
        validate_output_record(extra_routing, schema_filename="result.schema.json", schemas=schemas)


def test_result_schema_accepts_optional_pre_routing_checks_and_rejects_malformed_groups() -> None:
    """AC-1/AC-2: result.schema.json accepts optional pre_routing_checks and rejects malformed groups."""
    initialize_output_schemas(OUTPUT_SCHEMA_DIRECTORY)
    schemas = load_output_schemas()

    legacy_row = _sample_valid_result_row()
    assert "pre_routing_checks" not in legacy_row
    validate_output_record(legacy_row, schema_filename="result.schema.json", schemas=schemas)

    data_checks = [
        {
            "name": "classification_confidence_check",
            "status": "passed",
            "confidence": 0.91,
            "threshold": 0.65,
            "matched_rule": None,
            "reason": "Confidence meets threshold.",
        },
        {
            "name": "source_presence_check",
            "status": "passed",
            "confidence": 1.0,
            "retrieved_doc_ids": ["DOC-API-001"],
            "related_doc_ids": ["DOC-SEC-001"],
            "excluded_source_count": 0,
            "matched_rule": None,
            "reason": "Approved primary source present.",
        },
    ]
    safety_checks = [
        {
            "name": "protected_rule_check",
            "status": "passed",
            "confidence": 1.0,
            "matched_rule": None,
            "protected_case": False,
            "reason": "No protected rule matched.",
        },
        {
            "check": "input_safety_guardrail",
            "passed": True,
            "blocked": False,
            "evidence": "No safety risk detected.",
            "action": "allow",
        },
    ]
    grouped_row = {
        **_sample_valid_result_row(),
        "pre_routing_checks": {
            "data_checks": data_checks,
            "safety_checks": safety_checks,
        },
    }
    validate_output_record(grouped_row, schema_filename="result.schema.json", schemas=schemas)
    assert grouped_row["pre_routing_checks"]["data_checks"] == data_checks
    assert grouped_row["pre_routing_checks"]["safety_checks"] == safety_checks

    malformed_cases = [
        {"safety_checks": safety_checks},
        {"data_checks": data_checks},
        {"data_checks": ["scalar-item"], "safety_checks": safety_checks},
        {"data_checks": data_checks, "safety_checks": [123]},
    ]
    for malformed_pre_routing in malformed_cases:
        invalid_row = {
            **_sample_valid_result_row(),
            "pre_routing_checks": malformed_pre_routing,
        }
        with pytest.raises(ValidationError):
            validate_output_record(invalid_row, schema_filename="result.schema.json", schemas=schemas)


def test_ticket_label_comparison_excludes_related_sources_and_preserves_legacy_rows(
    tmp_path: Path,
) -> None:
    """AC-2: ticket_label_comparison excludes related sources while preserving legacy rows."""
    db_path = tmp_path / "runtime.sqlite3"
    connection = connect_runtime_database(db_path)
    try:
        create_processing_run(
            connection,
            run_id="run-compat",
            input_ref="tickets.json",
            started_at="2026-11-10T12:00:00Z",
            label=None,
            output_dir="outputs/run-compat",
        )
        legacy_sources = [
            _sample_kb_source("DOC-LEGACY-1", "DOC-LEGACY-1:p1", "Legacy 1", "Body 1."),
            _sample_kb_source("DOC-LEGACY-2", "DOC-LEGACY-2:p1", "Legacy 2", "Body 2."),
        ]
        split_sources = [
            _sample_kb_source("DOC-PRIMARY-1", "DOC-PRIMARY-1:p1", "Primary", "Primary body.", source_kind="primary"),
            _sample_kb_source("DOC-RELATED-1", "DOC-RELATED-1:p1", "Related", "Related body.", source_kind="related"),
        ]
        for index, (ticket_id, sources) in enumerate(
            [("T-LEGACY", legacy_sources), ("T-SPLIT", split_sources)]
        ):
            create_ticket_audit_records(
                connection,
                run_id="run-compat",
                storage_ticket_id=ticket_id,
                channel="email",
                received_at=None,
                customer_tier=None,
                language_fluency=None,
                context={},
                processing_status="completed",
                error_code=None,
                error_message=None,
                event_id=f"event-{index}",
                occurred_at="2026-11-10T12:00:01Z",
                event_type="ticket_completed",
                input_summary="Question",
                action_taken="draft_for_review",
                reason="Ready.",
                result_payload={
                    "input_index": index,
                    "ticket_id": ticket_id,
                    "result": {
                        "status": "completed",
                        "route": "draft_for_review",
                        "classification": {
                            "provider_status": "classified",
                            "intent": "api_key_issue",
                            "urgency": "medium",
                        },
                        "routing": {"evidence_ready": True, "protected_case": False},
                        "sources": sources,
                    },
                },
            )
        connection.commit()
    finally:
        connection.close()

    # Re-run initializer non-destructively and verify existing rows + view behavior.
    initialize_runtime_database(db_path)
    migrated = connect_runtime_database(db_path)
    try:
        assert migrated.execute("SELECT COUNT(*) FROM processing_runs").fetchone()[0] == 1
        assert migrated.execute("SELECT COUNT(*) FROM ticket_runs").fetchone()[0] == 2
        assert migrated.execute("SELECT COUNT(*) FROM audit_events").fetchone()[0] == 2
        rows = {
            row["ticket_id"]: row
            for row in migrated.execute(
                "SELECT ticket_id, generated_labels_json, generated_labels_status "
                "FROM ticket_label_comparison"
            ).fetchall()
        }
        assert rows["T-LEGACY"]["generated_labels_status"] == "available"
        assert json.loads(rows["T-LEGACY"]["generated_labels_json"])["expected_doc_ids"] == [
            "DOC-LEGACY-1",
            "DOC-LEGACY-2",
        ]
        assert rows["T-SPLIT"]["generated_labels_status"] == "available"
        assert json.loads(rows["T-SPLIT"]["generated_labels_json"])["expected_doc_ids"] == [
            "DOC-PRIMARY-1",
        ]
    finally:
        migrated.close()


def test_write_run_exports_validates_multi_route_results_with_policy_checks_and_source_kind(
    tmp_path: Path,
) -> None:
    """AC-3: write_run_exports validates results, decision_log, metrics, and failed_records."""
    run_id = "b" * 32
    policy = json.loads(TEST_ROUTING_POLICY_PATH.read_text(encoding="utf-8"))
    policy_fingerprint = fingerprint_policy_file(TEST_ROUTING_POLICY_PATH)
    classifier_fingerprint = fingerprint_classifier_model("gemini", "gemini-decision-1")
    settings = Settings.from_env({"DRAFT_GENERATION_PROVIDER": "offline"})
    evaluator = RoutingPolicyEvaluator(
        policy,
        claim_grounding_fn=lambda *, ticket, sources, related_sources: {
            "claims": [
                {
                    "text": "Rotate API keys from the dashboard settings page.",
                    "source_ids": ["DOC-API-001:p1"],
                }
            ],
            "unsupported_points": [],
            "conflicting_points": [],
            "confidence": 0.96,
            "reason": "Primary KB passage supports rotating API keys.",
        },
    )
    generator = OfflineDraftGenerationProvider()

    primary = _sample_kb_source(
        "DOC-API-001",
        "DOC-API-001:p1",
        "API Keys",
        "Rotate API keys from the dashboard settings page.",
    )
    related = _sample_kb_source(
        "DOC-SEC-001",
        "DOC-SEC-001:p1",
        "Token Security",
        "Store API keys in a secrets manager.",
    )
    raw_tickets = [
        {
            "ticket_id": "T-DRAFT",
            "channel": "email",
            "subject": "Rotate API key",
            "body": "How do I rotate API keys from the dashboard settings page?",
        },
        {
            "ticket_id": "T-ESCALATE",
            "channel": "chat",
            "subject": "Security breach",
            "body": "Our account was hacked and unauthorized users accessed data.",
        },
        {
            "ticket_id": "T-DEGRADE",
            "channel": "web_form",
            "subject": "Billing question",
            "body": "Where can I download my invoice?",
        },
    ]
    classifications = [
        TicketClassification(
            intent="api_key_issue",
            urgency="medium",
            confidence=0.92,
            provider="gemini",
            model_name="gemini-decision-1",
            provider_status="classified",
        ),
        TicketClassification(
            intent="security_incident",
            urgency="high",
            confidence=0.95,
            provider="gemini",
            model_name="gemini-decision-1",
            provider_status="classified",
        ),
        classification_failure(
            provider="gemini",
            model_name="gemini-decision-1",
            uncertainty="Classifier request timed out.",
        ),
    ]
    retrievals = [
        RetrievalResult(
            status="ready",
            reason_code="sources_ready",
            reason="Approved primary and related sources matched.",
            sources=(primary,),
            related_sources=(related,),
            eligible_candidate_count=1,
            excluded_candidate_counts={},
            minimum_similarity=MINIMUM_SIMILARITY,
            embedding_model="gemini-embedding-001",
            source_sha256="1" * 64,
            metadata_manifest_sha256="2" * 64,
        ),
        RetrievalResult(
            status="skipped",
            reason_code="protected_case",
            reason="Skipped retrieval for protected ticket.",
            sources=(),
            eligible_candidate_count=0,
            excluded_candidate_counts={},
            minimum_similarity=MINIMUM_SIMILARITY,
            embedding_model="gemini-embedding-001",
            source_sha256="1" * 64,
            metadata_manifest_sha256="2" * 64,
        ),
        RetrievalResult(
            status="skipped",
            reason_code="classification_unavailable",
            reason="Skipped retrieval because classification failed.",
            sources=(),
            eligible_candidate_count=0,
            excluded_candidate_counts={},
            minimum_similarity=MINIMUM_SIMILARITY,
            embedding_model="gemini-embedding-001",
            source_sha256="1" * 64,
            metadata_manifest_sha256="2" * 64,
        ),
    ]

    db_path = tmp_path / "runtime.sqlite3"
    output_dir = tmp_path / "outputs" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = tmp_path / "tickets.json"
    input_path.write_text(json.dumps(raw_tickets), encoding="utf-8")

    connection = connect_runtime_database(db_path)
    try:
        create_processing_run(
            connection,
            run_id=run_id,
            input_ref=str(input_path),
            started_at="2026-11-10T12:00:00Z",
            label=None,
            output_dir=str(output_dir),
        )
        for index, (raw, classification, retrieval) in enumerate(
            zip(raw_tickets, classifications, retrievals, strict=True)
        ):
            ticket = normalize_ticket_record(raw, input_index=index)
            policy_eval = evaluator.evaluate(
                ticket=ticket,
                classification=classification.as_dict(),
                retrieval=retrieval,
            )
            decision = RoutingPolicyEvaluator.to_routing_decision(policy_eval)
            outcome = build_ticket_outcome(
                run_id=run_id,
                input_index=index,
                ticket=ticket,
                classification=classification,
                retrieval=retrieval,
                decision=decision,
                policy_evaluation=policy_eval,
                policy=policy,
                policy_file_fingerprint=policy_fingerprint,
                classifier_implementation_fingerprint=classifier_fingerprint,
                settings=settings,
                generator=generator,
                issues=[],
                duplicate_ticket_id=False,
            )
            outcome["processing_duration_seconds"] = 0.05
            outcome["processing_latency_seconds"] = 0.05
            create_ticket_audit_records(
                connection,
                run_id=run_id,
                storage_ticket_id=ticket.ticket_id,
                channel=ticket.channel,
                received_at=None,
                customer_tier=None,
                language_fluency=None,
                context={},
                processing_status="completed",
                error_code=None,
                error_message=None,
                event_id=f"{index:032x}",
                occurred_at="2026-11-10T12:00:01Z",
                event_type="ticket_completed",
                input_summary=ticket.body,
                action_taken=outcome["route"],
                reason=outcome["reason"],
                result_payload={
                    "input_index": index,
                    "ticket_id": ticket.ticket_id,
                    "result": outcome,
                },
                guardrail_results=outcome["guardrails"],
            )
        connection.commit()

        write_run_exports(
            connection,
            run_id=run_id,
            output_directory=output_dir,
            input_records=raw_tickets,
            input_path=input_path,
            source_ref="external",
        )
        write_markdown_run_metrics(
            connection,
            run_id=run_id,
            output_directory=output_dir,
        )
    finally:
        connection.close()

    schemas = load_output_schemas()
    exported_results = [
        json.loads(line)
        for line in (output_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["routing"]["final_decision"]["route"] for row in exported_results] == [
        "draft_for_review",
        "escalate",
        "degrade",
    ]
    assert [row["predicted_route"] for row in exported_results] == [
        "draft_for_review",
        "escalate",
        None,
    ]
    for row in exported_results:
        validate_output_record(row, schema_filename="result.schema.json", schemas=schemas)
        assert isinstance(row["routing"]["checks"], list) and row["routing"]["checks"]
        assert isinstance(row["routing"]["final_decision"], dict)
    assert [source["source_kind"] for source in exported_results[0]["sources"]] == [
        "primary",
        "related",
    ]

    for line in (output_dir / "decision_log.jsonl").read_text(encoding="utf-8").splitlines():
        validate_output_record(json.loads(line), schema_filename="decision-log.schema.json", schemas=schemas)
    metrics_payload = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    validate_output_record(metrics_payload, schema_filename="metrics.schema.json", schemas=schemas)
    assert {path.name for path in output_dir.iterdir()} == {
        "results.jsonl",
        "decision_log.jsonl",
        "metrics.json",
        "metrics.md",
        "exceptions.jsonl",
        "failure_report.json",
        "failure_report.md",
    }
    for retired in ("failed_records.json", "failed_details.json", "fail_results.md"):
        assert not (output_dir / retired).exists()
    failure_report = json.loads((output_dir / "failure_report.json").read_text(encoding="utf-8"))
    assert failure_report["run"]["run_id"] == run_id
    assert failure_report["run"]["kpi_failed_record_count"] == 0
    assert "failed_records" not in failure_report


def _sample_output_records(tmp_path: Path, run_id: str) -> dict[str, dict[str, Any]]:
    """Build one schema-valid record per output contract for run_id validation tests."""
    result_row = {**_sample_valid_result_row(), "run_id": run_id}
    decision_row = {
        "event_id": "0123456789abcdef0123456789abcdef",
        "run_id": run_id,
        "ticket_occurrence_id": f"{run_id}:0",
        "input_index": 0,
        "ticket_id": "T-SCHEMA-1",
        "occurred_at": "2026-11-17T09:13:00Z",
        "event_type": "classification",
        "actor_type": "system",
        "terminal": False,
        "stage_result": {
            "stage": "classification",
            "status": "classified",
            "value": {"intent": "api_key_issue"},
            "reason": "Classified.",
        },
    }
    input_path = tmp_path / "external_tickets.json"
    input_path.write_text(
        json.dumps([{"ticket_id": "T-SCHEMA-1", "channel": "email", "subject": "API key", "body": "Rotate?"}]),
        encoding="utf-8",
    )
    evaluation = evaluate_completed_run(
        run_id=run_id,
        results=[result_row],
        input_records=json.loads(input_path.read_text(encoding="utf-8")),
        input_path=input_path,
        source_ref="external",
        decision_events=[decision_row],
    )
    metrics_row = {
        "run_id": run_id,
        "total_count": 1,
        "completed_count": 1,
        "failed_count": 0,
        "exception_count": 0,
        "route_counts": {"draft_for_review": 1, "escalate": 0, "degrade": 0},
        "batch_export_duration_seconds": 0.01,
        "evaluation": evaluation,
    }
    quality_judgment_row = {
        "run_id": run_id,
        "ticket_id": "T-SCHEMA-1",
        "input_index": 0,
        "sample_index": 0,
        "judge_model": "gemini-3.8-flash",
        "rubric_version": "v1",
        "prompt_version": "v1",
        "sample_seed": "seed-1",
        "input_identity_sha256": "a" * 64,
        "reference_available": False,
        "status": "evaluated",
        "dimension_scores": {
            "relevance": {"score": 5, "rationale": "Relevant."},
            "factual_correctness": {"score": 5, "rationale": "Accurate."},
            "completeness_actionability": {"score": 5, "rationale": "Complete."},
            "clarity": {"score": 5, "rationale": "Clear."},
        },
        "overall_quality_score": 5.0,
        "citation_judgments": [],
        "unsupported_factual_claims": [],
        "error": None,
    }
    exception_row = {
        "run_id": run_id,
        "input_index": 0,
        "ticket_id": "T-SCHEMA-1",
        "occurred_at": "2026-11-17T09:13:00Z",
        "error_code": "invalid_ticket_record",
        "message": "Malformed record.",
    }
    return {
        "result.schema.json": result_row,
        "decision-log.schema.json": decision_row,
        "metrics.schema.json": metrics_row,
        "quality-judgment.schema.json": quality_judgment_row,
        "exception.schema.json": exception_row,
    }


@pytest.mark.parametrize(
    "run_id",
    [
        "0123456789abcdef0123456789abcdef",
        "single-ticket_2026-11-17_09-13",
    ],
)
@pytest.mark.parametrize(
    "schema_filename",
    [
        "result.schema.json",
        "decision-log.schema.json",
        "metrics.schema.json",
        "quality-judgment.schema.json",
        "exception.schema.json",
    ],
)
def test_output_schemas_accept_uuid_and_safe_folder_run_ids(
    tmp_path: Path,
    run_id: str,
    schema_filename: str,
) -> None:
    """AC-1: Every output schema with run_id accepts both legacy UUIDs and safe folder names."""
    initialize_output_schemas(OUTPUT_SCHEMA_DIRECTORY)
    schemas = load_output_schemas()
    records = _sample_output_records(tmp_path, run_id)
    validate_output_record(records[schema_filename], schema_filename=schema_filename, schemas=schemas)


@pytest.mark.parametrize(
    "unsafe_run_id",
    [
        "bad/run",
        "bad\\run",
        "bad run",
        "bad\trun",
        ".hidden",
        "..",
        "",
    ],
)
@pytest.mark.parametrize(
    "schema_filename",
    [
        "result.schema.json",
        "decision-log.schema.json",
        "metrics.schema.json",
        "quality-judgment.schema.json",
        "exception.schema.json",
    ],
)
def test_output_schemas_reject_unsafe_run_ids(
    tmp_path: Path,
    unsafe_run_id: str,
    schema_filename: str,
) -> None:
    """AC-2: Every output schema rejects path separators, whitespace, and unsafe run_id values."""
    schemas = load_output_schemas()
    records = _sample_output_records(tmp_path, "single-ticket_2026-11-17_09-13")
    invalid_record = {**records[schema_filename], "run_id": unsafe_run_id}
    with pytest.raises(ValidationError):
        validate_output_record(invalid_record, schema_filename=schema_filename, schemas=schemas)


def test_output_schemas_reject_nested_unsafe_run_ids_and_non_uuid_event_id(tmp_path: Path) -> None:
    """AC-2: Nested metrics run_id fields reject unsafe IDs and event_id remains UUID-only."""
    schemas = load_output_schemas()
    records = _sample_output_records(tmp_path, "single-ticket_2026-11-17_09-13")

    metrics_bad_eval = json.loads(json.dumps(records["metrics.schema.json"]))
    metrics_bad_eval["evaluation"]["run_id"] = "bad/eval"
    with pytest.raises(ValidationError):
        validate_output_record(metrics_bad_eval, schema_filename="metrics.schema.json", schemas=schemas)

    metrics_bad_judge = json.loads(json.dumps(records["metrics.schema.json"]))
    scorecard = _build_quality_judge_scorecard(
        run_id="bad judge",
        input_identity_sha256="a" * 64,
        ground_truth_sha256=None,
        sample_seed=QUALITY_JUDGE_SAMPLE_SEED,
        available_draft_count=0,
        selected_ticket_ids=[],
        reference_available_count=0,
        citation_occurrence_count=0,
        judgment_rows=[],
        source_set="external",
        operational_artifact_sha256={
            "results.jsonl": "b" * 64,
            "decision_log.jsonl": "c" * 64,
        },
    )
    metrics_bad_judge["evaluation"]["criteria"]["llm_judge"] = scorecard
    with pytest.raises(ValidationError):
        validate_output_record(metrics_bad_judge, schema_filename="metrics.schema.json", schemas=schemas)

    decision_bad_event_id = {
        **records["decision-log.schema.json"],
        "event_id": "single-ticket_2026-11-17_09-13",
    }
    with pytest.raises(ValidationError):
        validate_output_record(
            decision_bad_event_id,
            schema_filename="decision-log.schema.json",
            schemas=schemas,
        )


def test_ticket_label_comparison_gates_expected_doc_ids_on_evidence_ready(
    tmp_path: Path,
) -> None:
    """AC-1 & AC-2: gate expected_doc_ids on evidence_ready while preserving sources."""
    run_id = "c" * 32
    policy = json.loads(TEST_ROUTING_POLICY_PATH.read_text(encoding="utf-8"))
    policy_fingerprint = fingerprint_policy_file(TEST_ROUTING_POLICY_PATH)
    classifier_fingerprint = fingerprint_classifier_model("gemini", "gemini-decision-1")
    settings = Settings.from_env({"DRAFT_GENERATION_PROVIDER": "offline"})
    generator = OfflineDraftGenerationProvider()

    doc_1 = _sample_kb_source(
        "DOC-API-001",
        "DOC-API-001:p1",
        "API Keys",
        "Rotate API keys from the dashboard settings page.",
    )
    doc_2 = _sample_kb_source(
        "DOC-API-002",
        "DOC-API-002:p1",
        "API Rate Limits",
        "Check API rate limits in workspace settings.",
    )
    related_doc = _sample_kb_source(
        "DOC-SEC-001",
        "DOC-SEC-001:p1",
        "Token Security",
        "Store API keys in a secrets manager.",
    )

    raw_tickets = [
        {
            "ticket_id": "T-NOT-READY",
            "channel": "email",
            "subject": "Rotate API key and custom SLA",
            "body": "How do I rotate API keys and get a custom SLA?",
        },
        {
            "ticket_id": "T-READY",
            "channel": "email",
            "subject": "Rotate API key",
            "body": "How do I rotate API keys from the dashboard settings page?",
        },
    ]
    classification = TicketClassification(
        intent="api_key_issue",
        urgency="medium",
        confidence=0.92,
        provider="gemini",
        model_name="gemini-decision-1",
        provider_status="classified",
    )

    def fake_grounding(*, ticket: dict[str, Any], sources: tuple[Any, ...], related_sources: tuple[Any, ...]) -> dict[str, Any]:
        if ticket["ticket_id"] == "T-NOT-READY":
            return {
                "claims": [],
                "unsupported_points": ["Custom SLA terms are not in the KB."],
                "conflicting_points": [],
                "confidence": 0.90,
                "reason": "Candidate docs do not answer custom SLA question.",
            }
        return {
            "claims": [
                {
                    "text": "Rotate API keys from the dashboard settings page.",
                    "source_ids": ["DOC-API-001:p1"],
                }
            ],
            "unsupported_points": [],
            "conflicting_points": [],
            "confidence": 0.96,
            "reason": "Primary KB passages support the request.",
        }

    evaluator = RoutingPolicyEvaluator(policy, claim_grounding_fn=fake_grounding)
    retrievals = [
        RetrievalResult(
            status="ready",
            reason_code="sources_ready",
            reason="Candidate primary sources matched.",
            sources=(doc_1, doc_2),
            related_sources=(),
            eligible_candidate_count=2,
            excluded_candidate_counts={},
            minimum_similarity=MINIMUM_SIMILARITY,
            embedding_model="gemini-embedding-001",
            source_sha256="1" * 64,
            metadata_manifest_sha256="2" * 64,
        ),
        RetrievalResult(
            status="ready",
            reason_code="sources_ready",
            reason="Approved primary and related sources matched.",
            sources=(doc_2, doc_1),
            related_sources=(related_doc,),
            eligible_candidate_count=2,
            excluded_candidate_counts={},
            minimum_similarity=MINIMUM_SIMILARITY,
            embedding_model="gemini-embedding-001",
            source_sha256="1" * 64,
            metadata_manifest_sha256="2" * 64,
        ),
    ]

    db_path = tmp_path / "runtime.sqlite3"
    output_dir = tmp_path / "outputs" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = tmp_path / "tickets.json"
    input_path.write_text(json.dumps(raw_tickets), encoding="utf-8")

    connection = connect_runtime_database(db_path)
    try:
        create_processing_run(
            connection,
            run_id=run_id,
            input_ref=str(input_path),
            started_at="2026-11-10T12:00:00Z",
            label=None,
            output_dir=str(output_dir),
        )
        for index, (raw, retrieval) in enumerate(zip(raw_tickets, retrievals, strict=True)):
            ticket = normalize_ticket_record(raw, input_index=index)
            policy_eval = evaluator.evaluate(
                ticket=ticket,
                classification=classification.as_dict(),
                retrieval=retrieval,
            )
            decision = RoutingPolicyEvaluator.to_routing_decision(policy_eval)
            outcome = build_ticket_outcome(
                run_id=run_id,
                input_index=index,
                ticket=ticket,
                classification=classification,
                retrieval=retrieval,
                decision=decision,
                policy_evaluation=policy_eval,
                policy=policy,
                policy_file_fingerprint=policy_fingerprint,
                classifier_implementation_fingerprint=classifier_fingerprint,
                settings=settings,
                generator=generator,
                issues=[],
                duplicate_ticket_id=False,
            )
            outcome["processing_duration_seconds"] = 0.05
            outcome["processing_latency_seconds"] = 0.05
            create_ticket_audit_records(
                connection,
                run_id=run_id,
                storage_ticket_id=ticket.ticket_id,
                channel=ticket.channel,
                received_at=None,
                customer_tier=None,
                language_fluency=None,
                context={},
                processing_status="completed",
                error_code=None,
                error_message=None,
                event_id=f"{index + 10:032x}",
                occurred_at="2026-11-10T12:00:01Z",
                event_type="ticket_completed",
                input_summary=ticket.body,
                action_taken=outcome["route"],
                reason=outcome["reason"],
                result_payload={
                    "input_index": index,
                    "ticket_id": ticket.ticket_id,
                    "sources": outcome["sources"],
                    "result": outcome,
                },
                guardrail_results=outcome["guardrails"],
            )
        connection.commit()

        comparison_rows = {
            row["ticket_id"]: row
            for row in connection.execute(
                "SELECT ticket_id, generated_labels_json, generated_labels_status, generated_labels_reason "
                "FROM ticket_label_comparison WHERE run_id = ?",
                (run_id,),
            ).fetchall()
        }
        assert (
            comparison_rows["T-NOT-READY"]["generated_labels_status"] == "available"
        ), comparison_rows["T-NOT-READY"]["generated_labels_reason"]
        not_ready_labels = json.loads(comparison_rows["T-NOT-READY"]["generated_labels_json"])
        assert not_ready_labels["answerable_from_docs"] is False
        assert not_ready_labels["expected_doc_ids"] == []

        assert comparison_rows["T-READY"]["generated_labels_status"] == "available"
        ready_labels = json.loads(comparison_rows["T-READY"]["generated_labels_json"])
        assert ready_labels["answerable_from_docs"] is True
        assert ready_labels["expected_doc_ids"] == ["DOC-API-001", "DOC-API-002"]

        audit_sources_raw = connection.execute(
            "SELECT sources_json FROM audit_events WHERE run_id = ? AND ticket_id = ?",
            (run_id, "T-NOT-READY"),
        ).fetchone()["sources_json"]
        assert [src["doc_id"] for src in json.loads(audit_sources_raw)] == [
            "DOC-API-001",
            "DOC-API-002",
        ]

        write_run_exports(
            connection,
            run_id=run_id,
            output_directory=output_dir,
            input_records=raw_tickets,
            input_path=input_path,
            source_ref="external",
        )
    finally:
        connection.close()

    exported = {
        row["ticket_id"]: row
        for row in (
            json.loads(line)
            for line in (output_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
        )
    }
    assert exported["T-NOT-READY"]["predicted_doc_ids"] == []
    assert [src["doc_id"] for src in exported["T-NOT-READY"]["sources"]] == [
        "DOC-API-001",
        "DOC-API-002",
    ]
    assert exported["T-READY"]["predicted_doc_ids"] == ["DOC-API-001", "DOC-API-002"]


def test_initialize_runtime_database_migrates_ticket_label_comparison_non_destructively(
    tmp_path: Path,
) -> None:
    """AC-3: Re-initializing an existing DB preserves rows and applies the evidence_ready gate."""
    db_path = tmp_path / "legacy-runtime.sqlite3"
    connection = connect_runtime_database(db_path)
    try:
        create_processing_run(
            connection,
            run_id="legacy-run",
            input_ref="tickets.json",
            started_at="2026-11-10T12:00:00Z",
            label=None,
            output_dir="outputs/legacy-run",
        )
        doc_1 = _sample_kb_source("DOC-API-001", "DOC-API-001:p1", "API 1", "Body 1.", source_kind="primary")
        doc_2 = _sample_kb_source("DOC-API-002", "DOC-API-002:p1", "API 2", "Body 2.", source_kind="primary")
        related = _sample_kb_source("DOC-SEC-001", "DOC-SEC-001:p1", "Sec 1", "Body 3.", source_kind="related")
        for index, (ticket_id, evidence_ready, route, sources) in enumerate(
            [
                ("T-LEGACY-FALSE", False, "escalate", [doc_1, doc_2]),
                ("T-LEGACY-TRUE", True, "draft_for_review", [doc_2, doc_1, related]),
            ]
        ):
            create_ticket_audit_records(
                connection,
                run_id="legacy-run",
                storage_ticket_id=ticket_id,
                channel="email",
                received_at=None,
                customer_tier=None,
                language_fluency=None,
                context={},
                processing_status="completed",
                error_code=None,
                error_message=None,
                event_id=f"legacy-event-{index}",
                occurred_at="2026-11-10T12:00:01Z",
                event_type="ticket_completed",
                input_summary="Question",
                action_taken=route,
                reason="Legacy row.",
                result_payload={
                    "input_index": index,
                    "ticket_id": ticket_id,
                    "result": {
                        "status": "completed",
                        "route": route,
                        "classification": {
                            "provider_status": "classified",
                            "intent": "api_key_issue",
                            "urgency": "medium",
                        },
                        "routing": {"evidence_ready": evidence_ready, "protected_case": False},
                        "sources": sources,
                    },
                },
            )
        # Simulate a pre-updated view that lacked the evidence_ready gate.
        connection.execute("DROP VIEW IF EXISTS ticket_label_comparison")
        connection.execute(
            """
            CREATE VIEW ticket_label_comparison AS
            SELECT ticket.run_id,
                   ticket.ticket_id,
                   ticket.input_labels_json,
                   json_object(
                       'intent', json_extract(event.event_payload_json, '$.result.classification.intent'),
                       'urgency', json_extract(event.event_payload_json, '$.result.classification.urgency'),
                       'expected_route', json_extract(event.event_payload_json, '$.result.route'),
                       'answerable_from_docs', json(CASE
                           WHEN json_extract(event.event_payload_json, '$.result.routing.evidence_ready') = 1
                           THEN 'true' ELSE 'false' END),
                       'expected_doc_ids', json(COALESCE((
                           SELECT json_group_array(doc_id)
                           FROM (
                               SELECT DISTINCT json_extract(source.value, '$.doc_id') AS doc_id
                               FROM json_each(event.event_payload_json, '$.result.sources') AS source
                               WHERE COALESCE(json_extract(source.value, '$.source_kind'), 'primary') = 'primary'
                               ORDER BY doc_id
                           )
                       ), '[]')),
                       'must_not_auto_respond', json(CASE
                           WHEN json_extract(event.event_payload_json, '$.result.routing.protected_case') = 1
                           THEN 'true' ELSE 'false' END)
                   ) AS generated_labels_json,
                   'available' AS generated_labels_status,
                   NULL AS generated_labels_reason
            FROM ticket_runs AS ticket
            JOIN audit_events AS event
              ON event.run_id = ticket.run_id
             AND event.ticket_id = ticket.ticket_id
            """
        )
        pre_migration_false = json.loads(
            connection.execute(
                "SELECT generated_labels_json FROM ticket_label_comparison WHERE ticket_id = 'T-LEGACY-FALSE'"
            ).fetchone()["generated_labels_json"]
        )
        assert pre_migration_false["expected_doc_ids"] == ["DOC-API-001", "DOC-API-002"]
        connection.commit()
    finally:
        connection.close()

    initialize_runtime_database(db_path)
    migrated = connect_runtime_database(db_path)
    try:
        assert migrated.execute("SELECT COUNT(*) FROM processing_runs").fetchone()[0] == 1
        assert migrated.execute("SELECT COUNT(*) FROM ticket_runs").fetchone()[0] == 2
        assert migrated.execute("SELECT COUNT(*) FROM audit_events").fetchone()[0] == 2

        rows = {
            row["ticket_id"]: json.loads(row["generated_labels_json"])
            for row in migrated.execute(
                "SELECT ticket_id, generated_labels_json FROM ticket_label_comparison"
            ).fetchall()
        }
        assert rows["T-LEGACY-FALSE"]["expected_doc_ids"] == []
        assert rows["T-LEGACY-TRUE"]["expected_doc_ids"] == ["DOC-API-001", "DOC-API-002"]
    finally:
        migrated.close()

