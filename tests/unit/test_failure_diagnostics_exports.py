"""Tests for per-run failure_report.json and failure_report.md diagnostic exports."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
import re
from typing import Any

from cloudserve_support.classification.ticket_classification import (
    TicketClassification,
    classification_failure,
)
from cloudserve_support.config import Settings
from cloudserve_support.domain.ticket import CanonicalTicket
from cloudserve_support.generation.draft_generation import (
    DraftGenerationError,
    OfflineDraftGenerationProvider,
)
from cloudserve_support.ingestion.ticket_normalization import normalize_ticket_record
from cloudserve_support.knowledge.retrieval import MINIMUM_SIMILARITY, RetrievalResult
from cloudserve_support.persistence.runtime_database import (
    connect_runtime_database,
    create_processing_run,
    create_ticket_audit_records,
)
from cloudserve_support.processing.run_provenance import (
    fingerprint_classifier_model,
    fingerprint_policy_file,
)
from cloudserve_support.processing.ticket_outcomes import build_ticket_outcome
from cloudserve_support.reporting.run_exports import write_run_exports
from cloudserve_support.routing.routing_policy import RoutingPolicyEvaluator

TEST_ROUTING_POLICY_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "decisions-routing-policy.json"
)


class _GuardrailTriggeringDraftProvider:
    """Test boundary provider returning a draft whose body triggers private_data_output."""

    name = "gemini"

    def generate_draft(
        self,
        ticket: CanonicalTicket,
        *,
        sources: tuple[dict[str, Any], ...],
        related_sources: tuple[dict[str, Any], ...] = (),
        policy_evaluation: Mapping[str, Any] | None = None,
        model_name: str = "",
    ) -> dict[str, Any]:
        draft = OfflineDraftGenerationProvider().generate_draft(
            ticket,
            sources=sources,
            related_sources=related_sources,
            policy_evaluation=policy_evaluation,
            model_name=model_name,
        )
        draft["plain_text"] = (
            f"{draft['plain_text']} Please email security@example.com for help."
        )
        return draft


class _FailingDraftProvider:
    """Test boundary provider simulating a Gemini draft generation failure."""

    name = "gemini"

    def generate_draft(
        self,
        ticket: CanonicalTicket,
        *,
        sources: tuple[dict[str, Any], ...],
        related_sources: tuple[dict[str, Any], ...] = (),
        policy_evaluation: Mapping[str, Any] | None = None,
        model_name: str = "",
    ) -> dict[str, Any]:
        del ticket, sources, related_sources, policy_evaluation, model_name
        raise DraftGenerationError("generation_timeout", "Gemini draft request timed out.")


def _sample_kb_source(doc_id: str, passage_id: str, title: str, text: str) -> dict[str, Any]:
    section = "Overview"
    return {
        "passage_id": passage_id,
        "doc_id": doc_id,
        "title": title,
        "section": section,
        "passage_text": text,
        "similarity": 0.91,
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


def _persist_ticket_outcome(
    connection: Any,
    *,
    run_id: str,
    input_index: int,
    raw_ticket: dict[str, Any],
    classification: TicketClassification,
    retrieval: RetrievalResult,
    evaluator: RoutingPolicyEvaluator,
    policy: dict[str, Any],
    policy_fingerprint: str,
    classifier_fingerprint: str,
    settings: Settings,
    generator: Any,
) -> None:
    ticket = normalize_ticket_record(raw_ticket, input_index=input_index)
    policy_eval = evaluator.evaluate(
        ticket=ticket,
        classification=classification.as_dict(),
        retrieval=retrieval,
    )
    decision = RoutingPolicyEvaluator.to_routing_decision(policy_eval)
    outcome = build_ticket_outcome(
        run_id=run_id,
        input_index=input_index,
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
        event_id=f"{input_index:032x}",
        occurred_at="2026-11-10T12:00:01Z",
        event_type="ticket_completed",
        input_summary=ticket.body,
        action_taken=outcome["route"],
        reason=outcome["reason"],
        result_payload={
            "input_index": input_index,
            "ticket_id": ticket.ticket_id,
            "result": outcome,
        },
        guardrail_results=outcome["guardrails"],
    )
    if raw_ticket.get("labels") is not None:
        connection.execute(
            "UPDATE ticket_runs SET input_labels_json = ? WHERE run_id = ? AND ticket_id = ?",
            (json.dumps(raw_ticket["labels"]), run_id, ticket.ticket_id),
        )


def test_write_run_exports_writes_consolidated_failure_report_and_retires_legacy_files(
    tmp_path: Path,
) -> None:
    """Export failure_report.json and failure_report.md for multi-stage failures and omit retired legacy files."""
    run_id = "c" * 32
    policy = json.loads(TEST_ROUTING_POLICY_PATH.read_text(encoding="utf-8"))
    policy_fingerprint = fingerprint_policy_file(TEST_ROUTING_POLICY_PATH)
    classifier_fingerprint = fingerprint_classifier_model("gemini", "gemini-decision-1")
    settings = Settings.from_env({"DRAFT_GENERATION_PROVIDER": "offline"})
    primary = _sample_kb_source(
        "DOC-API-001",
        "DOC-API-001:p1",
        "API Keys",
        "Rotate API keys from the dashboard settings page.",
    )

    def _grounding_fn(*, ticket: Any, sources: Any, related_sources: Any) -> dict[str, Any]:
        if ticket["ticket_id"] == "T-GROUNDING":
            return {
                "claims": [
                    {
                        "text": "Rotate API keys from the dashboard settings page.",
                        "source_ids": ["DOC-API-001:p1"],
                    }
                ],
                "unsupported_points": ["Whether legacy v1 keys auto-expire in 24 hours."],
                "conflicting_points": ["Customer claims keys never expire, KB says 90 days."],
                "confidence": 0.42,
                "reason": "KB does not confirm legacy v1 key auto-expiration.",
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
            "reason": "Primary KB passage supports rotating API keys.",
        }

    evaluator = RoutingPolicyEvaluator(policy, claim_grounding_fn=_grounding_fn)
    ready_retrieval = RetrievalResult(
        status="ready",
        reason_code="sources_ready",
        reason="Approved primary source matched.",
        sources=(primary,),
        eligible_candidate_count=1,
        excluded_candidate_counts={},
        minimum_similarity=MINIMUM_SIMILARITY,
        embedding_model="gemini-embedding-001",
        source_sha256="1" * 64,
        metadata_manifest_sha256="2" * 64,
    )
    skipped_retrieval = RetrievalResult(
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
    )

    raw_tickets = [
        {
            "ticket_id": "T-PASS",
            "channel": "email",
            "subject": "Rotate API key",
            "body": "How do I rotate API keys from the dashboard settings page?",
            "labels": {
                "intent": "api_key_issue",
                "urgency": "low",
                "expected_route": "draft_for_review",
                "answerable_from_docs": True,
                "expected_doc_ids": ["DOC-API-001"],
                "must_not_auto_respond": False,
            },
        },
        {
            "ticket_id": "T-PROTECTED",
            "channel": "chat",
            "subject": "Security breach",
            "body": "Unauthorized user accessed our workspace.",
        },
        {
            "ticket_id": "T-GROUNDING",
            "channel": "web_form",
            "subject": "Legacy v1 key expiry",
            "body": "Do legacy v1 keys auto-expire in 24 hours?",
        },
        {
            "ticket_id": "T-GUARDRAIL",
            "channel": "email",
            "subject": "Rotate API key safely",
            "body": "How do I rotate API keys from the dashboard settings page?",
        },
    ]
    classifications = [
        TicketClassification(
            intent="api_key_issue",
            urgency="medium",
            confidence=0.93,
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
        TicketClassification(
            intent="api_key_issue",
            urgency="medium",
            confidence=0.91,
            provider="gemini",
            model_name="gemini-decision-1",
            provider_status="classified",
        ),
        TicketClassification(
            intent="api_key_issue",
            urgency="medium",
            confidence=0.94,
            provider="gemini",
            model_name="gemini-decision-1",
            provider_status="classified",
        ),
    ]
    retrievals = [ready_retrieval, skipped_retrieval, ready_retrieval, ready_retrieval]
    generators = [
        OfflineDraftGenerationProvider(),
        OfflineDraftGenerationProvider(),
        OfflineDraftGenerationProvider(),
        _GuardrailTriggeringDraftProvider(),
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
        for index, (raw, classification, retrieval, generator) in enumerate(
            zip(raw_tickets, classifications, retrievals, generators, strict=True)
        ):
            _persist_ticket_outcome(
                connection,
                run_id=run_id,
                input_index=index,
                raw_ticket=raw,
                classification=classification,
                retrieval=retrieval,
                evaluator=evaluator,
                policy=policy,
                policy_fingerprint=policy_fingerprint,
                classifier_fingerprint=classifier_fingerprint,
                settings=settings,
                generator=generator,
            )
        connection.commit()

        # Mutate caller input_records after persistence to verify failure diagnostics derive strictly from SQLite.
        mutated_inputs = [{**record, "subject": "MUTATED_AFTER_PERSISTENCE"} for record in raw_tickets]
        write_run_exports(
            connection,
            run_id=run_id,
            output_directory=output_dir,
            input_records=mutated_inputs,
            input_path=input_path,
            source_ref="external",
        )
    finally:
        connection.close()

    assert {path.name for path in output_dir.iterdir()} == {
        "results.jsonl",
        "decision_log.jsonl",
        "metrics.json",
        "exceptions.jsonl",
        "failure_report.json",
        "failure_report.md",
    }
    for retired in ("failed_records.json", "failed_details.json", "fail_results.md"):
        assert not (output_dir / retired).exists()

    failure_report = json.loads((output_dir / "failure_report.json").read_text(encoding="utf-8"))
    assert failure_report["run"]["run_id"] == run_id
    assert failure_report["run"]["operational_failure_count"] == 3
    assert failure_report["run"]["kpi_failed_record_count"] == 1
    assert failure_report["run"]["reported_ticket_count"] == 4
    tickets_by_id = {item["ticket_id"]: item for item in failure_report["tickets"]}
    assert set(tickets_by_id) == {"T-PASS", "T-PROTECTED", "T-GROUNDING", "T-GUARDRAIL"}
    assert "MUTATED_AFTER_PERSISTENCE" not in json.dumps(failure_report)

    protected_item = tickets_by_id["T-PROTECTED"]
    assert protected_item["stopped_at"] == "before_drafting"
    assert protected_item["failure_stage"] == "routing_policy"
    assert protected_item["failed_check"] == "protected_rule_check"
    assert protected_item["before_drafting"]["first_blocking_check"] == "protected_rule_check"
    assert [c["name"] for c in protected_item["before_drafting"]["routing_checks"]] == [
        "protected_rule_check",
        "classification_confidence_check",
        "source_presence_check",
        "claim_grounding_check",
    ]
    assert [c["status"] for c in protected_item["before_drafting"]["routing_checks"]] == [
        "failed",
        "skipped",
        "skipped",
        "skipped",
    ]

    grounding_item = tickets_by_id["T-GROUNDING"]
    assert grounding_item["stopped_at"] == "before_drafting"
    assert grounding_item["failure_stage"] == "routing_policy"
    assert grounding_item["failed_check"] == "claim_grounding_check"
    assert grounding_item["before_drafting"]["first_blocking_check"] == "claim_grounding_check"
    grounding_check = next(
        c
        for c in grounding_item["before_drafting"]["routing_checks"]
        if c["name"] == "claim_grounding_check"
    )
    assert grounding_check["unsupported_points"] == [
        "Whether legacy v1 keys auto-expire in 24 hours."
    ]
    assert grounding_check["conflicting_points"] == [
        "Customer claims keys never expire, KB says 90 days."
    ]

    guardrail_item = tickets_by_id["T-GUARDRAIL"]
    assert guardrail_item["stopped_at"] == "after_drafting"
    assert guardrail_item["failure_stage"] == "guardrail"
    assert guardrail_item["failed_check"] == "private_data_output"
    assert guardrail_item["after_drafting"]["blocking_check"] == "private_data_output"

    markdown = (output_dir / "failure_report.md").read_text(encoding="utf-8")
    assert "MUTATED_AFTER_PERSISTENCE" not in markdown
    assert "T-PROTECTED" in markdown
    assert "T-GROUNDING" in markdown
    assert "T-GUARDRAIL" in markdown
    assert "KB does not confirm legacy v1 key auto-expiration." in markdown
    assert "private_data_output" in markdown

    assert failure_report["failed_records"] == [
        {
            "input_index": 0,
            "ticket_id": "T-PASS",
            "failed_kpis": ["urgency"],
        }
    ]


def test_write_run_exports_handles_zero_failure_run(tmp_path: Path) -> None:
    """AC-2: Zero-failure runs emit failure_count=0, empty failures list, and explicit Markdown notice."""
    run_id = "d" * 32
    policy = json.loads(TEST_ROUTING_POLICY_PATH.read_text(encoding="utf-8"))
    policy_fingerprint = fingerprint_policy_file(TEST_ROUTING_POLICY_PATH)
    classifier_fingerprint = fingerprint_classifier_model("gemini", "gemini-decision-1")
    settings = Settings.from_env({"DRAFT_GENERATION_PROVIDER": "offline"})
    primary = _sample_kb_source(
        "DOC-API-001",
        "DOC-API-001:p1",
        "API Keys",
        "Rotate API keys from the dashboard settings page.",
    )
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
    raw_tickets = [
        {
            "ticket_id": "T-PASS-ONLY",
            "channel": "email",
            "subject": "Rotate API key",
            "body": "How do I rotate API keys from the dashboard settings page?",
        }
    ]
    classification = TicketClassification(
        intent="api_key_issue",
        urgency="medium",
        confidence=0.95,
        provider="gemini",
        model_name="gemini-decision-1",
        provider_status="classified",
    )
    retrieval = RetrievalResult(
        status="ready",
        reason_code="sources_ready",
        reason="Approved primary source matched.",
        sources=(primary,),
        eligible_candidate_count=1,
        excluded_candidate_counts={},
        minimum_similarity=MINIMUM_SIMILARITY,
        embedding_model="gemini-embedding-001",
        source_sha256="1" * 64,
        metadata_manifest_sha256="2" * 64,
    )

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
        _persist_ticket_outcome(
            connection,
            run_id=run_id,
            input_index=0,
            raw_ticket=raw_tickets[0],
            classification=classification,
            retrieval=retrieval,
            evaluator=evaluator,
            policy=policy,
            policy_fingerprint=policy_fingerprint,
            classifier_fingerprint=classifier_fingerprint,
            settings=settings,
            generator=OfflineDraftGenerationProvider(),
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
    finally:
        connection.close()

    assert {path.name for path in output_dir.iterdir()} == {
        "results.jsonl",
        "decision_log.jsonl",
        "metrics.json",
        "exceptions.jsonl",
        "failure_report.json",
        "failure_report.md",
    }
    for retired in ("failed_records.json", "failed_details.json", "fail_results.md"):
        assert not (output_dir / retired).exists()

    failure_report = json.loads((output_dir / "failure_report.json").read_text(encoding="utf-8"))
    assert failure_report["run"]["run_id"] == run_id
    assert failure_report["run"]["operational_failure_count"] == 0
    assert failure_report["run"]["kpi_failed_record_count"] == 0
    assert failure_report["run"]["reported_ticket_count"] == 0
    assert failure_report["tickets"] == []
    assert failure_report["failed_records"] == []
    markdown = (output_dir / "failure_report.md").read_text(encoding="utf-8")
    assert "No operational failures or evaluation KPI mismatches occurred in this run." in markdown


def test_write_run_exports_escapes_unpaired_surrogates_and_control_characters(
    tmp_path: Path,
) -> None:
    """AC-3: Unpaired UTF-16 surrogates and control chars export without UnicodeEncodeError."""
    run_id = "e" * 32
    policy = json.loads(TEST_ROUTING_POLICY_PATH.read_text(encoding="utf-8"))
    policy_fingerprint = fingerprint_policy_file(TEST_ROUTING_POLICY_PATH)
    classifier_fingerprint = fingerprint_classifier_model("gemini", "gemini-decision-1")
    settings = Settings.from_env({"DRAFT_GENERATION_PROVIDER": "offline"})
    primary = _sample_kb_source(
        "DOC-API-001",
        "DOC-API-001:p1",
        "API Keys",
        "Rotate API keys from the dashboard settings page.",
    )
    evaluator = RoutingPolicyEvaluator(
        policy,
        claim_grounding_fn=lambda *, ticket, sources, related_sources: {
            "claims": [],
            "unsupported_points": ["Bad point \ud800 with control \x00\x07\x1b"],
            "conflicting_points": ["Conflict \ud800 text"],
            "confidence": 0.2,
            "reason": "Grounding failed with surrogate \ud800 and control \x07",
        },
    )
    raw_tickets = [
        {
            "ticket_id": "T-SURROGATE",
            "channel": "email",
            "subject": "Subject \ud800\x00\x1b",
            "body": "Body \ud800\x07 text",
        }
    ]
    classification = TicketClassification(
        intent="api_key_issue",
        urgency="medium",
        confidence=0.92,
        provider="gemini",
        model_name="gemini-decision-1",
        provider_status="classified",
    )
    retrieval = RetrievalResult(
        status="ready",
        reason_code="sources_ready",
        reason="Approved primary source matched.",
        sources=(primary,),
        eligible_candidate_count=1,
        excluded_candidate_counts={},
        minimum_similarity=MINIMUM_SIMILARITY,
        embedding_model="gemini-embedding-001",
        source_sha256="1" * 64,
        metadata_manifest_sha256="2" * 64,
    )

    db_path = tmp_path / "runtime.sqlite3"
    output_dir = tmp_path / "outputs" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = tmp_path / "tickets.json"
    # Escape surrogates when writing the raw fixture file as UTF-8, while passing raw surrogate in input_records.
    input_path.write_text("[]", encoding="utf-8")

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
        ticket = normalize_ticket_record(raw_tickets[0], input_index=0)
        policy_eval = evaluator.evaluate(
            ticket=ticket,
            classification=classification.as_dict(),
            retrieval=retrieval,
        )
        decision = RoutingPolicyEvaluator.to_routing_decision(policy_eval)
        outcome = build_ticket_outcome(
            run_id=run_id,
            input_index=0,
            ticket=ticket,
            classification=classification,
            retrieval=retrieval,
            decision=decision,
            policy_evaluation=policy_eval,
            policy=policy,
            policy_file_fingerprint=policy_fingerprint,
            classifier_implementation_fingerprint=classifier_fingerprint,
            settings=settings,
            generator=OfflineDraftGenerationProvider(),
            issues=[],
            duplicate_ticket_id=False,
        )
        outcome["processing_duration_seconds"] = 0.05
        outcome["processing_latency_seconds"] = 0.05
        connection.execute(
            """INSERT INTO ticket_runs
               (run_id, ticket_id, channel, context_json, processing_status)
               VALUES (?, ?, ?, '{}', 'completed')""",
            (run_id, ticket.ticket_id, ticket.channel),
        )
        connection.execute(
            """INSERT INTO audit_events
               (event_id, run_id, ticket_id, occurred_at, event_type, actor_type,
                input_summary, action_taken, reason, guardrail_results_json, sources_json,
                event_payload_json)
               VALUES (?, ?, ?, ?, 'ticket_completed', 'system', ?, ?, ?, ?, '[]', ?)""",
            (
                "0" * 32,
                run_id,
                ticket.ticket_id,
                "2026-11-10T12:00:01Z",
                "surrogate ticket",
                outcome["route"],
                "surrogate grounding failure",
                json.dumps(outcome["guardrails"], ensure_ascii=True),
                json.dumps(
                    {
                        "input_index": 0,
                        "ticket_id": ticket.ticket_id,
                        "result": outcome,
                    },
                    ensure_ascii=True,
                ),
            ),
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
    finally:
        connection.close()

    for retired in ("failed_records.json", "failed_details.json", "fail_results.md"):
        assert not (output_dir / retired).exists()

    failure_report = json.loads((output_dir / "failure_report.json").read_text(encoding="utf-8"))
    assert failure_report["run"]["operational_failure_count"] == 1
    assert failure_report["tickets"][0]["ticket_id"] == "T-SURROGATE"

    results_lines = (output_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(results_lines) == 1
    parsed_result = json.loads(results_lines[0])
    assert parsed_result["ticket_id"] == "T-SURROGATE"

    markdown = (output_dir / "failure_report.md").read_text(encoding="utf-8")
    assert "T-SURROGATE" in markdown
    assert "\\ud800" in markdown
    assert "\x00" not in markdown
    assert "\x07" not in markdown
    assert "\x1b" not in markdown


def test_write_run_exports_records_input_validation_and_provider_failures(
    tmp_path: Path,
) -> None:
    """Input-validation and generation/provider failures emit stage, check/error, and Markdown entries."""
    run_id = "f" * 32
    policy = json.loads(TEST_ROUTING_POLICY_PATH.read_text(encoding="utf-8"))
    policy_fingerprint = fingerprint_policy_file(TEST_ROUTING_POLICY_PATH)
    classifier_fingerprint = fingerprint_classifier_model("gemini", "gemini-decision-1")
    settings = Settings.from_env({"DRAFT_GENERATION_PROVIDER": "gemini"})
    primary = _sample_kb_source(
        "DOC-API-001",
        "DOC-API-001:p1",
        "API Keys",
        "Rotate API keys from the dashboard settings page.",
    )
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
    raw_tickets = [
        {
            "ticket_id": "T-INVALID",
            "channel": "email",
            "subject": "Missing body",
            "body": "",
        },
        {
            "ticket_id": "T-GEN-ERR",
            "channel": "email",
            "subject": "Rotate API key",
            "body": "How do I rotate API keys from the dashboard settings page?",
        },
    ]
    ready_retrieval = RetrievalResult(
        status="ready",
        reason_code="sources_ready",
        reason="Approved primary source matched.",
        sources=(primary,),
        eligible_candidate_count=1,
        excluded_candidate_counts={},
        minimum_similarity=MINIMUM_SIMILARITY,
        embedding_model="gemini-embedding-001",
        source_sha256="1" * 64,
        metadata_manifest_sha256="2" * 64,
    )
    skipped_retrieval = RetrievalResult(
        status="skipped",
        reason_code="input_validation_failed",
        reason="Skipped retrieval for invalid ticket.",
        sources=(),
        eligible_candidate_count=0,
        excluded_candidate_counts={},
        minimum_similarity=MINIMUM_SIMILARITY,
        embedding_model="gemini-embedding-001",
        source_sha256="1" * 64,
        metadata_manifest_sha256="2" * 64,
    )

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

        # 0: Input validation failure
        invalid_ticket = normalize_ticket_record(raw_tickets[0], input_index=0)
        invalid_cls = classification_failure(
            provider="gemini",
            model_name="gemini-decision-1",
            uncertainty="Input validation failed.",
        )
        invalid_eval = evaluator.evaluate(
            ticket=invalid_ticket,
            classification=invalid_cls.as_dict(),
            retrieval=skipped_retrieval,
        )
        invalid_outcome = build_ticket_outcome(
            run_id=run_id,
            input_index=0,
            ticket=invalid_ticket,
            classification=invalid_cls,
            retrieval=skipped_retrieval,
            decision=RoutingPolicyEvaluator.to_routing_decision(invalid_eval),
            policy_evaluation=invalid_eval,
            policy=policy,
            policy_file_fingerprint=policy_fingerprint,
            classifier_implementation_fingerprint=classifier_fingerprint,
            settings=settings,
            generator=OfflineDraftGenerationProvider(),
            issues=["Ticket body is required."],
            duplicate_ticket_id=False,
        )
        invalid_outcome["processing_duration_seconds"] = 0.01
        invalid_outcome["processing_latency_seconds"] = 0.01
        create_ticket_audit_records(
            connection,
            run_id=run_id,
            storage_ticket_id=invalid_ticket.ticket_id,
            channel=invalid_ticket.channel,
            received_at=None,
            customer_tier=None,
            language_fluency=None,
            context={},
            processing_status="failed",
            error_code=invalid_outcome["error"]["code"],
            error_message=invalid_outcome["error"]["message"],
            event_id=f"{0:032x}",
            occurred_at="2026-11-10T12:00:01Z",
            event_type="ticket_failed",
            input_summary="Missing body",
            action_taken=invalid_outcome["route"],
            reason=invalid_outcome["reason"],
            result_payload={
                "input_index": 0,
                "ticket_id": invalid_ticket.ticket_id,
                "result": invalid_outcome,
            },
            guardrail_results=invalid_outcome["guardrails"],
        )

        # 1: Generation provider failure
        gen_ticket = normalize_ticket_record(raw_tickets[1], input_index=1)
        gen_cls = TicketClassification(
            intent="api_key_issue",
            urgency="medium",
            confidence=0.95,
            provider="gemini",
            model_name="gemini-decision-1",
            provider_status="classified",
        )
        gen_eval = evaluator.evaluate(
            ticket=gen_ticket,
            classification=gen_cls.as_dict(),
            retrieval=ready_retrieval,
        )
        gen_outcome = build_ticket_outcome(
            run_id=run_id,
            input_index=1,
            ticket=gen_ticket,
            classification=gen_cls,
            retrieval=ready_retrieval,
            decision=RoutingPolicyEvaluator.to_routing_decision(gen_eval),
            policy_evaluation=gen_eval,
            policy=policy,
            policy_file_fingerprint=policy_fingerprint,
            classifier_implementation_fingerprint=classifier_fingerprint,
            settings=settings,
            generator=_FailingDraftProvider(),
            issues=[],
            duplicate_ticket_id=False,
        )
        gen_outcome["processing_duration_seconds"] = 0.02
        gen_outcome["processing_latency_seconds"] = 0.02
        create_ticket_audit_records(
            connection,
            run_id=run_id,
            storage_ticket_id=gen_ticket.ticket_id,
            channel=gen_ticket.channel,
            received_at=None,
            customer_tier=None,
            language_fluency=None,
            context={},
            processing_status="failed",
            error_code=gen_outcome["error"]["code"],
            error_message=gen_outcome["error"]["message"],
            event_id=f"{1:032x}",
            occurred_at="2026-11-10T12:00:02Z",
            event_type="ticket_failed",
            input_summary=gen_ticket.body,
            action_taken=gen_outcome["route"],
            reason=gen_outcome["reason"],
            result_payload={
                "input_index": 1,
                "ticket_id": gen_ticket.ticket_id,
                "result": gen_outcome,
            },
            guardrail_results=gen_outcome["guardrails"],
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
    finally:
        connection.close()

    for retired in ("failed_records.json", "failed_details.json", "fail_results.md"):
        assert not (output_dir / retired).exists()

    failure_report = json.loads((output_dir / "failure_report.json").read_text(encoding="utf-8"))
    assert failure_report["run"]["operational_failure_count"] == 2
    tickets_by_id = {item["ticket_id"]: item for item in failure_report["tickets"]}

    invalid_item = tickets_by_id["T-INVALID"]
    assert invalid_item["stopped_at"] == "before_drafting"
    assert invalid_item["failure_stage"] == "input_validation"
    assert invalid_item["failed_check"] == "invalid_ticket_record"
    assert invalid_item["error"] == {
        "code": "invalid_ticket_record",
        "message": "Ticket body is required.",
    }

    gen_item = tickets_by_id["T-GEN-ERR"]
    assert gen_item["stopped_at"] == "during_drafting"
    assert gen_item["failure_stage"] == "generation"
    assert gen_item["failed_check"] == "generation_timeout"
    assert gen_item["error"] == {
        "code": "generation_timeout",
        "message": "Gemini draft request timed out.",
    }

    markdown = (output_dir / "failure_report.md").read_text(encoding="utf-8")
    assert "T-INVALID" in markdown
    assert "invalid_ticket_record" in markdown
    assert "T-GEN-ERR" in markdown
    assert "generation_timeout" in markdown


def test_write_run_exports_builds_failure_report_json_strictly_from_sqlite_pre_draft_and_kpis(
    tmp_path: Path,
) -> None:
    """AC-1 & AC-2: failure_report.json derives metrics, KPIs, and before_drafting diagnostics strictly from SQLite."""
    run_id = "1" * 32
    policy = json.loads(TEST_ROUTING_POLICY_PATH.read_text(encoding="utf-8"))
    policy_fingerprint = fingerprint_policy_file(TEST_ROUTING_POLICY_PATH)
    classifier_fingerprint = fingerprint_classifier_model("gemini", "gemini-decision-1")
    settings = Settings.from_env({"DRAFT_GENERATION_PROVIDER": "offline"})
    doc_1 = _sample_kb_source(
        "DOC-API-001",
        "DOC-API-001:p1",
        "API Keys",
        "Rotate API keys from the dashboard settings page.",
    )
    doc_2 = _sample_kb_source(
        "DOC-API-002",
        "DOC-API-002:p1",
        "API Scopes",
        "Configure API key scopes in workspace permissions.",
    )

    def _grounding_fn(*, ticket: Any, sources: Any, related_sources: Any) -> dict[str, Any]:
        if ticket["ticket_id"] == "T-GROUNDING":
            return {
                "claims": [
                    {
                        "text": "Rotate API keys from the dashboard settings page.",
                        "source_ids": ["DOC-API-001:p1"],
                    }
                ],
                "unsupported_points": ["Whether legacy v1 keys auto-expire in 24 hours."],
                "conflicting_points": ["Customer claims keys never expire, KB says 90 days."],
                "confidence": 0.42,
                "reason": "KB does not confirm legacy v1 key auto-expiration.",
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
            "reason": "Primary KB passage supports rotating API keys.",
        }

    evaluator = RoutingPolicyEvaluator(policy, claim_grounding_fn=_grounding_fn)
    two_doc_retrieval = RetrievalResult(
        status="ready",
        reason_code="sources_ready",
        reason="Approved primary sources matched.",
        sources=(doc_1, doc_2),
        eligible_candidate_count=2,
        excluded_candidate_counts={},
        minimum_similarity=MINIMUM_SIMILARITY,
        embedding_model="gemini-embedding-001",
        source_sha256="1" * 64,
        metadata_manifest_sha256="2" * 64,
    )
    one_doc_retrieval = RetrievalResult(
        status="ready",
        reason_code="sources_ready",
        reason="Approved primary source matched.",
        sources=(doc_1,),
        eligible_candidate_count=1,
        excluded_candidate_counts={},
        minimum_similarity=MINIMUM_SIMILARITY,
        embedding_model="gemini-embedding-001",
        source_sha256="1" * 64,
        metadata_manifest_sha256="2" * 64,
    )
    skipped_retrieval = RetrievalResult(
        status="skipped",
        reason_code="low_confidence",
        reason="Skipped retrieval below confidence threshold.",
        sources=(),
        eligible_candidate_count=0,
        excluded_candidate_counts={},
        minimum_similarity=MINIMUM_SIMILARITY,
        embedding_model="gemini-embedding-001",
        source_sha256="1" * 64,
        metadata_manifest_sha256="2" * 64,
    )

    raw_tickets = [
        {
            "ticket_id": "T-KPI-ONLY",
            "channel": "email",
            "subject": "Rotate API key",
            "body": "How do I rotate API keys from the dashboard settings page?",
            "labels": {
                "intent": "api_key_issue",
                "urgency": "low",
                "expected_route": "draft_for_review",
                "answerable_from_docs": True,
                "expected_doc_ids": ["DOC-API-002"],
                "must_not_auto_respond": False,
            },
        },
        {
            "ticket_id": "T-LOW-CONF",
            "channel": "chat",
            "subject": "Unclear request",
            "body": "Something is not working.",
            "labels": {
                "intent": "api_key_issue",
                "urgency": "medium",
                "expected_route": "escalate",
                "answerable_from_docs": False,
                "expected_doc_ids": [],
                "must_not_auto_respond": False,
            },
        },
        {
            "ticket_id": "T-GROUNDING",
            "channel": "web_form",
            "subject": "Legacy v1 key expiry",
            "body": "Do legacy v1 keys auto-expire in 24 hours?",
        },
        {
            "ticket_id": "T-INVALID",
            "channel": "email",
            "subject": "Missing body",
            "body": "Placeholder",
        },
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
        for index, (classification, retrieval) in enumerate(
            [
                (
                    TicketClassification(
                        intent="api_key_issue",
                        urgency="medium",
                        confidence=0.93,
                        provider="gemini",
                        model_name="gemini-decision-1",
                        provider_status="classified",
                    ),
                    two_doc_retrieval,
                ),
                (
                    TicketClassification(
                        intent="api_key_issue",
                        urgency="medium",
                        confidence=0.541,
                        provider="gemini",
                        model_name="gemini-decision-1",
                        provider_status="classified",
                    ),
                    skipped_retrieval,
                ),
                (
                    TicketClassification(
                        intent="api_key_issue",
                        urgency="medium",
                        confidence=0.91,
                        provider="gemini",
                        model_name="gemini-decision-1",
                        provider_status="classified",
                    ),
                    one_doc_retrieval,
                ),
            ]
        ):
            _persist_ticket_outcome(
                connection,
                run_id=run_id,
                input_index=index,
                raw_ticket=raw_tickets[index],
                classification=classification,
                retrieval=retrieval,
                evaluator=evaluator,
                policy=policy,
                policy_fingerprint=policy_fingerprint,
                classifier_fingerprint=classifier_fingerprint,
                settings=settings,
                generator=OfflineDraftGenerationProvider(),
            )

        invalid_ticket = normalize_ticket_record(raw_tickets[3], input_index=3)
        invalid_classification = classification_failure(
            provider="gemini",
            model_name="gemini-decision-1",
            uncertainty="Ticket body is required.",
        )
        invalid_policy_eval = evaluator.evaluate(
            ticket=invalid_ticket,
            classification=invalid_classification.as_dict(),
            retrieval=skipped_retrieval,
        )
        invalid_outcome = build_ticket_outcome(
            run_id=run_id,
            input_index=3,
            ticket=invalid_ticket,
            classification=invalid_classification,
            retrieval=skipped_retrieval,
            decision=RoutingPolicyEvaluator.to_routing_decision(invalid_policy_eval),
            policy_evaluation=invalid_policy_eval,
            policy=policy,
            policy_file_fingerprint=policy_fingerprint,
            classifier_implementation_fingerprint=classifier_fingerprint,
            settings=settings,
            generator=OfflineDraftGenerationProvider(),
            issues=["Ticket body is required."],
            duplicate_ticket_id=False,
        )
        invalid_outcome["processing_duration_seconds"] = 0.02
        invalid_outcome["processing_latency_seconds"] = 0.02
        # Mutate in-memory outcome after persistence payload creation is not needed;
        # instead persist first, then mutate in-memory input_records and outcome dicts.
        create_ticket_audit_records(
            connection,
            run_id=run_id,
            storage_ticket_id=invalid_ticket.ticket_id,
            channel=invalid_ticket.channel,
            received_at=None,
            customer_tier=None,
            language_fluency=None,
            context={},
            processing_status="failed",
            error_code=invalid_outcome["error"]["code"],
            error_message=invalid_outcome["error"]["message"],
            event_id=f"{3:032x}",
            occurred_at="2026-11-10T12:00:02Z",
            event_type="ticket_failed",
            input_summary=invalid_ticket.body,
            action_taken=invalid_outcome["route"],
            reason=invalid_outcome["reason"],
            result_payload={
                "input_index": 3,
                "ticket_id": invalid_ticket.ticket_id,
                "result": invalid_outcome,
            },
            guardrail_results=invalid_outcome["guardrails"],
        )
        connection.commit()

        invalid_outcome["reason"] = "MUTATED_AFTER_PERSISTENCE"
        mutated_inputs = json.loads(json.dumps(raw_tickets))
        mutated_inputs[0]["subject"] = "MUTATED_AFTER_PERSISTENCE"
        mutated_inputs[0]["labels"]["intent"] = "billing_query"
        mutated_inputs[0]["labels"]["urgency"] = "high"

        write_run_exports(
            connection,
            run_id=run_id,
            output_directory=output_dir,
            input_records=mutated_inputs,
            input_path=input_path,
            source_ref="external",
        )
    finally:
        connection.close()

    report = json.loads((output_dir / "failure_report.json").read_text(encoding="utf-8"))
    assert "MUTATED_AFTER_PERSISTENCE" not in json.dumps(report)
    assert report["schema_version"] == 1
    assert report["run"] == {
        "run_id": run_id,
        "input_ref": str(input_path),
        "tickets_processed": 4,
        "completed": 3,
        "processing_failures": 1,
        "exceptions": 1,
        "drafts_for_review": 1,
        "escalated": 2,
        "degraded": 1,
        "blocked_by_guardrails": 2,
        "operational_failure_count": 3,
        "kpi_failed_record_count": 1,
        "reported_ticket_count": 4,
    }
    assert report["metrics"]["intent_accuracy"]["value"] == 1.0
    assert report["metrics"]["intent_accuracy"]["numerator"] == 2
    assert report["metrics"]["intent_accuracy"]["denominator"] == 2
    assert report["metrics"]["urgency_exact_match"]["value"] == 0.5
    assert report["metrics"]["route_accuracy"]["value"] == 1.0
    assert report["metrics"]["answerability_accuracy"]["value"] == 1.0
    assert report["metrics"]["retrieval_hit_rate"]["value"] == 1.0
    assert report["metrics"]["decision_audit_coverage"]["value"] == 1.0
    assert report["metrics"]["p95_runtime_latency_seconds"]["value"] is not None
    calibration = report["metrics"]["confidence_calibration"]
    assert {"status", "target", "value", "populated_bands"}.issubset(calibration)
    assert len(calibration["populated_bands"]) == 2

    assert report["failed_records"] == [
        {
            "input_index": 0,
            "ticket_id": "T-KPI-ONLY",
            "failed_kpis": ["urgency", "expected_doc_ids"],
        }
    ]
    by_id = {item["ticket_id"]: item for item in report["tickets"]}
    assert [item["ticket_id"] for item in report["tickets"]] == [
        "T-KPI-ONLY",
        "T-LOW-CONF",
        "T-GROUNDING",
        "T-INVALID",
    ]

    kpi_ticket = by_id["T-KPI-ONLY"]
    assert kpi_ticket["stopped_at"] == "evaluation_only"
    assert kpi_ticket["evaluation"]["expected"]["urgency"] == "low"
    assert kpi_ticket["evaluation"]["predicted"]["urgency"] == "medium"
    assert kpi_ticket["evaluation"]["predicted"]["retrieved_doc_ids"] == [
        "DOC-API-001",
        "DOC-API-002",
    ]
    assert kpi_ticket["evaluation"]["failed_kpis"] == ["urgency", "expected_doc_ids"]
    assert "DOC-API-002" in kpi_ticket["evaluation"]["expected_doc_note"]

    low_conf = by_id["T-LOW-CONF"]
    assert low_conf["stopped_at"] == "before_drafting"
    assert low_conf["before_drafting"]["first_blocking_check"] == "classification_confidence_check"
    assert [c["status"] for c in low_conf["before_drafting"]["routing_checks"]] == [
        "passed",
        "failed",
        "skipped",
        "skipped",
    ]
    assert low_conf["drafting"]["attempted"] is False
    assert low_conf["after_drafting"] == {
        "status": "not_run",
        "checks": [],
        "blocking_check": None,
        "distinct_from_pre_draft_claim_grounding_note": None,
        "withheld_claims": [],
        "unsupported_points": [],
        "conflicting_points": [],
        "failed_claims": [],
        "diagnostics_note": None,
    }

    grounding = by_id["T-GROUNDING"]
    assert grounding["stopped_at"] == "before_drafting"
    assert grounding["before_drafting"]["first_blocking_check"] == "claim_grounding_check"
    grounding_check = next(
        c for c in grounding["before_drafting"]["routing_checks"] if c["name"] == "claim_grounding_check"
    )
    assert grounding_check["unsupported_points"] == [
        "Whether legacy v1 keys auto-expire in 24 hours."
    ]
    assert grounding_check["conflicting_points"] == [
        "Customer claims keys never expire, KB says 90 days."
    ]

    invalid = by_id["T-INVALID"]
    assert invalid["stopped_at"] == "before_drafting"
    assert invalid["failure_stage"] == "input_validation"
    assert invalid["failed_check"] == "invalid_ticket_record"
    assert invalid["drafting"]["attempted"] is False


def test_write_run_exports_failure_report_zero_failures_and_unpaired_surrogate(
    tmp_path: Path,
) -> None:
    """AC-3 & AC-4: Clean runs emit zero counts/empty lists and unpaired surrogates serialize safely."""
    policy = json.loads(TEST_ROUTING_POLICY_PATH.read_text(encoding="utf-8"))
    policy_fingerprint = fingerprint_policy_file(TEST_ROUTING_POLICY_PATH)
    classifier_fingerprint = fingerprint_classifier_model("gemini", "gemini-decision-1")
    settings = Settings.from_env({"DRAFT_GENERATION_PROVIDER": "offline"})
    primary = _sample_kb_source(
        "DOC-API-001",
        "DOC-API-001:p1",
        "API Keys",
        "Rotate API keys from the dashboard settings page.",
    )
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
    ready_retrieval = RetrievalResult(
        status="ready",
        reason_code="sources_ready",
        reason="Approved primary source matched.",
        sources=(primary,),
        eligible_candidate_count=1,
        excluded_candidate_counts={},
        minimum_similarity=MINIMUM_SIMILARITY,
        embedding_model="gemini-embedding-001",
        source_sha256="1" * 64,
        metadata_manifest_sha256="2" * 64,
    )

    # AC-3: Zero operational failures and zero KPI mismatches.
    clean_run_id = "2" * 32
    clean_dir = tmp_path / "outputs" / clean_run_id
    clean_dir.mkdir(parents=True, exist_ok=True)
    clean_input = tmp_path / "clean.json"
    clean_ticket = {
        "ticket_id": "T-CLEAN",
        "channel": "email",
        "subject": "Rotate API key",
        "body": "How do I rotate API keys from the dashboard settings page?",
        "labels": {
            "intent": "api_key_issue",
            "urgency": "medium",
            "expected_route": "draft_for_review",
            "answerable_from_docs": True,
            "expected_doc_ids": ["DOC-API-001"],
            "must_not_auto_respond": False,
        },
    }
    clean_input.write_text(json.dumps([clean_ticket]), encoding="utf-8")
    conn = connect_runtime_database(tmp_path / "clean.sqlite3")
    try:
        create_processing_run(
            conn,
            run_id=clean_run_id,
            input_ref=str(clean_input),
            started_at="2026-11-10T12:00:00Z",
            label=None,
            output_dir=str(clean_dir),
        )
        _persist_ticket_outcome(
            conn,
            run_id=clean_run_id,
            input_index=0,
            raw_ticket=clean_ticket,
            classification=TicketClassification(
                intent="api_key_issue",
                urgency="medium",
                confidence=0.95,
                provider="gemini",
                model_name="gemini-decision-1",
                provider_status="classified",
            ),
            retrieval=ready_retrieval,
            evaluator=evaluator,
            policy=policy,
            policy_fingerprint=policy_fingerprint,
            classifier_fingerprint=classifier_fingerprint,
            settings=settings,
            generator=OfflineDraftGenerationProvider(),
        )
        conn.commit()
        write_run_exports(
            conn,
            run_id=clean_run_id,
            output_directory=clean_dir,
            input_records=[clean_ticket],
            input_path=clean_input,
            source_ref="external",
        )
    finally:
        conn.close()

    clean_report = json.loads((clean_dir / "failure_report.json").read_text(encoding="utf-8"))
    assert clean_report["run"]["operational_failure_count"] == 0
    assert clean_report["run"]["kpi_failed_record_count"] == 0
    assert clean_report["run"]["reported_ticket_count"] == 0
    assert clean_report["failed_records"] == []
    assert clean_report["tickets"] == []

    # AC-4: Unpaired UTF-16 surrogate in pre-draft failing ticket serializes safely.
    surrogate_run_id = "3" * 32
    surrogate_dir = tmp_path / "outputs" / surrogate_run_id
    surrogate_dir.mkdir(parents=True, exist_ok=True)
    surrogate_input = tmp_path / "surrogate.json"
    surrogate_ticket = {
        "ticket_id": "T-SURROGATE",
        "channel": "chat",
        "subject": "Bad surrogate",
        "body": "Text with SURROGATE_TOKEN surrogate",
    }
    surrogate_input.write_text(json.dumps([surrogate_ticket]), encoding="utf-8")
    surrogate_conn = connect_runtime_database(tmp_path / "surrogate.sqlite3")
    try:
        create_processing_run(
            surrogate_conn,
            run_id=surrogate_run_id,
            input_ref="surrogate.json",
            started_at="2026-11-10T12:00:00Z",
            label=None,
            output_dir=str(surrogate_dir),
        )
        _persist_ticket_outcome(
            surrogate_conn,
            run_id=surrogate_run_id,
            input_index=0,
            raw_ticket=surrogate_ticket,
            classification=TicketClassification(
                intent="api_key_issue",
                urgency="medium",
                confidence=0.40,
                provider="gemini",
                model_name="gemini-decision-1",
                provider_status="classified",
            ),
            retrieval=RetrievalResult(
                status="skipped",
                reason_code="low_confidence",
                reason="Blocked SURROGATE_TOKEN reason",
                sources=(),
                eligible_candidate_count=0,
                excluded_candidate_counts={},
                minimum_similarity=MINIMUM_SIMILARITY,
                embedding_model="gemini-embedding-001",
                source_sha256="1" * 64,
                metadata_manifest_sha256="2" * 64,
            ),
            evaluator=evaluator,
            policy=policy,
            policy_fingerprint=policy_fingerprint,
            classifier_fingerprint=classifier_fingerprint,
            settings=settings,
            generator=OfflineDraftGenerationProvider(),
        )
        surrogate_conn.execute(
            "UPDATE audit_events SET event_payload_json = replace(event_payload_json, 'below the frozen threshold', 'below \\ud800 threshold') WHERE run_id = ?",
            (surrogate_run_id,),
        )
        surrogate_conn.commit()
        write_run_exports(
            surrogate_conn,
            run_id=surrogate_run_id,
            output_directory=surrogate_dir,
            input_records=[surrogate_ticket],
            input_path=tmp_path / "surrogate.json",
            source_ref="external",
        )
    finally:
        surrogate_conn.close()

    raw_report_text = (surrogate_dir / "failure_report.json").read_text(encoding="utf-8")
    assert "\\ud800" in raw_report_text
    parsed_surrogate = json.loads(raw_report_text)
    assert parsed_surrogate["tickets"][0]["ticket_id"] == "T-SURROGATE"
    assert parsed_surrogate["tickets"][0]["stopped_at"] == "before_drafting"


class _CustomDraftProvider:
    """Test boundary provider returning caller-specified draft claims and points."""

    name = "gemini"

    def __init__(
        self,
        *,
        claims: list[dict[str, Any]],
        plain_text: str,
        unsupported_points: list[str] | None = None,
        conflicting_points: list[str] | None = None,
    ) -> None:
        self._claims = claims
        self._plain_text = plain_text
        self._unsupported_points = unsupported_points or []
        self._conflicting_points = conflicting_points or []

    def generate_draft(
        self,
        ticket: CanonicalTicket,
        *,
        sources: tuple[dict[str, Any], ...],
        related_sources: tuple[dict[str, Any], ...] = (),
        policy_evaluation: Mapping[str, Any] | None = None,
        model_name: str = "",
    ) -> dict[str, Any]:
        draft = OfflineDraftGenerationProvider().generate_draft(
            ticket,
            sources=sources,
            related_sources=related_sources,
            policy_evaluation=policy_evaluation,
            model_name=model_name,
        )
        source_by_id = {
            sid: src
            for src in (*sources, *related_sources)
            for sid in (src.get("passage_id"), src.get("doc_id"))
            if isinstance(sid, str) and sid
        }
        assembled_claims = []
        for item in self._claims:
            citations = [
                {
                    "passage_id": source_by_id[sid]["passage_id"],
                    "source_ref": source_by_id[sid]["source_ref"],
                }
                for sid in item["source_ids"]
            ]
            assembled_claims.append(
                {
                    "text": item["text"],
                    "source_ids": list(item["source_ids"]),
                    "citations": citations,
                }
            )
        draft["plain_text"] = self._plain_text
        draft["claims"] = assembled_claims
        draft["unsupported_points"] = list(self._unsupported_points)
        draft["conflicting_points"] = list(self._conflicting_points)
        return draft


def test_write_run_exports_populates_after_drafting_diagnostics_and_cited_docs(
    tmp_path: Path,
) -> None:
    """AC-1, AC-2, AC-3: Populate after_drafting checks, clause diagnostics, and cited_doc_ids."""
    run_id = "4" * 32
    policy = json.loads(TEST_ROUTING_POLICY_PATH.read_text(encoding="utf-8"))
    policy_fingerprint = fingerprint_policy_file(TEST_ROUTING_POLICY_PATH)
    classifier_fingerprint = fingerprint_classifier_model("gemini", "gemini-decision-1")
    settings = Settings.from_env({"DRAFT_GENERATION_PROVIDER": "gemini"})

    deploy_primary = _sample_kb_source(
        "DOC-DEPLOY-001",
        "DOC-DEPLOY-001:p1",
        "Blue Green Deployments",
        "Use blue green deployment slots in the deployment console.",
    )
    deploy_other = _sample_kb_source(
        "DOC-DEPLOY-003",
        "DOC-DEPLOY-003:p1",
        "Rollback Steps",
        "Trigger a manual rollback from the release history tab.",
    )
    api_primary = _sample_kb_source(
        "DOC-API-001",
        "DOC-API-001:p1",
        "API Keys",
        "Rotate API keys from the dashboard settings page.",
    )
    api_other = _sample_kb_source(
        "DOC-API-002",
        "DOC-API-002:p1",
        "API Scopes",
        "Configure API key scopes in workspace permissions.",
    )

    def _grounding_fn(*, ticket: Any, sources: Any, related_sources: Any) -> dict[str, Any]:
        del related_sources
        first_passage = sources[0]["passage_id"]
        if ticket["ticket_id"] == "T-POST-GROUNDING":
            return {
                "claims": [
                    {
                        "text": "P1 policy verified blue green deployment slot guidance.",
                        "source_ids": [first_passage],
                    }
                ],
                "unsupported_points": [],
                "conflicting_points": [],
                "confidence": 0.94,
                "reason": "Pre-draft policy claim P1 passed.",
            }
        if ticket["ticket_id"] == "T-EMITTED-MISMATCH":
            return {
                "claims": [
                    {
                        "text": "Configure API key scopes in workspace permissions.",
                        "source_ids": ["DOC-API-002:p1"],
                    }
                ],
                "unsupported_points": [],
                "conflicting_points": [],
                "confidence": 0.95,
                "reason": "Policy claim grounded in DOC-API-002.",
            }
        return {
            "claims": [
                {
                    "text": "Rotate API keys from the dashboard settings page.",
                    "source_ids": [first_passage],
                }
            ],
            "unsupported_points": [],
            "conflicting_points": [],
            "confidence": 0.95,
            "reason": "Policy claim passed.",
        }

    evaluator = RoutingPolicyEvaluator(policy, claim_grounding_fn=_grounding_fn)
    deploy_retrieval = RetrievalResult(
        status="ready",
        reason_code="sources_ready",
        reason="Approved primary sources matched.",
        sources=(deploy_primary, deploy_other),
        eligible_candidate_count=2,
        excluded_candidate_counts={},
        minimum_similarity=MINIMUM_SIMILARITY,
        embedding_model="gemini-embedding-001",
        source_sha256="1" * 64,
        metadata_manifest_sha256="2" * 64,
    )
    api_retrieval = RetrievalResult(
        status="ready",
        reason_code="sources_ready",
        reason="Approved primary sources matched.",
        sources=(api_primary, api_other),
        eligible_candidate_count=2,
        excluded_candidate_counts={},
        minimum_similarity=MINIMUM_SIMILARITY,
        embedding_model="gemini-embedding-001",
        source_sha256="1" * 64,
        metadata_manifest_sha256="2" * 64,
    )
    single_api_retrieval = RetrievalResult(
        status="ready",
        reason_code="sources_ready",
        reason="Approved primary source matched.",
        sources=(api_primary,),
        eligible_candidate_count=1,
        excluded_candidate_counts={},
        minimum_similarity=MINIMUM_SIMILARITY,
        embedding_model="gemini-embedding-001",
        source_sha256="1" * 64,
        metadata_manifest_sha256="2" * 64,
    )

    raw_tickets = [
        {
            "ticket_id": "T-POST-GROUNDING",
            "channel": "email",
            "subject": "Blue green deploy",
            "body": "How do I use blue green deployment slots?",
            "labels": {
                "intent": "configuration_help",
                "urgency": "medium",
                "expected_route": "draft_for_review",
                "answerable_from_docs": True,
                "expected_doc_ids": ["DOC-DEPLOY-001"],
                "must_not_auto_respond": False,
            },
        },
        {
            "ticket_id": "T-POST-PRIVACY",
            "channel": "email",
            "subject": "Rotate key",
            "body": "How do I rotate API keys?",
        },
        {
            "ticket_id": "T-GEN-FAIL",
            "channel": "web_form",
            "subject": "Rotate key",
            "body": "How do I rotate API keys?",
        },
        {
            "ticket_id": "T-EMITTED-MISMATCH",
            "channel": "email",
            "subject": "API scopes",
            "body": "How do I configure API key scopes?",
            "labels": {
                "intent": "api_key_issue",
                "urgency": "medium",
                "expected_route": "draft_for_review",
                "answerable_from_docs": True,
                "expected_doc_ids": ["DOC-API-001"],
                "must_not_auto_respond": False,
            },
        },
    ]
    classifications = [
        TicketClassification(
            intent="configuration_help",
            urgency="medium",
            confidence=0.94,
            provider="gemini",
            model_name="gemini-decision-1",
            provider_status="classified",
        ),
        TicketClassification(
            intent="api_key_issue",
            urgency="medium",
            confidence=0.94,
            provider="gemini",
            model_name="gemini-decision-1",
            provider_status="classified",
        ),
        TicketClassification(
            intent="api_key_issue",
            urgency="medium",
            confidence=0.94,
            provider="gemini",
            model_name="gemini-decision-1",
            provider_status="classified",
        ),
        TicketClassification(
            intent="api_key_issue",
            urgency="medium",
            confidence=0.95,
            provider="gemini",
            model_name="gemini-decision-1",
            provider_status="classified",
        ),
    ]
    retrievals = [
        deploy_retrieval,
        single_api_retrieval,
        single_api_retrieval,
        api_retrieval,
    ]
    generators = [
        _CustomDraftProvider(
            claims=[
                {
                    "text": "D1 ungrounded deployment quota guarantee.",
                    "source_ids": ["DOC-DEPLOY-003:p1"],
                },
                {
                    "text": "D2 ungrounded automatic cluster failover guarantee.",
                    "source_ids": ["DOC-DEPLOY-003:p1"],
                },
            ],
            plain_text="We guarantee zero-downtime automatic failover.",
            unsupported_points=["Zero-downtime SLA guarantee is not in KB."],
            conflicting_points=["Draft claims automatic failover, KB states manual rollback."],
        ),
        _GuardrailTriggeringDraftProvider(),
        _FailingDraftProvider(),
        _CustomDraftProvider(
            claims=[
                {
                    "text": "Configure API key scopes in workspace permissions.",
                    "source_ids": ["DOC-API-002:p1"],
                }
            ],
            plain_text="Configure API key scopes in workspace permissions.",
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
        for index, (raw, classification, retrieval, generator) in enumerate(
            zip(raw_tickets, classifications, retrievals, generators, strict=True)
        ):
            _persist_ticket_outcome(
                connection,
                run_id=run_id,
                input_index=index,
                raw_ticket=raw,
                classification=classification,
                retrieval=retrieval,
                evaluator=evaluator,
                policy=policy,
                policy_fingerprint=policy_fingerprint,
                classifier_fingerprint=classifier_fingerprint,
                settings=settings,
                generator=generator,
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
        events = connection.execute(
            "SELECT event_payload_json FROM audit_events WHERE run_id = ? AND event_type = 'ticket_completed'",
            (run_id,),
        ).fetchall()
    finally:
        connection.close()

    persisted_by_id = {
        json.loads(row["event_payload_json"])["ticket_id"]: json.loads(row["event_payload_json"])["result"]
        for row in events
    }

    report = json.loads((output_dir / "failure_report.json").read_text(encoding="utf-8"))
    by_id = {item["ticket_id"]: item for item in report["tickets"]}

    # AC-1(a) & AC-2: post-draft citation_grounding block
    post_grounding = by_id["T-POST-GROUNDING"]
    persisted_pg_diag = persisted_by_id["T-POST-GROUNDING"]["post_draft_diagnostics"]
    assert post_grounding["stopped_at"] == "after_drafting"
    assert post_grounding["before_drafting"]["first_blocking_check"] is None
    assert post_grounding["drafting"]["draft_disposition"] == "withheld"
    assert post_grounding["after_drafting"]["status"] == "failed"
    assert [c["name"] for c in post_grounding["after_drafting"]["checks"]] == [
        "output_structure",
        "citation_grounding",
        "private_data_output",
        "instruction_integrity",
    ]
    assert [c["status"] for c in post_grounding["after_drafting"]["checks"]] == [
        "passed",
        "failed",
        "passed",
        "passed",
    ]
    assert post_grounding["after_drafting"]["checks"][1]["minimum_ratio"] == 0.65
    assert post_grounding["after_drafting"]["blocking_check"] == "citation_grounding"
    assert post_grounding["after_drafting"]["distinct_from_pre_draft_claim_grounding_note"]
    assert post_grounding["after_drafting"]["withheld_claims"] == persisted_pg_diag["withheld_claims"]
    assert post_grounding["after_drafting"]["unsupported_points"] == persisted_pg_diag["unsupported_points"]
    assert post_grounding["after_drafting"]["conflicting_points"] == persisted_pg_diag["conflicting_points"]
    assert post_grounding["after_drafting"]["failed_claims"] == persisted_pg_diag["failed_claims"]
    assert len(post_grounding["after_drafting"]["failed_claims"]) == 2
    assert all(
        "P1 policy verified" not in item["claim"]
        for item in post_grounding["after_drafting"]["failed_claims"]
    )
    assert post_grounding["evaluation"]["predicted"]["retrieved_doc_ids"] == [
        "DOC-DEPLOY-001",
        "DOC-DEPLOY-003",
    ]
    assert post_grounding["evaluation"]["predicted"]["cited_doc_ids"] == ["DOC-DEPLOY-003"]
    assert "expected_doc_ids" in post_grounding["evaluation"]["failed_kpis"]
    assert "DOC-DEPLOY-001" in post_grounding["evaluation"]["expected_doc_note"]
    assert "DOC-DEPLOY-003" in post_grounding["evaluation"]["expected_doc_note"]

    # AC-1(b): post-draft private_data_output block
    post_privacy = by_id["T-POST-PRIVACY"]
    persisted_priv_diag = persisted_by_id["T-POST-PRIVACY"]["post_draft_diagnostics"]
    assert post_privacy["stopped_at"] == "after_drafting"
    assert post_privacy["drafting"]["draft_disposition"] == "withheld"
    assert post_privacy["after_drafting"]["status"] == "failed"
    assert [c["name"] for c in post_privacy["after_drafting"]["checks"]] == [
        "output_structure",
        "citation_grounding",
        "private_data_output",
        "instruction_integrity",
    ]
    assert [c["status"] for c in post_privacy["after_drafting"]["checks"]] == [
        "passed",
        "passed",
        "failed",
        "passed",
    ]
    assert post_privacy["after_drafting"]["blocking_check"] == "private_data_output"
    assert post_privacy["after_drafting"]["distinct_from_pre_draft_claim_grounding_note"] is None
    assert post_privacy["after_drafting"]["withheld_claims"] == persisted_priv_diag["withheld_claims"]
    assert post_privacy["after_drafting"]["failed_claims"] == []

    # AC-1(c): generation.status == "failed" provider error
    gen_fail = by_id["T-GEN-FAIL"]
    assert gen_fail["stopped_at"] == "during_drafting"
    assert gen_fail["drafting"]["draft_disposition"] == "failed"
    assert gen_fail["after_drafting"]["status"] == "failed"
    assert [c["name"] for c in gen_fail["after_drafting"]["checks"]] == [
        "output_structure",
        "citation_grounding",
        "private_data_output",
        "instruction_integrity",
    ]
    assert [c["status"] for c in gen_fail["after_drafting"]["checks"]] == [
        "failed",
        "failed",
        "passed",
        "passed",
    ]
    assert gen_fail["after_drafting"]["blocking_check"] == "output_structure"
    assert gen_fail["after_drafting"]["withheld_claims"] == []
    assert gen_fail["after_drafting"]["unsupported_points"] == []
    assert gen_fail["after_drafting"]["conflicting_points"] == []
    assert gen_fail["after_drafting"]["failed_claims"] == []

    # AC-3: emitted draft_for_review ticket with cited_doc_ids KPI mismatch
    emitted = by_id["T-EMITTED-MISMATCH"]
    assert emitted["stopped_at"] == "evaluation_only"
    assert emitted["drafting"]["draft_disposition"] == "emitted"
    assert emitted["after_drafting"]["status"] == "passed"
    assert [c["name"] for c in emitted["after_drafting"]["checks"]] == [
        "output_structure",
        "citation_grounding",
        "private_data_output",
        "instruction_integrity",
    ]
    assert all(c["status"] == "passed" for c in emitted["after_drafting"]["checks"])
    assert emitted["evaluation"]["predicted"]["retrieved_doc_ids"] == [
        "DOC-API-001",
        "DOC-API-002",
    ]
    assert emitted["evaluation"]["predicted"]["cited_doc_ids"] == ["DOC-API-002"]
    assert "expected_doc_ids" in emitted["evaluation"]["failed_kpis"]
    assert "DOC-API-001" in emitted["evaluation"]["expected_doc_note"]
    assert "DOC-API-002" in emitted["evaluation"]["expected_doc_note"]


def test_write_run_exports_handles_legacy_post_draft_block_without_diagnostics(
    tmp_path: Path,
) -> None:
    """AC-4: Legacy SQLite terminal events without post_draft_diagnostics export cleanly."""
    run_id = "5" * 32
    policy = json.loads(TEST_ROUTING_POLICY_PATH.read_text(encoding="utf-8"))
    policy_fingerprint = fingerprint_policy_file(TEST_ROUTING_POLICY_PATH)
    classifier_fingerprint = fingerprint_classifier_model("gemini", "gemini-decision-1")
    settings = Settings.from_env({"DRAFT_GENERATION_PROVIDER": "gemini"})
    api_primary = _sample_kb_source(
        "DOC-API-001",
        "DOC-API-001:p1",
        "API Keys",
        "Rotate API keys from the dashboard settings page.",
    )
    evaluator = RoutingPolicyEvaluator(
        policy,
        claim_grounding_fn=lambda *, ticket, sources, related_sources: {
            "claims": [
                {
                    "text": "P1 policy claim from pre-draft routing.",
                    "source_ids": ["DOC-API-001:p1"],
                }
            ],
            "unsupported_points": [],
            "conflicting_points": [],
            "confidence": 0.95,
            "reason": "Policy claim passed.",
        },
    )
    retrieval = RetrievalResult(
        status="ready",
        reason_code="sources_ready",
        reason="Approved primary source matched.",
        sources=(api_primary,),
        eligible_candidate_count=1,
        excluded_candidate_counts={},
        minimum_similarity=MINIMUM_SIMILARITY,
        embedding_model="gemini-embedding-001",
        source_sha256="1" * 64,
        metadata_manifest_sha256="2" * 64,
    )
    raw_ticket = {
        "ticket_id": "T-LEGACY-POST",
        "channel": "email",
        "subject": "Rotate key",
        "body": "How do I rotate API keys?",
    }
    db_path = tmp_path / "legacy.sqlite3"
    output_dir = tmp_path / "outputs" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = tmp_path / "legacy.json"
    input_path.write_text(json.dumps([raw_ticket]), encoding="utf-8")

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
        _persist_ticket_outcome(
            connection,
            run_id=run_id,
            input_index=0,
            raw_ticket=raw_ticket,
            classification=TicketClassification(
                intent="api_key_issue",
                urgency="medium",
                confidence=0.95,
                provider="gemini",
                model_name="gemini-decision-1",
                provider_status="classified",
            ),
            retrieval=retrieval,
            evaluator=evaluator,
            policy=policy,
            policy_fingerprint=policy_fingerprint,
            classifier_fingerprint=classifier_fingerprint,
            settings=settings,
            generator=_CustomDraftProvider(
                claims=[
                    {
                        "text": "Ungrounded guarantee not in KB.",
                        "source_ids": ["DOC-API-001:p1"],
                    }
                ],
                plain_text="Ungrounded guarantee not in KB.",
            ),
        )
        # Strip post_draft_diagnostics from the persisted SQLite event to simulate a legacy row.
        row = connection.execute(
            "SELECT event_id, event_payload_json FROM audit_events WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        payload = json.loads(row["event_payload_json"])
        payload["result"].pop("post_draft_diagnostics", None)
        connection.execute(
            "UPDATE audit_events SET event_payload_json = ? WHERE event_id = ?",
            (json.dumps(payload), row["event_id"]),
        )
        connection.commit()

        write_run_exports(
            connection,
            run_id=run_id,
            output_directory=output_dir,
            input_records=[raw_ticket],
            input_path=input_path,
            source_ref="external",
        )
    finally:
        connection.close()

    report = json.loads((output_dir / "failure_report.json").read_text(encoding="utf-8"))
    legacy_item = report["tickets"][0]
    assert legacy_item["stopped_at"] == "after_drafting"
    assert legacy_item["after_drafting"]["failed_claims"] == []
    assert legacy_item["after_drafting"]["withheld_claims"] == []
    assert legacy_item["evaluation"]["predicted"]["cited_doc_ids"] == []
    assert isinstance(legacy_item["after_drafting"]["diagnostics_note"], str)
    assert legacy_item["after_drafting"]["diagnostics_note"]


def test_write_run_exports_renders_simplified_failure_report_markdown_ac1(
    tmp_path: Path,
) -> None:
    """AC-1: Render failure_report.md header, summary table, per-ticket sections, and footer from failure_report.json."""
    run_id = "6" * 32
    policy = json.loads(TEST_ROUTING_POLICY_PATH.read_text(encoding="utf-8"))
    policy_fingerprint = fingerprint_policy_file(TEST_ROUTING_POLICY_PATH)
    classifier_fingerprint = fingerprint_classifier_model("gemini", "gemini-decision-1")
    settings = Settings.from_env({"DRAFT_GENERATION_PROVIDER": "gemini"})

    deploy_primary = _sample_kb_source(
        "DOC-DEPLOY-001",
        "DOC-DEPLOY-001:p1",
        "Blue Green Deployments",
        "Use blue green deployment slots in the deployment console.",
    )
    deploy_other = _sample_kb_source(
        "DOC-DEPLOY-003",
        "DOC-DEPLOY-003:p1",
        "Rollback Runbook",
        "Rollback from deployment history.",
    )

    def _grounding_fn(*, ticket: Any, sources: Any, related_sources: Any) -> dict[str, Any]:
        del ticket, sources, related_sources
        return {
            "claims": [
                {
                    "text": "Use blue green deployment slots in the deployment console.",
                    "source_ids": ["DOC-DEPLOY-001:p1"],
                }
            ],
            "unsupported_points": [],
            "conflicting_points": [],
            "confidence": 0.95,
            "reason": "Supported by deployment guide.",
        }

    evaluator = RoutingPolicyEvaluator(policy, claim_grounding_fn=_grounding_fn)
    primary_only = RetrievalResult(
        status="ready",
        reason_code="sources_ready",
        reason="Primary source matched.",
        sources=(deploy_primary,),
        eligible_candidate_count=1,
        excluded_candidate_counts={},
        minimum_similarity=MINIMUM_SIMILARITY,
        embedding_model="gemini-embedding-001",
        source_sha256="1" * 64,
        metadata_manifest_sha256="2" * 64,
    )
    primary_and_related = RetrievalResult(
        status="ready",
        reason_code="sources_ready",
        reason="Primary and related sources matched.",
        sources=(deploy_primary, {**deploy_other, "source_kind": "related"}),
        eligible_candidate_count=2,
        excluded_candidate_counts={},
        minimum_similarity=MINIMUM_SIMILARITY,
        embedding_model="gemini-embedding-001",
        source_sha256="1" * 64,
        metadata_manifest_sha256="2" * 64,
    )
    skipped_retrieval = RetrievalResult(
        status="skipped",
        reason_code="low_confidence",
        reason="Skipped retrieval below confidence threshold.",
        sources=(),
        eligible_candidate_count=0,
        excluded_candidate_counts={},
        minimum_similarity=MINIMUM_SIMILARITY,
        embedding_model="gemini-embedding-001",
        source_sha256="1" * 64,
        metadata_manifest_sha256="2" * 64,
    )

    three_rejected_provider = _CustomDraftProvider(
        claims=[
            {"text": "Use blue green deployment slots.", "source_ids": ["DOC-DEPLOY-003:p1"]},
            {"text": "Enable zero-downtime swap.", "source_ids": ["DOC-DEPLOY-003:p1"]},
            {"text": "Verify slot health.", "source_ids": ["DOC-DEPLOY-003:p1"]},
        ],
        plain_text="Use blue green deployment slots. Enable zero-downtime swap. Verify slot health. Extra ungrounded clause.",
    )
    # Use unknown passage_id on all 3 claims so all 3 clauses fail citation_grounding.
    three_rejected_provider_ungrounded = _CustomDraftProvider(
        claims=[],
        plain_text="First ungrounded clause.",
    )

    class _ThreeRejectedProvider:
        name = "gemini"

        def generate_draft(
            self,
            ticket: CanonicalTicket,
            *,
            sources: tuple[dict[str, Any], ...],
            related_sources: tuple[dict[str, Any], ...] = (),
            policy_evaluation: Mapping[str, Any] | None = None,
            model_name: str = "",
        ) -> dict[str, Any]:
            draft = three_rejected_provider.generate_draft(
                ticket,
                sources=sources,
                related_sources=related_sources,
                policy_evaluation=policy_evaluation,
                model_name=model_name,
            )
            # Mark claims as citing DOC-DEPLOY-003:p1 while stripping it from sources_used/passage map via unknown suffix.
            for claim in draft["claims"]:
                claim["citations"] = [
                    {
                        "passage_id": "DOC-DEPLOY-003:missing",
                        "source_ref": "data/kb/documentation.json#DOC-DEPLOY-003#Overview",
                    }
                ]
            return draft

    del three_rejected_provider_ungrounded

    raw_tickets = [
        {
            "ticket_id": "T-MD-CONF",
            "channel": "chat",
            "subject": "Unclear issue",
            "body": "Something failed.",
            "labels": {
                "intent": "billing_query",
                "urgency": "high",
                "expected_route": "escalate",
                "answerable_from_docs": False,
                "expected_doc_ids": [],
                "must_not_auto_respond": False,
            },
        },
        {
            "ticket_id": "T-MD-POST",
            "channel": "email",
            "subject": "Blue green deployment",
            "body": "How do I use blue green deployment slots?",
            "labels": {
                "intent": "deployment_failure",
                "urgency": "medium",
                "expected_route": "draft_for_review",
                "answerable_from_docs": True,
                "expected_doc_ids": ["DOC-DEPLOY-001"],
                "must_not_auto_respond": False,
            },
        },
        {
            "ticket_id": "T-MD-GEN",
            "channel": "email",
            "subject": "Blue green deployment",
            "body": "How do I use blue green deployment slots?",
        },
        {
            "ticket_id": "T-MD-EVAL",
            "channel": "email",
            "subject": "Blue green deployment",
            "body": "How do I use blue green deployment slots?",
            "labels": {
                "intent": "deployment_failure",
                "urgency": "low",
                "expected_route": "draft_for_review",
                "answerable_from_docs": True,
                "expected_doc_ids": ["DOC-DEPLOY-001"],
                "must_not_auto_respond": False,
            },
        },
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
        configs = [
            (
                TicketClassification(
                    intent="deployment_failure",
                    urgency="medium",
                    confidence=0.541,
                    provider="gemini",
                    model_name="gemini-decision-1",
                    provider_status="classified",
                ),
                skipped_retrieval,
                OfflineDraftGenerationProvider(),
            ),
            (
                TicketClassification(
                    intent="deployment_failure",
                    urgency="medium",
                    confidence=0.92,
                    provider="gemini",
                    model_name="gemini-decision-1",
                    provider_status="classified",
                ),
                primary_and_related,
                _ThreeRejectedProvider(),
            ),
            (
                TicketClassification(
                    intent="deployment_failure",
                    urgency="medium",
                    confidence=0.92,
                    provider="gemini",
                    model_name="gemini-decision-1",
                    provider_status="classified",
                ),
                primary_only,
                _FailingDraftProvider(),
            ),
            (
                TicketClassification(
                    intent="deployment_failure",
                    urgency="medium",
                    confidence=0.94,
                    provider="gemini",
                    model_name="gemini-decision-1",
                    provider_status="classified",
                ),
                primary_only,
                OfflineDraftGenerationProvider(),
            ),
        ]
        for index, (classification, retrieval, generator) in enumerate(configs):
            _persist_ticket_outcome(
                connection,
                run_id=run_id,
                input_index=index,
                raw_ticket=raw_tickets[index],
                classification=classification,
                retrieval=retrieval,
                evaluator=evaluator,
                policy=policy,
                policy_fingerprint=policy_fingerprint,
                classifier_fingerprint=classifier_fingerprint,
                settings=settings,
                generator=generator,
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
    finally:
        connection.close()

    md = (output_dir / "failure_report.md").read_text(encoding="utf-8")
    assert f"- Run: `{run_id}`" in md
    assert "- Processing: 3/4 completed (1 processing failures, 1 exceptions)" in md
    assert "- Outcome: 1 drafts for review, 2 escalated, 1 degraded, 3 drafts blocked by guardrails" in md
    assert "- Evaluation:" in md
    assert "| Ticket | Stopped at | Main reason | Result |" in md

    table_rows = [line for line in md.splitlines() if line.startswith("| T-MD-")]
    assert [row.split("|")[1].strip() for row in table_rows] == [
        "T-MD-CONF",
        "T-MD-POST",
        "T-MD-GEN",
        "T-MD-EVAL",
    ]
    assert "## T-MD-CONF — Before drafting" in md
    assert "0.541 < 0.65" in md
    assert "Skipped source_presence_check, claim_grounding_check" in md
    assert "intent: billing_query -> deployment_failure" in md
    assert "urgency: high -> medium" in md

    assert "## T-MD-POST — After drafting" in md
    assert "3 rejected clause(s)" in md
    assert "DOC-DEPLOY-003" in md
    assert "- Document note: Expected doc(s) DOC-DEPLOY-001 were retrieved" in md
    assert "- Document note:" not in md.split("## T-MD-EVAL")[1]

    assert "## T-MD-GEN — During drafting" in md
    assert "generation_timeout" in md

    assert "## T-MD-EVAL — Evaluation only" in md
    assert "urgency: low -> medium" in md
    assert "See `failure_report.json` for full check statuses, thresholds, claim/clause details, source IDs, and metrics." in md


def test_write_run_exports_renders_clean_zero_failure_markdown_and_sanitizes_table_cells(
    tmp_path: Path,
) -> None:
    """AC-2 & AC-3: Zero-failure runs omit ticket sections; surrogates, newlines, and pipes stay single-line."""
    policy = json.loads(TEST_ROUTING_POLICY_PATH.read_text(encoding="utf-8"))
    policy_fingerprint = fingerprint_policy_file(TEST_ROUTING_POLICY_PATH)
    classifier_fingerprint = fingerprint_classifier_model("gemini", "gemini-decision-1")
    settings = Settings.from_env({"DRAFT_GENERATION_PROVIDER": "offline"})
    primary = _sample_kb_source(
        "DOC-API-001",
        "DOC-API-001:p1",
        "API Keys",
        "Rotate API keys from the dashboard settings page.",
    )
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
    ready_retrieval = RetrievalResult(
        status="ready",
        reason_code="sources_ready",
        reason="Approved primary source matched.",
        sources=(primary,),
        eligible_candidate_count=1,
        excluded_candidate_counts={},
        minimum_similarity=MINIMUM_SIMILARITY,
        embedding_model="gemini-embedding-001",
        source_sha256="1" * 64,
        metadata_manifest_sha256="2" * 64,
    )

    # AC-2: Clean run.
    clean_run_id = "7" * 32
    clean_dir = tmp_path / "outputs" / clean_run_id
    clean_dir.mkdir(parents=True, exist_ok=True)
    clean_input = tmp_path / "clean.json"
    clean_ticket = {
        "ticket_id": "T-CLEAN-MD",
        "channel": "email",
        "subject": "Rotate API key",
        "body": "How do I rotate API keys from the dashboard settings page?",
    }
    clean_input.write_text(json.dumps([clean_ticket]), encoding="utf-8")
    conn = connect_runtime_database(tmp_path / "clean.sqlite3")
    try:
        create_processing_run(
            conn,
            run_id=clean_run_id,
            input_ref=str(clean_input),
            started_at="2026-11-10T12:00:00Z",
            label=None,
            output_dir=str(clean_dir),
        )
        _persist_ticket_outcome(
            conn,
            run_id=clean_run_id,
            input_index=0,
            raw_ticket=clean_ticket,
            classification=TicketClassification(
                intent="api_key_issue",
                urgency="medium",
                confidence=0.95,
                provider="gemini",
                model_name="gemini-decision-1",
                provider_status="classified",
            ),
            retrieval=ready_retrieval,
            evaluator=evaluator,
            policy=policy,
            policy_fingerprint=policy_fingerprint,
            classifier_fingerprint=classifier_fingerprint,
            settings=settings,
            generator=OfflineDraftGenerationProvider(),
        )
        conn.commit()
        write_run_exports(
            conn,
            run_id=clean_run_id,
            output_directory=clean_dir,
            input_records=[clean_ticket],
            input_path=clean_input,
            source_ref="external",
        )
    finally:
        conn.close()

    clean_md = (clean_dir / "failure_report.md").read_text(encoding="utf-8")
    assert f"- Run: `{clean_run_id}`" in clean_md
    assert "No operational failures or evaluation KPI mismatches occurred in this run." in clean_md
    assert "| Ticket | Stopped at | Main reason | Result |" not in clean_md
    assert "\n## " not in clean_md
    assert "failure_report.json" in clean_md

    # AC-3: Unpaired surrogate, newline, and pipe in failing ticket fields.
    bad_run_id = "8" * 32
    bad_dir = tmp_path / "outputs" / bad_run_id
    bad_dir.mkdir(parents=True, exist_ok=True)
    bad_input = tmp_path / "bad.json"
    bad_ticket = {
        "ticket_id": "T-PIPE|BS\\|LINE",
        "channel": "chat",
        "subject": "Bad chars",
        "body": "Help",
    }
    bad_input.write_text(json.dumps([bad_ticket]), encoding="utf-8")
    bad_conn = connect_runtime_database(tmp_path / "bad.sqlite3")
    try:
        create_processing_run(
            bad_conn,
            run_id=bad_run_id,
            input_ref=str(bad_input),
            started_at="2026-11-10T12:00:00Z",
            label=None,
            output_dir=str(bad_dir),
        )
        _persist_ticket_outcome(
            bad_conn,
            run_id=bad_run_id,
            input_index=0,
            raw_ticket=bad_ticket,
            classification=TicketClassification(
                intent="api_key_issue",
                urgency="medium",
                confidence=0.40,
                provider="gemini",
                model_name="gemini-decision-1",
                provider_status="classified",
            ),
            retrieval=RetrievalResult(
                status="skipped",
                reason_code="low_confidence",
                reason="Low confidence",
                sources=(),
                eligible_candidate_count=0,
                excluded_candidate_counts={},
                minimum_similarity=MINIMUM_SIMILARITY,
                embedding_model="gemini-embedding-001",
                source_sha256="1" * 64,
                metadata_manifest_sha256="2" * 64,
            ),
            evaluator=evaluator,
            policy=policy,
            policy_fingerprint=policy_fingerprint,
            classifier_fingerprint=classifier_fingerprint,
            settings=settings,
            generator=OfflineDraftGenerationProvider(),
        )
        bad_conn.execute(
            "UPDATE audit_events SET event_payload_json = replace(event_payload_json, 'below the frozen threshold', 'below \\ud800\\n|threshold') WHERE run_id = ?",
            (bad_run_id,),
        )
        bad_conn.commit()
        write_run_exports(
            bad_conn,
            run_id=bad_run_id,
            output_directory=bad_dir,
            input_records=[bad_ticket],
            input_path=bad_input,
            source_ref="external",
        )
    finally:
        bad_conn.close()

    bad_md = (bad_dir / "failure_report.md").read_text(encoding="utf-8")
    assert "\\ud800" in bad_md
    table_lines = [line for line in bad_md.splitlines() if line.startswith("| T-PIPE")]
    assert len(table_lines) == 1
    assert "T-PIPE\\|BS\\\\\\|LINE" in table_lines[0]
    assert re.sub(r"\\.", "", table_lines[0]).count("|") == 5



