"""Process one ticket batch into SQLite audit rows and isolated run exports."""

from __future__ import annotations

import json
import re
import sqlite3
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from cloudserve_support.classification.ticket_classification import (
    DECISIONS_CLASSIFIER_MODEL,
    TicketClassification,
    classify_ticket_batches,
    classification_failure,
)
from cloudserve_support.config import Settings
from cloudserve_support.domain.ticket import CanonicalTicket
from cloudserve_support.generation.draft_generation import (
    GENERATION_PROMPT_VERSION,
    DraftGenerationProvider,
    create_draft_generation_provider,
)
from cloudserve_support.guardrails.draft_safety import (
    contains_private_data,
    contains_prompt_injection,
    filter_evaluation_context,
    ticket_text_for_generation,
)
from cloudserve_support.ingestion.ticket_input import read_ticket_records
from cloudserve_support.ingestion.ticket_normalization import normalize_ticket_record
from cloudserve_support.knowledge.retrieval import (
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_QUERY_BATCH_SIZE,
    MINIMUM_SIMILARITY,
    KnowledgeRetriever,
    RetrievalResult,
)
from cloudserve_support.persistence.runtime_database import (
    connect_runtime_database,
    create_processing_run,
    create_audit_event_record,
    create_ticket_run_record,
    update_terminal_audit_event_payload,
    update_ticket_run_status,
    update_processing_run_status,
)
from cloudserve_support.reporting.output_schemas import load_output_schemas
from cloudserve_support.reporting.run_exports import (
    record_batch_export_duration,
    write_markdown_run_metrics,
    write_run_exports,
)
from cloudserve_support.processing.run_provenance import (
    fingerprint_classifier_implementation,
    fingerprint_policy_file,
)
from cloudserve_support.routing.policy_calibration import (
    load_frozen_routing_policy,
)
from cloudserve_support.processing.ticket_outcomes import (
    PreRoutingSafetyEvaluation,
    _report_pre_routing_progress,
    build_pre_routing_checks,
    build_ticket_outcome,
    evaluate_pre_routing_safety,
)
from cloudserve_support.routing.routing_policy import (
    CLAIM_GROUNDING_PROMPT_VERSION,
    RoutingDecision,
    RoutingPolicyEvaluator,
    create_claim_grounding_checker,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATABASE_PATH = REPO_ROOT / "data" / "runtime" / "cloudserve.sqlite3"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs"
DEFAULT_ROUTING_POLICY_CONFIG_PATH = REPO_ROOT / "conf" / "frozen_routing_policy.json"


@dataclass
class _PreparedTicket:
    """Carry each input position's typed outcomes across pipeline phases."""

    input_index: int
    ticket: CanonicalTicket
    storage_ticket_id: str
    input_labels_json: str | None
    issues: list[str]
    duplicate_ticket_id: bool
    pre_provider_guardrail_block: bool
    classification: TicketClassification | None
    retrieval: RetrievalResult | None = None
    decision: RoutingDecision | None = None
    policy_evaluation: dict[str, Any] | None = None
    pre_routing_safety: PreRoutingSafetyEvaluation | None = None
    classification_request_duration_seconds: float = 0.0


def run_ticket_batch(
    *,
    input_path: Path,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    database_path: Path = DEFAULT_DATABASE_PATH,
    policy_path: Path = DEFAULT_ROUTING_POLICY_CONFIG_PATH,
    label: str | None = None,
    skip_draft: bool = False,
    skip_draft_review: bool = False,
    retriever: KnowledgeRetriever | None = None,
    generator: DraftGenerationProvider | None = None,
    progress_callback: Callable[[str, str], None] | None = None,
) -> tuple[str, Path, dict[str, Any]]:
    """Classify, retrieve, draft or hand off, and persist every ticket occurrence."""
    input_records = read_ticket_records(input_path)
    resolved_policy_path = policy_path.expanduser().resolve()
    policy = load_frozen_routing_policy(resolved_policy_path)
    policy_file_fingerprint = fingerprint_policy_file(resolved_policy_path)
    classifier_implementation_fingerprint = fingerprint_classifier_implementation()
    settings = Settings.from_env()
    if settings.gemini_api_key and (
        policy["classifier_provider"] != "gemini"
        or policy["classifier_model"] != DECISIONS_CLASSIFIER_MODEL
    ):
        raise RuntimeError(
            "The frozen routing policy does not match the Decisions classifier model. "
            "Run `cloudserve calibrate --development-input PATH` before processing tickets."
        )
    knowledge_retriever = retriever or KnowledgeRetriever(settings)
    draft_generator = generator or create_draft_generation_provider(settings)
    load_output_schemas()
    resolved_input_path = input_path.expanduser().resolve()
    resolved_output_root = output_root.expanduser().resolve()
    resolved_database_path = database_path.expanduser().resolve()
    started_datetime = datetime.now(timezone.utc)
    started_at = started_datetime.isoformat().replace("+00:00", "Z")

    connection = connect_runtime_database(resolved_database_path)
    run_record_committed = False
    try:
        resolved_output_root.mkdir(parents=True, exist_ok=True)
        connection.execute("BEGIN IMMEDIATE")
        run_id, output_directory = _allocate_run_id_and_directory(
            connection=connection,
            output_root=resolved_output_root,
            label=label,
            started_datetime=started_datetime,
        )
        create_processing_run(
            connection,
            run_id=run_id,
            input_ref=_display_path(resolved_input_path),
            started_at=started_at,
            label=label,
            output_dir=_display_path(output_directory),
        )
        connection.commit()
        run_record_committed = True
        prepared_tickets = _prepare_ticket_records(input_records)
        connection.execute("BEGIN IMMEDIATE")
        for prepared_ticket in prepared_tickets:
            ticket = prepared_ticket.ticket
            create_ticket_run_record(
                connection,
                run_id=run_id,
                storage_ticket_id=prepared_ticket.storage_ticket_id,
                channel=ticket.channel,
                received_at=ticket.received_at,
                customer_tier=ticket.customer_tier,
                language_fluency=ticket.language_fluency,
                context=ticket.context,
                input_labels_json=prepared_ticket.input_labels_json,
            )
        connection.commit()
        classifiable_tickets = [
            prepared_ticket
            for prepared_ticket in prepared_tickets
            if prepared_ticket.classification is None
        ]
        _report_run_progress(progress_callback, "Classification", "in_progress")
        if classifiable_tickets:
            classification_batches = classify_ticket_batches(
                [prepared_ticket.ticket for prepared_ticket in classifiable_tickets],
                settings,
            )
            for classification_batch in classification_batches:
                for batch_offset, classification in enumerate(
                    classification_batch.classifications
                ):
                    prepared_ticket = classifiable_tickets[
                        classification_batch.start_index + batch_offset
                    ]
                    prepared_ticket.classification = classification
                    prepared_ticket.classification_request_duration_seconds = (
                        classification_batch.request_duration_seconds
                    )

        for prepared_ticket in prepared_tickets:
            classification = prepared_ticket.classification
            if classification is None:
                classification = classification_failure(
                    provider="not_attempted",
                    model_name=DECISIONS_CLASSIFIER_MODEL,
                    uncertainty="Ticket was not included in a Decisions classification batch.",
                )
                prepared_ticket.classification = classification
            _persist_stage_checkpoint(
                connection,
                run_id=run_id,
                prepared_ticket=prepared_ticket,
                stage="classification",
                status=classification.provider_status,
                value=classification.as_dict(),
                reason=(
                    classification.failure_reason
                    or (
                        "Classification was not attempted because input validation or a pre-provider guardrail blocked it."
                        if classification.provider == "not_attempted"
                        else "Ticket classification completed."
                    )
                ),
                model_name=classification.model_name,
                prediction=classification.as_dict(),
            )
        _report_run_progress(progress_callback, "Classification", "done")

        _report_run_progress(progress_callback, "Retrieval", "in_progress")
        retrieval_results = _retrieve_ticket_results(
            prepared_tickets,
            retriever=knowledge_retriever,
            settings=settings,
        )
        for prepared_ticket, retrieval in zip(prepared_tickets, retrieval_results, strict=True):
            prepared_ticket.retrieval = retrieval
            _persist_stage_checkpoint(
                connection,
                run_id=run_id,
                prepared_ticket=prepared_ticket,
                stage="retrieval",
                status=retrieval.status,
                value={
                    **retrieval.as_dict(),
                    "passages": [
                        {
                            "doc_id": source["doc_id"],
                            "passage_id": source["passage_id"],
                            "similarity": source.get("similarity"),
                        }
                        for source in retrieval.sources
                    ],
                },
                reason=retrieval.reason,
            )
        _report_run_progress(progress_callback, "Retrieval", "done")

        _report_run_progress(progress_callback, "Routing", "in_progress")
        routing_evaluator = RoutingPolicyEvaluator(
            policy,
            claim_grounding_fn=create_claim_grounding_checker(settings),
        )
        for routing_position, prepared_ticket in enumerate(prepared_tickets, start=1):
            routing_progress_label = (
                f"Routing ticket {routing_position}/{len(prepared_tickets)} "
                f"({prepared_ticket.ticket.ticket_id!r})"
            )
            _report_run_progress(
                progress_callback,
                routing_progress_label,
                "routing_ticket_started",
            )
            classification = prepared_ticket.classification
            if classification is None:
                classification = classification_failure(
                    provider="not_attempted",
                    model_name=DECISIONS_CLASSIFIER_MODEL,
                    uncertainty="Ticket was not included in a Decisions classification batch.",
                )
                prepared_ticket.classification = classification
            retrieval = prepared_ticket.retrieval
            if retrieval is None:
                raise RuntimeError("Ticket retrieval phase did not resolve every input position.")
            policy_evaluation = route_ticket_classification(
                routing_evaluator,
                ticket=prepared_ticket.ticket,
                classification=classification.as_dict(),
                retrieval=retrieval,
                pre_provider_guardrail_blocked=prepared_ticket.pre_provider_guardrail_block,
            )
            prepared_ticket.policy_evaluation = policy_evaluation
            prepared_ticket.decision = RoutingPolicyEvaluator.to_routing_decision(
                policy_evaluation
            )
            decision = prepared_ticket.decision
            if progress_callback is not None:
                prepared_ticket.pre_routing_safety = evaluate_pre_routing_safety(
                    prepared_ticket.ticket,
                    retrieval,
                )
            used_gemini_grounding = _routing_used_gemini_grounding(
                policy_evaluation,
                settings,
            )
            _persist_stage_checkpoint(
                connection,
                run_id=run_id,
                prepared_ticket=prepared_ticket,
                stage="routing",
                status="decided",
                value=policy_evaluation,
                reason=decision.reason,
                model_name=settings.llm_model if used_gemini_grounding else None,
                prompt_version=(
                    CLAIM_GROUNDING_PROMPT_VERSION if used_gemini_grounding else None
                ),
            )
            if progress_callback is not None:
                grounding_checker = routing_evaluator.claim_grounding_fn
                grounding_request_duration = getattr(
                    grounding_checker,
                    "last_request_duration_seconds",
                    0.0,
                )
                if (
                    used_gemini_grounding
                    and isinstance(grounding_request_duration, (int, float))
                    and not isinstance(grounding_request_duration, bool)
                    and grounding_request_duration > 0
                ):
                    _report_run_progress(
                        progress_callback,
                        "Gemini claim-grounding request took "
                        f"{grounding_request_duration:.2f}s",
                        "routing_ticket_detail",
                    )
                _report_run_progress(
                    progress_callback,
                    f"Routing decision: {decision.route} ({decision.matched_rule})",
                    "routing_ticket_detail",
                )
            _report_run_progress(
                progress_callback,
                routing_progress_label,
                "routing_ticket_finished",
            )
        _report_run_progress(progress_callback, "Routing", "done")

        _report_run_progress(progress_callback, "Ticket processing", "in_progress")

        def report_ticket_step(step: str) -> None:
            _report_run_progress(progress_callback, step, "ticket_step")

        for ticket_position, prepared_ticket in enumerate(prepared_tickets, start=1):
            ticket = prepared_ticket.ticket
            _report_run_progress(
                progress_callback,
                f"Ticket {ticket_position}/{len(prepared_tickets)} ({ticket.ticket_id!r})",
                "ticket",
            )
            if progress_callback is not None:
                policy_evaluation = prepared_ticket.policy_evaluation
                decision = prepared_ticket.decision
                safety_eval = prepared_ticket.pre_routing_safety
                if policy_evaluation is None or decision is None or safety_eval is None:
                    raise RuntimeError("Ticket pre-draft checks were not fully resolved.")
                pre_routing_checks = build_pre_routing_checks(
                    routing_checks=policy_evaluation.get("checks", ()),
                    safety_checks=safety_eval.safety_checks,
                )
                _report_pre_routing_progress(report_ticket_step, pre_routing_checks)
            ticket_started = time.perf_counter()
            result, storage_ticket_id, ticket = _build_ticket_result(
                prepared_ticket,
                run_id=run_id,
                settings=settings,
                policy=policy,
                policy_file_fingerprint=policy_file_fingerprint,
                classifier_implementation_fingerprint=classifier_implementation_fingerprint,
                generator=draft_generator,
                skip_draft=skip_draft,
                skip_draft_review=skip_draft_review,
                ticket_step_callback=(
                    report_ticket_step if progress_callback is not None else None
                ),
            )
            draft_request_duration = result["draft_generation_request_duration_seconds"]
            if progress_callback is not None and draft_request_duration > 0:
                report_ticket_step(
                    f"Draft generation provider request took {draft_request_duration:.2f}s"
                )
            _report_run_progress(
                progress_callback,
                "Persisting audit events and ticket status",
                "ticket_step",
            )
            occurred_at = _utc_timestamp()
            connection.execute("BEGIN IMMEDIATE")
            generation = result["generation"]
            _insert_audit_event(
                connection,
                run_id=run_id,
                storage_ticket_id=storage_ticket_id,
                event_id=uuid.uuid4().hex,
                occurred_at=occurred_at,
                event_type="generation",
                result_payload={
                    "input_index": prepared_ticket.input_index,
                    "ticket_id": ticket.ticket_id,
                    "stage": "generation",
                    "status": generation["status"],
                    "value": {
                        **generation,
                        "draft_present": result.get("draft") is not None,
                    },
                    "reason": (
                        result.get("error", {}).get("message")
                        or "Draft generation completed or was safely skipped."
                    ),
                },
                model_name=generation.get("model_name"),
                prompt_version=GENERATION_PROMPT_VERSION,
                reason=(
                    result.get("error", {}).get("message")
                    or "Draft generation completed or was safely skipped."
                ),
            )
            connection.commit()

            connection.execute("BEGIN IMMEDIATE")
            error = result.get("error", {})
            terminal_event_type = (
                "ticket_failed" if result["status"] == "failed" else "ticket_completed"
            )
            terminal_event_id = uuid.uuid4().hex
            terminal_payload = {
                "input_index": prepared_ticket.input_index,
                "ticket_id": ticket.ticket_id,
                "result": result,
            }
            _insert_audit_event(
                connection,
                run_id=run_id,
                storage_ticket_id=storage_ticket_id,
                event_id=terminal_event_id,
                occurred_at=_utc_timestamp(),
                event_type=terminal_event_type,
                result_payload=terminal_payload,
                input_summary=ticket.body,
                model_name=result["classification"]["model_name"],
                prompt_version=GENERATION_PROMPT_VERSION,
                prediction=result["classification"],
                requirement_ids=[
                    "A3", "A4", "A5", "A6", "A7", "A11", "FR-01", "FR-02",
                    "FR-03", "FR-04", "FR-05", "FR-06", "FR-07", "FR-08", "FR-11",
                ],
                threshold_applied=result["routing"]["confidence_threshold"],
                action_taken=result["route"],
                reason=result["reason"],
                guardrail_results={
                    "protected_case": result["routing"]["protected_case"],
                    "matched_policy_rule": result["routing"]["matched_rule"],
                    "evidence_ready": result["routing"]["evidence_ready"],
                    "retrieval": result["routing"]["retrieval"],
                    "checks": result["guardrails"]["checks"],
                },
                sources=result["sources"],
            )
            update_ticket_run_status(
                connection,
                run_id=run_id,
                storage_ticket_id=storage_ticket_id,
                processing_status=result["status"],
                error_code=error.get("code"),
                error_message=error.get("message"),
            )
            _commit_ticket_records(connection)
            result["processing_duration_seconds"] = max(
                prepared_ticket.classification_request_duration_seconds
                + time.perf_counter()
                - ticket_started,
                0.0,
            )
            generation_request_duration = result[
                "draft_generation_request_duration_seconds"
            ]
            result["processing_latency_seconds"] = max(
                result["processing_duration_seconds"] - generation_request_duration,
                0.0,
            )
            connection.execute("BEGIN IMMEDIATE")
            update_terminal_audit_event_payload(
                connection,
                event_id=terminal_event_id,
                result_payload=terminal_payload,
            )
            connection.commit()
            _report_run_progress(progress_callback, "Ticket result persisted", "ticket_step")
            _report_run_progress(progress_callback, "", "ticket_separator")
        _report_run_progress(progress_callback, "Ticket processing", "done")

        _report_run_progress(progress_callback, "Export finalization", "in_progress")
        export_started = time.perf_counter()
        metrics = write_run_exports(
            connection,
            run_id=run_id,
            output_directory=output_directory,
            input_records=input_records,
            input_path=resolved_input_path,
            source_ref=_display_path(resolved_input_path),
        )
        run_status = "completed_with_errors" if metrics["failed_count"] else "completed"
        update_processing_run_status(
            connection,
            run_id=run_id,
            completed_at=_utc_timestamp(),
            status=run_status,
        )
        metrics = record_batch_export_duration(
            metrics=metrics,
            output_directory=output_directory,
            duration_seconds=max(time.perf_counter() - export_started, 0.0),
        )
        write_markdown_run_metrics(
            connection,
            run_id=run_id,
            output_directory=output_directory,
        )
        _report_run_progress(progress_callback, "Export finalization", "done")
        return run_id, output_directory, metrics
    except Exception as error:
        if connection.in_transaction:
            connection.rollback()
        if run_record_committed:
            try:
                update_processing_run_status(
                    connection,
                    run_id=run_id,
                    completed_at=_utc_timestamp(),
                    status="failed",
                    error_message=f"Run export or completion failed: {type(error).__name__}",
                )
            except sqlite3.Error:
                pass
        raise
    finally:
        connection.close()


def _report_run_progress(
    progress_callback: Callable[[str, str], None] | None,
    stage: str,
    status: str,
) -> None:
    """Report a ticket-run stage transition when the CLI supplies a progress callback."""
    if progress_callback is not None:
        progress_callback(stage, status)


def _commit_ticket_records(connection: sqlite3.Connection) -> None:
    """Durably commit one matching result and audit pair before stopping its timer."""
    connection.commit()


def route_ticket_classification(
    evaluator: RoutingPolicyEvaluator,
    *,
    ticket: CanonicalTicket,
    classification: dict[str, Any],
    retrieval: RetrievalResult,
    pre_provider_guardrail_blocked: bool,
) -> dict[str, Any]:
    """Evaluate the ordered routing policy checks for one retrieved ticket."""
    return evaluator.evaluate(
        ticket=ticket,
        classification=classification,
        retrieval=retrieval,
        pre_provider_guardrail_blocked=pre_provider_guardrail_blocked,
    )


def _routing_used_gemini_grounding(
    policy_evaluation: dict[str, Any],
    settings: Settings,
) -> bool:
    """Return whether routing invoked the Gemini claim grounding check rather than skipping it."""
    if settings.draft_generation_provider == "OFFLINE":
        return False
    for check in policy_evaluation.get("checks", ()):
        if check.get("name") == "claim_grounding_check":
            return check.get("status") != "skipped"
    return False


def _persist_stage_checkpoint(
    connection: sqlite3.Connection,
    *,
    run_id: str,
    prepared_ticket: _PreparedTicket,
    stage: str,
    status: str,
    value: dict[str, Any],
    reason: str,
    model_name: str | None = None,
    prompt_version: str | None = None,
    prediction: dict[str, Any] | None = None,
) -> None:
    """Commit a source-safe typed result before the next pipeline stage begins."""
    ticket = prepared_ticket.ticket
    connection.execute("BEGIN IMMEDIATE")
    _insert_audit_event(
        connection,
        run_id=run_id,
        storage_ticket_id=prepared_ticket.storage_ticket_id,
        event_id=uuid.uuid4().hex,
        occurred_at=_utc_timestamp(),
        event_type=stage,
        result_payload={
            "input_index": prepared_ticket.input_index,
            "ticket_id": ticket.ticket_id,
            "stage": stage,
            "status": status,
            "value": value,
            "reason": reason,
        },
        input_summary=ticket.body if stage == "classification" else None,
        model_name=model_name,
        prompt_version=prompt_version,
        prediction=prediction,
        reason=reason,
    )
    connection.commit()


def _insert_audit_event(
    connection: sqlite3.Connection,
    *,
    run_id: str,
    storage_ticket_id: str,
    event_id: str,
    occurred_at: str,
    event_type: str,
    result_payload: dict[str, Any],
    input_summary: str | None = None,
    model_name: str | None = None,
    prompt_version: str | None = None,
    prediction: dict[str, Any] | None = None,
    requirement_ids: list[str] | None = None,
    alternatives: list[dict[str, Any]] | None = None,
    threshold_applied: float | None = None,
    action_taken: str | None = None,
    reason: str | None = None,
    guardrail_results: dict[str, Any] | None = None,
    sources: list[dict[str, Any]] | None = None,
) -> None:
    """Insert one audit row through the database's typed persistence boundary."""
    create_audit_event_record(
        connection,
        event_id=event_id,
        run_id=run_id,
        storage_ticket_id=storage_ticket_id,
        occurred_at=occurred_at,
        event_type=event_type,
        result_payload=result_payload,
        input_summary=input_summary,
        model_name=model_name,
        prompt_version=prompt_version,
        prediction=prediction,
        requirement_ids=requirement_ids,
        alternatives=alternatives,
        threshold_applied=threshold_applied,
        action_taken=action_taken,
        reason=reason,
        guardrail_results=guardrail_results,
        sources=sources,
    )


def _build_ticket_result(
    prepared_ticket: _PreparedTicket,
    *,
    run_id: str,
    settings: Settings,
    policy: dict[str, Any],
    policy_file_fingerprint: str,
    classifier_implementation_fingerprint: str,
    generator: DraftGenerationProvider,
    skip_draft: bool = False,
    skip_draft_review: bool = False,
    ticket_step_callback: Callable[[str], None] | None = None,
) -> tuple[dict[str, Any], str, CanonicalTicket]:
    """Compose one terminal outcome from its completed per-ticket pipeline phases."""
    ticket = prepared_ticket.ticket
    input_index = prepared_ticket.input_index
    storage_ticket_id = prepared_ticket.storage_ticket_id
    issues = prepared_ticket.issues
    duplicate = prepared_ticket.duplicate_ticket_id
    classification = prepared_ticket.classification
    if classification is None:
        classification = classification_failure(
            provider="not_attempted",
            model_name=DECISIONS_CLASSIFIER_MODEL,
            uncertainty="Ticket was not included in a Decisions classification batch.",
        )
    retrieval = prepared_ticket.retrieval
    decision = prepared_ticket.decision
    if retrieval is None or decision is None:
        raise RuntimeError("Ticket outcome requires completed retrieval and routing phases.")
    result = build_ticket_outcome(
        run_id=run_id,
        input_index=input_index,
        ticket=ticket,
        classification=classification,
        retrieval=retrieval,
        decision=decision,
        policy_evaluation=prepared_ticket.policy_evaluation,
        pre_routing_safety=prepared_ticket.pre_routing_safety,
        policy=policy,
        policy_file_fingerprint=policy_file_fingerprint,
        classifier_implementation_fingerprint=classifier_implementation_fingerprint,
        settings=settings,
        generator=generator,
        skip_draft=skip_draft,
        skip_draft_review=skip_draft_review,
        issues=issues,
        duplicate_ticket_id=duplicate,
        progress_callback=ticket_step_callback,
    )
    return result, storage_ticket_id, ticket


def _retrieve_ticket_result(
    prepared_ticket: _PreparedTicket,
    *,
    retriever: KnowledgeRetriever,
    settings: Settings,
) -> RetrievalResult:
    """Resolve one retrieval outcome, including explicit skips and provider failures."""
    if prepared_ticket.issues or prepared_ticket.pre_provider_guardrail_block:
        return _skipped_retrieval_result(prepared_ticket, settings)
    try:
        return retriever.retrieve(prepared_ticket.ticket)
    except Exception:
        return RetrievalResult(
            status="unavailable",
            reason_code="retrieval_provider_or_index_error",
            reason="Knowledge retrieval failed; no source was used.",
            sources=(),
            eligible_candidate_count=0,
            excluded_candidate_counts={},
            minimum_similarity=MINIMUM_SIMILARITY,
            embedding_model=settings.embedding_model,
            source_sha256=getattr(retriever, "source_sha256", None),
            metadata_manifest_sha256=getattr(
                retriever, "metadata_manifest_sha256", None
            ),
        )


def _retrieve_ticket_results(
    prepared_tickets: list[_PreparedTicket],
    *,
    retriever: KnowledgeRetriever,
    settings: Settings,
) -> tuple[RetrievalResult, ...]:
    """Resolve every retrieval outcome in input order before any routing begins."""
    results: list[RetrievalResult | None] = [None] * len(prepared_tickets)
    eligible_positions: list[int] = []
    for position, prepared_ticket in enumerate(prepared_tickets):
        if prepared_ticket.issues or prepared_ticket.pre_provider_guardrail_block:
            results[position] = _skipped_retrieval_result(prepared_ticket, settings)
        else:
            eligible_positions.append(position)

    if eligible_positions:
        # Legacy injected test retrievers remain serial; the production class uses bulk retrieval.
        retrieve_many = getattr(type(retriever), "retrieve_many", None)
        if retrieve_many is not None:
            try:
                batch_results = retriever.retrieve_many(
                    [prepared_tickets[position].ticket for position in eligible_positions],
                    embedding_batch_size=DEFAULT_EMBEDDING_BATCH_SIZE,
                    query_batch_size=DEFAULT_QUERY_BATCH_SIZE,
                )
                if len(batch_results) != len(eligible_positions):
                    raise ValueError("Bulk retrieval result count does not match ticket count.")
                for position, retrieval in zip(
                    eligible_positions, batch_results, strict=True
                ):
                    results[position] = retrieval
            except Exception:
                for position in eligible_positions:
                    results[position] = _unavailable_retrieval_result(retriever, settings)
        else:
            for position in eligible_positions:
                results[position] = _retrieve_ticket_result(
                    prepared_tickets[position], retriever=retriever, settings=settings
                )

    if any(result is None for result in results):
        raise RuntimeError("Retrieval phase did not resolve every input position.")
    return tuple(result for result in results if result is not None)


def _skipped_retrieval_result(
    prepared_ticket: _PreparedTicket,
    settings: Settings,
) -> RetrievalResult:
    """Build a no-provider retrieval result for an invalid or blocked ticket."""
    return RetrievalResult(
        status="skipped",
        reason_code=(
            "invalid_ticket_record"
            if prepared_ticket.issues
            else "pre_provider_guardrail_block"
        ),
        reason=(
            "Retrieval was skipped because the ticket record is invalid."
            if prepared_ticket.issues
            else "Retrieval was skipped because an input guardrail blocked external processing."
        ),
        sources=(),
        eligible_candidate_count=0,
        excluded_candidate_counts={},
        minimum_similarity=MINIMUM_SIMILARITY,
        embedding_model=settings.embedding_model,
        source_sha256=None,
        metadata_manifest_sha256=None,
    )


def _unavailable_retrieval_result(
    retriever: KnowledgeRetriever,
    settings: Settings,
) -> RetrievalResult:
    """Build a safe unavailable result when the bulk retrieval boundary raises."""
    return RetrievalResult(
        status="unavailable",
        reason_code="retrieval_provider_or_index_error",
        reason="Knowledge retrieval failed; no source was used.",
        sources=(),
        eligible_candidate_count=0,
        excluded_candidate_counts={},
        minimum_similarity=MINIMUM_SIMILARITY,
        embedding_model=settings.embedding_model,
        source_sha256=getattr(retriever, "source_sha256", None),
        metadata_manifest_sha256=getattr(retriever, "metadata_manifest_sha256", None),
    )


def _prepare_ticket_records(input_records: list[Any]) -> list[_PreparedTicket]:
    """Normalize, deduplicate, and hold unsafe tickets out of classification batches."""
    prepared_tickets: list[_PreparedTicket] = []
    seen_ticket_ids: set[str] = set()
    used_storage_ticket_ids: set[str] = set()
    for input_index, raw_ticket in enumerate(input_records):
        ticket = normalize_ticket_record(raw_ticket, input_index=input_index)
        raw_record = raw_ticket if isinstance(raw_ticket, dict) else {}
        input_labels_json = (
            json.dumps(raw_record["labels"], ensure_ascii=True, separators=(",", ":"))
            if "labels" in raw_record
            else None
        )
        issues = list(ticket.validation_errors)
        duplicate = ticket.has_ticket_id and ticket.ticket_id in seen_ticket_ids
        if ticket.has_ticket_id:
            seen_ticket_ids.add(ticket.ticket_id)
        if duplicate:
            issues.append("ticket_id duplicates an earlier record in this run")

        storage_ticket_id = ticket.ticket_id
        if duplicate or storage_ticket_id in used_storage_ticket_ids:
            storage_ticket_id = f"{ticket.ticket_id}#input-{input_index + 1:06d}"
            collision_suffix = 2
            while storage_ticket_id in used_storage_ticket_ids:
                storage_ticket_id = (
                    f"{ticket.ticket_id}#input-{input_index + 1:06d}-{collision_suffix}"
                )
                collision_suffix += 1
        used_storage_ticket_ids.add(storage_ticket_id)

        operational_text = ticket_text_for_generation(ticket)
        private_input_text = json.dumps(
            filter_evaluation_context(ticket.context), ensure_ascii=False, default=str
        )
        pre_provider_guardrail_block = contains_prompt_injection(
            operational_text
        ) or contains_private_data(operational_text) or contains_private_data(
            private_input_text
        )
        classification = None
        if issues or pre_provider_guardrail_block:
            classification = classification_failure(
                provider="not_attempted",
                model_name=DECISIONS_CLASSIFIER_MODEL,
                uncertainty=(
                    "Ticket validation failed before classification."
                    if issues
                    else "Private or instruction-like input was held for human review."
                ),
            )
        prepared_tickets.append(
            _PreparedTicket(
                input_index=input_index,
                ticket=ticket,
                storage_ticket_id=storage_ticket_id,
                input_labels_json=input_labels_json,
                issues=issues,
                duplicate_ticket_id=duplicate,
                pre_provider_guardrail_block=pre_provider_guardrail_block,
                classification=classification,
            )
        )
    return prepared_tickets


def _usable_run_label(label: str | None) -> str | None:
    """Return a filename-safe label prefix, or None when no usable label was supplied."""
    if label is None or not label.strip():
        return None
    safe_label = re.sub(r"[^A-Za-z0-9._-]+", "-", label.strip()).strip("-._")
    return safe_label or None


def _allocate_run_id_and_directory(
    *,
    connection: sqlite3.Connection,
    output_root: Path,
    label: str | None,
    started_datetime: datetime,
) -> tuple[str, Path]:
    """Allocate a collision-safe run_id and output directory matching its basename."""
    safe_label = _usable_run_label(label)
    if safe_label is None:
        run_id = uuid.uuid4().hex
        output_directory = output_root / run_id
        output_directory.mkdir(exist_ok=False)
        return run_id, output_directory

    base_name = f"{safe_label}_{started_datetime:%Y-%m-%d_%H-%M}"
    suffix = 1
    while True:
        candidate = base_name if suffix == 1 else f"{base_name}_{suffix}"
        existing_row = connection.execute(
            "SELECT 1 FROM processing_runs WHERE run_id = ?",
            (candidate,),
        ).fetchone()
        if existing_row is not None:
            suffix += 1
            continue
        candidate_directory = output_root / candidate
        try:
            candidate_directory.mkdir(exist_ok=False)
        except FileExistsError:
            suffix += 1
            continue
        return candidate, candidate_directory


def _utc_timestamp() -> str:
    """Return an ISO 8601 UTC timestamp suitable for JSON Schema date-time fields."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _display_path(path: Path) -> str:
    """Prefer repository-relative SQLite paths while preserving external input paths."""
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)
