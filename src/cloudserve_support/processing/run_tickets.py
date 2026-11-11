"""Process one ticket batch into SQLite audit rows and isolated run exports."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cloudserve_support.classification.ticket_classification import (
    classification_failure,
    classify_ticket,
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
from cloudserve_support.knowledge.retrieval import KnowledgeRetriever, RetrievalResult
from cloudserve_support.persistence.runtime_database import (
    connect_runtime_database,
    create_processing_run,
    create_ticket_audit_records,
    update_ticket_audit_result_payload,
    update_processing_run_status,
)
from cloudserve_support.reporting.output_schemas import load_output_schemas
from cloudserve_support.reporting.run_exports import (
    record_batch_export_duration,
    write_run_exports,
)
from cloudserve_support.processing.run_provenance import (
    fingerprint_classifier_implementation,
    fingerprint_policy_file,
)
from cloudserve_support.routing.policy_calibration import (
    load_frozen_routing_policy,
)
from cloudserve_support.processing.ticket_outcomes import build_ticket_outcome
from cloudserve_support.routing.routing_policy import route_ticket_classification


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATABASE_PATH = REPO_ROOT / "data" / "runtime" / "cloudserve.sqlite3"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs"
DEFAULT_ROUTING_POLICY_PATH = REPO_ROOT / "data" / "policy" / "frozen_routing_policy.json"


def run_ticket_batch(
    *,
    input_path: Path,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    database_path: Path = DEFAULT_DATABASE_PATH,
    policy_path: Path = DEFAULT_ROUTING_POLICY_PATH,
    label: str | None = None,
    retriever: KnowledgeRetriever | None = None,
    generator: DraftGenerationProvider | None = None,
) -> tuple[str, Path, dict[str, Any]]:
    """Classify, retrieve, draft or hand off, and persist every ticket occurrence."""
    input_records = read_ticket_records(input_path)
    resolved_policy_path = policy_path.expanduser().resolve()
    policy = load_frozen_routing_policy(resolved_policy_path)
    policy_file_fingerprint = fingerprint_policy_file(resolved_policy_path)
    classifier_implementation_fingerprint = fingerprint_classifier_implementation()
    settings = Settings.from_env()
    knowledge_retriever = retriever or KnowledgeRetriever(settings)
    draft_generator = generator or create_draft_generation_provider(settings)
    load_output_schemas()
    resolved_input_path = input_path.expanduser().resolve()
    resolved_output_root = output_root.expanduser().resolve()
    resolved_database_path = database_path.expanduser().resolve()
    run_id = uuid.uuid4().hex
    output_directory = resolved_output_root / run_id

    connection = connect_runtime_database(resolved_database_path)
    run_record_committed = False
    try:
        resolved_output_root.mkdir(parents=True, exist_ok=True)
        output_directory.mkdir(exist_ok=False)
        started_at = _utc_timestamp()
        connection.execute("BEGIN IMMEDIATE")
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
        seen_ticket_ids: set[str] = set()
        used_storage_ticket_ids: set[str] = set()
        for input_index, raw_ticket in enumerate(input_records):
            ticket_started = time.perf_counter()
            result, storage_ticket_id, ticket = _build_ticket_result(
                raw_ticket,
                run_id=run_id,
                input_index=input_index,
                seen_ticket_ids=seen_ticket_ids,
                used_storage_ticket_ids=used_storage_ticket_ids,
                settings=settings,
                policy=policy,
                policy_file_fingerprint=policy_file_fingerprint,
                classifier_implementation_fingerprint=classifier_implementation_fingerprint,
                retriever=knowledge_retriever,
                generator=draft_generator,
            )
            occurred_at = _utc_timestamp()
            connection.execute("BEGIN IMMEDIATE")
            create_ticket_audit_records(
                connection,
                run_id=run_id,
                storage_ticket_id=storage_ticket_id,
                channel=ticket.channel,
                received_at=ticket.received_at,
                customer_tier=ticket.customer_tier,
                language_fluency=ticket.language_fluency,
                context=ticket.context,
                processing_status=result["status"],
                error_code=result.get("error", {}).get("code"),
                error_message=result.get("error", {}).get("message"),
                event_id=uuid.uuid4().hex,
                occurred_at=occurred_at,
                event_type=(
                    "processing_error" if result["status"] == "failed" else "route_recommended"
                ),
                input_summary=ticket.body,
                action_taken=result["route"],
                reason=result["reason"],
                result_payload=result,
                model_name=result["classification"]["model_name"],
                model_version=None,
                prompt_version=GENERATION_PROMPT_VERSION,
                requirement_ids=[
                    "A3", "A4", "A5", "A6", "A7", "A11", "FR-01", "FR-02",
                    "FR-03", "FR-04", "FR-05", "FR-06", "FR-07", "FR-08", "FR-11",
                ],
                prediction=result["classification"],
                alternatives=result["classification"]["alternatives"],
                threshold_applied=result["routing"]["confidence_threshold"],
                guardrail_results={
                    "protected_case": result["routing"]["protected_case"],
                    "matched_policy_rule": result["routing"]["matched_rule"],
                    "evidence_ready": result["routing"]["evidence_ready"],
                    "retrieval": result["routing"]["retrieval"],
                    "checks": result["guardrails"]["checks"],
                },
            )
            _commit_ticket_records(connection)
            result["processing_duration_seconds"] = max(
                time.perf_counter() - ticket_started, 0.0
            )
            generation_request_duration = result[
                "draft_generation_request_duration_seconds"
            ]
            result["processing_latency_seconds"] = max(
                result["processing_duration_seconds"] - generation_request_duration,
                0.0,
            )
            connection.execute("BEGIN IMMEDIATE")
            update_ticket_audit_result_payload(
                connection,
                run_id=run_id,
                storage_ticket_id=storage_ticket_id,
                result_payload=result,
            )
            connection.commit()

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


def _commit_ticket_records(connection: sqlite3.Connection) -> None:
    """Durably commit one matching result and audit pair before stopping its timer."""
    connection.commit()


def _build_ticket_result(
    raw_ticket: Any,
    *,
    run_id: str,
    input_index: int,
    seen_ticket_ids: set[str],
    used_storage_ticket_ids: set[str],
    settings: Settings,
    policy: dict[str, Any],
    policy_file_fingerprint: str,
    classifier_implementation_fingerprint: str,
    retriever: KnowledgeRetriever,
    generator: DraftGenerationProvider,
) -> tuple[dict[str, Any], str, CanonicalTicket]:
    """Normalize, classify, retrieve, apply guardrails, and produce one final outcome."""
    ticket = normalize_ticket_record(raw_ticket, input_index=input_index)
    ticket_id = ticket.ticket_id
    issues = list(ticket.validation_errors)

    duplicate = ticket.has_ticket_id and ticket_id in seen_ticket_ids
    if ticket.has_ticket_id:
        seen_ticket_ids.add(ticket_id)
    if duplicate:
        issues.append("ticket_id duplicates an earlier record in this run")
    storage_ticket_id = ticket_id
    if duplicate or storage_ticket_id in used_storage_ticket_ids:
        storage_ticket_id = f"{ticket_id}#input-{input_index + 1:06d}"
        collision_suffix = 2
        while storage_ticket_id in used_storage_ticket_ids:
            storage_ticket_id = (
                f"{ticket_id}#input-{input_index + 1:06d}-{collision_suffix}"
            )
            collision_suffix += 1
    used_storage_ticket_ids.add(storage_ticket_id)

    operational_text = ticket_text_for_generation(ticket)
    private_input_text = json.dumps(
        filter_evaluation_context(ticket.context), ensure_ascii=False, default=str
    )
    ticket_injection_detected = contains_prompt_injection(operational_text)
    private_input_detected = contains_private_data(operational_text) or contains_private_data(
        private_input_text
    )
    pre_provider_guardrail_block = ticket_injection_detected or private_input_detected
    classification = (
        classification_failure(
            provider="not_attempted",
            model_name=settings.llm_model,
            uncertainty=(
                "Ticket validation failed before classification."
                if issues
                else "Private or instruction-like input was held for human review."
            ),
        )
        if issues or pre_provider_guardrail_block
        else classify_ticket(ticket, settings)
    )
    retrieval = (
        retriever.retrieve(ticket)
        if not issues and not pre_provider_guardrail_block
        else RetrievalResult(
            status="skipped",
            reason_code=(
                "invalid_ticket_record" if issues else "pre_provider_guardrail_block"
            ),
            reason=(
                "Retrieval was skipped because the ticket record is invalid."
                if issues
                else "Retrieval was skipped because an input guardrail blocked external processing."
            ),
            sources=(),
            eligible_candidate_count=0,
            excluded_candidate_counts={},
            minimum_similarity=0.45,
            embedding_model=settings.embedding_model,
            source_sha256=None,
            metadata_manifest_sha256=None,
        )
    )
    decision = route_ticket_classification(
        classification.as_dict(),
        policy,
        evidence_ready=retrieval.evidence_ready,
        ticket=ticket,
    )
    result = build_ticket_outcome(
        run_id=run_id,
        input_index=input_index,
        ticket=ticket,
        classification=classification,
        retrieval=retrieval,
        decision=decision,
        policy=policy,
        policy_file_fingerprint=policy_file_fingerprint,
        classifier_implementation_fingerprint=classifier_implementation_fingerprint,
        settings=settings,
        generator=generator,
        issues=issues,
        duplicate_ticket_id=duplicate,
    )
    return result, storage_ticket_id, ticket


def _utc_timestamp() -> str:
    """Return an ISO 8601 UTC timestamp suitable for JSON Schema date-time fields."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _display_path(path: Path) -> str:
    """Prefer repository-relative SQLite paths while preserving external input paths."""
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)
