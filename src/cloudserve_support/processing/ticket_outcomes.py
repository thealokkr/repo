"""Build guarded drafts, escalation handoffs, and degraded ticket outcomes."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any

from cloudserve_support.classification.ticket_classification import TicketClassification
from cloudserve_support.config import Settings
from cloudserve_support.domain.ticket import CanonicalTicket
from cloudserve_support.generation.draft_generation import (
    DraftGenerationError,
    DraftGenerationProvider,
    draft_source_contains_private_data,
    draft_source_contains_prompt_injection,
    extract_verified_policy_claims,
    validate_generated_draft,
)
from cloudserve_support.guardrails.draft_safety import (
    check_claim_grounding,
    contains_private_data,
    contains_private_source_data,
    contains_prompt_injection,
    eligible_source_subset,
    filter_evaluation_context,
    inspect_claim_grounding,
    sanitize_handoff_content,
    ticket_text_for_generation,
)
from cloudserve_support.knowledge.retrieval import MINIMUM_SIMILARITY, RetrievalResult
from cloudserve_support.processing.run_provenance import (
    fingerprint_classifier_model,
    fingerprint_policy_threshold,
)
from cloudserve_support.routing.routing_policy import RoutingDecision


def build_ticket_outcome(
    *,
    run_id: str,
    input_index: int,
    ticket: CanonicalTicket,
    classification: TicketClassification,
    retrieval: RetrievalResult,
    decision: RoutingDecision,
    policy_evaluation: Mapping[str, Any] | None = None,
    policy: dict[str, Any],
    policy_file_fingerprint: str,
    classifier_implementation_fingerprint: str,
    settings: Settings,
    generator: DraftGenerationProvider,
    issues: list[str],
    duplicate_ticket_id: bool,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Return the final structured draft, complete handoff, or safe degradation."""
    _report_ticket_step(progress_callback, "Checking ticket and source safety")
    classification_payload = classification.as_dict()
    retrieval_payload = retrieval.as_dict()
    current_sources, eligibility_evidence = eligible_source_subset(retrieval.sources)
    ticket_injection_detected = contains_prompt_injection(ticket_text_for_generation(ticket))
    injected_source_count = sum(
        draft_source_contains_prompt_injection(source) for source in current_sources
    )
    private_source_count = sum(
        contains_private_source_data(source) or draft_source_contains_private_data(source)
        for source in current_sources
    )
    eligible_sources = tuple(
        source
        for source in current_sources
        if not draft_source_contains_prompt_injection(source)
        and not contains_private_source_data(source)
        and not draft_source_contains_private_data(source)
    )
    eligible_related_sources = _screen_related_sources_for_drafting(
        retrieval.related_sources,
        primary_sources=eligible_sources,
    )
    retained_sources = (
        *(_tag_source_snapshot(source, source_kind="primary") for source in eligible_sources),
        *(
            _tag_source_snapshot(source, source_kind="related")
            for source in eligible_related_sources
        ),
    )
    eligibility_evidence += (
        f" {injected_source_count} instruction-bearing source(s) and "
        f"{private_source_count} private-data source(s) excluded."
    )
    scrubbed_context_value = filter_evaluation_context(ticket.context)
    scrubbed_context = (
        scrubbed_context_value if isinstance(scrubbed_context_value, dict) else {}
    )
    private_input_text = json.dumps(
        scrubbed_context, ensure_ascii=False, default=str
    )
    private_input_detected = contains_private_data(
        ticket_text_for_generation(ticket)
    ) or contains_private_data(private_input_text)
    pre_provider_guardrail_block = ticket_injection_detected or private_input_detected
    injection_detected = ticket_injection_detected
    evidence_ready = bool(eligible_sources) and retrieval.evidence_ready

    policy_provenance = {
        "policy_version": policy["policy_version"],
        "classification_confidence_threshold": policy[
            "classification_confidence_threshold"
        ],
        "threshold_source": policy["threshold_source"],
        "calibration_method": policy["calibration_method"],
        "calibration_source": policy["calibration_source"],
        "frozen": policy["frozen"],
        "policy_file_sha256": policy_file_fingerprint,
        "threshold_sha256": fingerprint_policy_threshold(policy),
        "classifier_implementation_sha256": classifier_implementation_fingerprint,
        "classifier_model_sha256": fingerprint_classifier_model(
            policy["classifier_provider"], policy["classifier_model"]
        ),
    }
    (
        policy_route,
        policy_matched_rule,
        policy_reason,
        policy_protected_case,
    ) = _resolve_policy_decision(policy_evaluation, decision)
    routing_fields = {
        "policy_version": policy["policy_version"],
        "matched_rule": policy_matched_rule,
        "protected_case": policy_protected_case,
        "evidence_ready": evidence_ready,
        "confidence_threshold": policy["classification_confidence_threshold"],
        "threshold_source": policy["threshold_source"],
        "calibration_method": policy["calibration_method"],
        "calibration_source": policy["calibration_source"],
        "policy_frozen": policy["frozen"],
        "policy_provenance": policy_provenance,
        "retrieval": retrieval_payload,
    }
    if isinstance(policy_evaluation, Mapping):
        policy_checks = policy_evaluation.get("checks")
        if isinstance(policy_checks, list):
            routing_fields["checks"] = [
                dict(item) for item in policy_checks if isinstance(item, Mapping)
            ]
        policy_final_decision = policy_evaluation.get("final_decision")
        if isinstance(policy_final_decision, Mapping):
            routing_fields["final_decision"] = dict(policy_final_decision)
    guardrail_checks = [
        _make_guardrail_check(
            "source_eligibility",
            passed=bool(eligible_sources),
            blocked=not bool(eligible_sources),
            evidence=eligibility_evidence,
            action="accepted" if eligible_sources else "no_source_available",
        ),
        _make_guardrail_check(
            "prompt_injection",
            passed=not injection_detected,
            blocked=injection_detected,
            evidence=(
                "Untrusted ticket text contains an instruction override pattern."
                if injection_detected
                else (
                    f"Ticket passed; {injected_source_count} instruction-bearing source(s) were excluded."
                    if injected_source_count
                    else "Ticket and eligible source text passed instruction screening."
                )
            ),
            action=(
                "block_and_escalate"
                if injection_detected
                else "exclude_untrusted_source" if injected_source_count else "continue"
            ),
        ),
        _make_guardrail_check(
            "private_data_input",
            passed=not private_input_detected,
            blocked=private_input_detected,
            evidence=(
                "Ticket input contains an email, credential-like value, or national identifier."
                if private_input_detected
                else "No supported private-data pattern was found in ticket input."
            ),
            action="block_and_escalate" if private_input_detected else "continue",
        ),
        _make_guardrail_check(
            "private_data_source",
            passed=private_source_count == 0,
            blocked=private_source_count > 0,
            evidence=(
                "One or more retrieved passages contained private data and were withheld before generation."
                if private_source_count
                else "Retrieved passage titles and text passed the private-data screen."
            ),
            action="block_and_escalate" if private_source_count else "continue",
        ),
    ]

    route = "degrade" if issues else policy_route
    reason = (
        f"Ticket could not be processed: {'; '.join(issues)}."
        if issues
        else policy_reason
    )
    error: dict[str, str] | None = None
    if issues:
        error = {
            "code": "duplicate_ticket_id" if duplicate_ticket_id else "invalid_ticket_record",
            "message": "; ".join(issues),
        }
    failed_classification = classification.provider_status != "classified"
    if error is None and not pre_provider_guardrail_block and (
        failed_classification
        or policy_matched_rule == "classification_unavailable_or_model_mismatch"
    ):
        error = {
            "code": "classification_unavailable",
            "message": classification.failure_reason
            or "Classification did not match the frozen policy model.",
        }
        if not policy_protected_case:
            route = "degrade"
            reason = "Classification is unavailable; the ticket is available for human review."
    elif error is None and pre_provider_guardrail_block:
        route = policy_route
        reason = policy_reason
    elif error is None and policy_route == "degrade":
        error = _resolve_routing_degrade_error(
            policy_evaluation=policy_evaluation,
            matched_rule=policy_matched_rule,
            reason=policy_reason,
        )

    generation = {
        "provider": generator.name,
        "model_name": settings.llm_model if generator.name == "gemini" else "offline-extractive-v1",
        "reasoning_effort": None,
        "status": "not_attempted",
        "error_code": None,
    }
    draft: dict[str, Any] | None = None
    post_draft_diagnostics: dict[str, Any] | None = None
    draft_generation_request_duration_seconds = 0.0
    _report_ticket_step(progress_callback, "Checking draft eligibility")
    if policy_route == "draft_for_review" and route == "draft_for_review" and error is None:
        if injection_detected:
            _report_ticket_step(
                progress_callback,
                "Draft withheld by the prompt-injection guardrail",
            )
            route = "escalate"
            reason = "Ticket contains an instruction override attempt and was held for human review."
        elif private_input_detected:
            _report_ticket_step(
                progress_callback,
                "Draft withheld by the private-data input guardrail",
            )
            route = "escalate"
            reason = "Ticket context contains private data that must be reviewed by an agent."
        elif private_source_count:
            _report_ticket_step(
                progress_callback,
                "Draft withheld because a source contains private data",
            )
            route = "escalate"
            reason = "Retrieved source content contains private data and was withheld for human review."
        elif not eligible_sources:
            _report_ticket_step(
                progress_callback,
                "Draft withheld because no source passed eligibility checks",
            )
            route = "escalate"
            reason = "No source passed the eligibility guardrail for drafting."
        else:
            operational_ticket = replace(ticket, context=scrubbed_context)
            sanitized_policy_evaluation = _sanitize_policy_evaluation(policy_evaluation)
            if not _has_verified_drafting_claims(
                sanitized_policy_evaluation,
                primary_sources=eligible_sources,
                related_sources=eligible_related_sources,
            ):
                _report_ticket_step(
                    progress_callback,
                    "Draft skipped because policy-verified claims are missing",
                )
                route = "degrade"
                reason = (
                    "Routing policy evaluation is missing verified grounding claims; "
                    "manual review is required."
                )
                error = {"code": "routing_policy_invalid", "message": reason}
            else:
                (
                    draft,
                    route,
                    reason,
                    error,
                    draft_generation_request_duration_seconds,
                    draft_guardrail_checks,
                    post_draft_diagnostics,
                ) = _attempt_draft_generation(
                    ticket=operational_ticket,
                    eligible_sources=eligible_sources,
                    eligible_related_sources=eligible_related_sources,
                    policy_evaluation=sanitized_policy_evaluation,
                    default_reason=reason,
                    settings=settings,
                    generator=generator,
                    generation=generation,
                    progress_callback=progress_callback,
                )
                guardrail_checks.extend(draft_guardrail_checks)
    else:
        _report_ticket_step(
            progress_callback,
            f"Draft generation skipped (route: {route})",
        )

    if not any(check["check"] == "output_structure" for check in guardrail_checks):
        guardrail_checks.extend(
            [
                _make_guardrail_check(
                    "output_structure",
                    passed=True,
                    blocked=False,
                    evidence="No draft was generated for this route.",
                    action="not_applicable",
                ),
                _make_guardrail_check(
                    "citation_grounding",
                    passed=True,
                    blocked=False,
                    evidence="No draft claims were emitted for this route.",
                    action="not_applicable",
                ),
                _make_guardrail_check(
                    "private_data_output",
                    passed=True,
                    blocked=False,
                    evidence="No draft text was emitted.",
                    action="not_applicable",
                ),
            ]
        )
    if not any(check["check"] == "instruction_integrity" for check in guardrail_checks):
        guardrail_checks.append(
            _make_guardrail_check(
                "instruction_integrity",
                passed=not injection_detected,
                blocked=injection_detected,
                evidence=(
                    "Instruction-like content was not followed and the ticket was escalated."
                    if injection_detected
                    else "Provider instructions and untrusted-data boundaries were preserved."
                ),
                action="block_and_escalate" if injection_detected else "enforced",
            )
        )

    if error and error["code"] in {"invalid_ticket_record", "duplicate_ticket_id"}:
        routing_fields["matched_rule"] = "ticket_validation_failed"
        routing_fields["protected_case"] = False
    elif route != policy_route:
        if route == "degrade" and error and error["code"] == "classification_unavailable":
            routing_fields["matched_rule"] = "classification_unavailable_or_model_mismatch"
        elif route == "degrade" and error and error["code"] == "routing_policy_invalid":
            routing_fields["matched_rule"] = "routing_policy_invalid"
        elif route == "degrade" and generation["status"] == "failed":
            routing_fields["matched_rule"] = "generation_unavailable"
        else:
            routing_fields["matched_rule"] = "generation_guardrail_blocked"
        routing_fields["evidence_ready"] = bool(eligible_sources)

    if route in {"escalate", "degrade"}:
        _report_ticket_step(progress_callback, "Building and sanitizing human handoff")
        handoff = _build_human_handoff(
            ticket,
            classification=classification_payload,
            retrieval=retrieval_payload,
            sources=retained_sources,
            reason=reason,
            classification_failure_reason=classification.failure_reason,
            attempted_steps=_describe_attempted_steps(
                classification=classification_payload,
                retrieval=retrieval_payload,
                generation=generation,
                draft_was_withheld=generation["status"] == "generated" and draft is None,
            ),
        )
    else:
        handoff = None
    handoff_private_data_withheld = False
    handoff_instruction_text_withheld = False
    if handoff is not None:
        (
            sanitized_handoff,
            handoff_private_data_withheld,
            handoff_instruction_text_withheld,
        ) = sanitize_handoff_content(handoff)
        handoff = sanitized_handoff
    if handoff_private_data_withheld or handoff_instruction_text_withheld:
        for check in guardrail_checks:
            if check["check"] == "private_data_output" and handoff_private_data_withheld:
                check.update(
                    {
                        "passed": False,
                        "blocked": True,
                        "evidence": "Private or instruction-like text was withheld from the handoff.",
                        "action": "redact_and_escalate",
                    }
                )
            elif check["check"] == "instruction_integrity" and handoff_instruction_text_withheld:
                check.update(
                    {
                        "passed": False,
                        "blocked": True,
                        "evidence": "Instruction-like text was withheld from the handoff.",
                        "action": "block_and_escalate",
                    }
                )
    _report_ticket_step(progress_callback, "Finalizing ticket outcome")
    result = {
        "run_id": run_id,
        "input_index": input_index,
        "ticket_id": ticket.ticket_id,
        "channel": ticket.channel,
        "route": route,
        "status": "failed" if error else "completed",
        "reason": reason,
        "draft_response": draft["plain_text"] if draft else None,
        "draft": draft,
        "handoff": handoff,
        "guardrails": {"checks": guardrail_checks},
        "generation": generation,
        "draft_generation_request_duration_seconds": (
            draft_generation_request_duration_seconds
        ),
        "sources": list(retained_sources),
        "classification": classification_payload,
        "routing": routing_fields,
    }
    if error:
        result["error"] = error
    if post_draft_diagnostics is not None:
        result["post_draft_diagnostics"] = post_draft_diagnostics
    return result


def _resolve_policy_decision(
    policy_evaluation: Mapping[str, Any] | None,
    decision: RoutingDecision,
) -> tuple[str, str, str, bool]:
    """Resolve route metadata from final_decision, failing closed if draft policy is missing."""
    final_decision = (
        policy_evaluation.get("final_decision")
        if isinstance(policy_evaluation, Mapping)
        else None
    )
    route = final_decision.get("route") if isinstance(final_decision, Mapping) else None
    if isinstance(final_decision, Mapping) and route in {"draft_for_review", "escalate", "degrade"}:
        matched_rule = final_decision.get("matched_rule")
        reason = final_decision.get("reason")
        if isinstance(matched_rule, str) and matched_rule and isinstance(reason, str) and reason:
            return route, matched_rule, reason, bool(final_decision.get("protected_case"))
    if policy_evaluation is None and decision.route in {"escalate", "degrade"}:
        return decision.route, decision.matched_rule, decision.reason, decision.protected_case
    return (
        "degrade",
        "routing_policy_invalid",
        "Routing policy final_decision is missing or invalid; manual review is required.",
        False,
    )


def _has_verified_drafting_claims(
    policy_evaluation: Mapping[str, Any] | None,
    *,
    primary_sources: tuple[dict[str, Any], ...],
    related_sources: tuple[dict[str, Any], ...],
) -> bool:
    """Confirm policy_evaluation supplies non-empty verified claims against safe sources."""
    allowed_ids = {
        identifier
        for source in (*primary_sources, *related_sources)
        for identifier in (source.get("passage_id"), source.get("doc_id"))
        if isinstance(identifier, str) and identifier
    }
    return bool(
        extract_verified_policy_claims(
            policy_evaluation,
            allowed_source_ids=allowed_ids,
        )
    )


def _sanitize_policy_evaluation(
    policy_evaluation: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Strip any evaluation keys from policy-check input before passing to drafting."""
    if policy_evaluation is None:
        return None
    scrubbed = filter_evaluation_context(dict(policy_evaluation))
    return scrubbed if isinstance(scrubbed, dict) else None


def _is_safe_eligible_related_source(candidate: object) -> bool:
    """Verify a related KB snapshot is approved, fresh, provenance-valid, and free of unsafe text."""
    if not isinstance(candidate, dict):
        return False
    validation_view = (
        {
            **candidate,
            "similarity": MINIMUM_SIMILARITY,
            "minimum_similarity": MINIMUM_SIMILARITY,
        }
        if candidate.get("similarity") is None and candidate.get("minimum_similarity") is None
        else candidate
    )
    eligible, _ = eligible_source_subset((validation_view,))
    if not eligible:
        return False
    return (
        not draft_source_contains_prompt_injection(candidate)
        and not contains_private_source_data(candidate)
        and not draft_source_contains_private_data(candidate)
    )


def _tag_source_snapshot(
    source: Mapping[str, Any],
    *,
    source_kind: str,
) -> dict[str, Any]:
    """Tag a retained source snapshot with source_kind and schema-valid numeric scores."""
    tagged = dict(source)
    tagged["source_kind"] = source_kind
    if source_kind == "related":
        if tagged.get("similarity") is None:
            tagged["similarity"] = 0.0
        if tagged.get("minimum_similarity") is None:
            tagged["minimum_similarity"] = MINIMUM_SIMILARITY
    return tagged


def _screen_related_sources_for_drafting(
    related_sources: tuple[dict[str, Any], ...],
    *,
    primary_sources: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    """Keep unique approved, fresh, safe related articles not already in primary sources."""
    seen_doc_ids = {source["doc_id"] for source in primary_sources}
    accepted: list[dict[str, Any]] = []
    for candidate in related_sources:
        if not _is_safe_eligible_related_source(candidate):
            continue
        doc_id = candidate["doc_id"]
        if doc_id in seen_doc_ids:
            continue
        seen_doc_ids.add(doc_id)
        accepted.append(candidate)
    return tuple(accepted)


def _attempt_draft_generation(
    *,
    ticket: CanonicalTicket,
    eligible_sources: tuple[dict[str, Any], ...],
    eligible_related_sources: tuple[dict[str, Any], ...],
    policy_evaluation: Mapping[str, Any] | None,
    default_reason: str,
    settings: Settings,
    generator: DraftGenerationProvider,
    generation: dict[str, Any],
    progress_callback: Callable[[str], None] | None,
) -> tuple[
    dict[str, Any] | None,
    str,
    str,
    dict[str, str] | None,
    float,
    list[dict[str, Any]],
    dict[str, Any] | None,
]:
    """Invoke closed-world drafting and validate structure, citations, and privacy."""
    try:
        _report_ticket_step(progress_callback, "Generating draft")
        raw_draft = generator.generate_draft(
            ticket,
            sources=eligible_sources,
            related_sources=eligible_related_sources,
            policy_evaluation=policy_evaluation,
            model_name=settings.llm_model,
        )
        duration = _read_generation_request_duration(generator)
        _report_ticket_step(
            progress_callback,
            "Validating draft structure, citations, and privacy",
        )
        draft_candidate = validate_generated_draft(raw_draft)
        generation["status"] = "generated"
    except DraftGenerationError as generation_error:
        duration = _read_generation_request_duration(generator)
        generation["status"] = "failed"
        generation["error_code"] = generation_error.code
        error = {"code": generation_error.code, "message": str(generation_error)}
        _report_ticket_step(
            progress_callback,
            "Draft generation or validation failed; degrading ticket",
        )
        return (
            None,
            "degrade",
            "Draft generation failed; the ticket is available for human review.",
            error,
            duration,
            _make_failed_draft_checks(str(generation_error)),
            None,
        )
    except Exception as unexpected_generation_error:
        duration = _read_generation_request_duration(generator)
        generation["status"] = "failed"
        generation["error_code"] = "generation_provider_error"
        error = {
            "code": "generation_provider_error",
            "message": "The configured generation provider could not complete this ticket.",
        }
        _report_ticket_step(
            progress_callback,
            "Draft generation or validation failed; degrading ticket",
        )
        return (
            None,
            "degrade",
            "Draft generation failed; the ticket is available for human review.",
            error,
            duration,
            _make_failed_draft_checks(
                f"Provider failure type: {type(unexpected_generation_error).__name__}."
            ),
            None,
        )

    all_safe_sources = (*eligible_sources, *eligible_related_sources)
    allowed_source_ids = {
        identifier
        for source in all_safe_sources
        for identifier in (source.get("passage_id"), source.get("doc_id"))
        if isinstance(identifier, str) and identifier
    }
    verified_claims = extract_verified_policy_claims(
        policy_evaluation,
        allowed_source_ids=allowed_source_ids,
    )
    draft, route, reason, checks, post_draft_diagnostics = (
        _evaluate_generated_draft_guardrails(
            draft_candidate=draft_candidate,
            all_safe_sources=all_safe_sources,
            verified_claims=verified_claims,
            default_reason=default_reason,
        )
    )
    return draft, route, reason, None, duration, checks, post_draft_diagnostics


def _evaluate_generated_draft_guardrails(
    *,
    draft_candidate: dict[str, Any],
    all_safe_sources: tuple[dict[str, Any], ...],
    verified_claims: list[dict[str, Any]],
    default_reason: str,
) -> tuple[
    dict[str, Any] | None,
    str,
    str,
    list[dict[str, Any]],
    dict[str, Any] | None,
]:
    """Screen generated draft claims and prose for grounding, privacy, and instruction safety."""
    claim_texts = " ".join(item["text"] for item in draft_candidate["claims"])
    private_output_detected = contains_private_data(claim_texts) or contains_private_data(
        draft_candidate["plain_text"]
    )
    output_injection_detected = contains_prompt_injection(
        claim_texts
    ) or contains_prompt_injection(draft_candidate["plain_text"])
    grounded, grounding_evidence = check_claim_grounding(
        draft_candidate["claims"],
        all_safe_sources,
        verified_claims=verified_claims,
    )
    if draft_candidate["unsupported_points"] or draft_candidate["conflicting_points"]:
        grounded = False
        grounding_evidence = (
            "The provider reported unsupported or conflicting points; draft withheld."
        )
    checks = [
        _make_guardrail_check(
            "output_structure",
            passed=True,
            blocked=False,
            evidence="Provider JSON matches the strict draft contract.",
            action="continue",
        ),
        _make_guardrail_check(
            "citation_grounding",
            passed=grounded,
            blocked=not grounded,
            evidence=grounding_evidence,
            action="continue" if grounded else "block_and_escalate",
        ),
        _make_guardrail_check(
            "private_data_output",
            passed=not private_output_detected,
            blocked=private_output_detected,
            evidence=(
                "Generated text contains an email, credential-like value, or national identifier."
                if private_output_detected
                else "Generated text passed the private-data screen."
            ),
            action="block_and_escalate" if private_output_detected else "continue",
        ),
        _make_guardrail_check(
            "instruction_integrity",
            passed=not output_injection_detected,
            blocked=output_injection_detected,
            evidence=(
                "Generated text contains an instruction-override pattern and was withheld."
                if output_injection_detected
                else "Provider instructions and untrusted-data boundaries were preserved."
            ),
            action="block_and_escalate" if output_injection_detected else "enforced",
        ),
    ]
    if private_output_detected or output_injection_detected or not grounded:
        post_draft_diagnostics = _build_sanitized_post_draft_diagnostics(
            draft_candidate=draft_candidate,
            all_safe_sources=all_safe_sources,
            verified_claims=verified_claims,
        )
        return (
            None,
            "escalate",
            "Generated content failed a privacy, instruction-integrity, or source-grounding check; "
            "the draft was withheld for human review.",
            checks,
            post_draft_diagnostics,
        )
    return draft_candidate, "draft_for_review", default_reason, checks, None


def _build_sanitized_post_draft_diagnostics(
    *,
    draft_candidate: Mapping[str, Any],
    all_safe_sources: tuple[dict[str, Any], ...],
    verified_claims: list[dict[str, Any]],
) -> dict[str, Any]:
    """Capture and redact withheld draft claims, unsupported points, and failed clauses."""
    claims = list(draft_candidate["claims"])
    grounding_inspection = inspect_claim_grounding(
        claims,
        all_safe_sources,
        verified_claims=verified_claims,
    )
    raw_diagnostics = {
        "withheld_claims": [
            {
                "text": item["text"],
                "source_ids": list(item["source_ids"]),
            }
            for item in claims
        ],
        "unsupported_points": list(draft_candidate["unsupported_points"]),
        "conflicting_points": list(draft_candidate["conflicting_points"]),
        "failed_claims": grounding_inspection["failed_claims"],
        "minimum_ratio": grounding_inspection["minimum_ratio"],
    }
    sanitized_diagnostics, _, _ = sanitize_handoff_content(raw_diagnostics)
    return sanitized_diagnostics  # type: ignore[return-value]


def _resolve_routing_degrade_error(
    *,
    policy_evaluation: Mapping[str, Any] | None,
    matched_rule: str,
    reason: str,
) -> dict[str, str]:
    """Extract a structured per-ticket error when a routing policy check fails closed."""
    if policy_evaluation is None:
        return {"code": matched_rule, "message": reason}
    for check in policy_evaluation.get("checks", ()):
        error_code = check.get("error_code")
        if check.get("status") == "failed" and isinstance(error_code, str) and error_code:
            return {"code": error_code, "message": reason}
    return {"code": matched_rule, "message": reason}


def _report_ticket_step(
    progress_callback: Callable[[str], None] | None,
    step: str,
) -> None:
    """Report one ticket-processing step when the CLI requested detailed progress."""
    if progress_callback is not None:
        progress_callback(step)


def _read_generation_request_duration(generator: DraftGenerationProvider) -> float:
    """Return a finite nonnegative provider request-wait duration, if available."""
    value = getattr(generator, "last_request_duration_seconds", 0.0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    duration = float(value)
    return duration if math.isfinite(duration) and duration >= 0 else 0.0


def _make_guardrail_check(
    check: str,
    *,
    passed: bool,
    blocked: bool,
    evidence: str,
    action: str,
) -> dict[str, Any]:
    """Create a searchable guardrail result without recording raw prompt text."""
    return {
        "check": check,
        "passed": passed,
        "blocked": blocked,
        "evidence": evidence,
        "action": action,
    }


def _make_failed_draft_checks(evidence: str) -> list[dict[str, Any]]:
    """Record the blocked output and grounding checks when generation fails."""
    return [
        _make_guardrail_check(
            "output_structure",
            passed=False,
            blocked=True,
            evidence=evidence,
            action="degrade",
        ),
        _make_guardrail_check(
            "citation_grounding",
            passed=False,
            blocked=True,
            evidence="No validated claims were produced.",
            action="degrade",
        ),
        _make_guardrail_check(
            "private_data_output",
            passed=True,
            blocked=False,
            evidence="No draft text was emitted.",
            action="not_applicable",
        ),
    ]


def _build_human_handoff(
    ticket: CanonicalTicket,
    *,
    classification: dict[str, Any],
    retrieval: dict[str, Any],
    sources: tuple[dict[str, Any], ...],
    reason: str,
    classification_failure_reason: str | None,
    attempted_steps: list[str],
) -> dict[str, Any]:
    """Assemble an honest handoff from observed ticket and pipeline facts."""
    uncertainties = (
        [classification_failure_reason] if classification_failure_reason else []
    )
    if retrieval["status"] != "ready":
        uncertainties.append(retrieval["reason"])
    if not sources:
        uncertainties.append(
            "No eligible source is available for the handoff."
        )
    if not ticket.subject.strip():
        uncertainties.append("Ticket subject is missing.")
    if not ticket.body.strip():
        uncertainties.append("Ticket body is missing.")
    if not uncertainties:
        uncertainties.append(f"Automated handling did not resolve the case: {reason}")
    issue = "\n".join(part for part in (ticket.subject, ticket.body) if part).strip()
    summary = issue or "Ticket issue details are incomplete."
    additional_context = filter_evaluation_context(ticket.context)
    if isinstance(additional_context, dict):
        additional_context = {
            key: value
            for key, value in additional_context.items()
            if str(key).casefold() not in {"ticket_id", "channel", "subject", "body"}
        }
    return {
        "summary": summary,
        "issue": summary,
        "context": {
            "channel": ticket.channel,
            "ticket_metadata": {
                "received_at": ticket.received_at,
                "customer_tier": ticket.customer_tier,
                "language_fluency": ticket.language_fluency,
            },
            "additional_context": additional_context,
        },
        "sources": list(sources),
        "attempted_steps": attempted_steps,
        "uncertainties": uncertainties,
        "reason": reason,
        "owner": "human support agent",
        "next_action": "Review the ticket and decide the response; no reply has been sent.",
    }


def _describe_attempted_steps(
    *,
    classification: dict[str, Any],
    retrieval: dict[str, Any],
    generation: dict[str, Any],
    draft_was_withheld: bool,
) -> list[str]:
    """Describe only pipeline stages that actually ran for this ticket."""
    steps = [
        "Ticket classification completed."
        if classification["provider_status"] == "classified"
        else "Ticket classification did not complete successfully."
    ]
    steps.append(f"Knowledge retrieval returned {retrieval['status']} evidence.")
    if generation["status"] == "generated" and draft_was_withheld:
        steps.append("Structured draft generation completed, then guardrails withheld the draft.")
    elif generation["status"] == "generated":
        steps.append("Structured draft generation completed, then route handling created a human handoff.")
    elif generation["status"] == "failed":
        steps.append("Structured draft generation failed and the batch continued.")
    else:
        steps.append("Draft generation was not attempted for this route.")
    return steps
