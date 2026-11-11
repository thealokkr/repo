"""Build guarded drafts, escalation handoffs, and degraded ticket outcomes."""

from __future__ import annotations

import json
import math
from typing import Any

from cloudserve_support.classification.ticket_classification import TicketClassification
from cloudserve_support.config import Settings
from cloudserve_support.domain.ticket import CanonicalTicket
from cloudserve_support.generation.draft_generation import (
    OPENAI_DRAFT_REASONING_EFFORT,
    DraftGenerationError,
    DraftGenerationProvider,
    parse_generated_claims,
)
from cloudserve_support.guardrails.draft_safety import (
    check_claim_grounding,
    contains_private_data,
    contains_private_source_data,
    contains_prompt_injection,
    eligible_source_subset,
    filter_evaluation_context,
    sanitize_handoff_content,
    compose_cited_draft,
    ticket_text_for_generation,
)
from cloudserve_support.knowledge.retrieval import RetrievalResult
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
    policy: dict[str, Any],
    policy_file_fingerprint: str,
    classifier_implementation_fingerprint: str,
    settings: Settings,
    generator: DraftGenerationProvider,
    issues: list[str],
    duplicate_ticket_id: bool,
) -> dict[str, Any]:
    """Return the final structured draft, complete handoff, or safe degradation."""
    classification_payload = classification.as_dict()
    retrieval_payload = retrieval.as_dict()
    current_sources, eligibility_evidence = eligible_source_subset(retrieval.sources)
    ticket_injection_detected = contains_prompt_injection(ticket_text_for_generation(ticket))
    injected_source_count = sum(
        contains_prompt_injection(source["passage_text"]) for source in current_sources
    )
    private_source_count = sum(
        contains_private_source_data(source)
        for source in current_sources
    )
    eligible_sources = tuple(
        source
        for source in current_sources
        if not contains_prompt_injection(source["passage_text"])
        and not contains_private_source_data(source)
    )
    eligibility_evidence += (
        f" {injected_source_count} instruction-bearing source(s) and "
        f"{private_source_count} private-data source(s) excluded."
    )
    private_input_text = json.dumps(
        filter_evaluation_context(ticket.context), ensure_ascii=False, default=str
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
    routing_fields = {
        "policy_version": policy["policy_version"],
        "matched_rule": decision.matched_rule,
        "protected_case": decision.protected_case,
        "evidence_ready": evidence_ready,
        "confidence_threshold": policy["classification_confidence_threshold"],
        "threshold_source": policy["threshold_source"],
        "calibration_method": policy["calibration_method"],
        "calibration_source": policy["calibration_source"],
        "policy_frozen": policy["frozen"],
        "policy_provenance": policy_provenance,
        "retrieval": retrieval_payload,
    }
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

    route = "degrade" if issues else decision.route
    reason = (
        f"Ticket could not be processed: {'; '.join(issues)}."
        if issues
        else decision.reason
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
        or decision.matched_rule == "classification_unavailable_or_model_mismatch"
    ):
        error = {
            "code": "classification_unavailable",
            "message": "; ".join(classification.uncertainties)
            or "Classification did not match the frozen policy model.",
        }
        route = "degrade"
    elif error is None and pre_provider_guardrail_block:
        route = "escalate"
        reason = (
            "Ticket contains private data that must be reviewed by an agent."
            if private_input_detected
            else "Ticket contains an instruction override attempt and was held for human review."
        )

    generation = {
        "provider": generator.name,
        "model_name": settings.llm_model if generator.name == "openai" else "offline-extractive-v1",
        "reasoning_effort": (
            OPENAI_DRAFT_REASONING_EFFORT if generator.name == "openai" else None
        ),
        "status": "not_attempted",
        "error_code": None,
    }
    draft: dict[str, Any] | None = None
    draft_generation_request_duration_seconds = 0.0
    if route == "draft_for_review" and error is None:
        if injection_detected:
            route = "escalate"
            reason = "Ticket contains an instruction override attempt and was held for human review."
        elif private_input_detected:
            route = "escalate"
            reason = "Ticket context contains private data that must be reviewed by an agent."
        elif private_source_count:
            route = "escalate"
            reason = "Retrieved source content contains private data and was withheld for human review."
        elif not eligible_sources:
            route = "escalate"
            reason = "No source passed the eligibility guardrail for drafting."
        else:
            try:
                raw_generated = generator.generate_claims(
                    ticket,
                    sources=eligible_sources,
                    model_name=settings.llm_model,
                )
                draft_generation_request_duration_seconds = (
                    _read_generation_request_duration(generator)
                )
                generated = parse_generated_claims(
                    raw_generated
                )
                generation["status"] = "generated"
                generated_text = " ".join(item["text"] for item in generated["claims"])
                private_output_detected = contains_private_data(generated_text)
                grounded, grounding_evidence = check_claim_grounding(
                    generated["claims"], eligible_sources
                )
                has_model_findings = bool(
                    generated["unsupported_points"] or generated["conflicting_points"]
                )
                if has_model_findings:
                    grounded = False
                    grounding_evidence = (
                        "The provider reported unsupported or conflicting points; draft withheld."
                    )
                draft_candidate = None
                if grounded and not has_model_findings:
                    draft_candidate = compose_cited_draft(
                        generated,
                        eligible_sources,
                        ticket_subject=ticket.subject,
                        ticket_text=ticket_text_for_generation(ticket),
                    )
                    private_output_detected = private_output_detected or contains_private_data(
                        draft_candidate["plain_text"]
                    )
                guardrail_checks.extend(
                    [
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
                    ]
                )
                if private_output_detected or not grounded:
                    route = "escalate"
                    reason = (
                        "Generated content failed a privacy or source-grounding check; "
                        "the draft was withheld for human review."
                    )
                else:
                    draft = draft_candidate
            except DraftGenerationError as generation_error:
                draft_generation_request_duration_seconds = (
                    _read_generation_request_duration(generator)
                )
                generation["status"] = "failed"
                generation["error_code"] = generation_error.code
                error = {
                    "code": generation_error.code,
                    "message": str(generation_error),
                }
                route = "degrade"
                reason = "Draft generation failed; the ticket is available for human review."
                guardrail_checks.extend(
                    _make_failed_draft_checks(str(generation_error))
                )
            except Exception as unexpected_generation_error:
                draft_generation_request_duration_seconds = (
                    _read_generation_request_duration(generator)
                )
                generation["status"] = "failed"
                generation["error_code"] = "generation_provider_error"
                error = {
                    "code": "generation_provider_error",
                    "message": "The configured generation provider could not complete this ticket.",
                }
                route = "degrade"
                reason = "Draft generation failed; the ticket is available for human review."
                guardrail_checks.extend(
                    _make_failed_draft_checks(
                        f"Provider failure type: {type(unexpected_generation_error).__name__}."
                    )
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

    if route != decision.route:
        if route == "degrade" and error and error["code"] == "classification_unavailable":
            routing_fields["matched_rule"] = "classification_unavailable_or_model_mismatch"
        elif route == "degrade" and error and error["code"] in {
            "invalid_ticket_record", "duplicate_ticket_id"
        }:
            routing_fields["matched_rule"] = "ticket_validation_failed"
        elif route == "degrade" and generation["status"] == "failed":
            routing_fields["matched_rule"] = "generation_unavailable"
        else:
            routing_fields["matched_rule"] = "generation_guardrail_blocked"
        routing_fields["protected_case"] = route == "escalate"
        routing_fields["evidence_ready"] = bool(eligible_sources)

    handoff = (
        _build_human_handoff(
            ticket,
            classification=classification_payload,
            retrieval=retrieval_payload,
            sources=eligible_sources,
            reason=reason,
            attempted_steps=_describe_attempted_steps(
                classification=classification_payload,
                retrieval=retrieval_payload,
                generation=generation,
                draft_was_withheld=generation["status"] == "generated" and draft is None,
            ),
        )
        if route in {"escalate", "degrade"}
        else None
    )
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
        "sources": list(eligible_sources),
        "classification": classification_payload,
        "routing": routing_fields,
    }
    if error:
        result["error"] = error
    return result


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
    attempted_steps: list[str],
) -> dict[str, Any]:
    """Assemble an honest handoff from observed ticket and pipeline facts."""
    uncertainties = list(classification["uncertainties"])
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
