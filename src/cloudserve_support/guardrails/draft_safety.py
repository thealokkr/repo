"""Validate provider drafts against source, privacy, and instruction boundaries."""

from __future__ import annotations

import json
import re
from datetime import date, timedelta
from typing import Any

from cloudserve_support.domain.ticket import CanonicalTicket
from cloudserve_support.guardrails.operational_safety import operational_ticket_text
from cloudserve_support.knowledge.retrieval import MINIMUM_SIMILARITY


_PROMPT_INJECTION_PATTERNS = (
    re.compile(r"\bignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions\b", re.I),
    re.compile(r"\bdisregard\s+(?:all\s+)?(?:previous|prior|system)\s+instructions\b", re.I),
    re.compile(r"\b(?:reveal|print|show|repeat|expose)\s+(?:the\s+)?(?:system|developer)\s+prompt\b", re.I),
    re.compile(r"\b(?:reveal|print|expose|send)\s+(?:the\s+)?(?:api|secret|access)\s+key\b", re.I),
    re.compile(r"\bact\s+as\s+(?:the\s+)?system\b", re.I),
)
_PRIVATE_DATA_PATTERNS = (
    re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    re.compile(r"(?<!\w)(?:\+\d{1,3}[ .-]?)?(?:\(\d{2,4}\)|\d{2,4})[ .-]\d{3,4}[ .-]\d{4}(?!\w)"),
    re.compile(r"\b(?:sk|pk|api|secret|token)[_-][A-Za-z0-9_-]{16,}\b", re.I),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/-]{12,}", re.I),
)
_EVALUATION_KEYS = frozenset({"labels", "history"})
CLARIFICATION_QUESTION_TEMPLATES = {
    "error_details": "What exact error message or code do you see?",
    "steps_tried": "What steps reproduce the issue?",
    "expected_observed": "What result did you expect, and what result did you observe?",
    "environment_version": "Which application version and environment are involved?",
    "occurrence_time": "When did the issue first occur?",
    "request_identifier": "If available, what request or trace ID is shown?",
    "none": "",
}
CLARIFICATION_KINDS = tuple(CLARIFICATION_QUESTION_TEMPLATES)


def clarification_question_for_ticket(kind: str, ticket_text: str) -> str:
    """Render only a fixed missing-detail question; never emit provider question text."""
    if kind not in CLARIFICATION_QUESTION_TEMPLATES or kind == "none":
        return ""
    if _clarification_detail_is_present(kind, ticket_text):
        return ""
    return CLARIFICATION_QUESTION_TEMPLATES[kind]


def default_clarification_kind(ticket_text: str) -> str:
    """Choose a deterministic missing diagnostic question for offline drafting."""
    text = ticket_text.casefold()
    if not re.search(
        r"\b(?:error|fail(?:ed|ure)?|unable|cannot|can't|not working|issue|problem)\b",
        text,
    ):
        return "none"
    for kind in ("error_details", "steps_tried"):
        if not _clarification_detail_is_present(kind, text):
            return kind
    return "none"


def _clarification_detail_is_present(kind: str, ticket_text: str) -> bool:
    """Detect common ticket evidence that already answers a requested detail type."""
    text = ticket_text.casefold()
    if kind == "error_details":
        return bool(
            re.search(r"\b(?:http\s*)?[45]\d\d\b", text)
            or re.search(r"\b(?:err|e)[_-][a-z0-9_-]{2,}\b", text)
            or re.search(
                r"\b(?:error(?:\s+message|\s+code)?|status\s+code|message)\s*(?:is|was|:|=)\s*['\"]?\w+",
                text,
            )
        )
    if kind == "steps_tried":
        return bool(
            re.search(
                r"\b(?:i|we)\s+(?:already\s+)?"
                r"(?:tried|attempted|restarted|retried|ran|cleared|refreshed)\b",
                text,
            )
            or re.search(r"\bfirst\b.{0,120}\bthen\b", text)
            or re.search(r"(?:^|\s)\d+[.)]\s+\S", text)
        )
    if kind == "expected_observed":
        has_expected = re.search(
            r"\b(?:expected|should|normally|used to|previously)\b", text
        )
        has_observed = re.search(
            r"\b(?:now|instead|actual(?:ly)?|received|got|returned|shows?|fails?|failed)\b",
            text,
        )
        return bool(has_expected and has_observed)
    if kind == "environment_version":
        return bool(
            re.search(
                r"\b(?:version|build|release)\s*[:#]?\s*v?\d+(?:\.\d+)+\b",
                text,
            )
            or re.search(
                r"\b(?:chrome|firefox|safari|edge|windows|macos|ios|android|linux)"
                r"\s+\d+(?:\.\d+)*\b",
                text,
            )
        )
    if kind == "occurrence_time":
        return bool(
            re.search(
                r"\b(?:today|yesterday|this morning|last night|last week|last month|"
                r"\d+\s+(?:minutes?|hours?|days?|weeks?)\s+ago)\b",
                text,
            )
            or re.search(
                r"\b\d{4}-\d{2}-\d{2}(?:[t ]\d{2}:\d{2})?\b", text
            )
        )
    if kind == "request_identifier":
        return bool(
            re.search(
                r"\b(?:request|trace|correlation)[ -]?(?:id|identifier)\s*"
                r"(?:is\s*)?[:=#]?\s*[a-z0-9][a-z0-9._-]{5,}\b",
                text,
            )
        )
    return False


def eligible_source_subset(
    sources: tuple[dict[str, Any], ...],
    *,
    today: date | None = None,
) -> tuple[tuple[dict[str, Any], ...], str]:
    """Keep only snapshots with current approval, freshness, provenance, and relevance."""
    current_date = today or date.today()
    eligible: list[dict[str, Any]] = []
    for source in sources:
        try:
            passage_id = source["passage_id"]
            passage_text = source["passage_text"]
            minimum_similarity = source["minimum_similarity"]
            similarity = source["similarity"]
            reviewed_at = date.fromisoformat(source["reviewed_at"])
            review_interval_days = source["review_interval_days"]
            eligible_now = (
                isinstance(passage_id, str)
                and bool(passage_id)
                and isinstance(passage_text, str)
                and bool(passage_text.strip())
                and source.get("approval_status") == "approved"
                and all(
                    isinstance(source.get(field), str) and bool(source[field])
                    for field in (
                        "doc_id",
                        "title",
                        "section",
                        "source_ref",
                        "approval_status_source",
                        "metadata_owner",
                        "metadata_source",
                        "article_version",
                        "content_sha256",
                        "metadata_manifest_sha256",
                    )
                )
                and str(passage_id).startswith(f"{source['doc_id']}:")
                and source["source_ref"]
                == f"data/kb/documentation.json#{source['doc_id']}#{source['section']}"
                and re.fullmatch(r"[0-9a-f]{64}", source["content_sha256"]) is not None
                and re.fullmatch(
                    r"[0-9a-f]{64}", source["metadata_manifest_sha256"]
                ) is not None
                and isinstance(minimum_similarity, (int, float))
                and not isinstance(minimum_similarity, bool)
                and isinstance(similarity, (int, float))
                and not isinstance(similarity, bool)
                and minimum_similarity == MINIMUM_SIMILARITY
                and similarity >= minimum_similarity
                and similarity <= 1
                and isinstance(review_interval_days, int)
                and not isinstance(review_interval_days, bool)
                and review_interval_days >= 0
                and reviewed_at <= current_date
                and reviewed_at + timedelta(days=review_interval_days) >= current_date
            )
        except (KeyError, TypeError, ValueError):
            eligible_now = False
        if eligible_now:
            eligible.append(source)
    detail = f"{len(eligible)} current eligible source(s) retained; {len(sources) - len(eligible)} excluded."
    return tuple(eligible), detail


def contains_prompt_injection(text: str) -> bool:
    """Detect common instruction override attempts in untrusted text."""
    return any(pattern.search(text) for pattern in _PROMPT_INJECTION_PATTERNS)


def contains_private_data(text: str) -> bool:
    """Detect common personal identifiers and credential-like strings."""
    return any(pattern.search(text) for pattern in _PRIVATE_DATA_PATTERNS)


def contains_private_source_data(source: dict[str, Any]) -> bool:
    """Detect private data in every retrieved field sent to the draft provider."""
    provider_input = {
        field: source.get(field)
        for field in ("passage_id", "title", "passage_text")
    }
    return contains_private_data(json.dumps(provider_input, ensure_ascii=False))


def sanitize_handoff_content(value: object) -> tuple[object, bool, bool]:
    """Withhold common private data and instruction overrides from handoff output."""
    if isinstance(value, dict):
        sanitized: dict[object, object] = {}
        private_data_withheld = False
        instruction_text_withheld = False
        for key, item in value.items():
            safe_item, private_item, instruction_item = sanitize_handoff_content(item)
            safe_key = key
            if isinstance(key, str):
                safe_key, private_key, instruction_key = sanitize_handoff_content(key)
                private_data_withheld = private_data_withheld or private_key
                instruction_text_withheld = instruction_text_withheld or instruction_key
            if safe_key in sanitized:
                original_safe_key = safe_key
                collision_index = 2
                while safe_key in sanitized:
                    safe_key = f"{original_safe_key} ({collision_index})"
                    collision_index += 1
            sanitized[safe_key] = safe_item
            private_data_withheld = private_data_withheld or private_item
            instruction_text_withheld = instruction_text_withheld or instruction_item
        return sanitized, private_data_withheld, instruction_text_withheld
    if isinstance(value, list):
        sanitized_items = [sanitize_handoff_content(item) for item in value]
        return (
            [item for item, _, _ in sanitized_items],
            any(private_item for _, private_item, _ in sanitized_items),
            any(instruction_item for _, _, instruction_item in sanitized_items),
        )
    if isinstance(value, str):
        sanitized_text = value
        private_data_withheld = False
        for pattern in _PRIVATE_DATA_PATTERNS:
            sanitized_text, replacements = pattern.subn(
                "[private data withheld]", sanitized_text
            )
            private_data_withheld = private_data_withheld or replacements > 0
        instruction_text_withheld = False
        for pattern in _PROMPT_INJECTION_PATTERNS:
            sanitized_text, replacements = pattern.subn(
                "[instruction-like text withheld]", sanitized_text
            )
            instruction_text_withheld = instruction_text_withheld or replacements > 0
        return sanitized_text, private_data_withheld, instruction_text_withheld
    return value, False, False


def check_claim_grounding(
    claims: list[dict[str, Any]],
    sources: tuple[dict[str, Any], ...],
) -> tuple[bool, str]:
    """Require every output claim to be an exact excerpt from a cited source."""
    sources_by_id = {source["passage_id"]: source for source in sources}
    if not claims:
        return False, "No source-grounded claims were returned."
    for claim in claims:
        source_ids = set(claim["source_ids"])
        if not source_ids or not source_ids.issubset(sources_by_id):
            return False, "A claim cited a passage outside the retrieved eligible source set."
        normalized_claim = _normalize_excerpt(claim["text"])
        if re.search(r"\[citation:[^\]]+\]", claim["text"], re.I):
            return False, "A claim contained an untrusted citation marker."
        if not all(
            normalized_claim in _normalize_excerpt(sources_by_id[source_id]["passage_text"])
            for source_id in source_ids
        ):
            return False, "A claim was not copied from every one of its cited source passages."
    return True, "Every claim is an exact excerpt with resolvable eligible passage citations."


def compose_cited_draft(
    generated: dict[str, Any],
    sources: tuple[dict[str, Any], ...],
    *,
    ticket_subject: str,
    ticket_text: str,
) -> dict[str, Any]:
    """Frame source excerpts and render only a fixed, missing-detail question."""
    sources_by_id = {source["passage_id"]: source for source in sources}
    claims = []
    text_parts = []
    general_possibilities = False
    for generated_claim in generated["claims"]:
        source_ids = list(dict.fromkeys(generated_claim["source_ids"]))
        citations = [
            {
                "passage_id": source_id,
                "source_ref": sources_by_id[source_id]["source_ref"],
            }
            for source_id in source_ids
        ]
        claim_text = generated_claim["text"].strip()
        if any(
            _describes_general_possibilities(claim_text, sources_by_id[source_id])
            for source_id in source_ids
        ):
            general_possibilities = True
        claims.append(
            {"text": claim_text, "source_ids": source_ids, "citations": citations}
        )
        text_parts.append(
            f"{claim_text} " + " ".join(f"[citation:{source_id}]" for source_id in source_ids)
        )
    subject = " ".join(ticket_subject.split()) or "your support question"
    if general_possibilities:
        preamble = (
            f"Thanks for contacting us about {subject}. "
            "The cited documentation lists possible explanations:"
        )
        context_note = (
            "These are general possibilities, not confirmation that any one cause applies "
            "to your case."
        )
    else:
        preamble = (
            f"Thanks for contacting us about {subject}. "
            "The cited documentation provides this guidance:"
        )
        context_note = (
            "This is general documentation and does not confirm what has happened in your "
            "specific environment."
        )
    clarifying_question = clarification_question_for_ticket(
        generated["clarification_kind"], ticket_text
    )
    message_parts = [preamble, *text_parts, context_note]
    if clarifying_question:
        message_parts.append(clarifying_question)
    plain_text = "\n".join(message_parts)
    return {
        "plain_text": plain_text,
        "claims": claims,
        "unsupported_points": generated["unsupported_points"],
        "conflicting_points": generated["conflicting_points"],
        "approval_required": True,
        "send_status": "not_sent",
    }


def _describes_general_possibilities(claim_text: str, source: dict[str, Any]) -> bool:
    """Recognize wording that lists symptoms, possible causes, or general scenarios."""
    text = " ".join(
        (
            str(source.get("title", "")),
            str(source.get("section", "")),
            claim_text,
        )
    ).casefold()
    return bool(
        re.search(
            r"\b(common|possible|potential|typical)\s+(?:causes?|reasons?|symptoms?|issues?)\b"
            r"|\b(?:may|might|could)\s+(?:cause|occur|happen|indicate)\b"
            r"|\bcommon symptoms?\b",
            text,
        )
    )


def filter_evaluation_context(value: object) -> object:
    """Remove labels/history keys recursively from handoff context snapshots."""
    if isinstance(value, dict):
        return {
            key: filter_evaluation_context(item)
            for key, item in value.items()
            if str(key).casefold() not in _EVALUATION_KEYS
        }
    if isinstance(value, list):
        return [filter_evaluation_context(item) for item in value]
    return value


def _normalize_excerpt(value: str) -> str:
    """Normalize whitespace and case only; do not permit semantic paraphrases."""
    return " ".join(value.split()).casefold()


def ticket_text_for_generation(ticket: CanonicalTicket) -> str:
    """Return the operational ticket projection used by pre-generation checks."""
    return operational_ticket_text(ticket)
