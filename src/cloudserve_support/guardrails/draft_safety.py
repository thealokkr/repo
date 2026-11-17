"""Validate provider drafts against source, privacy, and instruction boundaries."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
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
_TOKEN_PATTERN = re.compile(r"[a-z0-9]{3,}")
_CLAIM_STOPWORDS = frozenset(
    {
        "the", "and", "for", "with", "that", "this", "from", "your", "you",
        "are", "was", "were", "can", "will", "not", "but", "have", "has",
        "had", "into", "onto", "about", "after", "before", "when", "where",
        "which", "who", "what", "how", "all", "any", "each", "only", "also",
        "more", "most", "other", "such", "than", "then", "them", "they",
        "their", "there", "these", "those", "its", "our", "may", "might",
        "could", "should", "would", "use", "using", "used",
    }
)


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


def inspect_claim_grounding(
    claims: list[dict[str, Any]],
    sources: tuple[dict[str, Any], ...],
    *,
    verified_claims: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Evaluate all claims and return per-claim grounding failures without mutating state."""
    sources_by_id = {
        source["passage_id"]: source
        for source in sources
        if isinstance(source.get("passage_id"), str)
    }
    if not claims:
        return {
            "grounded": False,
            "failure_reason": "No source-grounded claims were returned.",
            "failed_claims": [],
            "minimum_ratio": 0.65,
        }
    verified_by_id = (
        _index_verified_claims_by_passage_id(sources, verified_claims)
        if verified_claims is not None
        else None
    )
    failed_claims: list[dict[str, Any]] = []
    for claim_position, claim in enumerate(claims, start=1):
        failure = _inspect_claim_grounding(
            claim,
            claim_position=claim_position,
            sources_by_id=sources_by_id,
            verified_by_id=verified_by_id,
        )
        if failure is not None:
            failed_claims.append(failure)
    return {
        "grounded": not failed_claims,
        "failure_reason": failed_claims[0]["failure_reason"] if failed_claims else None,
        "failed_claims": failed_claims,
        "minimum_ratio": 0.65,
    }


def check_claim_grounding(
    claims: list[dict[str, Any]],
    sources: tuple[dict[str, Any], ...],
    *,
    verified_claims: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[bool, str]:
    """Require every output claim to cite safe same-ticket sources and stay grounded."""
    sources_by_id = {
        source["passage_id"]: source
        for source in sources
        if isinstance(source.get("passage_id"), str)
    }
    if not claims:
        return False, "No source-grounded claims were returned."
    verified_by_id = (
        _index_verified_claims_by_passage_id(sources, verified_claims)
        if verified_claims is not None
        else None
    )
    for claim_position, claim in enumerate(claims, start=1):
        failure = _inspect_claim_grounding(
            claim,
            claim_position=claim_position,
            sources_by_id=sources_by_id,
            verified_by_id=verified_by_id,
        )
        if failure is not None:
            return False, failure["failure_reason"]
    return True, "Every claim is grounded in policy-verified claims and resolvable eligible sources."


def _index_verified_claims_by_passage_id(
    sources: tuple[dict[str, Any], ...],
    verified_claims: Sequence[Mapping[str, Any]],
) -> dict[str, list[str]]:
    """Map each safe source passage_id to policy-verified claim texts for that article."""
    by_passage_id: dict[str, list[str]] = {
        source["passage_id"]: []
        for source in sources
        if isinstance(source.get("passage_id"), str)
    }
    valid_claims = [
        (
            verified["text"].strip(),
            {sid for sid in verified["source_ids"] if isinstance(sid, str)},
        )
        for verified in verified_claims
        if isinstance(verified.get("text"), str)
        and verified["text"].strip()
        and isinstance(verified.get("source_ids"), list)
    ]
    for source in sources:
        passage_id = source.get("passage_id")
        doc_id = source.get("doc_id")
        if not isinstance(passage_id, str):
            continue
        by_passage_id[passage_id] = [
            text
            for text, id_set in valid_claims
            if passage_id in id_set or doc_id in id_set
        ]
    return by_passage_id


def _inspect_claim_grounding(
    claim: Mapping[str, Any],
    *,
    claim_position: int,
    sources_by_id: dict[str, dict[str, Any]],
    verified_by_id: dict[str, list[str]] | None,
) -> dict[str, Any] | None:
    """Return structured failure details for one claim, or None when it passes."""
    claim_text = claim.get("text")
    display_claim = claim_text if isinstance(claim_text, str) else ""
    raw_source_ids = claim.get("source_ids")
    display_source_ids = (
        [sid for sid in raw_source_ids if isinstance(sid, str)]
        if isinstance(raw_source_ids, list)
        else []
    )

    def _failure(
        failure_reason: str,
        *,
        failed_clauses: list[str] | None = None,
        minimum_ratio: float | None = None,
    ) -> dict[str, Any]:
        return {
            "claim_position": claim_position,
            "claim": display_claim,
            "source_ids": display_source_ids,
            "failed_clauses": failed_clauses or [],
            "failure_reason": failure_reason,
            "minimum_ratio": minimum_ratio,
        }

    if not isinstance(claim_text, str) or not claim_text.strip():
        return _failure("A claim was empty or malformed.")
    if re.search(r"\[citation:[^\]]+\]", claim_text, re.I):
        return _failure("A claim contained an untrusted citation marker.")
    if not isinstance(raw_source_ids, list) or not raw_source_ids or not all(
        isinstance(sid, str) for sid in raw_source_ids
    ):
        return _failure("A claim is missing valid source_ids.")
    ordered_source_ids = list(dict.fromkeys(raw_source_ids))
    source_ids = set(ordered_source_ids)
    if not source_ids.issubset(sources_by_id):
        return _failure("A claim cited a passage outside the retrieved eligible source set.")
    if "citations" in claim and not _citations_match_sources(
        claim["citations"],
        source_ids=source_ids,
        sources_by_id=sources_by_id,
    ):
        return _failure("A claim had unresolved or mismatched structured citations.")

    source_failed_clauses = _failed_text_support_clauses(
        claim_text,
        source_ids=ordered_source_ids,
        sources_by_id=sources_by_id,
        min_ratio=0.65,
    )
    if source_failed_clauses is not None:
        return _failure(
            "A claim was not supported by every one of its cited source articles.",
            failed_clauses=source_failed_clauses,
            minimum_ratio=0.65,
        )

    if verified_by_id is not None:
        verified_failed_clauses, verified_ratio = _failed_verified_claim_clauses(
            claim_text,
            source_ids=ordered_source_ids,
            sources_by_id=sources_by_id,
            verified_by_id=verified_by_id,
        )
        if verified_failed_clauses is not None:
            return _failure(
                "A claim was not grounded in the policy-verified claims for its cited articles.",
                failed_clauses=verified_failed_clauses,
                minimum_ratio=verified_ratio,
            )
    return None


def _failed_text_support_clauses(
    claim_text: str,
    *,
    source_ids: Sequence[str],
    sources_by_id: dict[str, dict[str, Any]],
    min_ratio: float,
) -> list[str] | None:
    """Return failing claim clauses against cited passages, or None when all pass."""
    segments = _claim_segments(claim_text, split_clauses=True)
    if not segments:
        return [claim_text.strip()] if claim_text.strip() else []
    failed: list[str] = []
    seen: set[str] = set()
    for sid in source_ids:
        reference_text = str(sources_by_id[sid].get("passage_text", ""))
        for segment in segments:
            if not _segment_supports_claim(
                segment,
                reference_text,
                min_ratio=min_ratio,
                use_overlap_coefficient=False,
            ) and segment not in seen:
                seen.add(segment)
                failed.append(segment)
    return failed if failed else None


def _failed_verified_claim_clauses(
    claim_text: str,
    *,
    source_ids: Sequence[str],
    sources_by_id: dict[str, dict[str, Any]],
    verified_by_id: dict[str, list[str]],
) -> tuple[list[str] | None, float]:
    """Return failing claim clauses against policy-verified claims, or None when all pass."""
    source_id_set = set(source_ids)
    is_verbatim_excerpt = _clause_is_verbatim_in_sources(
        claim_text,
        source_ids=source_id_set,
        sources_by_id=sources_by_id,
    )
    min_ratio = 0.25 if is_verbatim_excerpt else 0.65
    clauses = _claim_segments(claim_text, split_clauses=not is_verbatim_excerpt)
    combined_verified: list[str] = []
    for sid in source_ids:
        entries = verified_by_id.get(sid, [])
        if not entries:
            return (clauses or [claim_text.strip()]), min_ratio
        combined_verified.extend(entries)
    if not clauses:
        return ([claim_text.strip()] if claim_text.strip() else []), min_ratio
    verified_text = " ".join(combined_verified)
    failed = [
        clause
        for clause in clauses
        if not _segment_supports_claim(
            clause,
            verified_text,
            min_ratio=min_ratio,
            use_overlap_coefficient=is_verbatim_excerpt,
        )
    ]
    return (failed if failed else None), min_ratio


def _citations_match_sources(
    citations: object,
    *,
    source_ids: set[str],
    sources_by_id: dict[str, dict[str, Any]],
) -> bool:
    """Verify structured citations resolve to the same safe snapshots as source_ids."""
    if not isinstance(citations, list) or not citations:
        return False
    cited_ids: set[str] = set()
    for citation in citations:
        if not isinstance(citation, Mapping):
            return False
        passage_id = citation.get("passage_id")
        if not isinstance(passage_id, str) or passage_id not in sources_by_id:
            return False
        if citation.get("source_ref") != sources_by_id[passage_id].get("source_ref"):
            return False
        cited_ids.add(passage_id)
    return cited_ids == source_ids


def _verified_claims_support(
    claim_text: str,
    *,
    source_ids: set[str],
    sources_by_id: dict[str, dict[str, Any]],
    verified_by_id: dict[str, list[str]],
) -> bool:
    """Verify each cited source has policy-verified claims supporting every claim clause."""
    combined_verified: list[str] = []
    for sid in source_ids:
        entries = verified_by_id.get(sid, [])
        if not entries:
            return False
        combined_verified.extend(entries)
    verified_text = " ".join(combined_verified)
    is_verbatim_excerpt = _clause_is_verbatim_in_sources(
        claim_text,
        source_ids=source_ids,
        sources_by_id=sources_by_id,
    )
    clauses = _claim_segments(claim_text, split_clauses=not is_verbatim_excerpt)
    if not clauses:
        return False
    return all(
        _segment_supports_claim(
            clause,
            verified_text,
            min_ratio=0.25 if is_verbatim_excerpt else 0.65,
            use_overlap_coefficient=is_verbatim_excerpt,
        )
        for clause in clauses
    )


def _clause_is_verbatim_in_sources(
    clause: str,
    *,
    source_ids: set[str],
    sources_by_id: dict[str, dict[str, Any]],
) -> bool:
    """Return True when a clause is an exact substring of every cited source passage."""
    normalized_clause = _normalize_excerpt(clause)
    return bool(normalized_clause) and all(
        normalized_clause in _normalize_excerpt(str(sources_by_id[sid].get("passage_text", "")))
        for sid in source_ids
    )


def _content_tokens(value: str) -> set[str]:
    """Extract normalized content words for non-verbatim claim grounding."""
    tokens = set(_TOKEN_PATTERN.findall(value.casefold()))
    filtered = tokens - _CLAIM_STOPWORDS
    return filtered or tokens


def _claim_segments(claim_text: str, *, split_clauses: bool = False) -> list[str]:
    """Split a claim into independent sentence or clause assertions."""
    pattern = (
        r"(?<=[.!?])\s+|;\s+|,\s+"
        r"|\s+(?:which|who|where|when|while)\b"
        r"|\s+(?:and|but|so|or)\s+(?=(?:this|that|it|they|there|here|you|we|he|she)\b)"
        if split_clauses
        else r"(?<=[.!?])\s+|;\s+"
    )
    segments = [
        part.strip()
        for part in re.split(pattern, claim_text.strip(), flags=re.I)
        if part.strip() and _content_tokens(part)
    ]
    return segments or ([claim_text.strip()] if claim_text.strip() else [])


def _segment_supports_claim(
    segment: str,
    reference_text: str,
    *,
    min_ratio: float,
    use_overlap_coefficient: bool,
) -> bool:
    """Check one claim sentence/clause against individual lines/sentences of the reference."""
    normalized_segment = _normalize_excerpt(segment)
    normalized_ref = _normalize_excerpt(reference_text)
    if normalized_segment and normalized_segment in normalized_ref:
        return True
    segment_tokens = _content_tokens(segment)
    if not segment_tokens:
        return False
    ref_units = [
        unit.strip()
        for unit in re.split(r"(?<=[.!?])\s+|\n+", reference_text)
        if unit.strip()
    ] or [reference_text]
    min_shared = min(2, len(segment_tokens))
    for unit in ref_units:
        unit_tokens = _content_tokens(unit)
        if not unit_tokens:
            continue
        overlap = len(segment_tokens & unit_tokens)
        denominator = (
            min(len(segment_tokens), len(unit_tokens))
            if use_overlap_coefficient
            else len(segment_tokens)
        )
        if overlap >= min_shared and (overlap / denominator) >= min_ratio:
            return True
    return False


def _text_supports_claim(
    claim_text: str,
    reference_text: str,
    *,
    min_ratio: float,
    use_overlap_coefficient: bool = False,
) -> bool:
    """Require every asserted sentence/clause in a claim to be supported by the reference."""
    effective_ratio = 0.65 if not use_overlap_coefficient and min_ratio < 0.65 else min_ratio
    segments = _claim_segments(claim_text, split_clauses=True)
    if not segments:
        return False
    if use_overlap_coefficient:
        sentence_segments = _claim_segments(claim_text, split_clauses=False)
        return all(
            _segment_supports_claim(
                segment,
                reference_text,
                min_ratio=effective_ratio,
                use_overlap_coefficient=True,
            )
            for segment in sentence_segments
        )
    return all(
        _segment_supports_claim(
            segment,
            reference_text,
            min_ratio=effective_ratio,
            use_overlap_coefficient=False,
        )
        for segment in segments
    )


def compose_cited_draft(
    generated: dict[str, Any],
    sources: tuple[dict[str, Any], ...],
    *,
    ticket_subject: str,
    ticket_text: str = "",
) -> dict[str, Any]:
    """Frame source excerpts with passage citations for agent review."""
    del ticket_text
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
    plain_text = "\n".join([preamble, *text_parts, context_note])
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
