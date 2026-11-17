"""Measure post-run citation resolution and exact claim-to-source integrity."""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta, timezone
from typing import Any

from cloudserve_support.guardrails.draft_safety import _text_supports_claim


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
SOURCE_PROVENANCE_FIELDS = (
    "passage_id",
    "doc_id",
    "source_ref",
    "passage_text",
    "approval_status_source",
    "metadata_owner",
    "metadata_source",
    "article_version",
    "content_sha256",
    "metadata_manifest_sha256",
)
SEMANTIC_QUALITY_NOT_MEASURED = {
    "semantic_citation_support": (
        "This mechanical source-integrity section does not assess semantic support. "
        "See evaluation.criteria.llm_judge for the separate model-estimated sample."
    ),
    "whole_answer_correctness": (
        "Whole-answer correctness is not mechanically established by citation integrity. "
        "See evaluation.criteria.llm_judge for the separate model-estimated sample."
    ),
    "human_rated_citation_accuracy": (
        "Human-rated citation accuracy remains unmeasured; see evaluation.criteria.llm_judge "
        "for a separate model-estimated citation-support sample."
    ),
    "hallucination": (
        "Mechanical excerpt matching does not establish hallucination. See "
        "evaluation.criteria.llm_judge for a separate model-estimated sample."
    ),
}


def evaluate_citation_source_integrity(
    *,
    run_id: str,
    results: list[dict[str, Any]],
    labels_by_index: dict[int, dict[str, Any]],
    evaluated_at: datetime | None = None,
) -> dict[str, Any]:
    """Check every generated factual claim against its same-run eligible cited sources."""
    checked_at = evaluated_at or datetime.now(timezone.utc)
    as_of_date = checked_at.date()
    generated_draft_count = 0
    claim_count = 0
    citation_count = 0
    resolved_citation_count = 0
    eligible_citation_count = 0
    exact_excerpt_pair_count = 0
    traceable_claim_count = 0
    claims_missing_citations = 0
    claims_with_source_id_mismatch = 0
    failure_counts: dict[str, int] = {}

    for result in results:
        if not _is_generated_draft(result):
            continue
        generated_draft_count += 1
        draft = result.get("draft")
        claims = draft.get("claims", []) if isinstance(draft, dict) else []
        if not isinstance(claims, list):
            _increment(failure_counts, "invalid_claim_list")
            continue
        result_is_same_run = result.get("run_id") == run_id
        if not result_is_same_run:
            _increment(failure_counts, "cross_run_result")

        sources_by_id: dict[str, dict[str, Any]] = {}
        source_rows = result.get("sources", [])
        if not isinstance(source_rows, list):
            _increment(failure_counts, "invalid_source_snapshot_list")
            source_rows = []
        duplicate_source_ids: set[str] = set()
        for source in source_rows:
            if not isinstance(source, dict) or not isinstance(source.get("passage_id"), str):
                _increment(failure_counts, "invalid_source_snapshot")
                continue
            passage_id = source["passage_id"]
            if passage_id in sources_by_id:
                _increment(failure_counts, "duplicate_source_snapshot_id")
                duplicate_source_ids.add(passage_id)
                continue
            sources_by_id[passage_id] = source

        for claim in claims:
            if not isinstance(claim, dict):
                claim_count += 1
                _increment(failure_counts, "invalid_claim_record")
                continue
            claim_count += 1
            claim_text = claim.get("text")
            citations = claim.get("citations")
            source_ids = claim.get("source_ids")
            citation_rows = citations if isinstance(citations, list) else []
            citation_count += len(citation_rows)
            claim_has_citation = bool(citation_rows) and isinstance(source_ids, list) and bool(source_ids)
            if not claim_has_citation:
                claims_missing_citations += 1
                _increment(failure_counts, "claim_without_citation")

            citation_ids = [
                citation.get("passage_id") if isinstance(citation, dict) else None
                for citation in citation_rows
            ]
            source_id_mapping_matches = not (
                not isinstance(source_ids, list)
                or any(not isinstance(source_id, str) for source_id in source_ids)
                or set(source_ids) != {citation_id for citation_id in citation_ids if isinstance(citation_id, str)}
            )
            if not source_id_mapping_matches:
                claims_with_source_id_mismatch += 1
                _increment(failure_counts, "claim_source_id_mapping_mismatch")

            all_pairs_valid = (
                claim_has_citation
                and source_id_mapping_matches
                and isinstance(claim_text, str)
                and bool(claim_text.strip())
            )
            for citation in citation_rows:
                if not isinstance(citation, dict):
                    _increment(failure_counts, "invalid_citation_record")
                    all_pairs_valid = False
                    continue
                passage_id = citation.get("passage_id")
                source = sources_by_id.get(passage_id) if isinstance(passage_id, str) else None
                citation_resolves = (
                    result_is_same_run
                    and isinstance(source, dict)
                    and passage_id not in duplicate_source_ids
                    and isinstance(citation.get("source_ref"), str)
                    and citation.get("source_ref") == source.get("source_ref")
                )
                if citation_resolves:
                    resolved_citation_count += 1
                else:
                    _increment(
                        failure_counts,
                        "cross_run_citation" if not result_is_same_run else "unresolved_citation",
                    )
                    all_pairs_valid = False
                    continue

                eligibility_issue = source_eligibility_issue(source, as_of_date)
                if eligibility_issue is None:
                    eligible_citation_count += 1
                else:
                    _increment(failure_counts, eligibility_issue)
                    all_pairs_valid = False

                passage_text = source.get("passage_text")
                excerpt_matches = (
                    isinstance(claim_text, str)
                    and isinstance(passage_text, str)
                    and _normalize_excerpt(claim_text) in _normalize_excerpt(passage_text)
                )
                if excerpt_matches:
                    exact_excerpt_pair_count += 1
                claim_supported = (
                    isinstance(claim_text, str)
                    and isinstance(passage_text, str)
                    and _text_supports_claim(claim_text, passage_text, min_ratio=0.35)
                )
                if not claim_supported:
                    _increment(failure_counts, "claim_excerpt_mismatch")
                    all_pairs_valid = False
            if all_pairs_valid:
                traceable_claim_count += 1

    expected_document_count = 0
    expected_document_hit_count = 0
    missing_expected_document_label_count = 0
    for result in results:
        input_index = result.get("input_index")
        label = labels_by_index.get(input_index) if isinstance(input_index, int) else None
        if not isinstance(label, dict):
            missing_expected_document_label_count += 1
            continue
        expected_doc_ids = label.get("expected_doc_ids")
        if not isinstance(expected_doc_ids, list) or any(
            not isinstance(doc_id, str) for doc_id in expected_doc_ids
        ):
            if label.get("answerable_from_docs") is not False:
                missing_expected_document_label_count += 1
            continue
        expected_doc_set = set(expected_doc_ids)
        if not expected_doc_set and label.get("answerable_from_docs") is False:
            continue
        expected_document_count += len(expected_doc_set)
        source_rows = result.get("sources", [])
        if not isinstance(source_rows, list):
            source_rows = []
        available_doc_ids = {
            source.get("doc_id")
            for source in source_rows
            if isinstance(source, dict) and isinstance(source.get("doc_id"), str)
        }
        expected_document_hit_count += len(expected_doc_set & available_doc_ids)

    semantic_quality = {
        name: {"status": "not_measured", "reason": reason}
        for name, reason in SEMANTIC_QUALITY_NOT_MEASURED.items()
    }
    return {
        "checked_at_utc": checked_at.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        "run_result_count": len(results),
        "generated_draft_count": generated_draft_count,
        "claim_count": claim_count,
        "citation_count": citation_count,
        "citation_resolution": _integrity_metric(
            resolved_citation_count,
            citation_count,
            citation_count - resolved_citation_count,
            "No citations were generated.",
        ),
        "eligible_source_provenance": _integrity_metric(
            eligible_citation_count,
            citation_count,
            citation_count - eligible_citation_count,
            "No citations were generated.",
        ),
        "claim_excerpt_match": _integrity_metric(
            exact_excerpt_pair_count,
            citation_count,
            citation_count - exact_excerpt_pair_count,
            "No citation-to-claim pairs were generated.",
        ),
        "claim_traceability": _integrity_metric(
            traceable_claim_count,
            claim_count,
            claim_count - traceable_claim_count,
            "No factual claims were generated.",
        ),
        "claims_missing_citations": claims_missing_citations,
        "claims_with_source_id_mismatch": claims_with_source_id_mismatch,
        "failure_counts": dict(sorted(failure_counts.items())),
        "expected_document_coverage": _integrity_metric(
            expected_document_hit_count,
            expected_document_count,
            missing_expected_document_label_count,
            "No expected document IDs were available.",
        ),
        "expected_document_count": expected_document_count,
        "expected_document_hit_count": expected_document_hit_count,
        "missing_expected_document_label_count": missing_expected_document_label_count,
        "semantic_quality": semantic_quality,
        "note": (
            "Citation resolution, source eligibility, and exact excerpt checks are mechanical integrity checks. "
            "They do not establish semantic citation support, whole-answer correctness, or hallucination."
        ),
    }


def _is_generated_draft(result: dict[str, Any]) -> bool:
    """Recognize an emitted draft from the persisted route, provider status, and payload."""
    generation = result.get("generation")
    return (
        result.get("route") == "draft_for_review"
        and isinstance(generation, dict)
        and generation.get("status") == "generated"
        and isinstance(result.get("draft"), dict)
    )


def source_eligibility_issue(source: dict[str, Any], as_of_date: date) -> str | None:
    """Check whether one source snapshot has eligible provenance and is fresh as of a date."""
    reviewed_at_value = source.get("reviewed_at")
    if not isinstance(reviewed_at_value, str) or not reviewed_at_value.strip():
        return "invalid_source_review_date"
    if not all(isinstance(source.get(field), str) and source[field].strip() for field in SOURCE_PROVENANCE_FIELDS):
        return "missing_source_provenance"
    if source.get("approval_status") != "approved":
        return "source_not_approved"
    if not SHA256_PATTERN.fullmatch(source["content_sha256"]):
        return "invalid_source_content_hash"
    if not SHA256_PATTERN.fullmatch(source["metadata_manifest_sha256"]):
        return "invalid_source_metadata_hash"
    review_interval = source.get("review_interval_days")
    if isinstance(review_interval, bool) or not isinstance(review_interval, int) or review_interval < 0:
        return "invalid_source_review_interval"
    try:
        reviewed_at = date.fromisoformat(reviewed_at_value)
    except (TypeError, ValueError):
        return "invalid_source_review_date"
    if reviewed_at > as_of_date or reviewed_at + timedelta(days=review_interval) < as_of_date:
        return "stale_source_snapshot"
    return None


def _normalize_excerpt(value: str) -> str:
    """Normalize case and whitespace for exact source excerpt comparison."""
    return " ".join(value.split()).casefold()


def _integrity_metric(
    numerator: int,
    denominator: int,
    missing_count: int,
    not_applicable_note: str,
) -> dict[str, Any]:
    """Return a reproducible integrity ratio with explicit counts and empty-sample status."""
    return {
        "value": numerator / denominator if denominator else None,
        "status": "available" if denominator else "unavailable" if missing_count else "not_applicable",
        "numerator": numerator,
        "denominator": denominator,
        "sample_size": denominator,
        "missing_count": missing_count,
        "note": None if denominator else not_applicable_note,
    }


def _increment(counts: dict[str, int], key: str) -> None:
    """Count one concrete citation/source integrity failure code."""
    counts[key] = counts.get(key, 0) + 1
