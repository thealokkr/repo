"""Check post-run citation resolution, source eligibility, and explicit non-measures."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from jsonschema import Draft202012Validator

from cloudserve_support.evaluation.citation_source_integrity import (
    evaluate_citation_source_integrity,
)
from cloudserve_support.reporting.output_schemas import CITATION_SOURCE_INTEGRITY_SCHEMA


RUN_ID = "a" * 32
CHECKED_AT = datetime(2026, 11, 10, 12, tzinfo=timezone.utc)


def _source(
    *,
    passage_id: str,
    doc_id: str,
    passage_text: str,
    approval_status: str = "approved",
    reviewed_at: str = "2026-11-10",
) -> dict[str, Any]:
    """Build a current source snapshot with the full citation provenance contract."""
    return {
        "passage_id": passage_id,
        "doc_id": doc_id,
        "source_ref": f"data/kb/documentation.json#{doc_id}#Section",
        "passage_text": passage_text,
        "approval_status": approval_status,
        "approval_status_source": "test project metadata manifest",
        "metadata_owner": "test maintainer",
        "metadata_source": "test project metadata manifest",
        "article_version": "test-1.0",
        "content_sha256": "1" * 64,
        "metadata_manifest_sha256": "2" * 64,
        "reviewed_at": reviewed_at,
        "review_interval_days": 365,
    }


def _draft_result(
    *,
    ticket_id: str,
    input_index: int,
    claim_text: str,
    source: dict[str, Any] | None,
    include_citation: bool = True,
    run_id: str = RUN_ID,
) -> dict[str, Any]:
    """Build one generated draft result with optional source/citation evidence."""
    passage_id = source["passage_id"] if source else "missing-passage"
    citation = (
        [{"passage_id": passage_id, "source_ref": source["source_ref"]}]
        if include_citation and source
        else []
    )
    return {
        "run_id": run_id,
        "input_index": input_index,
        "ticket_id": ticket_id,
        "route": "draft_for_review",
        "generation": {"status": "generated"},
        "draft": {
            "claims": [
                {
                    "text": claim_text,
                    "source_ids": [passage_id] if citation else [],
                    "citations": citation,
                }
            ]
        },
        "sources": [source] if source else [],
    }


def test_citation_integrity_counts_all_claims_and_never_implies_semantic_quality() -> None:
    supported_text = "Rotate the key from the account security page."
    valid_source = _source(
        passage_id="DOC-1:section:0",
        doc_id="DOC-1",
        passage_text=supported_text,
    )
    expired_source = _source(
        passage_id="DOC-2:section:0",
        doc_id="DOC-2",
        passage_text="Approved account recovery steps appear on the account page.",
        approval_status="expired",
        reviewed_at="2020-01-01",
    )
    results = [
        _draft_result(
            ticket_id="VALID",
            input_index=0,
            claim_text=supported_text,
            source=valid_source,
        ),
        _draft_result(
            ticket_id="INELIGIBLE",
            input_index=1,
            claim_text="This answer does not appear in the cited passage.",
            source=expired_source,
        ),
        _draft_result(
            ticket_id="UNCITED",
            input_index=2,
            claim_text="This claim has no citation.",
            source=None,
            include_citation=False,
        ),
        _draft_result(
            ticket_id="CROSS-RUN",
            input_index=3,
            claim_text=supported_text,
            source=valid_source,
            run_id="b" * 32,
        ),
    ]
    labels = {
        0: {"expected_doc_ids": ["DOC-1"]},
        1: {"expected_doc_ids": ["DOC-2", "DOC-3"]},
        2: {"expected_doc_ids": [], "answerable_from_docs": False},
        3: {"expected_doc_ids": ["DOC-1"]},
    }

    report = evaluate_citation_source_integrity(
        run_id=RUN_ID,
        results=results,
        labels_by_index=labels,
        evaluated_at=CHECKED_AT,
    )

    assert report["generated_draft_count"] == 4
    assert report["claim_count"] == 4
    assert report["citation_count"] == 3
    assert report["citation_resolution"]["numerator"] == 2
    assert report["citation_resolution"]["denominator"] == 3
    assert report["eligible_source_provenance"]["numerator"] == 1
    assert report["eligible_source_provenance"]["denominator"] == 3
    assert report["claim_excerpt_match"]["numerator"] == 1
    assert report["claim_excerpt_match"]["denominator"] == 3
    assert report["claim_traceability"]["numerator"] == 1
    assert report["claim_traceability"]["denominator"] == 4
    assert report["claims_missing_citations"] == 1
    assert report["expected_document_coverage"]["numerator"] == 3
    assert report["expected_document_coverage"]["denominator"] == 4
    assert report["failure_counts"]["source_not_approved"] == 1
    assert report["failure_counts"]["claim_excerpt_mismatch"] == 1
    assert report["failure_counts"]["cross_run_citation"] == 1
    assert set(report["semantic_quality"]) == {
        "semantic_citation_support",
        "whole_answer_correctness",
        "human_rated_citation_accuracy",
        "hallucination",
    }
    assert {metric["status"] for metric in report["semantic_quality"].values()} == {
        "not_measured"
    }


def test_zero_draft_run_reports_not_applicable_integrity_and_unmeasured_semantics() -> None:
    report = evaluate_citation_source_integrity(
        run_id=RUN_ID,
        results=[
            {
                "run_id": RUN_ID,
                "input_index": 0,
                "ticket_id": "ESCALATED",
                "route": "escalate",
                "generation": {"status": "not_attempted"},
                "draft": None,
                "sources": [],
            }
        ],
        labels_by_index={0: {"expected_doc_ids": []}},
        evaluated_at=CHECKED_AT,
    )

    assert report["generated_draft_count"] == 0
    assert report["claim_count"] == 0
    assert report["citation_resolution"]["status"] == "not_applicable"
    assert report["claim_traceability"]["status"] == "not_applicable"
    assert all(
        metric["status"] == "not_measured"
        for metric in report["semantic_quality"].values()
    )


def test_missing_or_ambiguous_source_snapshots_are_counted_as_unresolved() -> None:
    source = _source(
        passage_id="DOC-1:section:0",
        doc_id="DOC-1",
        passage_text="The cited statement is present in this passage.",
    )
    missing_snapshot = _draft_result(
        ticket_id="MISSING-SNAPSHOT",
        input_index=0,
        claim_text="The cited statement is present in this passage.",
        source=source,
    )
    missing_snapshot["sources"] = None
    ambiguous_snapshot = _draft_result(
        ticket_id="DUPLICATE-SNAPSHOT",
        input_index=1,
        claim_text="The cited statement is present in this passage.",
        source=source,
    )
    ambiguous_snapshot["sources"].append({**source, "passage_text": "A duplicate record."})

    report = evaluate_citation_source_integrity(
        run_id=RUN_ID,
        results=[missing_snapshot, ambiguous_snapshot],
        labels_by_index={
            0: {"expected_doc_ids": ["DOC-1"]},
            1: {"expected_doc_ids": ["DOC-1"]},
        },
        evaluated_at=CHECKED_AT,
    )

    assert report["citation_resolution"]["numerator"] == 0
    assert report["citation_resolution"]["denominator"] == 2
    assert report["failure_counts"]["invalid_source_snapshot_list"] == 1
    assert report["failure_counts"]["duplicate_source_snapshot_id"] == 1
    assert report["failure_counts"]["unresolved_citation"] == 2


def test_malformed_review_dates_are_counted_as_source_failures() -> None:
    valid_source = _source(
        passage_id="DOC-1:section:0",
        doc_id="DOC-1",
        passage_text="The cited statement is present in this passage.",
    )
    missing_date_source = {key: value for key, value in valid_source.items() if key != "reviewed_at"}
    non_string_date_source = {**valid_source, "reviewed_at": None}
    invalid_date_source = {**valid_source, "reviewed_at": "not-a-date"}
    results = [
        _draft_result(
            ticket_id=f"BAD-DATE-{index}",
            input_index=index,
            claim_text="The cited statement is present in this passage.",
            source=source,
        )
        for index, source in enumerate(
            (missing_date_source, non_string_date_source, invalid_date_source)
        )
    ]

    report = evaluate_citation_source_integrity(
        run_id=RUN_ID,
        results=results,
        labels_by_index={
            index: {"expected_doc_ids": ["DOC-1"]}
            for index in range(len(results))
        },
        evaluated_at=CHECKED_AT,
    )

    assert report["eligible_source_provenance"]["numerator"] == 0
    assert report["eligible_source_provenance"]["denominator"] == 3
    assert report["failure_counts"]["invalid_source_review_date"] == 3


def test_citation_integrity_schema_requires_all_metrics_and_not_measured_statuses() -> None:
    source = _source(
        passage_id="DOC-1:section:0",
        doc_id="DOC-1",
        passage_text="The cited statement is present in this passage.",
    )
    report = evaluate_citation_source_integrity(
        run_id=RUN_ID,
        results=[
            _draft_result(
                ticket_id="SCHEMA-CHECK",
                input_index=0,
                claim_text="The cited statement is present in this passage.",
                source=source,
            )
        ],
        labels_by_index={0: {"expected_doc_ids": ["DOC-1"]}},
        evaluated_at=CHECKED_AT,
    )
    validator = Draft202012Validator(CITATION_SOURCE_INTEGRITY_SCHEMA)
    assert validator.is_valid(report)
    assert not validator.is_valid({})

    misleading_semantic_status = {
        **report,
        "semantic_quality": {
            **report["semantic_quality"],
            "hallucination": {
                "status": "available",
                "reason": "test invalid semantic status",
            },
        },
    }
    assert not validator.is_valid(misleading_semantic_status)
