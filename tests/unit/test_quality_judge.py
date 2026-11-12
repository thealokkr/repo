"""Check reproducible sampling, judge input isolation, and scorecard math."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from cloudserve_support.config import Settings
from cloudserve_support.evaluation.quality_judge import (
    QUALITY_JUDGE_SAMPLE_SIZE,
    QUALITY_JUDGE_SAMPLE_SEED,
    QualityJudgeError,
    _decision_events_by_key,
    _build_quality_judge_scorecard,
    _validated_judgment_row,
    build_quality_judge_input,
    select_quality_judge_sample,
)


def _source_snapshot(passage_id: str) -> dict[str, object]:
    return {
        "passage_id": passage_id,
        "doc_id": "DOC-1",
        "source_ref": "kb://DOC-1#p1",
        "passage_text": "The API token can be rotated from the security settings page.",
        "approval_status": "approved",
        "approval_status_source": "metadata_manifest",
        "metadata_owner": "capstone",
        "metadata_source": "project_manifest",
        "article_version": "capstone-1.0",
        "content_sha256": "a" * 64,
        "metadata_manifest_sha256": "b" * 64,
        "reviewed_at": "2026-11-01",
        "review_interval_days": 365,
        "title": "Rotate an API token",
    }


def _generated_result(index: int, run_id: str = "a" * 32) -> dict[str, object]:
    passage_id = f"PASSAGE-{index}"
    source = _source_snapshot(passage_id)
    return {
        "run_id": run_id,
        "input_index": index,
        "ticket_id": f"DEV-{index:04d}",
        "status": "completed",
        "route": "draft_for_review",
        "generation": {"status": "generated"},
        "draft": {
            "plain_text": "You can rotate the API token in security settings.",
            "claims": [
                {
                    "text": "The API token can be rotated from the security settings page.",
                    "citations": [
                        {"passage_id": passage_id, "source_ref": source["source_ref"]}
                    ],
                }
            ],
        },
        "sources": [source],
    }


def _judge_output(
    *,
    score: int = 4,
    citation_ids: list[str] | None = None,
    supported: bool = True,
    unsupported: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "dimensions": {
            name: {"score": score, "rationale": "The response is clear and supported by the supplied passage."}
            for name in (
                "relevance",
                "factual_correctness",
                "completeness_actionability",
                "clarity",
            )
        },
        "citations": [
            {"citation_id": citation_id, "supported": supported, "rationale": "The cited passage supports the claim."}
            for citation_id in (citation_ids or [])
        ],
        "unsupported_factual_claims": unsupported or [],
    }


def test_sample_selection_is_reproducible_and_uses_only_generated_drafts() -> None:
    results = [_generated_result(index) for index in range(27)]
    results.append(
        {
            **_generated_result(99),
            "ticket_id": "ESCALATED",
            "route": "escalate",
        }
    )
    results.append(
        {
            **_generated_result(100),
            "ticket_id": "NOT-COMPLETED",
            "status": "failed",
        }
    )
    input_identity = hashlib.sha256(b"development-input").hexdigest()

    first_sample, first_available = select_quality_judge_sample(
        results, input_identity_sha256=input_identity
    )
    second_sample, second_available = select_quality_judge_sample(
        results, input_identity_sha256=input_identity
    )

    assert first_available == second_available == 27
    assert len(first_sample) == QUALITY_JUDGE_SAMPLE_SIZE
    assert len({row["ticket_id"] for row in first_sample}) == QUALITY_JUDGE_SAMPLE_SIZE
    assert [row["ticket_id"] for row in first_sample] == [
        row["ticket_id"] for row in second_sample
    ]
    assert "ESCALATED" not in {row["ticket_id"] for row in first_sample}
    assert "NOT-COMPLETED" not in {row["ticket_id"] for row in first_sample}
    assert QUALITY_JUDGE_SAMPLE_SEED == "cloudserve-quality-judge-v1"


def test_judge_payload_allow_lists_ticket_draft_reference_and_eligible_source() -> None:
    result = _generated_result(1)
    raw_ticket = {
        "ticket_id": "DEV-0001",
        "channel": "email",
        "subject": "API token rotation",
        "body": "The labels on my account are incorrect.",
        "labels": {"intent": "evaluation-only"},
        "history": {"csat_rating": 1},
        "customer_name": "Not sent to judge",
    }

    payload, citations = build_quality_judge_input(
        result=result,
        raw_ticket=raw_ticket,
        run_id="a" * 32,
        occurred_at="2026-11-11T12:00:00Z",
        reference={
            "reference_response": "Rotate the token in security settings.",
            "must_mention": ["security settings"],
            "must_not_claim": ["a refund was issued"],
            "intent": "evaluation-only",
        },
    )

    assert set(payload) == {"ticket", "draft", "eligible_cited_sources", "matching_reference"}
    assert set(payload["ticket"]) == {"ticket_id", "channel", "subject", "body"}
    assert "labels" not in payload["ticket"]
    assert "history" not in payload["ticket"]
    assert "customer_name" not in payload["ticket"]
    assert set(payload["matching_reference"]) == {
        "reference_response", "must_mention", "must_not_claim"
    }
    assert "intent" not in payload["matching_reference"]
    assert len(payload["eligible_cited_sources"]) == 1
    assert list(citations) == ["claim-1-citation-1"]


def test_judge_rejects_cross_run_and_ineligible_source_snapshots() -> None:
    result = _generated_result(1)
    with pytest.raises(QualityJudgeError) as wrong_run:
        build_quality_judge_input(
            result={**result, "run_id": "c" * 32},
            raw_ticket={"ticket_id": "DEV-0001", "channel": "email", "subject": "API", "body": "Need a token."},
            run_id="a" * 32,
            occurred_at="2026-11-11T12:00:00Z",
            reference=None,
        )
    assert wrong_run.value.code == "quality_judge_cross_run_record"

    source = _source_snapshot("PASSAGE-1")
    source["approval_status"] = "expired"
    with pytest.raises(QualityJudgeError) as invalid_source:
        build_quality_judge_input(
            result={**result, "sources": [source]},
            raw_ticket={"ticket_id": "DEV-0001", "channel": "email", "subject": "API", "body": "Need a token."},
            run_id="a" * 32,
            occurred_at="2026-11-11T12:00:00Z",
            reference=None,
        )
    assert invalid_source.value.code == "quality_judge_ineligible_source_snapshot"


def test_judge_computes_unrounded_mean_and_validates_every_citation_id() -> None:
    result = _generated_result(1)
    payload, citation_index = build_quality_judge_input(
        result=result,
        raw_ticket={"ticket_id": "DEV-0001", "channel": "email", "subject": "API", "body": "Need a token."},
        run_id="a" * 32,
        occurred_at="2026-11-11T12:00:00Z",
        reference=None,
    )
    del payload
    output = _judge_output(citation_ids=list(citation_index), score=3)
    output["dimensions"]["clarity"]["score"] = 4
    row = _validated_judgment_row(
        result=result,
        run_id="a" * 32,
        input_identity_sha256="c" * 64,
        sample_seed=QUALITY_JUDGE_SAMPLE_SEED,
        sample_index=0,
        reference_available=False,
        citation_index=citation_index,
        model_output=output,
    )

    assert row["overall_quality_score"] == 3.25
    output["citations"] = []
    with pytest.raises(QualityJudgeError) as invalid_citation:
        _validated_judgment_row(
            result=result,
            run_id="a" * 32,
            input_identity_sha256="c" * 64,
            sample_seed=QUALITY_JUDGE_SAMPLE_SEED,
            sample_index=0,
            reference_available=False,
            citation_index=citation_index,
            model_output=output,
        )
    assert invalid_citation.value.code == "quality_judge_invalid_output"


def test_quality_scorecard_applies_targets_and_marks_partial_samples_incomplete() -> None:
    rows = []
    for index in range(20):
        rows.append(
            {
                "status": "evaluated",
                "ticket_id": f"DEV-{index:04d}",
                "overall_quality_score": 4.0,
                "citation_judgments": [{"supported": True}],
                "unsupported_factual_claims": [],
            }
        )
    kwargs = {
        "run_id": "a" * 32,
        "input_identity_sha256": "b" * 64,
        "ground_truth_sha256": "c" * 64,
        "sample_seed": QUALITY_JUDGE_SAMPLE_SEED,
        "available_draft_count": 24,
        "selected_ticket_ids": [row["ticket_id"] for row in rows],
        "reference_available_count": 8,
        "citation_occurrence_count": 20,
        "judgment_rows": rows,
        "source_set": "development",
        "operational_artifact_sha256": {
            "results.jsonl": "d" * 64,
            "decision_log.jsonl": "e" * 64,
        },
    }

    complete = _build_quality_judge_scorecard(**kwargs)
    assert complete["status"] == "complete"
    assert complete["response_quality_proxy"]["status"] == "pass"
    assert complete["citation_support_rate"]["numerator"] == 20
    assert complete["citation_support_rate"]["denominator"] == 20
    assert complete["hallucination_estimate"]["status"] == "pass"

    partial = _build_quality_judge_scorecard(**{**kwargs, "judgment_rows": rows[:-1]})
    assert partial["status"] == "incomplete"
    assert partial["response_quality_proxy"]["status"] == "incomplete"
    assert partial["citation_support_rate"]["status"] == "incomplete"
    assert partial["hallucination_estimate"]["missing_count"] == 1


def test_quality_judge_reconciliation_ignores_nonterminal_stage_events(
    tmp_path: Path,
) -> None:
    decision_log = tmp_path / "decision_log.jsonl"
    rows = [
        {
            "run_id": "a" * 32,
            "input_index": 0,
            "ticket_id": "T-1",
            "occurred_at": "2026-11-12T12:00:00Z",
            "event_type": "classification",
            "terminal": False,
        },
        {
            "run_id": "a" * 32,
            "input_index": 0,
            "ticket_id": "T-1",
            "occurred_at": "2026-11-12T12:00:01Z",
            "event_type": "ticket_completed",
            "terminal": True,
        },
    ]
    decision_log.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    events, cross_run_keys, ambiguous_keys = _decision_events_by_key(
        decision_log, run_id="a" * 32
    )

    assert events == {(0, "T-1"): "2026-11-12T12:00:01Z"}
    assert cross_run_keys == set()
    assert ambiguous_keys == set()


def test_quality_judge_model_setting_is_separate_from_operational_model() -> None:
    settings = Settings.from_env(
        {
            "OPENAI_API_KEY": "judge-secret",
            "OPENAI_LLM_MODEL": "operational-model",
        }
    )

    assert settings.llm_model == "operational-model"
    assert settings.quality_judge_model == "gpt-6-luna"
    assert settings.require_quality_judge_api_key() == "judge-secret"

    judge_settings = Settings.from_quality_judge_env(
        {
            "OPENAI_LLM_MODEL": "operational-model",
            "OPENAI_QUALITY_JUDGE_MODEL": "gpt-6-luna",
            "OPENAI_API_KEY": "judge-secret",
        }
    )
    assert judge_settings.llm_model == "gpt-6-luna"
    assert judge_settings.quality_judge_model == "gpt-6-luna"
