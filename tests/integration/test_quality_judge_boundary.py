"""Verify the post-run judge writes estimates without changing persisted decisions."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from cloudserve_support.evaluation.quality_judge import (
    QUALITY_JUDGE_SAMPLE_SIZE,
    evaluate_quality_run,
    select_quality_judge_sample,
)
from cloudserve_support.knowledge.retrieval import RetrievalResult
from cloudserve_support.processing.run_tickets import run_ticket_batch

TEST_ROUTING_POLICY_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "decisions-routing-policy.json"
)


class _NoEvidenceRetriever:
    """Avoid embedding calls while creating the completed test run."""

    def retrieve(self, ticket: object) -> RetrievalResult:
        del ticket
        return RetrievalResult(
            status="unavailable",
            reason_code="test_no_evidence",
            reason="No source evidence is available in this test.",
            sources=(),
            eligible_candidate_count=0,
            excluded_candidate_counts={},
            minimum_similarity=0.45,
            embedding_model="offline-test",
            source_sha256=None,
            metadata_manifest_sha256=None,
        )


class _SupportedJudge:
    """Return valid four-dimension judgments for every submitted citation."""

    def __init__(self) -> None:
        self.inputs: list[dict[str, Any]] = []

    def judge_quality_response(self, judge_input: dict[str, Any]) -> dict[str, Any]:
        self.inputs.append(judge_input)
        return {
            "dimensions": {
                name: {"score": 4, "rationale": "The response is useful and grounded."}
                for name in (
                    "relevance",
                    "factual_correctness",
                    "completeness_actionability",
                    "clarity",
                )
            },
            "citations": [
                {
                    "citation_id": citation["citation_id"],
                    "supported": True,
                    "rationale": "The cited source passage supports the claim.",
                }
                for citation in judge_input["draft"]["citations"]
            ],
            "unsupported_factual_claims": [],
        }


class _FailingJudge:
    """Raise a safe timeout for each selected response."""

    def judge_quality_response(self, judge_input: dict[str, Any]) -> dict[str, Any]:
        del judge_input
        from cloudserve_support.evaluation.quality_judge import QualityJudgeError

        raise QualityJudgeError("quality_judge_timeout")


def _source_snapshot(passage_id: str, reviewed_at: str) -> dict[str, Any]:
    return {
        "passage_id": passage_id,
        "doc_id": "DOC-JUDGE-001",
        "source_ref": f"kb://DOC-JUDGE-001#{passage_id}",
        "passage_text": "API tokens can be rotated from the security settings page.",
        "approval_status": "approved",
        "approval_status_source": "metadata_manifest",
        "metadata_owner": "capstone",
        "metadata_source": "project_manifest",
        "article_version": "capstone-1.0",
        "content_sha256": "a" * 64,
        "metadata_manifest_sha256": "b" * 64,
        "reviewed_at": reviewed_at,
        "review_interval_days": 365,
        "title": "Rotate API tokens",
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _prepare_run_artifacts(
    tmp_path: Path,
    monkeypatch,
    *,
    ticket_count: int = 20,
) -> tuple[Path, Path, list[dict[str, Any]]]:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    records = [
        {
            "ticket_id": f"JUDGE-{index:04d}",
            "channel": "email",
            "subject": "API token rotation question",
            "body": "Please explain how to rotate the API token. The word labels is ordinary text.",
            "labels": {"intent": "ignored-for-the-judge"},
            "history": {"csat_rating": 1},
        }
        for index in range(ticket_count)
    ]
    input_path = tmp_path / "development-input.json"
    input_path.write_text(json.dumps(records), encoding="utf-8")
    run_id, run_directory, _ = run_ticket_batch(
        input_path=input_path,
        output_root=tmp_path / "outputs",
        database_path=tmp_path / "runtime.sqlite3",
        policy_path=TEST_ROUTING_POLICY_PATH,
        retriever=_NoEvidenceRetriever(),
    )
    results = _read_jsonl(run_directory / "results.jsonl")
    decisions = _read_jsonl(run_directory / "decision_log.jsonl")
    event_by_index = {row["input_index"]: row for row in decisions}
    for result in results:
        event = event_by_index[result["input_index"]]
        event_time = datetime.fromisoformat(event["occurred_at"].replace("Z", "+00:00"))
        source = _source_snapshot(
            f"PASSAGE-{result['input_index']}", event_time.date().isoformat()
        )
        result["route"] = "draft_for_review"
        result["generation"] = {"status": "generated", "provider": "offline", "model_name": "test"}
        result["draft"] = {
            "plain_text": "API tokens can be rotated from the security settings page.",
            "claims": [
                {
                    "text": source["passage_text"],
                    "citations": [
                        {
                            "passage_id": source["passage_id"],
                            "source_ref": source["source_ref"],
                        }
                    ],
                }
            ],
        }
        result["sources"] = [source]
        event["action_taken"] = "draft_for_review"
        event["result"] = result
    _write_jsonl(run_directory / "results.jsonl", results)
    _write_jsonl(run_directory / "decision_log.jsonl", decisions)
    return run_directory, input_path, records


def test_quality_judge_command_artifacts_preserve_results_and_decisions(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_directory, input_path, records = _prepare_run_artifacts(tmp_path, monkeypatch)
    references = [
        {
            "ticket_id": record["ticket_id"],
            "reference_response": "Rotate API tokens from security settings.",
            "must_mention": ["security settings"],
            "must_not_claim": ["the token was already changed"],
            "intent": "not sent to the judge",
        }
        for record in records[:7]
    ]
    ground_truth_path = tmp_path / "references.json"
    ground_truth_path.write_text(json.dumps(references), encoding="utf-8")
    results_path = run_directory / "results.jsonl"
    decisions_path = run_directory / "decision_log.jsonl"
    results_before = results_path.read_bytes()
    decisions_before = decisions_path.read_bytes()
    judge = _SupportedJudge()

    scorecard = evaluate_quality_run(
        run_directory=run_directory,
        input_path=input_path,
        ground_truth_path=ground_truth_path,
        judge_provider=judge,
    )

    assert scorecard["status"] == "complete"
    assert scorecard["selected_count"] == scorecard["evaluated_count"] == QUALITY_JUDGE_SAMPLE_SIZE
    assert len(scorecard["selected_ticket_ids"]) == len(set(scorecard["selected_ticket_ids"])) == 20
    assert scorecard["reference_available_count"] == 7
    assert scorecard["response_quality_proxy"]["status"] == "pass"
    assert scorecard["citation_support_rate"]["numerator"] == 20
    assert scorecard["citation_support_rate"]["denominator"] == 20
    assert scorecard["hallucination_estimate"]["status"] == "pass"
    assert results_path.read_bytes() == results_before
    assert decisions_path.read_bytes() == decisions_before
    assert scorecard["operational_artifact_sha256"] == {
        "results.jsonl": hashlib.sha256(results_before).hexdigest(),
        "decision_log.jsonl": hashlib.sha256(decisions_before).hexdigest(),
    }
    assert len(judge.inputs) == 20
    assert all("labels" not in row["ticket"] and "history" not in row["ticket"] for row in judge.inputs)
    assert all(
        row["matching_reference"] is None or "intent" not in row["matching_reference"]
        for row in judge.inputs
    )
    judgment_rows = _read_jsonl(run_directory / "quality_judgments.jsonl")
    assert len(judgment_rows) == 20
    assert len({row["ticket_id"] for row in judgment_rows}) == 20
    metrics = json.loads((run_directory / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["evaluation"]["criteria"]["llm_judge"] == scorecard
    with sqlite3.connect(tmp_path / "runtime.sqlite3") as connection:
        audit_count = connection.execute("SELECT COUNT(*) FROM audit_events").fetchone()[0]
    assert audit_count == 20


def test_quality_judge_marks_short_samples_incomplete_without_provider_calls(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_directory, input_path, _ = _prepare_run_artifacts(tmp_path, monkeypatch, ticket_count=3)
    judge = _FailingJudge()

    scorecard = evaluate_quality_run(
        run_directory=run_directory,
        input_path=input_path,
        judge_provider=judge,
    )

    assert scorecard["status"] == "incomplete"
    assert scorecard["available_draft_count"] == 3
    assert scorecard["selected_count"] == 3
    assert scorecard["evaluated_count"] == 0
    assert scorecard["failed_count"] == 3
    assert scorecard["missing_count"] == 20
    assert scorecard["failure_counts"]["quality_judge_insufficient_sample"] == 3
    assert scorecard["response_quality_proxy"]["status"] == "incomplete"
    rows = _read_jsonl(run_directory / "quality_judgments.jsonl")
    assert len(rows) == 3
    assert all(row["error"]["code"] == "quality_judge_insufficient_sample" for row in rows)


def test_quality_judge_records_timeout_per_ticket_and_never_fills_missing_scores(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_directory, input_path, _ = _prepare_run_artifacts(tmp_path, monkeypatch)

    scorecard = evaluate_quality_run(
        run_directory=run_directory,
        input_path=input_path,
        judge_provider=_FailingJudge(),
    )

    assert scorecard["status"] == "incomplete"
    assert scorecard["failed_count"] == scorecard["missing_count"] == 20
    assert scorecard["failure_counts"]["quality_judge_timeout"] == 20
    assert scorecard["response_quality_proxy"]["value"] is None
    rows = _read_jsonl(run_directory / "quality_judgments.jsonl")
    assert len(rows) == 20
    assert all(row["status"] == "failed" for row in rows)
    assert all(row["overall_quality_score"] is None for row in rows)
    assert all(row["error"]["code"] == "quality_judge_timeout" for row in rows)


@pytest.mark.parametrize(
    ("event_problem", "expected_code"),
    [
        ("cross_run", "quality_judge_cross_run_audit_record"),
        ("duplicate", "quality_judge_ambiguous_audit_record"),
    ],
)
def test_quality_judge_rejects_cross_run_or_ambiguous_decision_event(
    tmp_path: Path,
    monkeypatch,
    event_problem: str,
    expected_code: str,
) -> None:
    run_directory, input_path, _ = _prepare_run_artifacts(tmp_path, monkeypatch)
    metrics = json.loads((run_directory / "metrics.json").read_text(encoding="utf-8"))
    results = _read_jsonl(run_directory / "results.jsonl")
    selected, _ = select_quality_judge_sample(
        results,
        input_identity_sha256=metrics["evaluation"]["input_sha256"],
    )
    sampled_result = selected[0]
    decisions_path = run_directory / "decision_log.jsonl"
    decisions = _read_jsonl(decisions_path)
    matching_event = next(
        event
        for event in decisions
        if event["input_index"] == sampled_result["input_index"]
        and event["ticket_id"] == sampled_result["ticket_id"]
    )
    if event_problem == "cross_run":
        matching_event["run_id"] = "f" * 32
    else:
        decisions.append(dict(matching_event))
    _write_jsonl(decisions_path, decisions)
    judge = _SupportedJudge()

    scorecard = evaluate_quality_run(
        run_directory=run_directory,
        input_path=input_path,
        judge_provider=judge,
    )

    assert scorecard["status"] == "incomplete"
    assert scorecard["evaluated_count"] == 19
    assert scorecard["failed_count"] == scorecard["missing_count"] == 1
    assert scorecard["failure_counts"][expected_code] == 1
    rows = _read_jsonl(run_directory / "quality_judgments.jsonl")
    failed_row = next(row for row in rows if row["ticket_id"] == sampled_result["ticket_id"])
    assert failed_row["error"]["code"] == expected_code
    assert len(judge.inputs) == 19
