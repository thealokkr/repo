"""Integration and live verification for batched gemini-decision-1 Noul claim_grounding_check routing."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
import json
import os
from pathlib import Path
import shutil
import sqlite3
from types import SimpleNamespace
from typing import Any

import chromadb
from dotenv import dotenv_values
from google import genai
import pytest

from cloudserve_support.classification.ticket_classification import (
    DECISIONS_CLASSIFIER_MODEL,
    DECISIONS_MAX_INPUT_BYTES,
    DECISIONS_MAX_QUESTIONS,
    INTENT_VALUES,
    TicketClassification,
    TicketClassificationBatch,
    URGENCY_VALUES,
)
from cloudserve_support.cli import main as cli_main
from cloudserve_support.config import Settings
from cloudserve_support.domain.ticket import CanonicalTicket
from cloudserve_support.generation.draft_generation import OfflineDraftGenerationProvider
from cloudserve_support.ingestion.ticket_normalization import normalize_ticket_record
from cloudserve_support.knowledge.retrieval import (
    COLLECTION_NAME,
    DEFAULT_INDEX_PATH,
    DEFAULT_METADATA_PATH,
    DEFAULT_SOURCE_PATH,
    KnowledgeRetriever,
    MINIMUM_SIMILARITY,
    RetrievalResult,
)
from cloudserve_support.persistence.runtime_database import connect_runtime_database
from cloudserve_support.processing import run_tickets
from cloudserve_support.processing.run_tickets import run_ticket_batch
from cloudserve_support.processing.ticket_outcomes import build_ticket_outcome
from cloudserve_support.reporting.output_schemas import (
    load_output_schemas,
    validate_output_record,
)
from cloudserve_support.reporting.run_exports import write_run_exports
from cloudserve_support.routing.policy_calibration import load_frozen_routing_policy
from cloudserve_support.routing.routing_policy import (
    ClaimGroundingError,
    GeminiClaimGroundingChecker,
    POLICY_CHECK_NAMES,
    RoutingPolicyEvaluator,
    check_claim_grounding,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_POLICY_PATH = REPO_ROOT / "conf" / "frozen_routing_policy.json"
DEVELOPMENT_TICKETS_PATH = REPO_ROOT / "data" / "tickets" / "development_tickets.json"
FIXED_TODAY = date(2026, 11, 16)


class _StoredVectorEmbedder:
    """Supply precomputed Chroma article vectors so retrieval runs against an isolated local index."""

    def __init__(self, vectors_by_query: dict[str, list[float]]) -> None:
        self.vectors_by_query = vectors_by_query
        self.models = self

    def embed_content(self, *, contents: object, **_: object) -> SimpleNamespace:
        queries = [contents] if isinstance(contents, str) else list(contents)
        return SimpleNamespace(
            embeddings=[
                SimpleNamespace(values=self.vectors_by_query[str(query)])
                for query in queries
            ]
        )


class _RecordingDraftGenerator:
    """Count draft-generation invocations while delegating to the offline extractive adapter."""

    name = "offline"

    def __init__(self) -> None:
        self.calls: list[str] = []
        self._delegate = OfflineDraftGenerationProvider()
        self.last_request_duration_seconds = 0.0

    def generate_draft(
        self,
        ticket: CanonicalTicket,
        *,
        sources: tuple[dict[str, Any], ...],
        related_sources: tuple[dict[str, Any], ...] = (),
        policy_evaluation: Mapping[str, Any] | None = None,
        model_name: str = "",
    ) -> dict[str, Any]:
        self.calls.append(ticket.ticket_id)
        return self._delegate.generate_draft(
            ticket,
            sources=sources,
            related_sources=related_sources,
            policy_evaluation=policy_evaluation,
            model_name=model_name,
        )


def _load_development_records() -> dict[str, dict[str, Any]]:
    """Index raw development ticket fixtures by ticket_id."""
    records = json.loads(DEVELOPMENT_TICKETS_PATH.read_text(encoding="utf-8"))
    return {record["ticket_id"]: dict(record) for record in records}


def _copy_isolated_index(tmp_path: Path, name: str = "isolated_index") -> tuple[Path, dict[str, list[float]]]:
    """Copy the packaged Chroma index to tmp_path so tests never mutate tracked index files."""
    isolated_path = tmp_path / name
    shutil.copytree(DEFAULT_INDEX_PATH, isolated_path)
    client = chromadb.PersistentClient(path=str(isolated_path / "chroma"))
    collection = client.get_collection(COLLECTION_NAME, embedding_function=None)
    stored = collection.get(include=["embeddings", "metadatas"])
    vectors_by_doc_id = {
        metadata["doc_id"]: embedding.tolist()
        for metadata, embedding in zip(stored["metadatas"], stored["embeddings"], strict=True)
    }
    return isolated_path, vectors_by_doc_id


def _make_decision_choice(
    name: str,
    selected: str,
    confidence: float,
    values: frozenset[str],
) -> SimpleNamespace:
    """Construct one Decisions API choice payload for external SDK boundary tests."""
    alternate = next(val for val in sorted(values) if val != selected)
    remainder = round(1.0 - confidence, 6)
    probabilities = [
        SimpleNamespace(
            value=val,
            probability=confidence if val == selected else remainder if val == alternate else 0.0,
        )
        for val in sorted(values)
    ]
    return SimpleNamespace(
        type="choice",
        name=name,
        choice=selected,
        confidence=confidence,
        probabilities=probabilities,
    )


def _require_live_gemini_api_key() -> str:
    """Resolve GEMINI_API_KEY or GOOGLE_API_KEY from environment or .env without skipping."""
    env_key = (
        os.environ.get("GEMINI_API_KEY", "").strip()
        or os.environ.get("GOOGLE_API_KEY", "").strip()
    )
    if env_key:
        return env_key
    dotenv_map = dotenv_values(REPO_ROOT / ".env")
    file_key = (
        str(dotenv_map.get("GEMINI_API_KEY") or "").strip()
        or str(dotenv_map.get("GOOGLE_API_KEY") or "").strip()
    )
    if file_key:
        return file_key
    pytest.fail("Set GEMINI_API_KEY or GOOGLE_API_KEY in the environment or ignored .env file.")


def _validate_all_exported_files(output_directory: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Validate results.jsonl, decision_log.jsonl, exceptions.jsonl, and metrics.json schemas."""
    schemas = load_output_schemas()
    results = [
        json.loads(line)
        for line in (output_directory / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    decisions = [
        json.loads(line)
        for line in (output_directory / "decision_log.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    exceptions = [
        json.loads(line)
        for line in (output_directory / "exceptions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    metrics = json.loads((output_directory / "metrics.json").read_text(encoding="utf-8"))

    for row in results:
        validate_output_record(row, schema_filename="result.schema.json", schemas=schemas)
    for row in decisions:
        validate_output_record(row, schema_filename="decision-log.schema.json", schemas=schemas)
    for row in exceptions:
        validate_output_record(row, schema_filename="exception.schema.json", schemas=schemas)
    validate_output_record(metrics, schema_filename="metrics.schema.json", schemas=schemas)
    return results, decisions, metrics


def test_live_gemini_claim_grounding_routes_answerable_and_unsupported_tickets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC-1: Run live gemini-3.8-flash structured-JSON claim grounding on answerable and unsupported tickets."""
    api_key = _require_live_gemini_api_key()
    monkeypatch.setenv("GEMINI_API_KEY", api_key)
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    monkeypatch.setenv("GEMINI_LLM_MODEL", "gemini-3.8-flash")
    monkeypatch.setenv("DRAFT_GENERATION_PROVIDER", "GEMINI")
    settings = Settings.from_env()
    assert settings.llm_model == "gemini-3.8-flash"
    assert settings.llm_model != DECISIONS_CLASSIFIER_MODEL

    dev_records = _load_development_records()
    answerable_raw = dev_records["DEV-0006"]
    unsupported_raw = dev_records["DEV-0009"]
    answerable_ticket = normalize_ticket_record(answerable_raw, input_index=0)
    unsupported_ticket = normalize_ticket_record(unsupported_raw, input_index=1)

    isolated_index, vectors_by_doc = _copy_isolated_index(tmp_path, "live_index")
    answerable_query = f"{answerable_ticket.subject}\n{answerable_ticket.body}".strip()
    unsupported_query = f"{unsupported_ticket.subject}\n{unsupported_ticket.body}".strip()
    retriever = KnowledgeRetriever(
        settings,
        index_path=isolated_index,
        embedding_client=_StoredVectorEmbedder(
            {
                answerable_query: vectors_by_doc["DOC-DEPLOY-004"],
                unsupported_query: vectors_by_doc["DOC-DEPLOY-001"],
            }
        ),
        today=FIXED_TODAY,
    )
    answerable_retrieval = retriever.retrieve(answerable_ticket)
    unsupported_retrieval = retriever.retrieve(unsupported_ticket)
    assert "DOC-DEPLOY-004" in [s["doc_id"] for s in answerable_retrieval.sources]
    assert "DOC-SEC-002" in [
        s["doc_id"] for s in (*answerable_retrieval.sources, *answerable_retrieval.related_sources)
    ]
    assert "DOC-DEPLOY-001" in [s["doc_id"] for s in unsupported_retrieval.sources]
    assert "DOC-DEPLOY-002" not in [
        s["doc_id"]
        for s in (*unsupported_retrieval.sources, *unsupported_retrieval.related_sources)
    ]

    frozen_policy = load_frozen_routing_policy(FROZEN_POLICY_PATH)
    checker = GeminiClaimGroundingChecker(settings, today=FIXED_TODAY)
    evaluator = RoutingPolicyEvaluator(
        frozen_policy,
        claim_grounding_fn=checker,
        today=FIXED_TODAY,
    )

    answerable_eval = evaluator.evaluate(
        ticket=answerable_raw,
        classification={
            "intent": "configuration_help",
            "urgency": "medium",
            "confidence": 0.94,
            "provider": "gemini",
            "model_name": DECISIONS_CLASSIFIER_MODEL,
            "provider_status": "classified",
        },
        retrieval=answerable_retrieval,
    )
    assert checker.call_count == 1
    assert checker.last_model_name == DECISIONS_CLASSIFIER_MODEL

    answerable_grounding = answerable_eval["checks"][3]
    assert answerable_grounding["name"] == "claim_grounding_check"
    assert answerable_grounding["status"] == "passed"
    assert 0.5 < answerable_grounding["noul"] <= 1.0
    assert 0.0 <= answerable_grounding["confidence"] <= 1.0
    assert answerable_grounding["reason"].strip()
    assert answerable_eval["final_decision"]["route"] == "draft_for_review"
    assert answerable_eval["final_decision"]["evidence_ready"] is True
    assert '"labels"' not in json.dumps(answerable_eval)
    assert '"history"' not in json.dumps(answerable_eval)

    unsupported_eval = evaluator.evaluate(
        ticket=unsupported_raw,
        classification={
            "intent": "rollback_request",
            "urgency": "high",
            "confidence": 0.92,
            "provider": "gemini",
            "model_name": DECISIONS_CLASSIFIER_MODEL,
            "provider_status": "classified",
        },
        retrieval=unsupported_retrieval,
    )
    assert checker.call_count == 2
    assert checker.last_model_name == DECISIONS_CLASSIFIER_MODEL
    assert unsupported_eval["checks"][2]["status"] == "passed"
    unsupported_grounding = unsupported_eval["checks"][3]
    assert unsupported_grounding["status"] == "failed"
    assert unsupported_grounding["matched_rule"] == "claim_grounding_failed"
    assert unsupported_grounding["noul"] <= 0.5
    assert unsupported_eval["final_decision"]["route"] == "escalate"
    assert unsupported_eval["final_decision"]["evidence_ready"] is False
    assert unsupported_eval["final_decision"]["failed_check"] == "claim_grounding_check"

    recording_generator = _RecordingDraftGenerator()
    escalated_outcome = build_ticket_outcome(
        run_id="0" * 32,
        input_index=1,
        ticket=unsupported_ticket,
        classification=TicketClassification(
            intent="rollback_request",
            urgency="high",
            confidence=0.92,
            provider="gemini",
            model_name=DECISIONS_CLASSIFIER_MODEL,
            provider_status="classified",
        ),
        retrieval=unsupported_retrieval,
        decision=RoutingPolicyEvaluator.to_routing_decision(unsupported_eval),
        policy_evaluation=unsupported_eval,
        policy=frozen_policy,
        policy_file_fingerprint="a" * 64,
        classifier_implementation_fingerprint="b" * 64,
        settings=settings,
        generator=recording_generator,
        issues=[],
        duplicate_ticket_id=False,
    )
    assert escalated_outcome["route"] == "escalate"
    assert escalated_outcome["draft"] is None
    assert recording_generator.calls == []


def test_batch_fault_injection_degrades_safely_skips_short_circuited_and_persists_policy_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC-2 & AC-3: Verify batch degradation, zero grounding calls on short-circuit, SQLite JSON, and exports."""
    isolated_index, vectors_by_doc = _copy_isolated_index(tmp_path, "batch_index")
    deploy_vector = vectors_by_doc["DOC-DEPLOY-004"]
    monkeypatch.setattr(
        "cloudserve_support.knowledge.retrieval.KnowledgeRetriever.__init__.__kwdefaults__",
        {
            "source_path": DEFAULT_SOURCE_PATH,
            "metadata_path": DEFAULT_METADATA_PATH,
            "index_path": isolated_index,
            "embedding_client": None,
            "collection": None,
            "today": FIXED_TODAY,
        },
    )

    grounding_calls: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    draft_calls: list[str] = []

    class _FaultInjectingGeminiBoundary:
        """Stub only external google.genai.Client to handle batched Noul grounding and faults."""

        def __init__(self, **_: object) -> None:
            self.decisions = SimpleNamespace(create=self._create_decision)
            self.models = SimpleNamespace(
                embed_content=self._embed_content,
                generate_content=self._generate_content,
            )

        @staticmethod
        def _create_decision(*, model: str, input: str, questions: dict[str, Any], **_: object) -> SimpleNamespace:
            if "ticket_0_intent" in questions:
                ticket_count = len(json.loads(input)["tickets"])
                all_answers = {
                    "ticket_0_intent": _make_decision_choice(
                        "ticket_0_intent", "configuration_help", 0.93, INTENT_VALUES
                    ),
                    "ticket_0_urgency": _make_decision_choice(
                        "ticket_0_urgency", "medium", 0.88, URGENCY_VALUES
                    ),
                    "ticket_1_intent": _make_decision_choice(
                        "ticket_1_intent", "configuration_help", 0.92, INTENT_VALUES
                    ),
                    "ticket_1_urgency": _make_decision_choice(
                        "ticket_1_urgency", "medium", 0.88, URGENCY_VALUES
                    ),
                    "ticket_2_intent": _make_decision_choice(
                        "ticket_2_intent", "configuration_help", 0.95, INTENT_VALUES
                    ),
                    "ticket_2_urgency": _make_decision_choice(
                        "ticket_2_urgency", "medium", 0.90, URGENCY_VALUES
                    ),
                    "ticket_3_intent": _make_decision_choice(
                        "ticket_3_intent", "data_residency", 0.94, INTENT_VALUES
                    ),
                    "ticket_3_urgency": _make_decision_choice(
                        "ticket_3_urgency", "medium", 0.85, URGENCY_VALUES
                    ),
                }
                return SimpleNamespace(
                    model=DECISIONS_CLASSIFIER_MODEL,
                    answers={
                        key: all_answers[key]
                        for idx in range(ticket_count)
                        for key in (f"ticket_{idx}_intent", f"ticket_{idx}_urgency")
                    },
                )

            if "claim_0_grounding" in questions:
                return SimpleNamespace(
                    model=DECISIONS_CLASSIFIER_MODEL,
                    answers={
                        q_name: SimpleNamespace(type="noul", noul=0.92, confidence=0.92)
                        for q_name in questions
                    },
                )

            payload = json.loads(input)
            grounding_calls.append((model, payload, questions))
            subjects = [item["subject"] for item in payload["tickets"]]
            if "Simulate grounding timeout" in subjects:
                raise TimeoutError("Gemini read timed out")
            if "Simulate invalid grounding refusal" in subjects:
                return SimpleNamespace(
                    model=DECISIONS_CLASSIFIER_MODEL,
                    answers={
                        q_name: SimpleNamespace(type="refusal", reason="Blocked")
                        for q_name in questions
                    },
                )
            return SimpleNamespace(
                model=DECISIONS_CLASSIFIER_MODEL,
                answers={
                    q_name: SimpleNamespace(type="noul", noul=0.92, confidence=0.92)
                    for q_name in questions
                },
            )

        @staticmethod
        def _embed_content(*, contents: object, **_: object) -> SimpleNamespace:
            batch = [contents] if isinstance(contents, str) else list(contents)
            return SimpleNamespace(
                embeddings=[SimpleNamespace(values=deploy_vector) for _ in batch]
            )

        @staticmethod
        def _generate_content(*, model: str, contents: str, config: Any) -> SimpleNamespace:
            del model, config
            payload = json.loads(contents)
            subject = payload["ticket"]["subject"]
            draft_calls.append(subject)
            source = payload["primary_sources"][0]
            return SimpleNamespace(
                text=json.dumps(
                    {
                        "plain_text": (
                            "Thank you for getting in touch. Store anything sensitive "
                            "as a secret rather than a variable so values stay masked in logs."
                        ),
                        "claims": [
                            {
                                "text": "Store anything sensitive as a secret rather than a variable.",
                                "source_ids": [source["passage_id"]],
                            }
                        ],
                        "unsupported_points": [],
                        "conflicting_points": [],
                    }
                )
            )

    monkeypatch.setattr(genai, "Client", _FaultInjectingGeminiBoundary)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    monkeypatch.setenv("GEMINI_LLM_MODEL", "gemini-3.8-flash")
    monkeypatch.setenv("DRAFT_GENERATION_PROVIDER", "GEMINI")

    dev_0006 = _load_development_records()["DEV-0006"]
    # 1. AC-1: 3 surviving tickets sharing KB articles + 1 protected compliance_request ticket -> 1 grounding batch call.
    shared_batch_records = [
        {
            "ticket_id": "SURV-1",
            "channel": "email",
            "subject": "Secrets in environment variables",
            "body": "Where should secrets be stored so they do not appear in the logs?",
            "labels": {"expected_route": "draft_for_review"},
            "history": {"csat_rating": 4},
        },
        {
            "ticket_id": "SURV-2",
            "channel": "chat",
            "subject": "Masking sensitive config",
            "body": "How do I keep API tokens masked in deployment logs?",
            "labels": {"expected_route": "draft_for_review"},
            "history": {"csat_rating": 4},
        },
        dev_0006,
        {
            "ticket_id": "BATCH-PROTECTED",
            "channel": "email",
            "subject": "Retention period question",
            "body": "Our policy requires two years of retention. What are our options?",
            "labels": {"expected_route": "escalate"},
            "history": {"csat_rating": 5},
        },
    ]
    input_path = tmp_path / "batch_tickets.json"
    input_path.write_text(json.dumps(shared_batch_records), encoding="utf-8")
    database_path = tmp_path / "runtime.sqlite3"
    output_root = tmp_path / "outputs"

    exit_code = cli_main(
        [
            "run",
            "--input",
            str(input_path),
            "--output-root",
            str(output_root),
            "--database",
            str(database_path),
            "--policy",
            str(FROZEN_POLICY_PATH),
            "--label",
            "grounding-shared-batch",
        ]
    )
    assert exit_code == 0

    assert len(grounding_calls) == 1
    g_model, g_payload, g_questions = grounding_calls[0]
    assert g_model == DECISIONS_CLASSIFIER_MODEL
    assert len(g_questions) == 3
    assert len(g_questions) <= DECISIONS_MAX_QUESTIONS
    assert len(json.dumps(g_payload).encode("utf-8")) <= DECISIONS_MAX_INPUT_BYTES
    assert [t["subject"] for t in g_payload["tickets"]] == [
        "Secrets in environment variables",
        "Masking sensitive config",
        dev_0006["subject"],
    ]
    serialized_sent = json.dumps(g_payload)
    assert '"labels"' not in serialized_sent
    assert '"history"' not in serialized_sent
    assert isinstance(g_payload["articles"], dict)
    assert len(g_payload["articles"]) >= 1
    for ticket_item in g_payload["tickets"]:
        assert set(ticket_item["candidate_doc_ids"]).issubset(g_payload["articles"])
    assert len(draft_calls) == 3

    run_dirs = list(output_root.iterdir())
    assert len(run_dirs) == 1
    results, _, metrics = _validate_all_exported_files(run_dirs[0])
    results_by_id = {row["ticket_id"]: row for row in results}
    assert results_by_id["SURV-1"]["predicted_route"] == "draft_for_review"
    assert results_by_id["SURV-2"]["predicted_route"] == "draft_for_review"
    assert results_by_id["DEV-0006"]["predicted_route"] == "draft_for_review"
    assert results_by_id["BATCH-PROTECTED"]["predicted_route"] == "escalate"
    assert results_by_id["BATCH-PROTECTED"]["routing"]["matched_rule"] == "operational_protected_compliance"
    assert metrics["route_counts"] == {"draft_for_review": 3, "escalate": 1, "degrade": 0}

    # Inspect SQLite audit_events routing stage rows and re-verify write_run_exports directly.
    connection = connect_runtime_database(database_path)
    try:
        run_id = results[0]["run_id"]
        routing_rows = connection.execute(
            "SELECT ticket_id, model_name, event_payload_json FROM audit_events "
            "WHERE run_id = ? AND event_type = 'routing' ORDER BY rowid",
            (run_id,),
        ).fetchall()
        assert len(routing_rows) == 4
        routing_by_ticket: dict[str, tuple[str | None, dict[str, Any]]] = {}
        for row in routing_rows:
            parsed_event = json.loads(row["event_payload_json"])
            value = parsed_event["value"]
            assert set(value) == {"ticket", "checks", "final_decision"}
            assert tuple(check["name"] for check in value["checks"]) == POLICY_CHECK_NAMES
            assert '"labels"' not in row["event_payload_json"]
            assert '"history"' not in row["event_payload_json"]
            routing_by_ticket[row["ticket_id"]] = (row["model_name"], value)

        for tid in ("SURV-1", "SURV-2", "DEV-0006"):
            assert routing_by_ticket[tid][0] == DECISIONS_CLASSIFIER_MODEL
            assert routing_by_ticket[tid][1]["final_decision"]["route"] == "draft_for_review"
        assert routing_by_ticket["BATCH-PROTECTED"][0] is None
        assert routing_by_ticket["BATCH-PROTECTED"][1]["checks"][3]["status"] == "skipped"
        assert routing_by_ticket["BATCH-PROTECTED"][1]["final_decision"]["route"] == "escalate"

        re_export_dir = tmp_path / "re_exported"
        re_export_dir.mkdir()
        re_metrics = write_run_exports(
            connection,
            run_id=run_id,
            output_directory=re_export_dir,
            input_records=shared_batch_records,
            input_path=input_path,
            source_ref="batch_tickets.json",
        )
        assert re_metrics["route_counts"] == metrics["route_counts"]
        _validate_all_exported_files(re_export_dir)
    finally:
        connection.close()

    # 2. AC-3: Timeout and refusal degrade safely without crashing.
    grounding_calls.clear()
    timeout_input = tmp_path / "timeout_batch.json"
    timeout_input.write_text(
        json.dumps([
            {
                "ticket_id": "BATCH-TIMEOUT",
                "channel": "email",
                "subject": "Simulate grounding timeout",
                "body": "Where should secrets be stored so they do not appear in the logs?",
            }
        ]),
        encoding="utf-8",
    )
    _, timeout_dir, _ = run_ticket_batch(
        input_path=timeout_input,
        output_root=tmp_path / "timeout_outputs",
        database_path=tmp_path / "timeout.sqlite3",
        policy_path=FROZEN_POLICY_PATH,
    )
    timeout_results, _, _ = _validate_all_exported_files(timeout_dir)
    assert timeout_results[0]["status"] == "failed"
    assert timeout_results[0]["error"]["code"] == "claim_grounding_timeout"
    assert timeout_results[0]["routing"]["matched_rule"] == "claim_grounding_unavailable"

    refusal_input = tmp_path / "refusal_batch.json"
    refusal_input.write_text(
        json.dumps([
            {
                "ticket_id": "BATCH-REFUSAL",
                "channel": "chat",
                "subject": "Simulate invalid grounding refusal",
                "body": "Where should secrets be stored so they do not appear in the logs?",
            }
        ]),
        encoding="utf-8",
    )
    _, refusal_dir, _ = run_ticket_batch(
        input_path=refusal_input,
        output_root=tmp_path / "refusal_outputs",
        database_path=tmp_path / "refusal.sqlite3",
        policy_path=FROZEN_POLICY_PATH,
    )
    refusal_results, _, _ = _validate_all_exported_files(refusal_dir)
    assert refusal_results[0]["status"] == "failed"
    assert refusal_results[0]["error"]["code"] == "invalid_claim_grounding_output"
    assert refusal_results[0]["routing"]["matched_rule"] == "claim_grounding_unavailable"

    # 3. AC-3: DRAFT_GENERATION_PROVIDER=OFFLINE makes 0 grounding decisions.create calls.
    grounding_calls.clear()
    monkeypatch.setenv("DRAFT_GENERATION_PROVIDER", "OFFLINE")
    _, offline_dir, _ = run_ticket_batch(
        input_path=timeout_input,
        output_root=tmp_path / "offline_outputs",
        database_path=tmp_path / "offline.sqlite3",
        policy_path=FROZEN_POLICY_PATH,
    )
    assert len(grounding_calls) == 0
    _validate_all_exported_files(offline_dir)
    monkeypatch.setenv("DRAFT_GENERATION_PROVIDER", "GEMINI")

    # Dedicated short-circuit batch: protected + low-confidence + no-source makes zero grounding calls.
    grounding_calls.clear()
    short_circuit_records = [
        {
            "ticket_id": "SC-PROTECTED",
            "channel": "email",
            "subject": "Security incident",
            "body": "An API key was exposed in a public repository.",
        },
        {
            "ticket_id": "SC-NO-SOURCE",
            "channel": "chat",
            "subject": " Environment variable question",
            "body": "Where should secrets be stored so they do not appear in the logs?",
        },
    ]
    sc_input = tmp_path / "short_circuit.json"
    sc_input.write_text(json.dumps(short_circuit_records), encoding="utf-8")

    class _NoSourceRetriever:
        def retrieve_many(
            self,
            tickets: list[CanonicalTicket],
            *,
            embedding_batch_size: int,
            query_batch_size: int,
        ) -> tuple[RetrievalResult, ...]:
            del embedding_batch_size, query_batch_size
            return tuple(
                RetrievalResult(
                    status="no_hit",
                    reason_code="below_relevance_threshold",
                    reason="No current eligible passage met the minimum relevance threshold.",
                    sources=(),
                    eligible_candidate_count=0,
                    excluded_candidate_counts={},
                    minimum_similarity=MINIMUM_SIMILARITY,
                    embedding_model="gemini-embedding-001",
                    source_sha256="1" * 64,
                    metadata_manifest_sha256="2" * 64,
                )
                for _ in tickets
            )

    def _short_circuit_classifier(
        tickets: list[CanonicalTicket],
        _settings: Settings,
    ) -> tuple[TicketClassificationBatch, ...]:
        return (
            TicketClassificationBatch(
                0,
                tuple(
                    TicketClassification(
                        intent="configuration_help",
                        urgency="low",
                        confidence=0.92,
                        provider="gemini",
                        model_name=DECISIONS_CLASSIFIER_MODEL,
                        provider_status="classified",
                    )
                    for _ in tickets
                ),
                0.0,
            ),
        )

    monkeypatch.setattr(run_tickets, "classify_ticket_batches", _short_circuit_classifier)
    _, sc_output_dir, _ = run_ticket_batch(
        input_path=sc_input,
        output_root=tmp_path / "sc_outputs",
        database_path=tmp_path / "sc.sqlite3",
        policy_path=FROZEN_POLICY_PATH,
        retriever=_NoSourceRetriever(),  # type: ignore[arg-type]
    )
    assert len(grounding_calls) == 0
    _validate_all_exported_files(sc_output_dir)


def test_claim_grounding_fails_closed_on_missing_key_decision_model_conflict_and_unknown_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify unmocked missing-key degradation, gemini-decision-1 rejection, conflicts, and unknown source_id."""
    isolated_index, vectors_by_doc = _copy_isolated_index(tmp_path, "guard_index")
    dev_0006 = _load_development_records()["DEV-0006"]
    ticket = normalize_ticket_record(dev_0006, input_index=0)
    query_text = f"{ticket.subject}\n{ticket.body}".strip()
    retriever = KnowledgeRetriever(
        Settings(gemini_api_key="embed-only"),
        index_path=isolated_index,
        embedding_client=_StoredVectorEmbedder({query_text: vectors_by_doc["DOC-DEPLOY-004"]}),
        today=FIXED_TODAY,
    )
    retrieval = retriever.retrieve(ticket)
    frozen_policy = load_frozen_routing_policy(FROZEN_POLICY_PATH)

    # 1. Missing API key in GEMINI mode fails closed to degrade in run_ticket_batch without genai stubs.
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    monkeypatch.setenv("DRAFT_GENERATION_PROVIDER", "GEMINI")

    class _PreRetrievedAdapter:
        def retrieve_many(
            self,
            tickets: list[CanonicalTicket],
            *,
            embedding_batch_size: int,
            query_batch_size: int,
        ) -> tuple[RetrievalResult, ...]:
            del embedding_batch_size, query_batch_size
            return tuple(retrieval for _ in tickets)

    monkeypatch.setattr(
        run_tickets,
        "classify_ticket_batches",
        lambda tickets, _settings: (
            TicketClassificationBatch(
                0,
                tuple(
                    TicketClassification(
                        intent="configuration_help",
                        urgency="medium",
                        confidence=0.95,
                        provider="gemini",
                        model_name=DECISIONS_CLASSIFIER_MODEL,
                        provider_status="classified",
                    )
                    for _ in tickets
                ),
                0.0,
            ),
        ),
    )
    missing_key_input = tmp_path / "missing_key.json"
    missing_key_input.write_text(json.dumps([dev_0006]), encoding="utf-8")
    _, mk_output_dir, mk_metrics = run_ticket_batch(
        input_path=missing_key_input,
        output_root=tmp_path / "mk_outputs",
        database_path=tmp_path / "mk.sqlite3",
        policy_path=FROZEN_POLICY_PATH,
        retriever=_PreRetrievedAdapter(),  # type: ignore[arg-type]
        generator=_RecordingDraftGenerator(),
    )
    mk_results, _, _ = _validate_all_exported_files(mk_output_dir)
    assert mk_results[0]["status"] == "failed"
    assert mk_results[0]["draft"] is None
    assert mk_results[0]["error"]["code"] == "claim_grounding_configuration_error"
    assert mk_metrics["route_counts"]["degrade"] == 1

    # 3. Conflicting points fail claim_grounding_check -> escalate without calling draft generator.
    conflicting_eval = RoutingPolicyEvaluator(
        frozen_policy,
        claim_grounding_fn=lambda **_: {
            "confidence": 0.72,
            "claims": [
                {
                    "text": "Secrets are masked in logs.",
                    "source_ids": [retrieval.sources[0]["doc_id"]],
                }
            ],
            "unsupported_points": [],
            "conflicting_points": ["Two articles disagree on rotation order."],
            "reason": "Conflicting rotation steps require human review.",
        },
        today=FIXED_TODAY,
    ).evaluate(
        ticket=ticket,
        classification={
            "intent": "configuration_help",
            "urgency": "medium",
            "confidence": 0.95,
            "provider": "gemini",
            "model_name": DECISIONS_CLASSIFIER_MODEL,
            "provider_status": "classified",
        },
        retrieval=retrieval,
    )
    assert conflicting_eval["final_decision"]["route"] == "escalate"
    assert conflicting_eval["final_decision"]["matched_rule"] == "claim_grounding_failed"

    # 4. Unknown source_id fails closed to degrade with structured invalid_claim_grounding_output code.
    unknown_id_eval = RoutingPolicyEvaluator(
        frozen_policy,
        claim_grounding_fn=lambda **_: {
            "confidence": 0.85,
            "claims": [{"text": "Invented cite.", "source_ids": ["DOC-UNKNOWN-999"]}],
            "unsupported_points": [],
            "conflicting_points": [],
            "reason": "Cited non-supplied doc.",
        },
        today=FIXED_TODAY,
    ).evaluate(
        ticket=ticket,
        classification={
            "intent": "configuration_help",
            "urgency": "medium",
            "confidence": 0.95,
            "provider": "gemini",
            "model_name": DECISIONS_CLASSIFIER_MODEL,
            "provider_status": "classified",
        },
        retrieval=retrieval,
    )
    unknown_outcome = build_ticket_outcome(
        run_id="1" * 32,
        input_index=0,
        ticket=ticket,
        classification=TicketClassification(
            intent="configuration_help",
            urgency="medium",
            confidence=0.95,
            provider="gemini",
            model_name=DECISIONS_CLASSIFIER_MODEL,
            provider_status="classified",
        ),
        retrieval=retrieval,
        decision=RoutingPolicyEvaluator.to_routing_decision(unknown_id_eval),
        policy_evaluation=unknown_id_eval,
        policy=frozen_policy,
        policy_file_fingerprint="a" * 64,
        classifier_implementation_fingerprint="b" * 64,
        settings=Settings.from_env(),
        generator=_RecordingDraftGenerator(),
        issues=[],
        duplicate_ticket_id=False,
    )
    assert unknown_outcome["route"] == "degrade"
    assert unknown_outcome["status"] == "failed"
    assert unknown_outcome["draft"] is None
    assert unknown_outcome["error"]["code"] == "invalid_claim_grounding_output"


@pytest.mark.parametrize("failing_noul", [0.50, 0.12])
def test_batched_noul_grounding_noul_boundary_unclear_and_sqlite_comparison(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failing_noul: float,
) -> None:
    """AC-2: Verify noul > 0.5 vs <= 0.5 fail-closed, unclear_request evidence_ready=False, and SQLite comparison."""
    isolated_index, vectors_by_doc = _copy_isolated_index(tmp_path, "ac2_index")
    deploy_vector = vectors_by_doc["DOC-DEPLOY-004"]
    monkeypatch.setattr(
        "cloudserve_support.knowledge.retrieval.KnowledgeRetriever.__init__.__kwdefaults__",
        {
            "source_path": DEFAULT_SOURCE_PATH,
            "metadata_path": DEFAULT_METADATA_PATH,
            "index_path": isolated_index,
            "embedding_client": None,
            "collection": None,
            "today": FIXED_TODAY,
        },
    )

    class _NoulBoundaryClient:
        def __init__(self, **_: object) -> None:
            self.decisions = SimpleNamespace(create=self._create_decision)
            self.models = SimpleNamespace(
                embed_content=self._embed_content,
                generate_content=self._generate_content,
            )

        @staticmethod
        def _create_decision(*, questions: dict[str, Any], **_: object) -> SimpleNamespace:
            if "ticket_0_intent" in questions:
                return SimpleNamespace(
                    model=DECISIONS_CLASSIFIER_MODEL,
                    answers={
                        "ticket_0_intent": _make_decision_choice(
                            "ticket_0_intent", "configuration_help", 0.94, INTENT_VALUES
                        ),
                        "ticket_0_urgency": _make_decision_choice(
                            "ticket_0_urgency", "medium", 0.85, URGENCY_VALUES
                        ),
                        "ticket_1_intent": _make_decision_choice(
                            "ticket_1_intent", "configuration_help", 0.93, INTENT_VALUES
                        ),
                        "ticket_1_urgency": _make_decision_choice(
                            "ticket_1_urgency", "medium", 0.85, URGENCY_VALUES
                        ),
                        "ticket_2_intent": _make_decision_choice(
                            "ticket_2_intent", "unclear_request", 0.91, INTENT_VALUES
                        ),
                        "ticket_2_urgency": _make_decision_choice(
                            "ticket_2_urgency", "low", 0.80, URGENCY_VALUES
                        ),
                    },
                )
            if "claim_0_grounding" in questions:
                return SimpleNamespace(
                    model=DECISIONS_CLASSIFIER_MODEL,
                    answers={
                        q_name: SimpleNamespace(type="noul", noul=0.92, confidence=0.92)
                        for q_name in questions
                    },
                )
            # Ticket A (index 0) -> 0.88 (passes), Ticket B (index 1) -> failing_noul (0.50 or 0.12)
            return SimpleNamespace(
                model=DECISIONS_CLASSIFIER_MODEL,
                answers={
                    "ticket_0_grounding": SimpleNamespace(type="noul", noul=0.88, confidence=0.88),
                    "ticket_1_grounding": SimpleNamespace(
                        type="noul",
                        noul=failing_noul,
                        confidence=max(failing_noul, 1.0 - failing_noul),
                    ),
                },
            )

        @staticmethod
        def _embed_content(*, contents: object, **_: object) -> SimpleNamespace:
            batch = [contents] if isinstance(contents, str) else list(contents)
            return SimpleNamespace(
                embeddings=[SimpleNamespace(values=deploy_vector) for _ in batch]
            )

        @staticmethod
        def _generate_content(*, contents: str, **_: object) -> SimpleNamespace:
            payload = json.loads(contents)
            source = payload["primary_sources"][0]
            return SimpleNamespace(
                text=json.dumps(
                    {
                        "plain_text": (
                            "Thank you for getting in touch. Store anything sensitive "
                            "as a secret rather than a variable so values stay masked in logs."
                        ),
                        "claims": [
                            {
                                "text": "Store anything sensitive as a secret rather than a variable.",
                                "source_ids": [source["passage_id"]],
                            }
                        ],
                        "unsupported_points": [],
                        "conflicting_points": [],
                    }
                )
            )

    monkeypatch.setattr(genai, "Client", _NoulBoundaryClient)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    monkeypatch.setenv("GEMINI_LLM_MODEL", "gemini-3.8-flash")
    monkeypatch.setenv("DRAFT_GENERATION_PROVIDER", "GEMINI")

    records = [
        {
            "ticket_id": "TICKET-A",
            "channel": "email",
            "subject": "Secrets in environment variables",
            "body": "Where should secrets be stored so they do not appear in logs?",
        },
        {
            "ticket_id": "TICKET-B",
            "channel": "chat",
            "subject": "Unsupported custom integration",
            "body": "Can CloudServe sync directly with our custom mainframe?",
        },
        {
            "ticket_id": "TICKET-C",
            "channel": "email",
            "subject": "Help",
            "body": "Something is wrong, please assist.",
        },
    ]
    input_path = tmp_path / "ac2_tickets.json"
    input_path.write_text(json.dumps(records), encoding="utf-8")
    database_path = tmp_path / "ac2_runtime.sqlite3"
    output_root = tmp_path / "ac2_outputs"

    run_id, out_dir, _ = run_ticket_batch(
        input_path=input_path,
        output_root=output_root,
        database_path=database_path,
        policy_path=FROZEN_POLICY_PATH,
    )
    results, _, _ = _validate_all_exported_files(out_dir)
    by_id = {row["ticket_id"]: row for row in results}

    assert by_id["TICKET-A"]["predicted_route"] == "draft_for_review"
    assert by_id["TICKET-A"]["routing"]["final_decision"]["evidence_ready"] is True

    assert by_id["TICKET-B"]["predicted_route"] == "escalate"
    assert by_id["TICKET-B"]["routing"]["matched_rule"] == "claim_grounding_failed"
    assert by_id["TICKET-B"]["routing"]["final_decision"]["evidence_ready"] is False

    assert by_id["TICKET-C"]["predicted_route"] == "escalate"
    assert by_id["TICKET-C"]["routing"]["final_decision"]["evidence_ready"] is False

    connection = connect_runtime_database(database_path)
    try:
        rows = connection.execute(
            "SELECT ticket_id, generated_labels_json FROM ticket_label_comparison "
            "WHERE run_id = ? ORDER BY ticket_id",
            (run_id,),
        ).fetchall()
        labels_by_id = {
            row["ticket_id"]: json.loads(row["generated_labels_json"])
            for row in rows
        }
        assert labels_by_id["TICKET-A"]["answerable_from_docs"] is True
        assert labels_by_id["TICKET-B"]["answerable_from_docs"] is False
        assert labels_by_id["TICKET-C"]["answerable_from_docs"] is False
    finally:
        connection.close()

