"""Integration and live verification for gemini-3.8-flash claim_grounding_check routing."""

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
    CLAIM_GROUNDING_SYSTEM_INSTRUCTION,
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
    assert checker.last_model_name == "gemini-3.8-flash"
    assert checker.last_model_name != DECISIONS_CLASSIFIER_MODEL

    answerable_grounding = answerable_eval["checks"][3]
    allowed_answerable_ids = {
        item["doc_id"]
        for item in (*answerable_retrieval.sources, *answerable_retrieval.related_sources)
    } | {
        item["passage_id"]
        for item in (*answerable_retrieval.sources, *answerable_retrieval.related_sources)
    }
    assert answerable_grounding["name"] == "claim_grounding_check"
    assert answerable_grounding["status"] == "passed"
    assert 0.0 <= answerable_grounding["confidence"] <= 1.0
    assert len(answerable_grounding["claims"]) >= 1
    for claim in answerable_grounding["claims"]:
        assert claim["text"].strip()
        assert set(claim["source_ids"]).issubset(allowed_answerable_ids)
    assert answerable_grounding["unsupported_points"] == []
    assert answerable_grounding["conflicting_points"] == []
    assert answerable_grounding["reason"].strip()
    assert answerable_eval["final_decision"]["route"] == "draft_for_review"
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
    assert checker.last_model_name == "gemini-3.8-flash"
    assert unsupported_eval["checks"][2]["status"] == "passed"
    unsupported_grounding = unsupported_eval["checks"][3]
    assert unsupported_grounding["status"] == "failed"
    assert unsupported_grounding["matched_rule"] == "claim_grounding_failed"
    assert (
        not unsupported_grounding["claims"]
        or len(unsupported_grounding["unsupported_points"]) >= 1
        or len(unsupported_grounding["conflicting_points"]) >= 1
    )
    assert unsupported_eval["final_decision"]["route"] == "escalate"
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

    grounding_calls: list[tuple[str, dict[str, Any]]] = []
    draft_calls: list[str] = []

    class _FaultInjectingGeminiBoundary:
        """Stub only external google.genai.Client to inject deterministic timeout and bad-JSON faults."""

        def __init__(self, **_: object) -> None:
            self.decisions = SimpleNamespace(create=self._create_decision)
            self.models = SimpleNamespace(
                embed_content=self._embed_content,
                generate_content=self._generate_content,
            )

        @staticmethod
        def _create_decision(**_: object) -> SimpleNamespace:
            return SimpleNamespace(
                model=DECISIONS_CLASSIFIER_MODEL,
                answers={
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
                        "ticket_3_intent", "data", 0.94, INTENT_VALUES
                    ),
                    "ticket_3_urgency": _make_decision_choice(
                        "ticket_3_urgency", "medium", 0.85, URGENCY_VALUES
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
        def _generate_content(*, model: str, contents: str, config: Any) -> SimpleNamespace:
            payload = json.loads(contents)
            subject = payload["ticket"]["subject"]
            if config.system_instruction == CLAIM_GROUNDING_SYSTEM_INSTRUCTION:
                grounding_calls.append((model, payload))
                if subject == "Simulate grounding timeout":
                    raise TimeoutError("Gemini read timed out")
                if subject == "Simulate invalid grounding JSON":
                    return SimpleNamespace(text="{malformed-json")
                primary_id = payload["primary_sources"][0]["passage_id"]
                return SimpleNamespace(
                    text=json.dumps(
                        {
                            "confidence": 0.96,
                            "claims": [
                                {
                                    "text": "Store anything sensitive as a secret rather than a variable.",
                                    "source_ids": [primary_id],
                                }
                            ],
                            "unsupported_points": [],
                            "conflicting_points": [],
                            "reason": "Secrets storage and log masking are documented in DOC-DEPLOY-004.",
                        }
                    )
                )

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
    batch_records = [
        {
            "ticket_id": "BATCH-TIMEOUT",
            "channel": "email",
            "subject": "Simulate grounding timeout",
            "body": "Where should secrets be stored so they do not appear in the logs?",
            "labels": {"expected_route": "draft_for_review"},
            "history": {"csat_rating": 4},
        },
        {
            "ticket_id": "BATCH-BAD-JSON",
            "channel": "chat",
            "subject": "Simulate invalid grounding JSON",
            "body": "Where should secrets be stored so they do not appear in the logs?",
            "labels": {"expected_route": "draft_for_review"},
            "history": {"csat_rating": 4},
        },
        dev_0006,
        {
            "ticket_id": "BATCH-PROTECTED",
            "channel": "email",
            "subject": "Data storage location",
            "body": "Where is our customer data stored and which region hosts it?",
            "labels": {"expected_route": "auto_respond"},
            "history": {"csat_rating": 5},
        },
    ]
    input_path = tmp_path / "batch_tickets.json"
    input_path.write_text(json.dumps(batch_records), encoding="utf-8")
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
            "grounding-fault-batch",
        ]
    )
    assert exit_code == 0

    # Protected ticket short-circuited before claim_grounding_check; only the 3 non-protected tickets called it.
    assert [model for model, _ in grounding_calls] == ["gemini-3.8-flash"] * 3
    assert [payload["ticket"]["subject"] for _, payload in grounding_calls] == [
        "Simulate grounding timeout",
        "Simulate invalid grounding JSON",
        dev_0006["subject"],
    ]
    for _, sent_payload in grounding_calls:
        serialized_sent = json.dumps(sent_payload)
        assert '"labels"' not in serialized_sent
        assert '"history"' not in serialized_sent
        assert len(sent_payload["primary_sources"]) >= 1
        assert len(sent_payload["related_sources"]) >= 1
    assert draft_calls == [dev_0006["subject"]]

    run_dirs = list(output_root.iterdir())
    assert len(run_dirs) == 1
    results, decisions, metrics = _validate_all_exported_files(run_dirs[0])
    results_by_id = {row["ticket_id"]: row for row in results}

    assert results_by_id["BATCH-TIMEOUT"]["status"] == "failed"
    assert results_by_id["BATCH-TIMEOUT"]["draft"] is None
    assert results_by_id["BATCH-TIMEOUT"]["error"]["code"] == "claim_grounding_timeout"
    assert results_by_id["BATCH-TIMEOUT"]["routing"]["matched_rule"] == "claim_grounding_unavailable"

    assert results_by_id["BATCH-BAD-JSON"]["status"] == "failed"
    assert results_by_id["BATCH-BAD-JSON"]["draft"] is None
    assert results_by_id["BATCH-BAD-JSON"]["error"]["code"] == "invalid_claim_grounding_output"
    assert results_by_id["BATCH-BAD-JSON"]["routing"]["matched_rule"] == "claim_grounding_unavailable"

    assert results_by_id["DEV-0006"]["status"] == "completed"
    assert results_by_id["DEV-0006"]["predicted_route"] == "draft_for_review"
    assert results_by_id["DEV-0006"]["draft"] is not None

    assert results_by_id["BATCH-PROTECTED"]["status"] == "completed"
    assert results_by_id["BATCH-PROTECTED"]["predicted_route"] == "escalate"
    assert results_by_id["BATCH-PROTECTED"]["draft"] is None
    assert (
        results_by_id["BATCH-PROTECTED"]["routing"]["matched_rule"]
        == "operational_protected_data_location"
    )
    assert metrics["total_count"] == 4
    assert metrics["route_counts"] == {
        "draft_for_review": 1,
        "escalate": 1,
        "degrade": 2,
    }

    # AC-3: Inspect SQLite audit_events routing stage rows and re-verify write_run_exports directly.
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

        assert routing_by_ticket["BATCH-TIMEOUT"][0] == "gemini-3.8-flash"
        assert routing_by_ticket["BATCH-TIMEOUT"][1]["final_decision"]["route"] == "degrade"
        assert routing_by_ticket["BATCH-BAD-JSON"][0] == "gemini-3.8-flash"
        assert routing_by_ticket["BATCH-BAD-JSON"][1]["final_decision"]["route"] == "degrade"
        assert routing_by_ticket["DEV-0006"][0] == "gemini-3.8-flash"
        assert routing_by_ticket["DEV-0006"][1]["final_decision"]["route"] == "draft_for_review"
        assert routing_by_ticket["BATCH-PROTECTED"][0] is None
        assert routing_by_ticket["BATCH-PROTECTED"][1]["checks"][3]["status"] == "skipped"
        assert routing_by_ticket["BATCH-PROTECTED"][1]["final_decision"]["route"] == "escalate"

        re_export_dir = tmp_path / "re_exported"
        re_export_dir.mkdir()
        re_metrics = write_run_exports(
            connection,
            run_id=run_id,
            output_directory=re_export_dir,
            input_records=batch_records,
            input_path=input_path,
            source_ref="batch_tickets.json",
        )
        assert re_metrics["route_counts"] == metrics["route_counts"]
        _validate_all_exported_files(re_export_dir)
    finally:
        connection.close()

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

    # 1. Misconfiguring GEMINI_LLM_MODEL to gemini-decision-1 is rejected before any client call.
    bad_model_settings = Settings(
        gemini_api_key="test-key",
        llm_model=DECISIONS_CLASSIFIER_MODEL,
    )
    with pytest.raises(ClaimGroundingError) as exc_info:
        check_claim_grounding(
            ticket=ticket,
            sources=retrieval.sources,
            related_sources=retrieval.related_sources,
            settings=bad_model_settings,
            today=FIXED_TODAY,
        )
    assert exc_info.value.code == "claim_grounding_configuration_error"

    # 2. Missing API key in GEMINI mode fails closed to degrade in run_ticket_batch without genai stubs.
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
