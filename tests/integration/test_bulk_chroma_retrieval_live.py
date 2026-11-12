"""Live acceptance checks for batched OpenAI embeddings and persistent Chroma queries."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from dotenv import dotenv_values
from openai import OpenAI

from cloudserve_support.classification.ticket_classification import (
    DECISIONS_CLASSIFIER_MODEL,
    TicketClassificationBatch,
    classification_failure,
)
from cloudserve_support.config import Settings
from cloudserve_support.domain.ticket import CanonicalTicket
from cloudserve_support.generation.draft_generation import OfflineDraftGenerationProvider
from cloudserve_support.guardrails.draft_safety import (
    contains_private_data,
    contains_prompt_injection,
    filter_evaluation_context,
    ticket_text_for_generation,
)
from cloudserve_support.ingestion.ticket_normalization import normalize_ticket_record
from cloudserve_support.knowledge.retrieval import (
    MINIMUM_SIMILARITY,
    KnowledgeRetriever,
    RetrievalResult,
)
from cloudserve_support.processing import run_tickets
from cloudserve_support.processing.run_tickets import run_ticket_batch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEVELOPMENT_INPUT = REPOSITORY_ROOT / "data" / "tickets" / "development_tickets.json"


class RecordingOpenAIEmbeddings:
    """Record live request sizes and sanitized failures without exposing request data."""

    def __init__(self, client: OpenAI, *, fail_call_number: int | None = None) -> None:
        self.client = client
        self.fail_call_number = fail_call_number
        self.calls: list[int] = []
        self.diagnostics: list[dict[str, str | int]] = []
        self.capture_phase: str | None = None
        self.response_vectors: dict[str, dict[str, tuple[float, ...]]] = {}

    def create(self, *, model: str, input: str | list[str]) -> Any:
        inputs = input if isinstance(input, list) else [input]
        self.calls.append(len(inputs))
        try:
            if len(self.calls) == self.fail_call_number:
                response = self.client.embeddings.create(
                    model="cloudserve-invalid-embedding-model",
                    input=input,
                )
            else:
                response = self.client.embeddings.create(model=model, input=input)
            try:
                self._capture_vectors(inputs, response)
            except Exception:
                pass
            return response
        except Exception as error:
            self.diagnostics.append(_safe_failure_summary("openai_embeddings", error))
            raise

    def _capture_vectors(self, inputs: list[str], response: Any) -> None:
        """Keep successful vectors in memory, indexed by response row and query hash."""
        if self.capture_phase is None:
            return
        data = getattr(response, "data", None)
        if not isinstance(data, list):
            return
        phase_vectors = self.response_vectors.setdefault(self.capture_phase, {})
        for row in data:
            index = getattr(row, "index", None)
            vector = getattr(row, "embedding", None)
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or index < 0
                or index >= len(inputs)
                or not isinstance(vector, (list, tuple))
                or len(vector) != 1536
            ):
                continue
            try:
                captured_vector = tuple(float(value) for value in vector)
            except (TypeError, ValueError, OverflowError):
                continue
            if not all(math.isfinite(value) for value in captured_vector):
                continue
            query_hash = _query_text_hash(inputs[index])
            phase_vectors[query_hash] = captured_vector


class RecordingPersistentCollection:
    """Record actual Chroma calls and optionally invalidate one real query vector."""

    def __init__(self, collection: Any, *, fail_call_number: int | None = None) -> None:
        self.collection = collection
        self.fail_call_number = fail_call_number
        self.calls: list[int] = []
        self.diagnostics: list[dict[str, str | int]] = []

    def count(self) -> int:
        return self.collection.count()

    def get(self, **options: Any) -> Any:
        return self.collection.get(**options)

    def query(self, *, query_embeddings: list[list[float]], **options: Any) -> Any:
        self.calls.append(len(query_embeddings))
        try:
            if len(self.calls) == self.fail_call_number:
                invalid_vectors = [list(vector) for vector in query_embeddings]
                invalid_vectors[0] = invalid_vectors[0][:-1]
                return self.collection.query(query_embeddings=invalid_vectors, **options)
            return self.collection.query(query_embeddings=query_embeddings, **options)
        except Exception as error:
            self.diagnostics.append(_safe_failure_summary("chroma_query", error))
            raise


class RecordingBulkRetriever:
    """Delegate bulk retrieval and emit completion events only after every result returns."""

    def __init__(
        self,
        delegate: KnowledgeRetriever,
        events: list[tuple[str, str]],
    ) -> None:
        self.delegate = delegate
        self.events = events
        self.results: tuple[RetrievalResult, ...] | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)

    def retrieve_many(
        self,
        tickets: list[CanonicalTicket],
        *,
        embedding_batch_size: int,
        query_batch_size: int,
    ) -> tuple[RetrievalResult, ...]:
        results = self.delegate.retrieve_many(
            tickets,
            embedding_batch_size=embedding_batch_size,
            query_batch_size=query_batch_size,
        )
        self.results = results
        for ticket in tickets:
            self.events.append(("retrieval_completed", ticket.ticket_id))
        return results


def _live_api_key() -> str:
    """Load the already configured live API key without exposing it in test output."""
    key = dotenv_values(REPOSITORY_ROOT / ".env").get("OPENAI_API_KEY")
    if not isinstance(key, str) or not key.strip():
        pytest.fail("Set OPENAI_API_KEY in the environment or ignored .env file before live acceptance.")
    return key.strip()


def _safe_failure_summary(component: str, error: Exception) -> dict[str, str | int]:
    """Return component, exception class, and numeric HTTP status only."""
    summary: dict[str, str | int] = {
        "component": component,
        "exception_class": type(error).__name__,
    }
    status_code = getattr(error, "status_code", None)
    if (
        isinstance(status_code, int)
        and not isinstance(status_code, bool)
        and 100 <= status_code <= 599
    ):
        summary["http_status"] = status_code
    return summary


def _retrieval_diagnostics(
    embeddings: RecordingOpenAIEmbeddings,
    collection: RecordingPersistentCollection,
) -> dict[str, list[dict[str, str | int]]]:
    """Summarize sanitized OpenAI and Chroma failures without request payloads."""
    return {
        "openai_embeddings": list(embeddings.diagnostics),
        "chroma_query": list(collection.diagnostics),
    }


def _query_text_hash(query_text: str) -> str:
    """Identify a normalized subject/body string without retaining its contents."""
    return hashlib.sha256(query_text.encode("utf-8")).hexdigest()


def _query_text_for_ticket(ticket: CanonicalTicket) -> str:
    """Build the subject/body-only retrieval text used for diagnostic association."""
    return "\n".join(
        part for part in (ticket.subject.strip(), ticket.body.strip()) if part
    )


def _ticket_without_persisted_text(ticket: CanonicalTicket) -> CanonicalTicket:
    """Keep typed retrieval fields while removing subject/body from stored context."""
    safe_context = {
        key: value
        for key, value in ticket.context.items()
        if str(key).casefold() not in {"subject", "body"}
    }
    return replace(ticket, context=safe_context)


def _comparison_diagnostics(
    tickets: list[CanonicalTicket],
    serial_results: tuple[RetrievalResult, ...],
    batch_results: tuple[RetrievalResult, ...],
    embeddings: RecordingOpenAIEmbeddings,
) -> list[dict[str, Any]]:
    """Build per-ticket deltas and equality flags, then discard captured vectors."""
    serial_vectors = embeddings.response_vectors.get("serial", {})
    batch_vectors = embeddings.response_vectors.get("batched", {})
    summaries = []
    for ticket, serial, batch in zip(
        tickets,
        serial_results,
        batch_results,
        strict=True,
    ):
        query_hash = _query_text_hash(_query_text_for_ticket(ticket))
        serial_vector = serial_vectors.get(query_hash)
        batch_vector = batch_vectors.get(query_hash)
        max_vector_delta = None
        if (
            serial_vector is not None
            and batch_vector is not None
            and len(serial_vector) == len(batch_vector)
        ):
            max_vector_delta = max(
                abs(serial_value - batch_value)
                for serial_value, batch_value in zip(
                    serial_vector,
                    batch_vector,
                    strict=True,
                )
            )

        serial_scores = {
            source["passage_id"]: source["similarity"] for source in serial.sources
        }
        batch_scores = {
            source["passage_id"]: source["similarity"] for source in batch.sources
        }
        score_deltas = [
            abs(serial_scores[passage_id] - batch_scores[passage_id])
            for passage_id in serial_scores
            if passage_id in batch_scores
        ]
        summaries.append(
            {
                "ticket_id": ticket.ticket_id,
                "serial_status": serial.status,
                "batch_status": batch.status,
                "status_equal": serial.status == batch.status,
                "source_ids_equal": [
                    source["passage_id"] for source in serial.sources
                ]
                == [source["passage_id"] for source in batch.sources],
                "embedding_max_abs_component_delta": max_vector_delta,
                "score_deltas": score_deltas,
            }
        )
    embeddings.response_vectors.clear()
    return summaries


def test_live_failure_diagnostics_exclude_exception_text_and_codes() -> None:
    """Keep sensitive provider messages and opaque codes out of smoke diagnostics."""
    class SimulatedProviderError(Exception):
        status_code = 503
        code = "opaque-internal-code"

    summary = _safe_failure_summary(
        "openai_embeddings",
        SimulatedProviderError("request body and credential details"),
    )

    assert summary == {
        "component": "openai_embeddings",
        "exception_class": "SimulatedProviderError",
        "http_status": 503,
    }


def test_live_embedding_capture_maps_response_indexes_without_retaining_queries() -> None:
    """Capture vector deltas by response index and query hash without storing text."""
    class FakeEmbeddingClient:
        class Embeddings:
            def create(self, *, model: str, input: str | list[str]) -> SimpleNamespace:
                assert model == "offline-capture-test"
                inputs = [input] if isinstance(input, str) else input
                rows = [
                    SimpleNamespace(
                        index=index,
                        embedding=[float(index + 1)] * 1536,
                    )
                    for index in reversed(range(len(inputs)))
                ]
                return SimpleNamespace(data=rows)

        def __init__(self) -> None:
            self.embeddings = self.Embeddings()

    recorder = RecordingOpenAIEmbeddings(FakeEmbeddingClient())  # type: ignore[arg-type]
    inputs = ["subject 0\nbody 0", "subject 1\nbody 1"]
    recorder.capture_phase = "batched"
    recorder.create(model="offline-capture-test", input=inputs)

    vectors = recorder.response_vectors["batched"]
    assert vectors[_query_text_hash(inputs[0])][0] == 1.0
    assert vectors[_query_text_hash(inputs[1])][0] == 2.0
    assert recorder.calls == [2]
    assert all(query not in repr(recorder.response_vectors) for query in inputs)


def test_smoke_ticket_context_redaction_preserves_live_fields() -> None:
    """Ensure diagnostic persistence excludes text but retrieval inputs stay intact."""
    ticket = CanonicalTicket(
        ticket_id="DEV-0001",
        has_ticket_id=True,
        channel="email",
        subject="Private test subject",
        body="Private test body",
        received_at=None,
        customer_tier=None,
        language_fluency=None,
        context={
            "subject": "Private test subject",
            "body": "Private test body",
            "channel": "email",
            "requester": "test-user",
        },
    )

    sanitized = _ticket_without_persisted_text(ticket)

    assert sanitized.subject == ticket.subject
    assert sanitized.body == ticket.body
    assert sanitized.context == {"channel": "email", "requester": "test-user"}
    assert ticket.subject not in json.dumps(sanitized.context)
    assert ticket.body not in json.dumps(sanitized.context)


def _smoke_tickets() -> tuple[list[dict[str, Any]], list[CanonicalTicket]]:
    """Select DEV-0001..0005 and pass only normalized operational fields onward."""
    source_records = json.loads(DEVELOPMENT_INPUT.read_text(encoding="utf-8"))
    selected_records = source_records[:5]
    assert [record["ticket_id"] for record in selected_records] == [
        f"DEV-{index:04d}" for index in range(1, 6)
    ]
    operational_records = [
        {key: value for key, value in record.items() if key not in {"labels", "history"}}
        for record in selected_records
    ]
    tickets = [
        normalize_ticket_record(record, input_index=index)
        for index, record in enumerate(operational_records)
    ]
    assert all(not ticket.validation_errors for ticket in tickets)
    assert all(
        not contains_prompt_injection(ticket_text_for_generation(ticket))
        and not contains_private_data(ticket_text_for_generation(ticket))
        and not contains_private_data(
            json.dumps(filter_evaluation_context(ticket.context), ensure_ascii=False)
        )
        for ticket in tickets
    )
    return operational_records, tickets


def _build_live_retriever(
    settings: Settings,
    *,
    embedding_recorder: RecordingOpenAIEmbeddings | None = None,
    collection_recorder: RecordingPersistentCollection | None = None,
) -> tuple[KnowledgeRetriever, RecordingOpenAIEmbeddings, RecordingPersistentCollection]:
    """Use the real OpenAI client and packaged persistent Chroma collection."""
    client = OpenAI(
        api_key=settings.require_openai_api_key(),
        timeout=15.0,
        max_retries=0,
    )
    embeddings = embedding_recorder or RecordingOpenAIEmbeddings(client)
    if collection_recorder is None:
        import chromadb

        chroma_client = chromadb.PersistentClient(
            path=str(REPOSITORY_ROOT / "data" / "index" / "chroma")
        )
        collection = chroma_client.get_collection(
            name="cloudserve_kb",
            embedding_function=None,
        )
        collection_recorder = RecordingPersistentCollection(collection)
    retriever = KnowledgeRetriever(
        settings,
        source_path=REPOSITORY_ROOT / "data" / "kb" / "documentation.json",
        metadata_path=REPOSITORY_ROOT / "data" / "kb" / "metadata_manifest.json",
        index_path=REPOSITORY_ROOT / "data" / "index",
        embedding_client=EmbeddingClientAdapter(embeddings),
        collection=collection_recorder,
    )
    assert retriever.unavailable_reason is None
    return retriever, embeddings, collection_recorder


class EmbeddingClientAdapter:
    """Expose a recording embeddings resource through the OpenAI client contract."""

    def __init__(self, embeddings: RecordingOpenAIEmbeddings) -> None:
        self.embeddings = embeddings


def _assert_results_match_serial(
    batched: tuple[RetrievalResult, ...],
    serial: tuple[RetrievalResult, ...],
) -> None:
    """Compare independent requests by status, source order, and provenance."""
    assert len(batched) == len(serial), "Bulk and serial result counts differ."
    for batched_result, serial_result in zip(batched, serial, strict=True):
        assert batched_result.status == serial_result.status, "Bulk and serial statuses differ."
        for field_name in (
            "reason_code",
            "eligible_candidate_count",
            "excluded_candidate_counts",
            "minimum_similarity",
            "embedding_model",
            "source_sha256",
            "metadata_manifest_sha256",
        ):
            assert getattr(batched_result, field_name) == getattr(
                serial_result,
                field_name,
            ), f"Bulk and serial retrieval field {field_name} differs."
        assert [source["passage_id"] for source in batched_result.sources] == [
            source["passage_id"] for source in serial_result.sources
        ], "Bulk and serial passage IDs or ordering differ."
        assert len(batched_result.sources) == len(serial_result.sources), (
            "Bulk and serial source counts differ."
        )
        for batched_source, serial_source in zip(
            batched_result.sources, serial_result.sources, strict=True
        ):
            assert batched_source.keys() == serial_source.keys(), "Bulk source fields differ."
            for key in batched_source:
                if key != "similarity":
                    assert batched_source[key] == serial_source[key], (
                        f"Bulk and serial source field {key} differs."
                    )


def test_live_result_comparison_allows_independent_score_deltas() -> None:
    """Independent API calls compare IDs/provenance while reporting score deltas."""
    source = {
        "passage_id": "DOC-SEC-003::p1",
        "doc_id": "DOC-SEC-003",
        "passage_text": "Approved source passage.",
        "approval_status": "approved",
        "approval_status_source": "metadata_manifest",
        "reviewed_at": "2026-10-01",
        "minimum_similarity": MINIMUM_SIMILARITY,
        "similarity": 0.602720797,
    }
    batch_source = {**source, "similarity": 0.602796139}
    result_fields = {
        "status": "ready",
        "reason_code": "sources_found",
        "reason": "Approved sources found.",
        "eligible_candidate_count": 1,
        "excluded_candidate_counts": {},
        "minimum_similarity": MINIMUM_SIMILARITY,
        "embedding_model": "text-embedding-3-small",
        "source_sha256": "source-hash",
        "metadata_manifest_sha256": "metadata-hash",
    }
    serial = RetrievalResult(sources=(source,), **result_fields)
    batched = RetrievalResult(sources=(batch_source,), **result_fields)

    _assert_results_match_serial((batched,), (serial,))


def test_live_five_ticket_retrieval_smoke_matches_serial_and_commits_before_routing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compare five live results and prove the CLI routes after retrieval commits."""
    api_key = _live_api_key()
    monkeypatch.setenv("OPENAI_API_KEY", api_key)
    monkeypatch.setenv("DRAFT_GENERATION_PROVIDER", "OFFLINE")
    settings = Settings.from_env()
    operational_records, tickets = _smoke_tickets()
    sanitized_hash = hashlib.sha256(
        json.dumps(
            [
                {"ticket_id": ticket.ticket_id, "subject": ticket.subject, "body": ticket.body}
                for ticket in tickets
            ],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert len(sanitized_hash) == 64
    input_path = tmp_path / "retrieval-smoke-5.json"

    live_retriever, embeddings, collection = _build_live_retriever(settings)
    embeddings.capture_phase = "serial"
    serial_results = tuple(live_retriever.retrieve(ticket) for ticket in tickets)
    embeddings.capture_phase = None
    diagnostics = _retrieval_diagnostics(embeddings, collection)
    assert len(embeddings.calls) == len(tickets)
    assert all(
        result.status != "unavailable" for result in serial_results
    ), f"Serial retrieval unavailable; sanitized diagnostics={diagnostics}; reason_codes={[result.reason_code for result in serial_results]}"
    assert any(
        result.status == "ready" for result in serial_results
    ), f"No approved source result was ready; sanitized diagnostics={diagnostics}; reason_codes={[result.reason_code for result in serial_results]}"
    assert serial_results[3].sources == (), "DEV-0004 must not cite expired DOC-DEPLOY-002."
    serial_query_call_count = len(collection.calls)
    events: list[tuple[str, str]] = []
    recording_retriever = RecordingBulkRetriever(live_retriever, events)
    database_path = tmp_path / "runtime.sqlite3"
    monkeypatch.setattr(run_tickets, "DEFAULT_EMBEDDING_BATCH_SIZE", len(tickets))
    monkeypatch.setattr(run_tickets, "DEFAULT_QUERY_BATCH_SIZE", len(tickets))
    real_router = run_tickets.route_ticket_classification
    real_audit_insert = run_tickets._insert_audit_event
    real_build_ticket_result = run_tickets._build_ticket_result
    real_prepare_ticket_records = run_tickets._prepare_ticket_records

    def insert_without_ticket_body(*args: Any, **kwargs: Any) -> None:
        kwargs["input_summary"] = None
        real_audit_insert(*args, **kwargs)

    def build_result_without_ticket_body(*args: Any, **kwargs: Any) -> Any:
        result, storage_ticket_id, ticket = real_build_ticket_result(*args, **kwargs)
        handoff = result.get("handoff")
        if isinstance(handoff, dict):
            handoff["summary"] = "Ticket details omitted from retrieval diagnostics."
            handoff["issue"] = "Ticket details omitted from retrieval diagnostics."
        return result, storage_ticket_id, ticket

    def prepare_without_persisted_ticket_text(
        input_records: list[dict[str, Any]],
    ) -> list[Any]:
        prepared_tickets = real_prepare_ticket_records(input_records)
        for prepared_ticket in prepared_tickets:
            prepared_ticket.ticket = _ticket_without_persisted_text(
                prepared_ticket.ticket
            )
        return prepared_tickets

    def deterministic_classifier(
        classifiable_tickets: list[CanonicalTicket],
        _settings: Settings,
    ) -> tuple[TicketClassificationBatch, ...]:
        classifications = tuple(
            classification_failure(
                provider="not_attempted",
                model_name=DECISIONS_CLASSIFIER_MODEL,
                uncertainty="Classification was intentionally skipped during retrieval acceptance.",
            )
            for _ in classifiable_tickets
        )
        return (TicketClassificationBatch(0, classifications, 0.0),)

    def recording_router(*args: Any, **kwargs: Any) -> Any:
        ticket = kwargs["ticket"]
        with sqlite3.connect(database_path) as connection:
            assert connection.execute(
                "SELECT COUNT(*) FROM audit_events WHERE event_type = 'retrieval'"
            ).fetchone()[0] == len(tickets)
        events.append(("route_started", ticket.ticket_id))
        return real_router(*args, **kwargs)

    monkeypatch.setattr(run_tickets, "classify_ticket_batches", deterministic_classifier)
    monkeypatch.setattr(run_tickets, "route_ticket_classification", recording_router)
    monkeypatch.setattr(run_tickets, "_insert_audit_event", insert_without_ticket_body)
    monkeypatch.setattr(run_tickets, "_build_ticket_result", build_result_without_ticket_body)
    monkeypatch.setattr(
        run_tickets,
        "_prepare_ticket_records",
        prepare_without_persisted_ticket_text,
    )
    input_path.write_text(json.dumps(operational_records), encoding="utf-8")
    embeddings.capture_phase = "batched"
    try:
        run_id, output_directory, _ = run_ticket_batch(
            input_path=input_path,
            output_root=tmp_path / "outputs",
            database_path=database_path,
            retriever=recording_retriever,
            generator=OfflineDraftGenerationProvider(),
        )
    finally:
        embeddings.capture_phase = None
        input_path.unlink(missing_ok=True)
    assert not input_path.exists()

    retrieval_events = [event for event in events if event[0] == "retrieval_completed"]
    route_events = [event for event in events if event[0] == "route_started"]
    assert [ticket_id for _, ticket_id in retrieval_events] == [ticket.ticket_id for ticket in tickets]
    assert [ticket_id for _, ticket_id in route_events] == [ticket.ticket_id for ticket in tickets]
    assert events.index(route_events[0]) == len(tickets)
    assert len(collection.calls) - serial_query_call_count == 1
    assert collection.calls[-1] == len(tickets)
    assert len(embeddings.calls) == 6
    assert sum(embeddings.calls) == 10
    assert recording_retriever.results is not None
    batched_results = recording_retriever.results
    diagnostics = _retrieval_diagnostics(embeddings, collection)
    assert all(
        result.status != "unavailable" for result in batched_results
    ), f"Batched retrieval unavailable; sanitized diagnostics={diagnostics}; reason_codes={[result.reason_code for result in batched_results]}"
    assert any(
        result.status == "ready" for result in batched_results
    ), f"No approved batched result was ready; sanitized diagnostics={diagnostics}; reason_codes={[result.reason_code for result in batched_results]}"
    assert all(
        source["approval_status"] == "approved"
        for result in batched_results
        for source in result.sources
    )
    comparison_report = _comparison_diagnostics(
        tickets,
        serial_results,
        batched_results,
        embeddings,
    )
    print(json.dumps({"retrieval_diagnostics": comparison_report}, sort_keys=True))
    assert live_retriever.article_metadata["DOC-DEPLOY-002"]["approval_status"] == "expired"
    assert all(
        source["doc_id"] != "DOC-DEPLOY-002"
        for result in batched_results
        for source in result.sources
    )
    assert all(
        source["similarity"] >= MINIMUM_SIMILARITY
        for result in batched_results
        for source in result.sources
    )
    with sqlite3.connect(database_path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM audit_events WHERE run_id = ? AND event_type = 'retrieval'",
            (run_id,),
        ).fetchone()[0] == len(tickets)
        assert connection.execute(
            "SELECT COUNT(*) FROM audit_events WHERE run_id = ? AND input_summary IS NOT NULL",
            (run_id,),
        ).fetchone()[0] == 0
    saved_artifacts = [database_path, *output_directory.iterdir()]
    artifact_bytes = b"".join(
        artifact.read_bytes() for artifact in saved_artifacts if artifact.is_file()
    )
    contains_ticket_content = any(
        content.encode("utf-8") in artifact_bytes
        for ticket in tickets
        for content in (ticket.subject, ticket.body)
        if content
    )
    assert not contains_ticket_content, (
        "SQLite and CLI outputs must not persist ticket subjects or bodies."
    )
    assert (output_directory / "results.jsonl").is_file()
    _assert_results_match_serial(batched_results, serial_results)


def test_live_bulk_embedding_failure_is_chunk_local_and_later_chunks_continue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Let OpenAI reject one invalid-model chunk and verify later real chunks still work."""
    monkeypatch.setenv("OPENAI_API_KEY", _live_api_key())
    settings = Settings.from_env()
    _, tickets = _smoke_tickets()
    client = OpenAI(api_key=settings.require_openai_api_key(), timeout=15.0, max_retries=0)
    embeddings = RecordingOpenAIEmbeddings(client, fail_call_number=2)
    retriever, _, _ = _build_live_retriever(settings, embedding_recorder=embeddings)

    results = retriever.retrieve_many(tickets, embedding_batch_size=2, query_batch_size=2)

    assert len(results) == 5
    assert all(
        result.status == "unavailable"
        and result.reason_code == "retrieval_provider_or_index_error"
        for result in results[2:4]
    )
    assert all(result.status != "unavailable" for result in results[:2] + results[4:])
    assert len(embeddings.calls) == 3


def test_live_bulk_chroma_dimension_failure_is_chunk_local_and_later_chunks_continue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Let persistent Chroma reject one malformed vector chunk, then continue querying."""
    monkeypatch.setenv("OPENAI_API_KEY", _live_api_key())
    settings = Settings.from_env()
    _, tickets = _smoke_tickets()
    import chromadb

    chroma_client = chromadb.PersistentClient(
        path=str(REPOSITORY_ROOT / "data" / "index" / "chroma")
    )
    real_collection = chroma_client.get_collection("cloudserve_kb", embedding_function=None)
    collection = RecordingPersistentCollection(real_collection, fail_call_number=1)
    retriever, embeddings, _ = _build_live_retriever(settings, collection_recorder=collection)

    results = retriever.retrieve_many(tickets, embedding_batch_size=2, query_batch_size=2)

    assert len(results) == 5
    assert all(
        result.status == "unavailable"
        and result.reason_code == "retrieval_provider_or_index_error"
        for result in results[:2]
    )
    assert all(result.status != "unavailable" for result in results[2:])
    assert len(embeddings.calls) == 3
    assert len(collection.calls) == 3
