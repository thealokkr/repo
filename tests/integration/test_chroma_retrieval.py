"""Exercise whole-article retrieval through real packaged Chroma and ticket data."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import chromadb
import jsonschema
import pytest

from cloudserve_support.classification.ticket_classification import (
    DECISIONS_CLASSIFIER_MODEL,
    TicketClassification,
    TicketClassificationBatch,
)
from cloudserve_support.config import Settings
from cloudserve_support.generation.draft_generation import OfflineDraftGenerationProvider
from cloudserve_support.ingestion.ticket_normalization import normalize_ticket_record
from cloudserve_support.knowledge.kb_source import (
    CHUNKER_VERSION,
    build_passage_records,
    load_kb_articles,
    load_kb_metadata_manifest,
    passage_records_sha256,
)
from cloudserve_support.processing import run_tickets
from cloudserve_support.processing.run_tickets import run_ticket_batch
from cloudserve_support.reporting.output_schemas import initialize_output_schemas
from cloudserve_support.knowledge.retrieval import (
    COLLECTION_NAME,
    DEFAULT_INDEX_PATH,
    DEFAULT_METADATA_PATH,
    DEFAULT_SOURCE_PATH,
    EMBEDDING_DIMENSIONS,
    MAXIMUM_SOURCES,
    MINIMUM_SIMILARITY,
    KnowledgeRetriever,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEVELOPMENT_TICKETS_PATH = REPOSITORY_ROOT / "data" / "tickets" / "development_tickets.json"


class PackagedArticleEmbeddings:
    """Use stored article vectors as deterministic inputs at the external embedding boundary."""

    def __init__(self, vectors_by_query: dict[str, list[float]]) -> None:
        self.vectors_by_query = vectors_by_query
        self.calls: list[list[str]] = []
        self.embeddings = self

    def create(self, *, model: str, input: str | list[str]) -> SimpleNamespace:
        assert model == "gemini-embedding-001"
        query_texts = [input] if isinstance(input, str) else input
        self.calls.append(query_texts)
        rows = [
            SimpleNamespace(index=index, embedding=self.vectors_by_query[query_text])
            for index, query_text in enumerate(query_texts)
        ]
        return SimpleNamespace(data=rows)


def _operational_ticket(ticket_id: str) -> Any:
    """Normalize one real fixture ticket without copying evaluation labels or history."""
    records = json.loads(DEVELOPMENT_TICKETS_PATH.read_text(encoding="utf-8"))
    record = next(item for item in records if item["ticket_id"] == ticket_id)
    operational_fields = {
        field: record[field]
        for field in ("ticket_id", "channel", "subject", "body")
    }
    return normalize_ticket_record(operational_fields, input_index=0)


def _health_check_ticket() -> Any:
    """Select a real deployment health-check ticket using only operational text."""
    records = json.loads(DEVELOPMENT_TICKETS_PATH.read_text(encoding="utf-8"))
    record = next(
        item
        for item in records
        if "health check timeout" in (item.get("subject", "") + " " + item["body"]).casefold()
    )
    operational_fields = {
        field: record[field]
        for field in ("ticket_id", "channel", "subject", "body")
    }
    return normalize_ticket_record(operational_fields, input_index=0)


def _article_vectors() -> tuple[dict[str, list[float]], dict[str, str]]:
    """Read exact article vectors from the packaged collection without modifying it."""
    articles, _ = load_kb_articles(DEFAULT_SOURCE_PATH)
    metadata_manifest, metadata_by_id, _ = load_kb_metadata_manifest(DEFAULT_METADATA_PATH)
    records = build_passage_records(
        articles,
        metadata_by_id,
        metadata_owner=metadata_manifest["metadata_owner"],
        metadata_source=metadata_manifest["metadata_source"],
        review_interval_days=metadata_manifest["review_interval_days"],
    )
    record_ids = {record["metadata"]["doc_id"]: record["id"] for record in records}
    client = chromadb.PersistentClient(path=str(DEFAULT_INDEX_PATH / "chroma"))
    collection = client.get_collection(COLLECTION_NAME, embedding_function=None)
    vectors: dict[str, list[float]] = {}
    for doc_id in ("DOC-ACCT-001", "DOC-DEPLOY-001", "DOC-DEPLOY-002"):
        stored = collection.get(ids=[record_ids[doc_id]], include=["embeddings"])
        assert stored["ids"] == [record_ids[doc_id]]
        vector = stored["embeddings"][0].tolist()
        assert len(vector) == EMBEDDING_DIMENSIONS
        vectors[doc_id] = vector
    return vectors, record_ids


def _retriever(
    embedding_client: Any,
    *,
    today: date = date(2026, 11, 10),
) -> KnowledgeRetriever:
    """Open the actual packaged index with fixed embeddings for deterministic tests."""
    return KnowledgeRetriever(
        Settings(
            gemini_api_key="test-key",
            embedding_model="gemini-embedding-001",
            llm_model="gemini-3.8-flash",
        ),
        embedding_client=embedding_client,
        today=today,
    )


def _temporary_chroma_retriever(
    tmp_path: Path,
    query_text: str,
    *,
    primary_doc_ids: set[str] | None = None,
    metadata_path: Path = DEFAULT_METADATA_PATH,
    today: date = date(2026, 11, 10),
) -> KnowledgeRetriever:
    """Build a disposable real Chroma collection with controlled article ranking."""
    articles, source_sha256 = load_kb_articles(DEFAULT_SOURCE_PATH)
    metadata_manifest, metadata_by_id, metadata_sha256 = load_kb_metadata_manifest(
        metadata_path
    )
    records = build_passage_records(
        articles,
        metadata_by_id,
        metadata_owner=metadata_manifest["metadata_owner"],
        metadata_source=metadata_manifest["metadata_source"],
        review_interval_days=metadata_manifest["review_interval_days"],
    )
    index_path = tmp_path / "index"
    index_path.mkdir(parents=True)
    (index_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 3,
                "collection": COLLECTION_NAME,
                "source_path": "data/kb/documentation.json",
                "source_sha256": source_sha256,
                "metadata_manifest_path": "data/kb/metadata_manifest.json",
                "metadata_manifest_sha256": metadata_sha256,
                "passage_records_sha256": passage_records_sha256(records),
                "chunker_version": CHUNKER_VERSION,
                "embedding_provider": "gemini",
                "embedding_model": "gemini-embedding-001",
                "embedding_dimensions": EMBEDDING_DIMENSIONS,
                "distance_metric": "cosine",
                "article_count": len(articles),
                "passage_count": len(records),
            }
        ),
        encoding="utf-8",
    )
    client = chromadb.PersistentClient(path=str(index_path / "chroma"))
    collection = client.create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
        embedding_function=None,
    )
    if primary_doc_ids is None:
        primary_doc_ids = {
            "DOC-AUTH-001",
            "DOC-AUTH-002",
            "DOC-AUTH-004",
            "DOC-DEPLOY-001",
            "DOC-DEPLOY-003",
            "DOC-API-001",
        }
    vectors = []
    for index, record in enumerate(records):
        vector = [0.0] * EMBEDDING_DIMENSIONS
        vector[0 if record["metadata"]["doc_id"] in primary_doc_ids else index + 1] = 1.0
        vectors.append(vector)
    collection.add(
        ids=[record["id"] for record in records],
        documents=[record["document"] for record in records],
        metadatas=[record["metadata"] for record in records],
        embeddings=vectors,
    )
    embedding_client = PackagedArticleEmbeddings(
        {query_text: [1.0, *([0.0] * (EMBEDDING_DIMENSIONS - 1))]}
    )
    return KnowledgeRetriever(
        Settings(
            gemini_api_key="test-key",
            embedding_model="gemini-embedding-001",
            llm_model="gemini-3.8-flash",
        ),
        metadata_path=metadata_path,
        index_path=index_path,
        collection=collection,
        embedding_client=embedding_client,
        today=today,
    )


def test_real_chroma_caps_primary_articles_at_three(tmp_path: Path) -> None:
    ticket = _operational_ticket("DEV-0017")
    query_text = f"{ticket.subject}\n{ticket.body}".strip()
    retriever = _temporary_chroma_retriever(tmp_path, query_text)

    for result in (retriever.retrieve(ticket), retriever.retrieve_many([ticket])[0]):
        assert result.status == "ready", result.as_dict()
        assert len(result.sources) == MAXIMUM_SOURCES == 3
        assert [source["doc_id"] for source in result.sources] == [
            "DOC-API-001",
            "DOC-AUTH-001",
            "DOC-AUTH-002",
        ]
        assert [source["similarity"] for source in result.sources] == sorted(
            (source["similarity"] for source in result.sources), reverse=True
        )
        assert all(source["similarity"] >= MINIMUM_SIMILARITY for source in result.sources)


def test_real_chroma_returns_whole_articles_and_filtered_related_docs() -> None:
    vectors, _ = _article_vectors()
    account_ticket = _operational_ticket("DEV-0017")
    deployment_ticket = _health_check_ticket()
    expired_ticket = _operational_ticket("DEV-0004")
    query_texts = [
        f"{ticket.subject}\n{ticket.body}".strip()
        for ticket in (account_ticket, deployment_ticket, expired_ticket)
    ]
    embeddings = PackagedArticleEmbeddings(
        dict(zip(query_texts, (vectors["DOC-ACCT-001"], vectors["DOC-DEPLOY-001"], vectors["DOC-DEPLOY-002"]), strict=True))
    )
    retriever = _retriever(embeddings)
    assert retriever.unavailable_reason is None

    tickets = [account_ticket, deployment_ticket, expired_ticket]
    results = retriever.retrieve_many(
        tickets,
        embedding_batch_size=3,
        query_batch_size=3,
    )
    serial_results = tuple(retriever.retrieve(ticket) for ticket in tickets)
    assert results == serial_results
    articles, _ = load_kb_articles(DEFAULT_SOURCE_PATH)
    article_content = {article["doc_id"]: article["content"] for article in articles}

    assert len(results) == 3
    for result in results[:2]:
        assert result.status == "ready", result.as_dict()
        assert 1 <= len(result.sources) <= MAXIMUM_SOURCES
        assert [source["similarity"] for source in result.sources] == sorted(
            (source["similarity"] for source in result.sources), reverse=True
        )
        for source in result.sources + result.related_sources:
            assert source["passage_text"] == article_content[source["doc_id"]]
            assert source["approval_status"] == "approved"
            assert source["reviewed_at"] <= "2026-11-10"
            assert all(
                field in source
                for field in (
                    "passage_id",
                    "doc_id",
                    "title",
                    "section",
                    "category",
                    "applies_to",
                    "metadata_owner",
                    "metadata_source",
                    "article_version",
                    "review_interval_days",
                    "content_sha256",
                    "source_ref",
                    "metadata_manifest_sha256",
                )
            )
        for source in result.sources:
            assert source["similarity"] >= result.minimum_similarity
            assert all(
                field in source
                for field in (
                    "passage_id",
                    "doc_id",
                    "title",
                    "section",
                    "category",
                    "applies_to",
                    "metadata_owner",
                    "metadata_source",
                    "article_version",
                    "review_interval_days",
                    "content_sha256",
                    "source_ref",
                    "metadata_manifest_sha256",
                )
            )
    account, deployment, expired = results
    account_primary_ids = {source["doc_id"] for source in account.sources}
    account_related_ids = {source["doc_id"] for source in account.related_sources}
    assert "DOC-ACCT-001" in account_primary_ids
    assert len(account_primary_ids) == len(account.sources)
    assert len(account_related_ids) == len(account.related_sources)
    assert all(
        source["similarity"] is None and source["minimum_similarity"] is None
        for source in account.related_sources
    )
    assert account_primary_ids.isdisjoint(account_related_ids)
    assert "DOC-AUTH-003" not in account_primary_ids | account_related_ids
    assert "DOC-SEC-001" in account_related_ids | account_primary_ids

    deployment_primary_ids = {source["doc_id"] for source in deployment.sources}
    deployment_related_ids = {source["doc_id"] for source in deployment.related_sources}
    assert "DOC-DEPLOY-001" in deployment_primary_ids
    assert len(deployment_primary_ids) == len(deployment.sources)
    assert len(deployment_related_ids) == len(deployment.related_sources)
    assert all(
        source["similarity"] is None and source["minimum_similarity"] is None
        for source in deployment.related_sources
    )
    assert deployment_primary_ids.isdisjoint(deployment_related_ids)
    assert "DOC-DEPLOY-002" not in deployment_primary_ids | deployment_related_ids
    assert "DOC-DEPLOY-004" in deployment_related_ids | deployment_primary_ids
    assert account.excluded_candidate_counts["approval_status_unapproved"] == 1
    assert deployment.excluded_candidate_counts["approval_status_expired"] == 1
    assert expired.excluded_candidate_counts.get("approval_status_expired", 0) >= 1
    assert "DOC-DEPLOY-002" not in {
        source["doc_id"] for source in expired.sources + expired.related_sources
    }
    assert all(not result.related_sources for result in results if result.status != "ready")
    assert [len(call) for call in embeddings.calls] == [3, 1, 1, 1]
    assert [query for call in embeddings.calls for query in call] == query_texts * 2


def test_related_sources_are_explicit_and_deduplicated_across_hits(
    tmp_path: Path,
) -> None:
    account_ticket = _operational_ticket("DEV-0017")
    account_query = f"{account_ticket.subject}\n{account_ticket.body}".strip()
    account_retriever = _temporary_chroma_retriever(
        tmp_path / "account",
        account_query,
        primary_doc_ids={"DOC-ACCT-001"},
    )

    account_result = account_retriever.retrieve(account_ticket)

    assert {source["doc_id"] for source in account_result.sources} == {"DOC-ACCT-001"}
    assert {source["doc_id"] for source in account_result.related_sources} == {
        "DOC-SEC-001"
    }
    assert account_result.excluded_candidate_counts["approval_status_unapproved"] == 1
    assert "DOC-AUTH-003" not in {
        source["doc_id"] for source in account_result.related_sources
    }

    deployment_ticket = _operational_ticket("DEV-0006")
    deployment_query = f"{deployment_ticket.subject}\n{deployment_ticket.body}".strip()
    deployment_retriever = _temporary_chroma_retriever(
        tmp_path / "deployment",
        deployment_query,
        primary_doc_ids={"DOC-DEPLOY-001", "DOC-ACCT-002"},
    )

    deployment_result = deployment_retriever.retrieve(deployment_ticket)
    related_doc_ids = [source["doc_id"] for source in deployment_result.related_sources]

    assert {source["doc_id"] for source in deployment_result.sources} == {
        "DOC-DEPLOY-001",
        "DOC-ACCT-002",
    }
    assert set(related_doc_ids) == {"DOC-ACCT-001", "DOC-DEPLOY-004"}
    assert len(related_doc_ids) == len(set(related_doc_ids))
    assert "DOC-DEPLOY-002" not in related_doc_ids
    assert deployment_result.excluded_candidate_counts["approval_status_expired"] == 1


def test_review_expiry_excludes_primary_and_related_articles(tmp_path: Path) -> None:
    ticket = _operational_ticket("DEV-0017")
    query_text = f"{ticket.subject}\n{ticket.body}".strip()
    vectors, _ = _article_vectors()
    primary_expired = _retriever(
        PackagedArticleEmbeddings({query_text: vectors["DOC-ACCT-001"]}),
        today=date(2027, 11, 11),
    ).retrieve(ticket)

    assert primary_expired.status == "no_hit"
    assert primary_expired.sources == primary_expired.related_sources == ()
    assert primary_expired.excluded_candidate_counts["review_expired"] >= 1

    metadata_manifest, _, _ = load_kb_metadata_manifest(DEFAULT_METADATA_PATH)
    sec_article_metadata = next(
        row for row in metadata_manifest["articles"] if row["doc_id"] == "DOC-SEC-001"
    )
    sec_article_metadata["reviewed_at"] = "2025-11-10"
    metadata_path = tmp_path / "metadata_manifest.json"
    metadata_path.write_text(json.dumps(metadata_manifest), encoding="utf-8")
    related_expired_retriever = _temporary_chroma_retriever(
        tmp_path / "related-expired",
        query_text,
        primary_doc_ids={"DOC-ACCT-001"},
        metadata_path=metadata_path,
        today=date(2026, 11, 11),
    )

    related_expired = related_expired_retriever.retrieve(ticket)

    assert related_expired.status == "ready"
    assert {source["doc_id"] for source in related_expired.sources} == {"DOC-ACCT-001"}
    assert related_expired.related_sources == ()
    assert related_expired.excluded_candidate_counts["review_expired"] == 1


def test_empty_query_and_unavailable_manifest_have_no_primary_or_related_sources(
    tmp_path: Path,
) -> None:
    vectors, _ = _article_vectors()
    ticket = _operational_ticket("DEV-0017")
    query_text = f"{ticket.subject}\n{ticket.body}".strip()
    embeddings = PackagedArticleEmbeddings({query_text: vectors["DOC-ACCT-001"]})
    retriever = _retriever(embeddings)

    empty = retriever.retrieve(replace(ticket, subject=" ", body="\n"))
    assert empty.status == "no_hit"
    assert empty.sources == empty.related_sources == ()
    assert embeddings.calls == []

    tampered_index = tmp_path / "index"
    tampered_index.mkdir()
    manifest = json.loads((DEFAULT_INDEX_PATH / "manifest.json").read_text(encoding="utf-8"))
    manifest["passage_records_sha256"] = "0" * 64
    (tampered_index / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    unavailable = KnowledgeRetriever(
        Settings(
            gemini_api_key="test-key",
            embedding_model="gemini-embedding-001",
            llm_model="gemini-3.8-flash",
        ),
        index_path=tampered_index,
        embedding_client=embeddings,
    ).retrieve(ticket)
    assert unavailable.status == "unavailable"
    assert unavailable.sources == unavailable.related_sources == ()


def test_missing_key_and_embedding_timeout_return_empty_results() -> None:
    ticket = _operational_ticket("DEV-0017")
    missing_key = KnowledgeRetriever(Settings()).retrieve(ticket)

    class TimedOutEmbeddings:
        def __init__(self) -> None:
            self.embeddings = self

        def create(self, *, model: str, input: str | list[str]) -> Any:
            raise TimeoutError("private provider detail")

    timed_out = _retriever(TimedOutEmbeddings()).retrieve(ticket)

    assert missing_key.status == "unavailable"
    assert missing_key.reason_code == "query_embeddings_unavailable"
    assert missing_key.sources == missing_key.related_sources == ()
    assert timed_out.status == "unavailable"
    assert timed_out.reason_code == "query_embedding_timeout"
    assert timed_out.sources == timed_out.related_sources == ()
    assert "private provider detail" not in timed_out.reason


def test_ticket_batch_exports_real_chroma_article_snapshots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify real ticket-to-Chroma retrieval snapshots reach local exports and SQLite."""
    ticket = _operational_ticket("DEV-0017")
    query_text = f"{ticket.subject}\n{ticket.body}".strip()
    vectors, _ = _article_vectors()
    retriever = _retriever(
        PackagedArticleEmbeddings({query_text: vectors["DOC-ACCT-001"]})
    )
    input_path = tmp_path / "development-ticket.json"
    input_path.write_text(
        json.dumps(
            [
                {
                    "ticket_id": ticket.ticket_id,
                    "channel": ticket.channel,
                    "subject": ticket.subject,
                    "body": ticket.body,
                }
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    monkeypatch.setenv("DRAFT_GENERATION_PROVIDER", "OFFLINE")
    schema_directory = tmp_path / "schemas" / "outputs"
    initialize_output_schemas(schema_directory)
    monkeypatch.setattr(
        "cloudserve_support.reporting.output_schemas.OUTPUT_SCHEMA_DIRECTORY",
        schema_directory,
    )

    def local_classifier(tickets: list[Any], _settings: Settings) -> tuple[Any, ...]:
        classifications = tuple(
            TicketClassification(
                intent="account_access",
                urgency="low",
                confidence=0.95,
                provider="gemini",
                model_name=DECISIONS_CLASSIFIER_MODEL,
                provider_status="classified",
            )
            for _ticket in tickets
        )
        return (TicketClassificationBatch(0, classifications, 0.0),)

    monkeypatch.setattr(run_tickets, "classify_ticket_batches", local_classifier)
    run_id, output_directory, metrics = run_ticket_batch(
        input_path=input_path,
        output_root=tmp_path / "outputs",
        database_path=tmp_path / "runtime.sqlite3",
        policy_path=REPOSITORY_ROOT / "conf" / "frozen_routing_policy.json",
        retriever=retriever,
        generator=OfflineDraftGenerationProvider(),
    )
    result = json.loads(
        (output_directory / "results.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    article_content = {
        article["doc_id"]: article["content"]
        for article in load_kb_articles(DEFAULT_SOURCE_PATH)[0]
    }

    assert result["ticket_id"] == "DEV-0017"
    assert result["status"] == "completed"
    assert result["sources"]
    assert "DOC-ACCT-001" in {source["doc_id"] for source in result["sources"]}
    assert all(
        source["passage_text"] == article_content[source["doc_id"]]
        for source in result["sources"]
    )
    assert metrics["total_count"] == metrics["completed_count"] == 1
    with sqlite3.connect(tmp_path / "runtime.sqlite3") as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM ticket_runs WHERE run_id = ?", (run_id,)
        ).fetchone()[0] == 1
        assert connection.execute(
            "SELECT COUNT(*) FROM audit_events WHERE run_id = ? AND event_type = 'retrieval'",
            (run_id,),
        ).fetchone()[0] == 1


def test_retrieval_diagnostics_remain_result_schema_compatible() -> None:
    vectors, _ = _article_vectors()
    ticket = _operational_ticket("DEV-0017")
    query_text = f"{ticket.subject}\n{ticket.body}".strip()
    result = _retriever(
        PackagedArticleEmbeddings({query_text: vectors["DOC-ACCT-001"]})
    ).retrieve(ticket)
    schema = json.loads(
        (REPOSITORY_ROOT / "schemas" / "outputs" / "result.schema.json").read_text(
            encoding="utf-8"
        )
    )
    retrieval_schema = schema["properties"]["routing"]["properties"]["retrieval"]

    jsonschema.validate(result.as_dict(), retrieval_schema)


def test_minimum_similarity_floor_excludes_candidates_just_below_threshold(
    tmp_path: Path,
) -> None:
    """Verify real Chroma cosine candidates just below MINIMUM_SIMILARITY are excluded."""
    assert MINIMUM_SIMILARITY == 0.57
    ticket = _operational_ticket("DEV-0017")
    query_text = f"{ticket.subject}\n{ticket.body}".strip()

    below_sim = MINIMUM_SIMILARITY - 0.01
    below_vec = [0.0] * EMBEDDING_DIMENSIONS
    below_vec[0] = below_sim
    below_vec[EMBEDDING_DIMENSIONS - 1] = (1.0 - below_sim**2) ** 0.5

    above_sim = MINIMUM_SIMILARITY + 0.01
    above_vec = [0.0] * EMBEDDING_DIMENSIONS
    above_vec[0] = above_sim
    above_vec[EMBEDDING_DIMENSIONS - 1] = (1.0 - above_sim**2) ** 0.5

    articles, source_sha256 = load_kb_articles(DEFAULT_SOURCE_PATH)
    metadata_manifest, metadata_by_id, metadata_sha256 = load_kb_metadata_manifest(
        DEFAULT_METADATA_PATH
    )
    records = build_passage_records(
        articles,
        metadata_by_id,
        metadata_owner=metadata_manifest["metadata_owner"],
        metadata_source=metadata_manifest["metadata_source"],
        review_interval_days=metadata_manifest["review_interval_days"],
    )
    index_path = tmp_path / "index"
    index_path.mkdir(parents=True)
    (index_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 3,
                "collection": COLLECTION_NAME,
                "source_path": "data/kb/documentation.json",
                "source_sha256": source_sha256,
                "metadata_manifest_path": "data/kb/metadata_manifest.json",
                "metadata_manifest_sha256": metadata_sha256,
                "passage_records_sha256": passage_records_sha256(records),
                "chunker_version": CHUNKER_VERSION,
                "embedding_provider": "gemini",
                "embedding_model": "gemini-embedding-001",
                "embedding_dimensions": EMBEDDING_DIMENSIONS,
                "distance_metric": "cosine",
                "article_count": len(articles),
                "passage_count": len(records),
            }
        ),
        encoding="utf-8",
    )
    client = chromadb.PersistentClient(path=str(index_path / "chroma"))
    collection = client.create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
        embedding_function=None,
    )
    vectors = []
    for index, record in enumerate(records):
        vector = [0.0] * EMBEDDING_DIMENSIONS
        vector[0 if record["metadata"]["doc_id"] == "DOC-AUTH-001" else index + 1] = 1.0
        vectors.append(vector)
    collection.add(
        ids=[record["id"] for record in records],
        documents=[record["document"] for record in records],
        metadatas=[record["metadata"] for record in records],
        embeddings=vectors,
    )

    def _run_with_vector(vec: list[float]) -> Any:
        return KnowledgeRetriever(
            Settings(
                gemini_api_key="test-key",
                embedding_model="gemini-embedding-001",
                llm_model="gemini-3.8-flash",
            ),
            index_path=index_path,
            embedding_client=PackagedArticleEmbeddings({query_text: vec}),
            today=date(2026, 11, 10),
        ).retrieve(ticket)

    below_result = _run_with_vector(below_vec)
    assert below_result.status == "no_hit"
    assert below_result.reason_code == "below_relevance_threshold"
    assert below_result.sources == ()
    assert below_result.minimum_similarity == MINIMUM_SIMILARITY
    assert below_result.excluded_candidate_counts["below_relevance_threshold"] == 27

    above_result = _run_with_vector(above_vec)
    assert above_result.status == "ready"
    assert above_result.reason_code == "relevant_sources_found"
    assert [source["doc_id"] for source in above_result.sources] == ["DOC-AUTH-001"]
    assert above_result.excluded_candidate_counts["below_relevance_threshold"] == 26

