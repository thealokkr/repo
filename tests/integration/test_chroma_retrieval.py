"""Exercise source-resolved ranked retrieval against a local Chroma collection."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import chromadb
import pytest

from cloudserve_support.config import Settings
from cloudserve_support.ingestion.ticket_normalization import normalize_ticket_record
from cloudserve_support.knowledge.kb_source import (
    CHUNKER_VERSION,
    build_passage_records,
    load_kb_articles,
    load_kb_metadata_manifest,
    passage_records_sha256,
)
from cloudserve_support.knowledge.retrieval import (
    COLLECTION_NAME,
    EMBEDDING_DIMENSIONS,
    KnowledgeRetriever,
)


class _QueryEmbeddings:
    """Return a deterministic query vector without contacting an embedding API."""

    def __init__(self, expected_query: str) -> None:
        self.expected_query = expected_query

    def create(self, *, model: str, input: str) -> SimpleNamespace:
        assert model == "text-embedding-3-small"
        assert input == self.expected_query
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[1.0, *([0.0] * (EMBEDDING_DIMENSIONS - 1))])]
        )


class _StoredPassageEmbedding:
    """Return one packaged passage vector so retrieval can be tested offline."""

    def __init__(self, vector: list[float]) -> None:
        self.vector = vector

    def create(self, *, model: str, input: str) -> SimpleNamespace:
        assert model == "text-embedding-3-small"
        assert "labels" not in input
        return SimpleNamespace(data=[SimpleNamespace(embedding=self.vector)])


def test_normalized_ticket_returns_ranked_chroma_passages_from_packaged_source(
    tmp_path: Path,
) -> None:
    """A real Chroma query returns exact, ranked passages from the packaged KB."""
    source_path = Path(__file__).resolve().parents[2] / "data" / "kb" / "documentation.json"
    metadata_path = (
        Path(__file__).resolve().parents[2] / "data" / "kb" / "metadata_manifest.json"
    )
    articles, source_sha256 = load_kb_articles(source_path)
    metadata_manifest, article_metadata, metadata_sha256 = load_kb_metadata_manifest(
        metadata_path
    )
    records = build_passage_records(
        articles,
        article_metadata,
        metadata_owner=metadata_manifest["metadata_owner"],
        metadata_source=metadata_manifest["metadata_source"],
        review_interval_days=metadata_manifest["review_interval_days"],
    )
    eligible_targets = [
        next(
            record
            for record in records
            if record["metadata"]["doc_id"] == doc_id
            and record["metadata"]["approval_status"] == "approved"
        )
        for doc_id in ("DOC-DEPLOY-001", "DOC-API-001", "DOC-BILL-001")
    ]

    index_path = tmp_path / "index"
    index_path.mkdir()
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
                "embedding_provider": "openai",
                "embedding_model": "text-embedding-3-small",
                "embedding_dimensions": EMBEDDING_DIMENSIONS,
                "distance_metric": "cosine",
                "passage_count": len(records),
                "article_count": len(articles),
            }
        ),
        encoding="utf-8",
    )

    chroma_client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
        embedding_function=None,
    )
    target_vectors = {
        eligible_targets[0]["id"]: [1.0, *([0.0] * (EMBEDDING_DIMENSIONS - 1))],
        eligible_targets[1]["id"]: [0.8, 0.6, *([0.0] * (EMBEDDING_DIMENSIONS - 2))],
        eligible_targets[2]["id"]: [0.6, 0.8, *([0.0] * (EMBEDDING_DIMENSIONS - 2))],
    }
    fallback_vectors: list[list[float]] = []
    next_fallback_dimension = 3
    for record in records:
        if record["id"] in target_vectors:
            fallback_vectors.append(target_vectors[record["id"]])
            continue
        vector = [0.0] * EMBEDDING_DIMENSIONS
        vector[next_fallback_dimension] = 1.0
        next_fallback_dimension += 1
        fallback_vectors.append(vector)
    collection.add(
        ids=[record["id"] for record in records],
        documents=[record["document"] for record in records],
        metadatas=[record["metadata"] for record in records],
        embeddings=fallback_vectors,
    )

    raw_ticket = {
        "ticket_id": "CS-010-CHROMA-001",
        "channel": "email",
        "subject": "Rollback after a failing deployment",
        "body": "How can I roll back a deployment after release?",
        "labels": {"expected_doc_ids": ["this must not be queried"]},
        "history": {"note": "this must not be queried"},
    }
    ticket = normalize_ticket_record(raw_ticket, input_index=0)
    query_text = f"{ticket.subject}\n{ticket.body}"
    retriever = KnowledgeRetriever(
        Settings(
            openai_api_key="test-key",
            embedding_model="text-embedding-3-small",
            llm_model="gpt-6-luna",
        ),
        source_path=source_path,
        metadata_path=metadata_path,
        index_path=index_path,
        embedding_client=SimpleNamespace(embeddings=_QueryEmbeddings(query_text)),
        collection=collection,
        today=date(2026, 11, 10),
    )

    result = retriever.retrieve(ticket)

    assert result.status == "ready", result.as_dict()
    assert [source["passage_id"] for source in result.sources] == [
        record["id"] for record in eligible_targets
    ]
    assert [source["doc_id"] for source in result.sources] == [
        record["metadata"]["doc_id"] for record in eligible_targets
    ]
    assert [source["passage_text"] for source in result.sources] == [
        record["document"].split("\n\n", 1)[1] for record in eligible_targets
    ]
    assert [source["similarity"] for source in result.sources] == pytest.approx(
        [1.0, 0.8, 0.6]
    )


def test_packaged_index_returns_current_source_ids_without_embedding_service() -> None:
    """The checked-in index is source-current and can retrieve with a local test vector."""
    repo_root = Path(__file__).resolve().parents[2]
    source_path = repo_root / "data" / "kb" / "documentation.json"
    metadata_path = repo_root / "data" / "kb" / "metadata_manifest.json"
    articles, _ = load_kb_articles(source_path)
    metadata_manifest, article_metadata, _ = load_kb_metadata_manifest(metadata_path)
    records = build_passage_records(
        articles,
        article_metadata,
        metadata_owner=metadata_manifest["metadata_owner"],
        metadata_source=metadata_manifest["metadata_source"],
        review_interval_days=metadata_manifest["review_interval_days"],
    )
    target = next(
        record
        for record in records
        if record["metadata"]["doc_id"] == "DOC-DEPLOY-001"
        and record["metadata"]["approval_status"] == "approved"
    )
    chroma_client = chromadb.PersistentClient(path=str(repo_root / "data" / "index" / "chroma"))
    collection = chroma_client.get_collection(COLLECTION_NAME, embedding_function=None)
    stored = collection.get(ids=[target["id"]], include=["embeddings"])
    retriever = KnowledgeRetriever(
        Settings(
            openai_api_key="test-key",
            embedding_model="text-embedding-3-small",
            llm_model="gpt-6-luna",
        ),
        source_path=source_path,
        metadata_path=metadata_path,
        index_path=repo_root / "data" / "index",
        embedding_client=SimpleNamespace(
            embeddings=_StoredPassageEmbedding(stored["embeddings"][0].tolist())
        ),
        collection=collection,
        today=date(2026, 11, 10),
    )
    ticket = normalize_ticket_record(
        {
            "ticket_id": "PACKAGED-INDEX-001",
            "channel": "email",
            "subject": "Deployment rollback",
            "body": "How do I roll back a deployment?",
            "labels": {"expected_doc_ids": ["DOC-DEPLOY-001"]},
        },
        input_index=0,
    )

    result = retriever.retrieve(ticket)

    assert result.status == "ready", result.as_dict()
    assert result.sources[0]["doc_id"] == "DOC-DEPLOY-001"
    assert result.sources[0]["passage_id"] in {
        record["id"] for record in records if record["metadata"]["doc_id"] == "DOC-DEPLOY-001"
    }
