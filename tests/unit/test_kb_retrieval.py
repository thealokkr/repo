"""Checks for project-owned KB metadata and fail-safe passage retrieval."""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from cloudserve_support.classification.ticket_classification import (
    DECISIONS_CLASSIFIER_MODEL,
    TicketClassification,
    TicketClassificationBatch,
)
from cloudserve_support.config import Settings
from cloudserve_support.ingestion.ticket_normalization import normalize_ticket_record
from cloudserve_support.knowledge.kb_source import (
    build_passage_records,
    load_kb_articles,
    load_kb_metadata_manifest,
    passage_records_sha256,
    validate_kb_metadata_coverage,
)
from cloudserve_support.knowledge import retrieval as retrieval_module
from cloudserve_support.knowledge.retrieval import (
    DEFAULT_METADATA_PATH,
    DEFAULT_SOURCE_PATH,
    EMBEDDING_DIMENSIONS,
    MAXIMUM_SOURCES,
    MINIMUM_SIMILARITY,
    QUERY_EMBEDDING_MAX_RETRIES,
    QUERY_EMBEDDING_TIMEOUT_SECONDS,
    KnowledgeRetriever,
)
from cloudserve_support.processing.run_tickets import run_ticket_batch
from cloudserve_support.reporting.output_schemas import initialize_output_schemas

TEST_ROUTING_POLICY_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "decisions-routing-policy.json"
)


def test_initial_retrieval_batch_defaults_are_five() -> None:
    """Use the owner's initial measured chunk-size settings in the CLI."""
    assert retrieval_module.DEFAULT_EMBEDDING_BATCH_SIZE == 5
    assert retrieval_module.DEFAULT_QUERY_BATCH_SIZE == 5


class _FakeEmbeddings:
    """Return one fixed-dimension query vector without calling a provider."""

    def create(self, *, model: str, input: str | list[str]) -> SimpleNamespace:
        assert model == "gemini-embedding-001"
        inputs = input if isinstance(input, list) else [input]
        assert all("labels" not in text for text in inputs)
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=index, embedding=[0.0] * EMBEDDING_DIMENSIONS)
                for index in range(len(inputs))
            ]
        )


class _TimeoutOnceEmbeddings:
    """Simulate one stalled provider request followed by a successful response."""

    def __init__(self) -> None:
        self.calls = 0

    def create(self, *, model: str, input: str | list[str]) -> SimpleNamespace:
        self.calls += 1
        assert model == "gemini-embedding-001"
        inputs = input if isinstance(input, list) else [input]
        assert all("labels" not in text for text in inputs)
        if self.calls == 1:
            raise TimeoutError("private stalled-provider detail")
        rows = [
            SimpleNamespace(index=index, embedding=[0.0] * EMBEDDING_DIMENSIONS)
            for index in range(len(inputs))
        ]
        return SimpleNamespace(data=rows)


class _IndexedBatchEmbeddings:
    """Return indexed batch vectors in reverse order to exercise response mapping."""

    def __init__(self, *, invalid_vectors: dict[str, list[float]] | None = None) -> None:
        self.calls: list[list[str]] = []
        self.invalid_vectors = invalid_vectors or {}
        self.fail_texts: set[str] = set()
        self.embeddings = self

    def create(self, *, model: str, input: str | list[str]) -> SimpleNamespace:
        assert model == "gemini-embedding-001"
        assert isinstance(input, list)
        self.calls.append(input)
        assert all("labels" not in text and "history" not in text for text in input)
        if any(text in self.fail_texts for text in input):
            raise RuntimeError("private provider detail")
        rows = []
        for index in reversed(range(len(input))):
            query_text = input[index]
            vector = self.invalid_vectors.get(query_text, [0.0] * EMBEDDING_DIMENSIONS)
            rows.append(SimpleNamespace(index=index, embedding=vector))
        return SimpleNamespace(data=rows)


class _BatchFakeCollection:
    """Return one deterministic ranked passage row per explicit Chroma query vector."""

    def __init__(
        self,
        records: list[dict[str, Any]],
        distances: dict[str, Any],
        *,
        query_records: list[dict[str, Any]] | None = None,
    ) -> None:
        self.records = records
        self.query_records = query_records or records
        self.distances = distances
        self.calls: list[list[list[float]]] = []
        self.fail_calls: set[int] = set()
        self.malformed_calls: dict[int, dict[str, Any]] = {}

    def count(self) -> int:
        return len(self.records)

    def get(self, *, include: list[str]) -> dict[str, list[Any]]:
        assert include == ["documents", "metadatas"]
        return {
            "ids": [record["id"] for record in self.records],
            "documents": [record["document"] for record in self.records],
            "metadatas": [dict(record["metadata"]) for record in self.records],
        }

    def query(
        self,
        *,
        query_embeddings: list[list[float]],
        n_results: int,
        include: list[str],
    ) -> dict[str, list[list[Any]]]:
        assert query_embeddings
        assert all(len(vector) == EMBEDDING_DIMENSIONS for vector in query_embeddings)
        assert n_results == len(self.records)
        assert include == ["documents", "metadatas", "distances"]
        self.calls.append(query_embeddings)
        call_index = len(self.calls)
        if call_index in self.fail_calls:
            raise RuntimeError("private Chroma detail")
        if call_index in self.malformed_calls:
            return self.malformed_calls[call_index]  # type: ignore[return-value]
        ranked = sorted(
            self.query_records,
            key=lambda record: (self.distances.get(record["id"], 0.99), record["id"]),
        )
        rows = {
            "ids": [[record["id"] for record in ranked] for _ in query_embeddings],
            "documents": [[record["document"] for record in ranked] for _ in query_embeddings],
            "metadatas": [
                [dict(record["metadata"]) for record in ranked] for _ in query_embeddings
            ],
            "distances": [
                [self.distances.get(record["id"], 0.99) for record in ranked]
                for _ in query_embeddings
            ],
        }
        return rows


class _FakeCollection:
    """Expose deterministic stored passage records to the Chroma retrieval boundary."""

    def __init__(
        self,
        records: list[dict[str, Any]],
        distances: dict[str, Any],
        *,
        query_records: list[dict[str, Any]] | None = None,
    ) -> None:
        self.records = records
        self.query_records = query_records or records
        self.distances = distances

    def count(self) -> int:
        return len(self.records)

    def get(self, *, include: list[str]) -> dict[str, list[Any]]:
        assert include == ["documents", "metadatas"]
        return {
            "ids": [record["id"] for record in self.records],
            "documents": [record["document"] for record in self.records],
            "metadatas": [dict(record["metadata"]) for record in self.records],
        }

    def query(
        self,
        *,
        query_embeddings: list[list[float]],
        n_results: int,
        include: list[str],
    ) -> dict[str, list[list[Any]]]:
        assert len(query_embeddings) == 1
        assert len(query_embeddings[0]) == EMBEDDING_DIMENSIONS
        assert n_results == len(self.records)
        assert include == ["documents", "metadatas", "distances"]
        ranked = sorted(
            self.query_records,
            key=lambda record: (self.distances.get(record["id"], 0.99), record["id"]),
        )
        return {
            "ids": [[record["id"] for record in ranked]],
            "documents": [[record["document"] for record in ranked]],
            "metadatas": [[dict(record["metadata"]) for record in ranked]],
            "distances": [[self.distances.get(record["id"], 0.99) for record in ranked]],
        }


def _source_records() -> tuple[list[dict[str, Any]], str, str, dict[str, Any]]:
    """Build expected records from the two packaged source files."""
    articles, source_sha256 = load_kb_articles(DEFAULT_SOURCE_PATH)
    manifest, article_metadata, metadata_sha256 = load_kb_metadata_manifest(
        DEFAULT_METADATA_PATH
    )
    records = build_passage_records(
        articles,
        article_metadata,
        metadata_owner=manifest["metadata_owner"],
        metadata_source=manifest["metadata_source"],
        review_interval_days=manifest["review_interval_days"],
    )
    return records, source_sha256, metadata_sha256, manifest


def _retriever(
    tmp_path: Path,
    records: list[dict[str, Any]],
    distances: dict[str, Any],
    *,
    today: date = date(2026, 11, 10),
    index_metadata_sha256: str | None = None,
    collection_records: list[dict[str, Any]] | None = None,
    embedding_client: Any | None = None,
    collection: Any | None = None,
    use_gemini_client: bool = False,
) -> KnowledgeRetriever:
    """Create a test retriever with a source-matched manifest and fake dependencies."""
    expected_records, source_sha256, metadata_sha256, _ = _source_records()
    assert len(records) == len(expected_records)
    index_path = tmp_path / "index"
    index_path.mkdir(parents=True)
    (index_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 3,
                "collection": "cloudserve_kb",
                "source_path": "data/kb/documentation.json",
                "source_sha256": source_sha256,
                "metadata_manifest_path": "data/kb/metadata_manifest.json",
                "metadata_manifest_sha256": index_metadata_sha256 or metadata_sha256,
                "passage_records_sha256": passage_records_sha256(expected_records),
                "chunker_version": "markdown-h2-v1",
                "embedding_provider": "gemini",
                "embedding_model": "gemini-embedding-001",
                "embedding_dimensions": EMBEDDING_DIMENSIONS,
                "distance_metric": "cosine",
                "passage_count": len(expected_records),
                "article_count": 29,
            }
        ),
        encoding="utf-8",
    )
    settings = Settings(
        gemini_api_key="test-key",
        embedding_model="gemini-embedding-001",
        llm_model="gemini-3.8-flash",
    )
    if use_gemini_client:
        fake_embedding_client = None
    elif embedding_client is None:
        fake_embedding_client = SimpleNamespace(embeddings=_FakeEmbeddings())
    else:
        fake_embedding_client = embedding_client
    return KnowledgeRetriever(
        settings,
        index_path=index_path,
        collection=collection or _FakeCollection(
            collection_records or expected_records,
            distances,
            query_records=records,
        ),
        embedding_client=fake_embedding_client,
        today=today,
    )


def _ticket() -> Any:
    """Build one operational ticket with evaluation-only fixture fields."""
    return normalize_ticket_record(
        {
            "ticket_id": "RETRIEVAL-001",
            "channel": "email",
            "subject": "Rollback after a failing deployment",
            "body": "How can I roll back a deployment that failed after release?",
            "labels": {"expected_doc_ids": ["DOC-DEPLOY-001"]},
            "history": {"note": "evaluation history must not enter the query"},
        },
        input_index=0,
    )


def test_project_manifest_supplies_citation_metadata_and_exact_passage_source(
    tmp_path: Path,
) -> None:
    records, _, _, manifest = _source_records()
    target = next(
        record
        for record in records
        if record["metadata"]["doc_id"] == "DOC-DEPLOY-001"
        and record["metadata"]["section"] == "Resolution"
    )
    distances = {record["id"]: 0.9 for record in records}
    distances[target["id"]] = 0.15
    distances[next(record["id"] for record in records if record["metadata"]["doc_id"] == "DOC-DEPLOY-002")] = 0.1
    distances[next(record["id"] for record in records if record["metadata"]["doc_id"] == "DOC-AUTH-003")] = 0.11

    result = _retriever(tmp_path, records, distances).retrieve(_ticket())

    assert result.status == "ready"
    assert result.evidence_ready is True
    assert result.sources[0]["passage_id"] == target["id"]
    assert result.sources[0]["doc_id"] == "DOC-DEPLOY-001"
    assert result.sources[0]["passage_text"] == target["document"].split("\n\n", 1)[1]
    assert result.sources[0]["article_version"] == "capstone-1.0"
    assert result.sources[0]["reviewed_at"] == "2026-11-10"
    assert result.sources[0]["metadata_owner"] == manifest["metadata_owner"]
    assert result.sources[0]["approval_status"] == "approved"
    assert result.sources[0]["similarity"] == 0.85
    assert result.excluded_candidate_counts["approval_status_expired"] == 4
    assert result.excluded_candidate_counts["approval_status_unapproved"] == 4
    assert result.excluded_candidate_counts["below_relevance_threshold"] > 0
    assert len(result.sources) <= MAXIMUM_SOURCES


def test_project_manifest_governs_every_article_with_complete_metadata() -> None:
    articles, _ = load_kb_articles(DEFAULT_SOURCE_PATH)
    manifest, article_metadata, _ = load_kb_metadata_manifest(DEFAULT_METADATA_PATH)

    assert manifest["metadata_owner"].strip()
    assert manifest["metadata_source"].strip()
    assert "capstone" in manifest["authority_scope"].lower()
    assert "production" in manifest["authority_scope"].lower()
    assert manifest["review_procedure"].strip()
    assert manifest["update_rule"].strip()
    assert set(article_metadata) == {article["doc_id"] for article in articles}
    for metadata in article_metadata.values():
        assert metadata["approval_status"] in {"approved", "expired", "draft", "unapproved", "unknown"}
        assert metadata["article_version"].strip()
        assert date.fromisoformat(metadata["reviewed_at"]).isoformat() == metadata["reviewed_at"]


def test_missing_or_malformed_governance_metadata_is_rejected(
    tmp_path: Path,
) -> None:
    _, _, _, manifest = _source_records()
    articles, _ = load_kb_articles(DEFAULT_SOURCE_PATH)
    invalid_manifests: list[tuple[dict[str, Any], str]] = []

    missing_version = json.loads(json.dumps(manifest))
    del missing_version["articles"][0]["article_version"]
    invalid_manifests.append((missing_version, "missing fields"))

    malformed_version = json.loads(json.dumps(manifest))
    malformed_version["articles"][0]["article_version"] = "not a version"
    invalid_manifests.append((malformed_version, "invalid article_version"))

    malformed_date = json.loads(json.dumps(manifest))
    malformed_date["articles"][0]["reviewed_at"] = "2026-13-40"
    invalid_manifests.append((malformed_date, "ISO-8601 date"))

    missing_procedure = json.loads(json.dumps(manifest))
    del missing_procedure["review_procedure"]
    invalid_manifests.append((missing_procedure, "review_procedure"))

    def assert_runtime_has_no_eligible_source(metadata_path: Path, index: int) -> None:
        retriever = KnowledgeRetriever(
            Settings(embedding_model="gemini-embedding-001"),
            metadata_path=metadata_path,
            index_path=tmp_path / f"missing-index-{index}",
            collection=_FakeCollection([], {}),
            embedding_client=SimpleNamespace(embeddings=_FakeEmbeddings()),
        )
        result = retriever.retrieve(_ticket())
        assert result.status == "unavailable"
        assert result.sources == ()

    for index, (invalid_manifest, error_message) in enumerate(invalid_manifests):
        metadata_path = tmp_path / f"invalid-metadata-{index}.json"
        metadata_path.write_text(json.dumps(invalid_manifest), encoding="utf-8")
        with pytest.raises(ValueError, match=error_message):
            load_kb_metadata_manifest(metadata_path)
        assert_runtime_has_no_eligible_source(metadata_path, index)

    missing_article_metadata = json.loads(json.dumps(manifest))
    missing_article_metadata["articles"].pop()
    missing_path = tmp_path / "missing-article-metadata.json"
    missing_path.write_text(json.dumps(missing_article_metadata), encoding="utf-8")
    _, incomplete_article_metadata, _ = load_kb_metadata_manifest(missing_path)
    with pytest.raises(ValueError, match="coverage mismatch"):
        validate_kb_metadata_coverage(articles, incomplete_article_metadata)
    assert_runtime_has_no_eligible_source(missing_path, len(invalid_manifests))


def test_weak_match_returns_explicit_empty_evidence(tmp_path: Path) -> None:
    records, _, _, _ = _source_records()
    result = _retriever(
        tmp_path,
        records,
        {record["id"]: 0.9 for record in records},
    ).retrieve(_ticket())

    assert result.status == "no_hit"
    assert result.reason_code == "below_relevance_threshold"
    assert result.evidence_ready is False
    assert result.sources == ()
    assert result.as_dict()["minimum_similarity"] == MINIMUM_SIMILARITY
    assert result.excluded_candidate_counts["below_relevance_threshold"] > 0


def test_index_freshness_uses_passage_records_not_governance_file_bytes(
    tmp_path: Path,
) -> None:
    records, _, _, _ = _source_records()
    stale_build_file_hash = "5" * 64
    retriever = _retriever(
        tmp_path,
        records,
        {record["id"]: 0.1 for record in records},
        index_metadata_sha256=stale_build_file_hash,
    )

    result = retriever.retrieve(_ticket())

    assert result.status == "ready"
    assert result.metadata_manifest_sha256 == _source_records()[2]


def test_passage_record_digest_is_order_independent_and_detects_stale_content() -> None:
    records, _, _, _ = _source_records()
    digest = passage_records_sha256(records)

    assert passage_records_sha256(list(reversed(records))) == digest
    changed_records = [dict(record) for record in records]
    changed_records[0] = {
        **changed_records[0],
        "document": changed_records[0]["document"] + " stale",
    }
    assert passage_records_sha256(changed_records) != digest


def test_chroma_record_mismatch_makes_index_unavailable(tmp_path: Path) -> None:
    records, _, _, _ = _source_records()
    mutated_records = [
        {
            **record,
            "metadata": dict(record["metadata"]),
        }
        for record in records
    ]
    mutated_records[0]["metadata"]["article_version"] = "capstone-9.9"
    retriever = _retriever(
        tmp_path,
        records,
        {record["id"]: 0.1 for record in records},
        collection_records=mutated_records,
    )

    result = retriever.retrieve(_ticket())

    assert result.status == "unavailable"
    assert result.sources == ()


def test_invalid_cosine_distances_cannot_become_relevant_sources(
    tmp_path: Path,
) -> None:
    records, _, _, _ = _source_records()
    approved_records = [
        record for record in records if record["metadata"]["approval_status"] == "approved"
    ]
    invalid_distances: list[Any] = [-0.01, 2.01, False, True, 10**1000]
    distances: dict[str, Any] = {record["id"]: 0.99 for record in records}
    for record, distance in zip(
        approved_records[: len(invalid_distances)], invalid_distances, strict=True
    ):
        distances[record["id"]] = distance

    result = _retriever(tmp_path, records, distances).retrieve(_ticket())

    assert result.status == "no_hit"
    assert result.reason_code == "below_relevance_threshold"
    assert result.evidence_ready is False
    assert result.sources == ()
    assert result.excluded_candidate_counts["invalid_similarity"] == len(invalid_distances)
    assert result.as_dict()["minimum_similarity"] == MINIMUM_SIMILARITY


def test_missing_metadata_and_failed_provenance_cannot_support_a_source(
    tmp_path: Path,
) -> None:
    records, _, _, _ = _source_records()
    target = next(record for record in records if record["metadata"]["approval_status"] == "approved")
    target["metadata"].pop("article_version")
    distances = {record["id"]: 0.99 for record in records}
    distances[target["id"]] = 0.1
    result = _retriever(
        tmp_path / "missing",
        records,
        distances,
    ).retrieve(_ticket())
    assert result.sources == ()
    assert result.excluded_candidate_counts["missing_required_metadata"] == 1

    records, _, _, _ = _source_records()
    target = next(record for record in records if record["metadata"]["approval_status"] == "approved")
    distances = {record["id"]: 0.99 for record in records}
    distances[target["id"]] = 0.1
    target["metadata"]["content_sha256"] = "0" * 64
    result = _retriever(
        tmp_path / "mismatch",
        records,
        distances,
    ).retrieve(_ticket())
    assert result.sources == ()
    assert result.excluded_candidate_counts["provenance_mismatch"] == 1


def test_stale_review_and_unavailable_index_return_no_sources(tmp_path: Path) -> None:
    records, _, _, _ = _source_records()
    stale = _retriever(
        tmp_path / "stale",
        records,
        {record["id"]: 0.1 for record in records},
        today=date(2028, 1, 1),
    ).retrieve(_ticket())
    assert stale.sources == ()
    assert stale.excluded_candidate_counts["review_expired"] == 108

    unavailable = KnowledgeRetriever(
        Settings(embedding_model="gemini-embedding-001"),
        index_path=tmp_path / "missing-index",
    ).retrieve(_ticket())
    assert unavailable.status == "unavailable"
    assert unavailable.sources == ()
    assert unavailable.evidence_ready is False


def test_retrieval_query_uses_only_operational_ticket_subject_and_body(
    tmp_path: Path,
) -> None:
    records, _, _, _ = _source_records()
    result = _retriever(
        tmp_path,
        records,
        {record["id"]: 0.9 for record in records},
    ).retrieve(_ticket())

    assert result.sources == ()
    assert result.reason_code == "below_relevance_threshold"


def test_empty_and_invalid_queries_return_explicit_no_hit_results(
    tmp_path: Path,
) -> None:
    records, _, _, _ = _source_records()
    retriever = _retriever(
        tmp_path,
        records,
        {record["id"]: 0.9 for record in records},
    )

    empty = retriever.retrieve(replace(_ticket(), subject="  ", body="\n"))
    invalid_type = retriever.retrieve(None)  # type: ignore[arg-type]
    invalid_field = retriever.retrieve(replace(_ticket(), subject=None))  # type: ignore[arg-type]

    assert empty.status == "no_hit"
    assert empty.reason_code == "empty_query"
    assert empty.sources == ()
    for invalid in (invalid_type, invalid_field):
        assert invalid.status == "no_hit"
        assert invalid.reason_code == "invalid_query"
        assert invalid.sources == ()


def test_empty_chroma_response_returns_explicit_zero_candidates(
    tmp_path: Path,
) -> None:
    records, _, _, _ = _source_records()
    retriever = _retriever(
        tmp_path,
        records,
        {record["id"]: 0.9 for record in records},
    )
    retriever.collection = SimpleNamespace(
        count=lambda: len(records),
        query=lambda **_: {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }
    )

    result = retriever.retrieve(_ticket())

    assert result.status == "no_hit", result.as_dict()
    assert result.reason_code == "no_index_candidates"
    assert result.sources == ()
    assert result.eligible_candidate_count == 0


def test_retrieve_many_maps_unordered_embedding_rows_and_keeps_empty_positions(
    tmp_path: Path,
) -> None:
    records, _, _, _ = _source_records()
    target = next(
        record
        for record in records
        if record["metadata"]["doc_id"] == "DOC-DEPLOY-001"
        and record["metadata"]["section"] == "Resolution"
    )
    expired = next(
        record
        for record in records
        if record["metadata"]["doc_id"] == "DOC-DEPLOY-002"
    )
    distances = {record["id"]: 0.99 for record in records}
    distances[target["id"]] = 0.15
    distances[expired["id"]] = 0.01
    embedding_client = _IndexedBatchEmbeddings()
    collection = _BatchFakeCollection(records, distances)
    retriever = _retriever(
        tmp_path,
        records,
        distances,
        embedding_client=embedding_client,
        collection=collection,
    )
    tickets = [
        _ticket(),
        replace(_ticket(), subject=" ", body="\n"),
        replace(_ticket(), ticket_id="RETRIEVAL-002", subject="Second subject"),
        replace(_ticket(), ticket_id="RETRIEVAL-003", subject="Third subject"),
    ]

    results = retriever.retrieve_many(
        tickets,
        embedding_batch_size=2,
        query_batch_size=3,
    )

    assert len(results) == len(tickets)
    assert [result.status for result in results] == ["ready", "no_hit", "ready", "ready"]
    assert results[1].reason_code == "empty_query"
    assert [[source["passage_id"] for source in result.sources] for result in results] == [
        [target["id"]],
        [],
        [target["id"]],
        [target["id"]],
    ]
    assert all(
        source["doc_id"] != "DOC-DEPLOY-002"
        for result in results
        for source in result.sources
    )
    assert [len(call) for call in embedding_client.calls] == [2, 1]
    assert [len(call) for call in collection.calls] == [3]
    assert all("labels" not in text and "history" not in text for call in embedding_client.calls for text in call)


def test_retrieve_many_matches_serial_for_ticket_specific_deterministic_vectors(
    tmp_path: Path,
) -> None:
    """Match exact sources/scores while mapping distinct indexed vectors and query rows."""
    records, _, _, _ = _source_records()
    approved_targets = [
        record
        for record in records
        if record["metadata"]["approval_status"] == "approved"
        and date.fromisoformat(record["metadata"]["reviewed_at"])
        <= date(2026, 11, 10)
        and date.fromisoformat(record["metadata"]["reviewed_at"])
        + timedelta(days=record["metadata"]["review_interval_days"])
        >= date(2026, 11, 10)
    ][:3]
    assert len(approved_targets) == 3
    distances = {record["id"]: 0.99 for record in records}

    class TicketSpecificEmbeddings:
        def __init__(self) -> None:
            self.embeddings = self
            self.calls: list[list[str]] = []

        def create(self, *, model: str, input: str | list[str]) -> SimpleNamespace:
            assert model == "gemini-embedding-001"
            inputs = [input] if isinstance(input, str) else input
            self.calls.append(inputs)
            rows = []
            for response_index in reversed(range(len(inputs))):
                query_number = int(inputs[response_index].split("\n", 1)[0].split()[-1])
                vector = [0.0] * EMBEDDING_DIMENSIONS
                vector[0] = float(query_number + 1)
                rows.append(
                    SimpleNamespace(index=response_index, embedding=vector)
                )
            return SimpleNamespace(data=rows)

    class TicketSpecificCollection(_BatchFakeCollection):
        def query(
            self,
            *,
            query_embeddings: list[list[float]],
            n_results: int,
            include: list[str],
        ) -> dict[str, list[list[Any]]]:
            assert n_results == len(self.records)
            assert include == ["documents", "metadatas", "distances"]
            self.calls.append(query_embeddings)
            rows = []
            for vector in query_embeddings:
                query_number = int(vector[0]) - 1
                target = approved_targets[query_number]
                ranked = [target, *(record for record in self.records if record != target)]
                rows.append(
                    (
                        [record["id"] for record in ranked],
                        [record["document"] for record in ranked],
                        [dict(record["metadata"]) for record in ranked],
                        [0.15, *([0.99] * (len(ranked) - 1))],
                    )
                )
            return {
                "ids": [row[0] for row in rows],
                "documents": [row[1] for row in rows],
                "metadatas": [row[2] for row in rows],
                "distances": [row[3] for row in rows],
            }

    tickets = [
        replace(
            _ticket(),
            ticket_id=f"RETRIEVAL-{query_number:03d}",
            subject=f"Query {query_number}",
            body=f"Body {query_number}",
        )
        for query_number in range(3)
    ]
    serial_embeddings = TicketSpecificEmbeddings()
    serial_collection = TicketSpecificCollection(records, distances)
    serial_retriever = _retriever(
        tmp_path / "serial",
        records,
        distances,
        embedding_client=serial_embeddings,
        collection=serial_collection,
    )
    batch_embeddings = TicketSpecificEmbeddings()
    batch_collection = TicketSpecificCollection(records, distances)
    batch_retriever = _retriever(
        tmp_path / "bulk",
        records,
        distances,
        embedding_client=batch_embeddings,
        collection=batch_collection,
    )

    serial_results = tuple(serial_retriever.retrieve(ticket) for ticket in tickets)
    batch_results = batch_retriever.retrieve_many(
        tickets,
        embedding_batch_size=3,
        query_batch_size=3,
    )

    expected_passages = [record["id"] for record in approved_targets]
    assert [result.status for result in serial_results] == ["ready"] * 3
    assert [result.status for result in batch_results] == ["ready"] * 3
    assert [result.sources[0]["passage_id"] for result in serial_results] == expected_passages
    assert [result.sources[0]["passage_id"] for result in batch_results] == expected_passages
    assert batch_results == serial_results
    for ticket_index, (serial_result, batch_result) in enumerate(
        zip(serial_results, batch_results, strict=True)
    ):
        serial_vector = serial_collection.calls[ticket_index][0]
        batch_vector = batch_collection.calls[0][ticket_index]
        assert serial_vector == batch_vector
        assert [source["passage_id"] for source in serial_result.sources] == [
            source["passage_id"] for source in batch_result.sources
        ]
        assert all(
            abs(serial_source["similarity"] - batch_source["similarity"]) <= 1e-6
            for serial_source, batch_source in zip(
                serial_result.sources,
                batch_result.sources,
                strict=True,
            )
        )
    assert [result.sources[0]["similarity"] for result in batch_results] == [0.85] * 3
    assert [len(call) for call in serial_embeddings.calls] == [1, 1, 1]
    assert [len(call) for call in batch_embeddings.calls] == [3]
    assert [len(call) for call in batch_collection.calls] == [3]


def test_retrieve_many_keeps_later_embedding_and_chroma_chunks_after_failures(
    tmp_path: Path,
) -> None:
    records, _, _, _ = _source_records()
    distances = {record["id"]: 0.99 for record in records}
    embedding_client = _IndexedBatchEmbeddings()
    tickets = [
        replace(_ticket(), ticket_id=f"BATCH-{index}", subject=f"Subject {index}")
        for index in range(4)
    ]
    embedding_client.fail_texts.add(f"Subject 0\n{tickets[0].body}")
    collection = _BatchFakeCollection(records, distances)
    retriever = _retriever(
        tmp_path,
        records,
        distances,
        embedding_client=embedding_client,
        collection=collection,
    )

    embedding_results = retriever.retrieve_many(
        tickets,
        embedding_batch_size=2,
        query_batch_size=2,
    )
    assert [result.status for result in embedding_results] == [
        "unavailable",
        "unavailable",
        "no_hit",
        "no_hit",
    ]
    assert all(
        result.reason_code == "retrieval_provider_or_index_error"
        for result in embedding_results[:2]
    )
    assert len(embedding_client.calls) == 2
    assert len(collection.calls) == 1

    embedding_client = _IndexedBatchEmbeddings()
    collection = _BatchFakeCollection(records, distances)
    collection.fail_calls.add(1)
    retriever = _retriever(
        tmp_path / "chroma-failure",
        records,
        distances,
        embedding_client=embedding_client,
        collection=collection,
    )
    chroma_results = retriever.retrieve_many(
        tickets,
        embedding_batch_size=4,
        query_batch_size=2,
    )
    assert [result.status for result in chroma_results] == [
        "unavailable",
        "unavailable",
        "no_hit",
        "no_hit",
    ]
    assert [result.reason_code for result in chroma_results[:2]] == [
        "retrieval_provider_or_index_error",
        "retrieval_provider_or_index_error",
    ]
    assert len(collection.calls) == 2


def test_retrieve_many_isolates_bad_embedding_vectors_and_bad_query_rows(
    tmp_path: Path,
) -> None:
    records, _, _, _ = _source_records()
    distances = {record["id"]: 0.99 for record in records}
    first_ticket = _ticket()
    second_ticket = replace(first_ticket, ticket_id="BAD-VECTOR", subject="bad vector")
    invalid_vectors = {
        f"bad vector\n{second_ticket.body}": [math.nan] * EMBEDDING_DIMENSIONS,
    }
    wrong_dimension_ticket = replace(
        first_ticket,
        ticket_id="WRONG-DIMENSION",
        subject="wrong dimension",
    )
    invalid_vectors[f"wrong dimension\n{wrong_dimension_ticket.body}"] = [
        0.0
    ] * (EMBEDDING_DIMENSIONS - 1)
    embedding_client = _IndexedBatchEmbeddings(invalid_vectors=invalid_vectors)
    collection = _BatchFakeCollection(records, distances)
    retriever = _retriever(
        tmp_path,
        records,
        distances,
        embedding_client=embedding_client,
        collection=collection,
    )
    vector_results = retriever.retrieve_many(
        [first_ticket, second_ticket, wrong_dimension_ticket],
        embedding_batch_size=3,
        query_batch_size=3,
    )
    assert [result.status for result in vector_results] == [
        "no_hit",
        "unavailable",
        "unavailable",
    ]
    assert vector_results[1].reason_code == "invalid_embedding_response"
    assert vector_results[2].reason_code == "invalid_embedding_response"
    assert [len(call) for call in collection.calls] == [1]

    embedding_client = _IndexedBatchEmbeddings()
    collection = _BatchFakeCollection(records, distances)
    target_record_count = len(records)
    collection.malformed_calls[1] = {
        "ids": [[record["id"] for record in records], [record["id"] for record in records]],
        "documents": [
            [record["document"] for record in records],
            [record["document"] for record in records[:-1]],
        ],
        "metadatas": [
            [record["metadata"] for record in records],
            [record["metadata"] for record in records],
        ],
        "distances": [[0.99] * target_record_count, [0.99] * target_record_count],
    }
    retriever = _retriever(
        tmp_path / "row-shape",
        records,
        distances,
        embedding_client=embedding_client,
        collection=collection,
    )
    row_results = retriever.retrieve_many(
        [first_ticket, second_ticket],
        embedding_batch_size=2,
        query_batch_size=2,
    )
    assert [result.status for result in row_results] == ["no_hit", "unavailable"]
    assert row_results[1].reason_code == "invalid_index_response"


@pytest.mark.parametrize(
    "indexes",
    [
        [0, 0],
        [0, 2],
        [-1, 1],
        [False, 1],
    ],
)
def test_parse_embedding_response_rejects_ambiguous_indexes(indexes: list[object]) -> None:
    response = SimpleNamespace(
        data=[
            SimpleNamespace(index=index, embedding=[0.0] * EMBEDDING_DIMENSIONS)
            for index in indexes
        ]
    )

    with pytest.raises(ValueError, match="indexes"):
        retrieval_module._parse_embedding_response(response, 2)


def test_parse_embedding_response_rejects_a_missing_response_position() -> None:
    response = SimpleNamespace(
        data=[SimpleNamespace(index=0, embedding=[0.0] * EMBEDDING_DIMENSIONS)]
    )

    with pytest.raises(ValueError, match="count"):
        retrieval_module._parse_embedding_response(response, 2)


def test_parse_chroma_query_rows_rejects_unalignable_outer_result_counts() -> None:
    response = {
        "ids": [[], []],
        "documents": [[]],
        "metadatas": [[], []],
        "distances": [[], []],
    }

    with pytest.raises(ValueError, match="rows do not align"):
        retrieval_module._parse_chroma_query_rows(response, 2)


def test_retrieve_many_validates_batch_sizes_and_empty_input_before_provider_calls(
    tmp_path: Path,
) -> None:
    records, _, _, _ = _source_records()
    distances = {record["id"]: 0.99 for record in records}
    embedding_client = _IndexedBatchEmbeddings()
    collection = _BatchFakeCollection(records, distances)
    retriever = _retriever(
        tmp_path,
        records,
        distances,
        embedding_client=embedding_client,
        collection=collection,
    )

    assert retriever.retrieve_many([]) == ()
    for options in (
        {"embedding_batch_size": 0},
        {"embedding_batch_size": True},
        {"query_batch_size": -1},
        {"query_batch_size": 1.5},
    ):
        with pytest.raises(ValueError, match="positive integer"):
            retriever.retrieve_many([_ticket()], **options)  # type: ignore[arg-type]
    assert embedding_client.calls == []
    assert collection.calls == []


def test_gemini_query_embedding_client_uses_bounded_timeout_and_no_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records, _, _, _ = _source_records()
    options: dict[str, Any] = {}

    class _GenAIClient:
        def __init__(self, **client_options: Any) -> None:
            options.update(client_options)
            self.embeddings = _FakeEmbeddings()

    monkeypatch.setattr(retrieval_module.genai, "Client", _GenAIClient)
    _retriever(
        tmp_path,
        records,
        {record["id"]: 0.9 for record in records},
        use_gemini_client=True,
    )

    assert options == {
        "api_key": "test-key",
    }


def test_l2_normalize_rejects_zero_and_non_finite_norms() -> None:
    """Ensure _l2_normalize refuses to normalize zero or non-finite vectors."""
    with pytest.raises(ValueError, match="zero or non-finite L2 norm"):
        retrieval_module._l2_normalize([0.0] * EMBEDDING_DIMENSIONS)
    with pytest.raises(ValueError, match="non-finite"):
        retrieval_module._l2_normalize([float("nan")] * EMBEDDING_DIMENSIONS)
    with pytest.raises(ValueError, match="non-finite"):
        retrieval_module._l2_normalize([float("inf")] * EMBEDDING_DIMENSIONS)


@pytest.mark.parametrize(
    "invalid_vector",
    [
        [0.0] * EMBEDDING_DIMENSIONS,
        [float("nan")] * EMBEDDING_DIMENSIONS,
        [float("inf")] * EMBEDDING_DIMENSIONS,
    ],
)
def test_gemini_query_embedding_invalid_norm_returns_safe_unavailable(
    tmp_path: Path,
    invalid_vector: list[float],
) -> None:
    records, _, _, _ = _source_records()

    class _InvalidGeminiClient:
        class Models:
            @staticmethod
            def embed_content(*args: Any, **kwargs: Any) -> Any:
                return SimpleNamespace(
                    embeddings=[SimpleNamespace(values=invalid_vector)]
                )

        models = Models()

    retriever = _retriever(
        tmp_path,
        records,
        {record["id"]: 0.1 for record in records},
        embedding_client=_InvalidGeminiClient(),
    )

    result = retriever.retrieve(_ticket())

    assert result.status == "unavailable"
    assert result.reason_code == "retrieval_provider_or_index_error"
    assert result.sources == ()
    assert result.evidence_ready is False


@pytest.mark.parametrize(
    "invalid_vector",
    [
        [0.0] * EMBEDDING_DIMENSIONS,
        [float("nan")] * EMBEDDING_DIMENSIONS,
    ],
)
def test_gemini_batch_embedding_invalid_norm_degrades_chunk_safely(
    tmp_path: Path,
    invalid_vector: list[float],
) -> None:
    records, _, _, _ = _source_records()

    class _InvalidBatchGeminiClient:
        class Models:
            @staticmethod
            def embed_content(*args: Any, **kwargs: Any) -> Any:
                contents = kwargs.get("contents", [])
                return SimpleNamespace(
                    embeddings=[
                        SimpleNamespace(values=invalid_vector) for _ in contents
                    ]
                )

        models = Models()

    retriever = _retriever(
        tmp_path,
        records,
        {record["id"]: 0.1 for record in records},
        embedding_client=_InvalidBatchGeminiClient(),
    )

    results = retriever.retrieve_many([_ticket(), _ticket()], embedding_batch_size=2)

    assert len(results) == 2
    for result in results:
        assert result.status == "unavailable"
        assert result.reason_code == "retrieval_provider_or_index_error"
        assert result.sources == ()
        assert result.evidence_ready is False


def test_query_embedding_timeout_returns_safe_unavailable(
    tmp_path: Path,
) -> None:
    records, _, _, _ = _source_records()
    embeddings = _TimeoutOnceEmbeddings()
    retriever = _retriever(
        tmp_path,
        records,
        {record["id"]: 0.1 for record in records},
        embedding_client=SimpleNamespace(embeddings=embeddings),
    )

    result = retriever.retrieve(_ticket())

    assert result.status == "unavailable"
    assert result.reason_code == "query_embedding_timeout"
    assert result.sources == ()
    assert result.evidence_ready is False
    assert "private stalled-provider detail" not in result.reason


def test_batch_escalates_after_query_embedding_timeout_and_processes_later_ticket(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Isolate each ticket so the second query is a later chunk after the first timeout.
    monkeypatch.setattr(
        "cloudserve_support.processing.run_tickets.DEFAULT_EMBEDDING_BATCH_SIZE",
        1,
    )
    monkeypatch.setattr(
        "cloudserve_support.processing.run_tickets.DEFAULT_QUERY_BATCH_SIZE",
        1,
    )
    records, _, _, _ = _source_records()
    target = next(
        record
        for record in records
        if record["metadata"]["doc_id"] == "DOC-DEPLOY-001"
        and record["metadata"]["approval_status"] == "approved"
    )
    distances = {record["id"]: 0.99 for record in records}
    distances[target["id"]] = 0.1
    embeddings = _TimeoutOnceEmbeddings()
    retriever = _retriever(
        tmp_path / "retrieval-index",
        records,
        distances,
        embedding_client=SimpleNamespace(embeddings=embeddings),
    )
    schema_directory = tmp_path / "schemas" / "outputs"
    initialize_output_schemas(schema_directory)
    monkeypatch.setattr(
        "cloudserve_support.reporting.output_schemas.OUTPUT_SCHEMA_DIRECTORY",
        schema_directory,
    )
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("DRAFT_GENERATION_PROVIDER", "OFFLINE")
    monkeypatch.setattr(
        "cloudserve_support.processing.run_tickets.classify_ticket_batches",
        lambda tickets, _settings: (
            TicketClassificationBatch(
                0,
                tuple(
                    TicketClassification(
                        intent="deployment_failure",
                        urgency="medium",
                        confidence=0.95,
                        provider="gemini",
                        model_name=DECISIONS_CLASSIFIER_MODEL,
                        provider_status="classified",
                    )
                    for _ticket in tickets
                ),
                0.0,
            ),
        ),
    )
    input_path = tmp_path / "two-tickets.json"
    input_path.write_text(
        json.dumps(
            [
                {
                    "ticket_id": "T-EMBEDDING-TIMEOUT",
                    "channel": "email",
                    "subject": "Rollback after a failing deployment",
                    "body": "How can I roll back a deployment that failed after release?",
                },
                {
                    "ticket_id": "T-AFTER-EMBEDDING-TIMEOUT",
                    "channel": "email",
                    "subject": "Rollback after a failing deployment",
                    "body": "How can I roll back a deployment that failed after release?",
                },
            ]
        ),
        encoding="utf-8",
    )

    run_id, output_directory, metrics = run_ticket_batch(
        input_path=input_path,
        output_root=tmp_path / "outputs",
        database_path=tmp_path / "runtime.sqlite3",
        retriever=retriever,
        policy_path=TEST_ROUTING_POLICY_PATH,
    )

    results = [
        json.loads(line)
        for line in (output_directory / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [result["ticket_id"] for result in results] == [
        "T-EMBEDDING-TIMEOUT",
        "T-AFTER-EMBEDDING-TIMEOUT",
    ]
    assert results[0]["routing"]["retrieval"]["status"] == "unavailable"
    assert results[0]["routing"]["retrieval"]["reason_code"] == "query_embedding_timeout"
    assert results[0]["sources"] == []
    assert results[0]["route"] == "escalate"
    assert results[1]["routing"]["retrieval"]["status"] == "ready"
    assert results[1]["sources"][0]["doc_id"] == "DOC-DEPLOY-001"
    assert results[1]["status"] == "completed"
    assert embeddings.calls == 2
    assert metrics["total_count"] == metrics["completed_count"] == 2
    assert metrics["failed_count"] == 0
    assert "private stalled-provider detail" not in str(results[0])
    connection = sqlite3.connect(tmp_path / "runtime.sqlite3")
    try:
        assert connection.execute(
            "SELECT COUNT(*) FROM ticket_runs WHERE run_id = ?", (run_id,)
        ).fetchone()[0] == 2
        assert connection.execute(
            "SELECT COUNT(*) FROM audit_events WHERE run_id = ?", (run_id,)
        ).fetchone()[0] == 10
    finally:
        connection.close()
