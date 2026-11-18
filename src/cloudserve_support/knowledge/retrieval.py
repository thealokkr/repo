"""Query the current, project-governed KB and return traceable passage snapshots."""

from __future__ import annotations

import json
import math
from numbers import Real
from collections import Counter
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Sequence

import chromadb
from google import genai

from cloudserve_support.config import Settings
from cloudserve_support.domain.ticket import CanonicalTicket
from cloudserve_support.knowledge.kb_source import (
    CHUNKER_VERSION,
    build_passage_records,
    load_kb_articles,
    load_kb_metadata_manifest,
    passage_records_sha256,
    validate_kb_metadata_coverage,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE_PATH = REPO_ROOT / "data" / "kb" / "documentation.json"
DEFAULT_METADATA_PATH = REPO_ROOT / "data" / "kb" / "metadata_manifest.json"
DEFAULT_INDEX_PATH = REPO_ROOT / "data" / "index"
COLLECTION_NAME = "cloudserve_kb"
EMBEDDING_DIMENSIONS = 768
MINIMUM_SIMILARITY = 0.57
MAXIMUM_SOURCES = 3
QUERY_EMBEDDING_TIMEOUT_SECONDS = 15.0
QUERY_EMBEDDING_MAX_RETRIES = 0
# Initial measured five-ticket settings; the downstream 5/50 CLI acceptance verifies them.
DEFAULT_EMBEDDING_BATCH_SIZE = 5
DEFAULT_QUERY_BATCH_SIZE = 5
REQUIRED_PASSAGE_METADATA = (
    "doc_id",
    "title",
    "category",
    "applies_to",
    "section",
    "related_docs",
    "content_sha256",
    "approval_status",
    "approval_status_source",
    "metadata_owner",
    "metadata_source",
    "article_version",
    "reviewed_at",
    "review_interval_days",
)


@dataclass(frozen=True)
class RetrievalResult:
    """A retrieval outcome with a safe explanation and citation-ready snapshots."""

    status: str
    reason_code: str
    reason: str
    sources: tuple[dict[str, Any], ...]
    eligible_candidate_count: int
    excluded_candidate_counts: dict[str, int]
    minimum_similarity: float
    embedding_model: str
    source_sha256: str | None
    metadata_manifest_sha256: str | None
    related_sources: tuple[dict[str, Any], ...] = ()

    @property
    def evidence_ready(self) -> bool:
        """Return whether at least one current relevant source can support a draft."""
        return self.status == "ready" and bool(self.sources)

    def as_dict(self) -> dict[str, Any]:
        """Return the structured retrieval evidence written with each ticket result."""
        return {
            "status": self.status,
            "reason_code": self.reason_code,
            "reason": self.reason,
            "eligible_candidate_count": self.eligible_candidate_count,
            "excluded_candidate_counts": dict(self.excluded_candidate_counts),
            "minimum_similarity": self.minimum_similarity,
            "embedding_model": self.embedding_model,
            "source_sha256": self.source_sha256,
            "metadata_manifest_sha256": self.metadata_manifest_sha256,
        }


class KnowledgeRetriever:
    """Validate the packaged index once, then retrieve eligible source snapshots."""

    def __init__(
        self,
        settings: Settings,
        *,
        source_path: Path = DEFAULT_SOURCE_PATH,
        metadata_path: Path = DEFAULT_METADATA_PATH,
        index_path: Path = DEFAULT_INDEX_PATH,
        embedding_client: Any | None = None,
        collection: Any | None = None,
        today: date | None = None,
    ) -> None:
        self.settings = settings
        self.today = today or date.today()
        self.embedding_client = embedding_client
        self.collection = collection
        self.source_sha256: str | None = None
        self.metadata_manifest_sha256: str | None = None
        self.expected_passages: dict[str, dict[str, Any]] = {}
        self.article_records_by_doc_id: dict[str, dict[str, Any]] = {}
        self.article_metadata: dict[str, dict[str, Any]] = {}
        self.review_interval_days = 0
        self.unavailable_reason: tuple[str, str] | None = None
        try:
            self._load_current_index(
                source_path=source_path,
                metadata_path=metadata_path,
                index_path=index_path,
            )
        except Exception:
            self.unavailable_reason = (
                "kb_index_unavailable",
                "The packaged knowledge index is unavailable or does not match its source files.",
            )
            return

        if self.embedding_client is None and not settings.gemini_api_key:
            self.unavailable_reason = (
                "query_embeddings_unavailable",
                "Query embeddings are unavailable; no knowledge source was used.",
            )
            return
        if self.embedding_client is None:
            try:
                self.embedding_client = genai.Client(
                    api_key=settings.gemini_api_key,
                )
            except Exception:
                self.unavailable_reason = (
                    "query_embeddings_unavailable",
                    "Query embeddings are unavailable; no knowledge source was used.",
                )

    def _load_current_index(
        self,
        *,
        source_path: Path,
        metadata_path: Path,
        index_path: Path,
    ) -> None:
        """Check source hashes, metadata coverage, embedding model, and collection."""
        articles, self.source_sha256 = load_kb_articles(source_path)
        metadata_manifest, self.article_metadata, self.metadata_manifest_sha256 = (
            load_kb_metadata_manifest(metadata_path)
        )
        validate_kb_metadata_coverage(articles, self.article_metadata)
        self.review_interval_days = metadata_manifest["review_interval_days"]
        expected_records = build_passage_records(
            articles,
            self.article_metadata,
            metadata_owner=metadata_manifest["metadata_owner"],
            metadata_source=metadata_manifest["metadata_source"],
            review_interval_days=self.review_interval_days,
        )
        self.expected_passages = {record["id"]: record for record in expected_records}
        self.article_records_by_doc_id = {
            record["metadata"]["doc_id"]: record for record in expected_records
        }
        expected_passage_records_hash = passage_records_sha256(expected_records)

        index_manifest = json.loads((index_path / "manifest.json").read_text(encoding="utf-8"))
        if (
            index_manifest.get("schema_version") != 3
            or index_manifest.get("collection") != COLLECTION_NAME
            or index_manifest.get("source_path") != "data/kb/documentation.json"
            or index_manifest.get("source_sha256") != self.source_sha256
            or index_manifest.get("metadata_manifest_path") != "data/kb/metadata_manifest.json"
            or index_manifest.get("metadata_manifest_sha256") != self.metadata_manifest_sha256
            or not isinstance(index_manifest.get("metadata_manifest_sha256"), str)
            or len(index_manifest["metadata_manifest_sha256"]) != 64
            or any(
                character not in "0123456789abcdef"
                for character in index_manifest["metadata_manifest_sha256"]
            )
            or index_manifest.get("passage_records_sha256") != expected_passage_records_hash
            or index_manifest.get("chunker_version") != CHUNKER_VERSION
            or index_manifest.get("embedding_provider") != "gemini"
            or index_manifest.get("embedding_model") != self.settings.embedding_model
            or index_manifest.get("embedding_dimensions") != EMBEDDING_DIMENSIONS
            or index_manifest.get("distance_metric") != "cosine"
            or index_manifest.get("passage_count") != len(expected_records)
            or index_manifest.get("article_count") != len(articles)
        ):
            raise ValueError("Index manifest does not match current source and query settings.")

        if self.collection is None:
            chroma_client = chromadb.PersistentClient(path=str(index_path / "chroma"))
            self.collection = chroma_client.get_collection(
                name=COLLECTION_NAME,
                embedding_function=None,
            )
        if self.collection.count() != len(expected_records):
            raise ValueError("Chroma passage count does not match the source corpus.")
        stored_records = self.collection.get(include=["documents", "metadatas"])
        stored_ids = stored_records.get("ids")
        stored_documents = stored_records.get("documents")
        stored_metadatas = stored_records.get("metadatas")
        if not (
            isinstance(stored_ids, list)
            and isinstance(stored_documents, list)
            and isinstance(stored_metadatas, list)
            and len(stored_ids) == len(stored_documents) == len(stored_metadatas)
        ):
            raise ValueError("Chroma passage records are incomplete.")
        indexed_records = [
            {"id": record_id, "document": document, "metadata": metadata}
            for record_id, document, metadata in zip(
                stored_ids, stored_documents, stored_metadatas, strict=True
            )
        ]
        if passage_records_sha256(indexed_records) != expected_passage_records_hash:
            raise ValueError("Chroma passage records do not match the current source corpus.")

    def retrieve(self, ticket: CanonicalTicket) -> RetrievalResult:
        """Return up to three ranked articles or an explicit safe no-evidence result."""
        if self.unavailable_reason:
            return self._empty_result("unavailable", *self.unavailable_reason)

        if not isinstance(ticket, CanonicalTicket):
            return self._empty_result(
                "no_hit",
                "invalid_query",
                "The ticket query is invalid; no knowledge source was used.",
            )
        if not isinstance(ticket.subject, str) or not isinstance(ticket.body, str):
            return self._empty_result(
                "no_hit",
                "invalid_query",
                "The ticket query is invalid; no knowledge source was used.",
            )

        query_text = "\n".join(
            part for part in (ticket.subject.strip(), ticket.body.strip()) if part
        )
        if not query_text:
            return self._empty_result(
                "no_hit",
                "empty_query",
                "The ticket has no searchable subject or body text.",
            )

        try:
            query_embedding = _embed_query_vector(
                self.embedding_client,
                model=self.settings.embedding_model,
                query_text=query_text,
            )
            if len(query_embedding) != EMBEDDING_DIMENSIONS:
                raise ValueError("Query embedding dimensions do not match the index.")
        except Exception as error:
            if "timeout" in type(error).__name__.casefold():
                return self._empty_result(
                    "unavailable",
                    "query_embedding_timeout",
                    "Query embedding request timed out; no knowledge source was used.",
                )
            return self._empty_result(
                "unavailable",
                "retrieval_provider_or_index_error",
                "Knowledge retrieval failed; no source was used.",
            )

        try:
            query_result = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=self.collection.count(),
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            return self._empty_result(
                "unavailable",
                "retrieval_provider_or_index_error",
                "Knowledge retrieval failed; no source was used.",
            )

        ids = _first_query_list(query_result, "ids")
        documents = _first_query_list(query_result, "documents")
        metadatas = _first_query_list(query_result, "metadatas")
        distances = _first_query_list(query_result, "distances")
        if not (len(ids) == len(documents) == len(metadatas) == len(distances)):
            return self._empty_result(
                "unavailable",
                "invalid_index_response",
                "Knowledge retrieval returned incomplete index records; no source was used.",
            )

        return self._result_from_candidates(ids, documents, metadatas, distances)

    def retrieve_many(
        self,
        tickets: Sequence[CanonicalTicket],
        *,
        embedding_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
        query_batch_size: int = DEFAULT_QUERY_BATCH_SIZE,
    ) -> tuple[RetrievalResult, ...]:
        """Retrieve ordered ticket results using bounded embedding and Chroma batches."""
        _require_positive_batch_size(embedding_batch_size, "embedding_batch_size")
        _require_positive_batch_size(query_batch_size, "query_batch_size")
        if not tickets:
            return ()
        if self.unavailable_reason:
            return tuple(
                self._empty_result("unavailable", *self.unavailable_reason)
                for _ in tickets
            )

        results: list[RetrievalResult | None] = [None] * len(tickets)
        searchable: list[tuple[int, str]] = []
        for position, ticket in enumerate(tickets):
            query_text, query_error = _ticket_query_text(ticket)
            if query_error is not None:
                status, reason_code, reason = query_error
                results[position] = self._empty_result(status, reason_code, reason)
            else:
                searchable.append((position, query_text))

        vectors: dict[int, list[float]] = {}
        for chunk_start in range(0, len(searchable), embedding_batch_size):
            chunk = searchable[chunk_start : chunk_start + embedding_batch_size]
            try:
                chunk_texts = [text for _, text in chunk]
                vectors_chunk = _embed_query_vectors(
                    self.embedding_client,
                    model=self.settings.embedding_model,
                    query_texts=chunk_texts,
                )
            except Exception as error:
                reason_code = (
                    "query_embedding_timeout"
                    if "timeout" in type(error).__name__.casefold()
                    else "retrieval_provider_or_index_error"
                )
                reason = (
                    "Query embedding request timed out; no knowledge source was used."
                    if reason_code == "query_embedding_timeout"
                    else "Knowledge retrieval failed; no source was used."
                )
                for position, _ in chunk:
                    results[position] = self._empty_result("unavailable", reason_code, reason)
                continue

            for local_position, (position, _) in enumerate(chunk):
                vector = vectors_chunk[local_position]
                if not _is_valid_embedding_vector(vector):
                    results[position] = self._empty_result(
                        "unavailable",
                        "invalid_embedding_response",
                        "Query embedding response was invalid; no knowledge source was used.",
                    )
                else:
                    vectors[position] = vector

        query_positions = list(vectors)
        for chunk_start in range(0, len(query_positions), query_batch_size):
            chunk_positions = query_positions[chunk_start : chunk_start + query_batch_size]
            query_vectors = [vectors[position] for position in chunk_positions]
            try:
                query_result = self.collection.query(
                    query_embeddings=query_vectors,
                    n_results=self.collection.count(),
                    include=["documents", "metadatas", "distances"],
                )
            except Exception:
                for position in chunk_positions:
                    results[position] = self._empty_result(
                        "unavailable",
                        "retrieval_provider_or_index_error",
                        "Knowledge retrieval failed; no source was used.",
                    )
                continue
            try:
                query_rows = _parse_chroma_query_rows(query_result, len(chunk_positions))
            except Exception:
                for position in chunk_positions:
                    results[position] = self._empty_result(
                        "unavailable",
                        "invalid_index_response",
                        "Knowledge retrieval returned incomplete index records; no source was used.",
                    )
                continue

            for position, query_row in zip(chunk_positions, query_rows, strict=True):
                if query_row is None:
                    results[position] = self._empty_result(
                        "unavailable",
                        "invalid_index_response",
                        "Knowledge retrieval returned incomplete index records; no source was used.",
                    )
                else:
                    results[position] = self._result_from_candidates(*query_row)

        if any(result is None for result in results):
            raise RuntimeError("Bulk retrieval failed to resolve every ticket position.")
        return tuple(result for result in results if result is not None)

    def _result_from_candidates(
        self,
        ids: list[Any],
        documents: list[Any],
        metadatas: list[Any],
        distances: list[Any],
    ) -> RetrievalResult:
        """Apply current provenance, freshness, and similarity rules to one query row."""
        if not (len(ids) == len(documents) == len(metadatas) == len(distances)):
            return self._empty_result(
                "unavailable",
                "invalid_index_response",
                "Knowledge retrieval returned incomplete index records; no source was used.",
            )

        excluded = Counter()
        excluded_article_ids: set[str] = set()
        current_eligible_count = 0
        eligible: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
        for passage_id, document, metadata, distance in zip(
            ids, documents, metadatas, distances, strict=True
        ):
            reason = self._source_exclusion_reason(passage_id, document, metadata)
            if reason:
                excluded[reason] += 1
                if isinstance(passage_id, str):
                    excluded_article_ids.add(passage_id)
                continue
            current_eligible_count += 1
            if (
                isinstance(distance, bool)
                or not isinstance(distance, (int, float))
                or not 0 <= distance <= 2
            ):
                excluded["invalid_similarity"] += 1
                continue
            similarity = 1.0 - float(distance)
            if similarity < MINIMUM_SIMILARITY:
                excluded["below_relevance_threshold"] += 1
                continue
            record = self.expected_passages[passage_id]
            eligible.append((similarity, self._source_snapshot(passage_id, record, similarity), record))

        eligible.sort(key=lambda item: (-item[0], item[1]["passage_id"]))
        primary_hits = eligible[:MAXIMUM_SOURCES]
        snapshots = tuple(source for _, source, _ in primary_hits)
        if snapshots:
            related_sources = self._related_source_snapshots(
                primary_hits,
                excluded,
                excluded_article_ids,
            )
            return RetrievalResult(
                status="ready",
                reason_code="relevant_sources_found",
                reason="Current capstone-approved articles met the relevance threshold.",
                sources=snapshots,
                related_sources=related_sources,
                eligible_candidate_count=current_eligible_count,
                excluded_candidate_counts=dict(sorted(excluded.items())),
                minimum_similarity=MINIMUM_SIMILARITY,
                embedding_model=self.settings.embedding_model,
                source_sha256=self.source_sha256,
                metadata_manifest_sha256=self.metadata_manifest_sha256,
            )

        if not ids:
            reason_code = "no_index_candidates"
            reason = "The knowledge index returned no candidates; no source was used."
        elif excluded.get("below_relevance_threshold"):
            reason_code = "below_relevance_threshold"
            reason = (
                "No current eligible passage met the minimum relevance threshold; "
                "no source was used."
            )
        elif sum(excluded.values()):
            reason_code = "no_eligible_current_sources"
            reason = "No current approved source was eligible for this query."
        else:
            reason_code = "no_relevant_source"
            reason = "No relevant source was found; no source was used."
        return RetrievalResult(
            status="no_hit",
            reason_code=reason_code,
            reason=reason,
            sources=(),
            eligible_candidate_count=current_eligible_count,
            excluded_candidate_counts=dict(sorted(excluded.items())),
            minimum_similarity=MINIMUM_SIMILARITY,
            embedding_model=self.settings.embedding_model,
            source_sha256=self.source_sha256,
            metadata_manifest_sha256=self.metadata_manifest_sha256,
        )

    def _source_exclusion_reason(
        self,
        passage_id: Any,
        document: Any,
        metadata: Any,
    ) -> str | None:
        """Check index records against source text and project metadata before use."""
        if not isinstance(passage_id, str):
            return "unresolvable_source_reference"
        if not isinstance(metadata, dict):
            return "missing_required_metadata"
        missing = [key for key in REQUIRED_PASSAGE_METADATA if key not in metadata]
        if missing:
            return "missing_required_metadata"
        expected = self.expected_passages.get(passage_id)
        if expected is None:
            return "unresolvable_source_reference"
        if document != expected["document"]:
            return "passage_content_mismatch"
        expected_metadata = expected["metadata"]
        if any(metadata.get(key) != value for key, value in expected_metadata.items()):
            return "provenance_mismatch"
        status = metadata.get("approval_status")
        if status != "approved":
            return f"approval_status_{status or 'unknown'}"
        reviewed_at = date.fromisoformat(metadata["reviewed_at"])
        if reviewed_at > self.today:
            return "review_date_in_future"
        current_until = reviewed_at + timedelta(days=metadata["review_interval_days"])
        if current_until < self.today:
            return "review_expired"
        return None

    def _source_snapshot(
        self,
        passage_id: str,
        record: dict[str, Any],
        similarity: float | None,
    ) -> dict[str, Any]:
        """Build source snapshots; related articles intentionally have no vector score."""
        metadata = record["metadata"]
        return {
            "passage_id": passage_id,
            "doc_id": metadata["doc_id"],
            "title": metadata["title"],
            "section": metadata["section"],
            "passage_text": record["document"].split("\n\n", 1)[1],
            "category": metadata["category"],
            "applies_to": metadata["applies_to"],
            "similarity": similarity,
            "minimum_similarity": MINIMUM_SIMILARITY if similarity is not None else None,
            "approval_status": metadata["approval_status"],
            "approval_status_source": metadata["approval_status_source"],
            "metadata_owner": metadata["metadata_owner"],
            "metadata_source": metadata["metadata_source"],
            "article_version": metadata["article_version"],
            "reviewed_at": metadata["reviewed_at"],
            "review_interval_days": metadata["review_interval_days"],
            "content_sha256": metadata["content_sha256"],
            "related_docs": json.loads(metadata["related_docs"]),
            "source_ref": f"data/kb/documentation.json#{metadata['doc_id']}#{metadata['section']}",
            "metadata_manifest_sha256": self.metadata_manifest_sha256,
        }

    def _related_source_snapshots(
        self,
        primary_hits: list[tuple[float, dict[str, Any], dict[str, Any]]],
        excluded: Counter[str],
        excluded_article_ids: set[str],
    ) -> tuple[dict[str, Any], ...]:
        """Resolve approved fresh related articles from the source-validated corpus."""
        primary_doc_ids = {source["doc_id"] for _, source, _ in primary_hits}
        seen_doc_ids = set(primary_doc_ids)
        related_sources: list[dict[str, Any]] = []
        for _, _, primary_record in primary_hits:
            related_doc_ids = json.loads(primary_record["metadata"]["related_docs"])
            for related_doc_id in related_doc_ids:
                if related_doc_id in seen_doc_ids:
                    continue
                seen_doc_ids.add(related_doc_id)
                related_record = self.article_records_by_doc_id[related_doc_id]
                reason = self._source_exclusion_reason(
                    related_record["id"],
                    related_record["document"],
                    related_record["metadata"],
                )
                if reason:
                    if related_record["id"] not in excluded_article_ids:
                        excluded[reason] += 1
                        excluded_article_ids.add(related_record["id"])
                    continue
                related_sources.append(
                    self._source_snapshot(related_record["id"], related_record, None)
                )
        return tuple(related_sources)

    def _empty_result(
        self,
        status: str,
        reason_code: str,
        reason: str,
    ) -> RetrievalResult:
        """Construct a no-source retrieval result without leaking provider details."""
        return RetrievalResult(
            status=status,
            reason_code=reason_code,
            reason=reason,
            sources=(),
            eligible_candidate_count=0,
            excluded_candidate_counts={},
            minimum_similarity=MINIMUM_SIMILARITY,
            embedding_model=self.settings.embedding_model,
            source_sha256=self.source_sha256,
            metadata_manifest_sha256=self.metadata_manifest_sha256,
        )


def _first_query_list(query_result: dict[str, Any], key: str) -> list[Any]:
    """Extract the single query's candidate list from Chroma's nested response."""
    values = query_result.get(key)
    if not isinstance(values, list) or not values or not isinstance(values[0], list):
        return []
    return values[0]


def _require_positive_batch_size(batch_size: int, argument_name: str) -> None:
    """Reject invalid batch sizes before any embedding or Chroma requests."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError(f"{argument_name} must be a positive integer.")


def _ticket_query_text(
    ticket: CanonicalTicket,
) -> tuple[str, tuple[str, str, str] | None]:
    """Build subject/body-only retrieval text or return its explicit local outcome."""
    if not isinstance(ticket, CanonicalTicket):
        return "", (
            "no_hit",
            "invalid_query",
            "The ticket query is invalid; no knowledge source was used.",
        )
    if not isinstance(ticket.subject, str) or not isinstance(ticket.body, str):
        return "", (
            "no_hit",
            "invalid_query",
            "The ticket query is invalid; no knowledge source was used.",
        )
    query_text = "\n".join(
        part for part in (ticket.subject.strip(), ticket.body.strip()) if part
    )
    if not query_text:
        return "", (
            "no_hit",
            "empty_query",
            "The ticket has no searchable subject or body text.",
        )
    return query_text, None


def _l2_normalize(vector: Sequence[float]) -> list[float]:
    """Manually L2-normalize reduced-dimensional vectors; reject zero or non-finite norms."""
    for x in vector:
        if not isinstance(x, Real) or isinstance(x, bool) or not math.isfinite(float(x)):
            raise ValueError("Embedding vector contains non-finite values.")
    norm = math.sqrt(sum(float(x) * float(x) for x in vector))
    if norm == 0.0 or not math.isfinite(norm):
        raise ValueError("Embedding vector has zero or non-finite L2 norm.")
    return [float(x / norm) for x in vector]


def _embed_query_vector(
    client: Any,
    *,
    model: str,
    query_text: str,
) -> list[float]:
    """Create one normalized query vector using Gemini or test adapter."""
    if hasattr(client, "models") and hasattr(client.models, "embed_content"):
        from google.genai import types

        response = client.models.embed_content(
            model=model,
            contents=query_text,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=EMBEDDING_DIMENSIONS,
            ),
        )
        if hasattr(response, "embeddings") and response.embeddings:
            return _l2_normalize(response.embeddings[0].values)
        if hasattr(response, "embedding") and response.embedding:
            return _l2_normalize(response.embedding.values)
        raise ValueError("Invalid embedding response.")
    if hasattr(client, "embeddings") and hasattr(client.embeddings, "create"):
        response = client.embeddings.create(model=model, input=query_text)
        data = getattr(response, "data", None)
        if data and len(data) > 0:
            return getattr(data[0], "embedding")
    raise ValueError("Unsupported embedding client.")


def _embed_query_vectors(
    client: Any,
    *,
    model: str,
    query_texts: list[str],
) -> list[list[float]]:
    """Create batch normalized query vectors using Gemini or test adapter."""
    if hasattr(client, "models") and hasattr(client.models, "embed_content"):
        from google.genai import types

        response = client.models.embed_content(
            model=model,
            contents=query_texts,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=EMBEDDING_DIMENSIONS,
            ),
        )
        if not hasattr(response, "embeddings") or len(response.embeddings) != len(query_texts):
            raise ValueError("Embedding count mismatch.")
        return [_l2_normalize(item.values) for item in response.embeddings]
    if hasattr(client, "embeddings") and hasattr(client.embeddings, "create"):
        response = client.embeddings.create(model=model, input=query_texts)
        return _parse_embedding_response(response, len(query_texts))
    raise ValueError("Unsupported embedding client.")


def _parse_embedding_response(response: Any, expected_count: int) -> list[Any]:
    """Map embedding rows by strict response index, rejecting ambiguous chunks."""
    data = getattr(response, "data", None)
    if not isinstance(data, list) or len(data) != expected_count:
        raise ValueError("Embedding response count does not match the request.")
    indexed_rows: list[Any] = [None] * expected_count
    seen_indexes: set[int] = set()
    for row in data:
        index = getattr(row, "index", None)
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index >= expected_count
            or index in seen_indexes
        ):
            raise ValueError("Embedding response indexes are ambiguous.")
        seen_indexes.add(index)
        indexed_rows[index] = getattr(row, "embedding", None)
    if len(seen_indexes) != expected_count:
        raise ValueError("Embedding response indexes are incomplete.")
    return indexed_rows


def _is_valid_embedding_vector(vector: Any) -> bool:
    """Require a finite 768-dimension numeric vector for the packaged Chroma index."""
    if not isinstance(vector, (list, tuple)) or len(vector) != EMBEDDING_DIMENSIONS:
        return False
    for value in vector:
        if not isinstance(value, Real) or isinstance(value, bool):
            return False
        try:
            if not math.isfinite(float(value)):
                return False
        except (OverflowError, ValueError):
            return False
    return True


def _parse_chroma_query_rows(
    query_result: Any,
    expected_count: int,
) -> list[tuple[list[Any], list[Any], list[Any], list[Any]] | None]:
    """Align Chroma query rows and isolate identifiable row-level shape failures."""
    if not isinstance(query_result, dict):
        raise ValueError("Chroma query response must be an object.")
    arrays = [query_result.get(key) for key in ("ids", "documents", "metadatas", "distances")]
    if any(not isinstance(rows, list) or len(rows) != expected_count for rows in arrays):
        raise ValueError("Chroma query response rows do not align with request positions.")

    query_rows: list[tuple[list[Any], list[Any], list[Any], list[Any]] | None] = []
    for row_values in zip(*arrays, strict=True):
        if any(not isinstance(values, list) for values in row_values):
            query_rows.append(None)
            continue
        ids, documents, metadatas, distances = row_values
        if not (len(ids) == len(documents) == len(metadatas) == len(distances)):
            query_rows.append(None)
            continue
        query_rows.append((ids, documents, metadatas, distances))
    return query_rows
