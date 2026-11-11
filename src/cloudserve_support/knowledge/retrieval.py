"""Query the current, project-governed KB and return traceable passage snapshots."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import chromadb
from openai import OpenAI

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
EMBEDDING_DIMENSIONS = 1536
MINIMUM_SIMILARITY = 0.45
MAXIMUM_SOURCES = 3
QUERY_EMBEDDING_TIMEOUT_SECONDS = 15.0
QUERY_EMBEDDING_MAX_RETRIES = 0
REQUIRED_PASSAGE_METADATA = (
    "doc_id",
    "title",
    "category",
    "applies_to",
    "section",
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

        if self.embedding_client is None and not settings.openai_api_key:
            self.unavailable_reason = (
                "query_embeddings_unavailable",
                "Query embeddings are unavailable; no knowledge source was used.",
            )
            return
        if self.embedding_client is None:
            try:
                self.embedding_client = OpenAI(
                    api_key=settings.openai_api_key,
                    timeout=QUERY_EMBEDDING_TIMEOUT_SECONDS,
                    max_retries=QUERY_EMBEDDING_MAX_RETRIES,
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
        expected_passage_records_hash = passage_records_sha256(expected_records)

        index_manifest = json.loads((index_path / "manifest.json").read_text(encoding="utf-8"))
        if (
            index_manifest.get("schema_version") != 3
            or index_manifest.get("collection") != COLLECTION_NAME
            or index_manifest.get("source_path") != "data/kb/documentation.json"
            or index_manifest.get("source_sha256") != self.source_sha256
            or index_manifest.get("metadata_manifest_path") != "data/kb/metadata_manifest.json"
            or not isinstance(index_manifest.get("metadata_manifest_sha256"), str)
            or len(index_manifest["metadata_manifest_sha256"]) != 64
            or any(
                character not in "0123456789abcdef"
                for character in index_manifest["metadata_manifest_sha256"]
            )
            or index_manifest.get("passage_records_sha256") != expected_passage_records_hash
            or index_manifest.get("chunker_version") != CHUNKER_VERSION
            or index_manifest.get("embedding_provider") != "openai"
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
        """Return up to three ranked sources or an explicit safe no-evidence result."""
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
            response = self.embedding_client.embeddings.create(
                model=self.settings.embedding_model,
                input=query_text,
            )
            query_embedding = response.data[0].embedding
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

        excluded = Counter()
        current_eligible_count = 0
        eligible: list[tuple[float, dict[str, Any]]] = []
        for passage_id, document, metadata, distance in zip(
            ids, documents, metadatas, distances, strict=True
        ):
            reason = self._source_exclusion_reason(passage_id, document, metadata)
            if reason:
                excluded[reason] += 1
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
            eligible.append((similarity, self._source_snapshot(passage_id, record, similarity)))

        eligible.sort(key=lambda item: (-item[0], item[1]["passage_id"]))
        snapshots = tuple(source for _, source in eligible[:MAXIMUM_SOURCES])
        if snapshots:
            return RetrievalResult(
                status="ready",
                reason_code="relevant_sources_found",
                reason="Current capstone-approved passages met the relevance threshold.",
                sources=snapshots,
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
        passage_id: str,
        document: Any,
        metadata: Any,
    ) -> str | None:
        """Check index records against source text and project metadata before use."""
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
        similarity: float,
    ) -> dict[str, Any]:
        """Build citation and audit fields from the source-resolved passage record."""
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
            "minimum_similarity": MINIMUM_SIMILARITY,
            "approval_status": metadata["approval_status"],
            "approval_status_source": metadata["approval_status_source"],
            "metadata_owner": metadata["metadata_owner"],
            "metadata_source": metadata["metadata_source"],
            "article_version": metadata["article_version"],
            "reviewed_at": metadata["reviewed_at"],
            "review_interval_days": metadata["review_interval_days"],
            "content_sha256": metadata["content_sha256"],
            "source_ref": f"data/kb/documentation.json#{metadata['doc_id']}#{metadata['section']}",
            "metadata_manifest_sha256": self.metadata_manifest_sha256,
        }

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
