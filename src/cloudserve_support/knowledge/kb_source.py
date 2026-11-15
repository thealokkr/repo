"""Load whole KB articles and build canonical source records."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date
from pathlib import Path
from typing import Any


CHUNKER_VERSION = "whole-article-v1"
ARTICLE_VERSION_PATTERN = re.compile(r"^[a-z][a-z0-9]*-\d+\.\d+$")
APPROVAL_STATUSES = {"approved", "expired", "draft", "unapproved", "unknown"}
REQUIRED_METADATA_FIELDS = {
    "doc_id",
    "approval_status",
    "article_version",
    "reviewed_at",
}


def load_kb_articles(source_path: Path) -> tuple[list[dict[str, Any]], str]:
    """Load and validate source articles, returning their raw-file SHA-256."""
    source_bytes = source_path.read_bytes()
    articles = json.loads(source_bytes)
    if not isinstance(articles, list) or not articles:
        raise ValueError("KB source must be a non-empty JSON array of articles.")

    required_fields = ("doc_id", "title", "category", "applies_to", "content")
    article_ids: set[str] = set()
    for article_number, article in enumerate(articles, start=1):
        if not isinstance(article, dict):
            raise ValueError(f"KB article {article_number} must be a JSON object.")
        missing_fields = [
            field
            for field in required_fields
            if not isinstance(article.get(field), str) or not article[field].strip()
        ]
        if missing_fields:
            raise ValueError(
                f"KB article {article_number} is missing non-empty fields: "
                + ", ".join(missing_fields)
            )
        if article["doc_id"] in article_ids:
            raise ValueError(f"KB source contains duplicate doc_id {article['doc_id']!r}.")
        article_ids.add(article["doc_id"])

    return articles, hashlib.sha256(source_bytes).hexdigest()


def load_kb_metadata_manifest(
    manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], str]:
    """Load the project-owned metadata overlay and return article metadata by ID."""
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("KB metadata manifest must be a schema-version-1 JSON object.")

    for field in (
        "metadata_owner",
        "metadata_source",
        "authority_scope",
        "review_procedure",
        "update_rule",
    ):
        if not isinstance(manifest.get(field), str) or not manifest[field].strip():
            raise ValueError(f"KB metadata manifest requires non-empty {field}.")

    review_interval_days = manifest.get("review_interval_days")
    if not isinstance(review_interval_days, int) or review_interval_days < 1:
        raise ValueError("KB metadata manifest review_interval_days must be a positive integer.")

    raw_metadata = manifest.get("articles")
    if not isinstance(raw_metadata, list) or not raw_metadata:
        raise ValueError("KB metadata manifest articles must be a non-empty JSON array.")

    metadata_by_id: dict[str, dict[str, Any]] = {}
    for article_number, metadata in enumerate(raw_metadata, start=1):
        if not isinstance(metadata, dict):
            raise ValueError(f"KB metadata row {article_number} must be a JSON object.")
        missing_fields = REQUIRED_METADATA_FIELDS - metadata.keys()
        if missing_fields:
            raise ValueError(
                f"KB metadata row {article_number} is missing fields: "
                + ", ".join(sorted(missing_fields))
            )
        doc_id = metadata["doc_id"]
        if not isinstance(doc_id, str) or not doc_id.strip():
            raise ValueError(f"KB metadata row {article_number} has an invalid doc_id.")
        if doc_id in metadata_by_id:
            raise ValueError(f"KB metadata manifest contains duplicate doc_id {doc_id!r}.")
        if (
            not isinstance(metadata["approval_status"], str)
            or metadata["approval_status"] not in APPROVAL_STATUSES
        ):
            raise ValueError(
                f"KB metadata for {doc_id} has unsupported approval_status "
                f"{metadata['approval_status']!r}."
            )
        if (
            not isinstance(metadata["article_version"], str)
            or not ARTICLE_VERSION_PATTERN.fullmatch(metadata["article_version"])
        ):
            raise ValueError(f"KB metadata for {doc_id} has an invalid article_version.")
        if not isinstance(metadata["reviewed_at"], str):
            raise ValueError(f"KB metadata for {doc_id} has an invalid reviewed_at date.")
        try:
            parsed_review_date = date.fromisoformat(metadata["reviewed_at"])
            if parsed_review_date.isoformat() != metadata["reviewed_at"]:
                raise ValueError("date must use YYYY-MM-DD format")
        except ValueError as error:
            raise ValueError(
                f"KB metadata for {doc_id} reviewed_at must be an ISO-8601 date."
            ) from error
        metadata_by_id[doc_id] = metadata

    return (
        manifest,
        metadata_by_id,
        hashlib.sha256(manifest_bytes).hexdigest(),
    )


def validate_kb_metadata_coverage(
    articles: list[dict[str, Any]],
    metadata_by_id: dict[str, dict[str, Any]],
) -> None:
    """Require one project metadata row for each article and no orphan rows."""
    source_ids = {article["doc_id"] for article in articles}
    metadata_ids = set(metadata_by_id)
    missing = sorted(source_ids - metadata_ids)
    extra = sorted(metadata_ids - source_ids)
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append("missing metadata for " + ", ".join(missing))
        if extra:
            details.append("metadata references unknown articles " + ", ".join(extra))
        raise ValueError("KB metadata coverage mismatch: " + "; ".join(details) + ".")


def _related_doc_ids(article: dict[str, Any], article_ids: set[str]) -> list[str]:
    """Validate related article references against the packaged corpus."""
    related_docs = article.get("related_docs", [])
    if not isinstance(related_docs, list) or any(
        not isinstance(doc_id, str) or not doc_id.strip() for doc_id in related_docs
    ):
        raise ValueError(f"KB article {article['doc_id']} related_docs must be a string array.")
    if len(related_docs) != len(set(related_docs)):
        raise ValueError(f"KB article {article['doc_id']} has duplicate related_docs.")
    unknown_ids = sorted(set(related_docs) - article_ids)
    if unknown_ids:
        raise ValueError(
            f"KB article {article['doc_id']} references unknown related docs: "
            + ", ".join(unknown_ids)
            + "."
        )
    return related_docs


def build_passage_records(
    articles: list[dict[str, Any]],
    metadata_by_id: dict[str, dict[str, Any]],
    *,
    metadata_owner: str,
    metadata_source: str,
    review_interval_days: int,
) -> list[dict[str, Any]]:
    """Build one canonical full-article record per article with scalar metadata."""
    validate_kb_metadata_coverage(articles, metadata_by_id)
    article_ids = {article["doc_id"] for article in articles}
    records: list[dict[str, Any]] = []
    for article in articles:
        related_docs = _related_doc_ids(article, article_ids)
        article_hash = hashlib.sha256(article["content"].encode("utf-8")).hexdigest()
        article_metadata = metadata_by_id[article["doc_id"]]
        document = (
            f"Title: {article['title']}\n"
            f"Category: {article['category']}\n"
            f"Applies to: {article['applies_to']}\n\n"
            f"{article['content']}"
        )
        record_metadata: dict[str, str | int] = {
            "doc_id": article["doc_id"],
            "title": article["title"],
            "category": article["category"],
            "applies_to": article["applies_to"],
            "section": "Article",
            "related_docs": json.dumps(related_docs, separators=(",", ":")),
            "content_sha256": article_hash,
            "approval_status": article_metadata["approval_status"],
            "approval_status_source": metadata_source,
            "metadata_owner": metadata_owner,
            "metadata_source": metadata_source,
            "article_version": article_metadata["article_version"],
            "reviewed_at": article_metadata["reviewed_at"],
            "review_interval_days": review_interval_days,
        }
        records.append(
            {
                "id": f"{article['doc_id']}:{article_hash}",
                "document": document,
                "metadata": record_metadata,
            }
        )
    return records


def passage_records_sha256(records: list[dict[str, Any]]) -> str:
    """Hash the canonical source-derived IDs, article text, and metadata for an index."""
    canonical_records = [
        {
            "id": record["id"],
            "document": record["document"],
            "metadata": record["metadata"],
        }
        for record in sorted(records, key=lambda item: item["id"])
    ]
    canonical_bytes = json.dumps(
        canonical_records,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical_bytes).hexdigest()
