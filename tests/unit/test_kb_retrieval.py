"""Core whole-article KB record and retrieval contract checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from cloudserve_support.knowledge.kb_source import (
    CHUNKER_VERSION,
    build_passage_records,
    load_kb_articles,
    load_kb_metadata_manifest,
    passage_records_sha256,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = REPOSITORY_ROOT / "data" / "kb" / "documentation.json"
METADATA_PATH = REPOSITORY_ROOT / "data" / "kb" / "metadata_manifest.json"


def _build_source_records() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build records directly from packaged source files."""
    articles, _ = load_kb_articles(SOURCE_PATH)
    manifest, metadata_by_id, _ = load_kb_metadata_manifest(METADATA_PATH)
    records = build_passage_records(
        articles,
        metadata_by_id,
        metadata_owner=manifest["metadata_owner"],
        metadata_source=manifest["metadata_source"],
        review_interval_days=manifest["review_interval_days"],
    )
    return articles, records


def test_kb_builds_one_full_article_record_with_scalar_metadata() -> None:
    articles, records = _build_source_records()
    records_by_doc_id = {record["metadata"]["doc_id"]: record for record in records}

    assert CHUNKER_VERSION == "whole-article-v1"
    assert len(articles) == len(records) == 29
    assert len(records_by_doc_id) == 29
    for article in articles:
        record = records_by_doc_id[article["doc_id"]]
        assert article["content"] in record["document"]
        assert record["metadata"]["section"] == "Article"
        assert json.loads(record["metadata"]["related_docs"]) == article["related_docs"]
        assert all(
            isinstance(value, (str, int, float, bool))
            for value in record["metadata"].values()
        )


def test_related_doc_references_must_exist_in_the_packaged_corpus() -> None:
    articles, _ = load_kb_articles(SOURCE_PATH)
    manifest, metadata_by_id, _ = load_kb_metadata_manifest(METADATA_PATH)
    invalid_articles = [dict(article) for article in articles]
    invalid_articles[0] = {**invalid_articles[0], "related_docs": ["DOC-MISSING-999"]}

    with pytest.raises(ValueError, match="references unknown related docs: DOC-MISSING-999"):
        build_passage_records(
            invalid_articles,
            metadata_by_id,
            metadata_owner=manifest["metadata_owner"],
            metadata_source=manifest["metadata_source"],
            review_interval_days=manifest["review_interval_days"],
        )


def test_article_record_digest_is_order_independent_and_detects_content_changes() -> None:
    _, records = _build_source_records()
    digest = passage_records_sha256(records)
    changed_records = [dict(record) for record in records]
    changed_records[0] = {
        **changed_records[0],
        "document": changed_records[0]["document"] + " stale",
    }

    assert passage_records_sha256(list(reversed(records))) == digest
    assert passage_records_sha256(changed_records) != digest
