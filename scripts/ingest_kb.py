"""Build or refresh the persistent Chroma index from the packaged KB articles."""

from __future__ import annotations

import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types

from cloudserve_support.knowledge.kb_source import (
    CHUNKER_VERSION,
    build_passage_records,
    load_kb_articles,
    load_kb_metadata_manifest,
    passage_records_sha256,
    validate_kb_metadata_coverage,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_PATH = REPO_ROOT / "data" / "kb" / "documentation.json"
DEFAULT_METADATA_PATH = REPO_ROOT / "data" / "kb" / "metadata_manifest.json"
DEFAULT_INDEX_PATH = REPO_ROOT / "data" / "index"
DEFAULT_CHROMA_PATH = DEFAULT_INDEX_PATH / "chroma"
DEFAULT_MANIFEST_PATH = DEFAULT_INDEX_PATH / "manifest.json"
COLLECTION_NAME = "cloudserve_kb"
COLLECTION_METADATA = {
    "hnsw:space": "cosine",
    "schema_version": 1,
    "purpose": "CloudServe approved KB passage retrieval",
}
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSIONS = 768
EMBEDDING_BATCH_SIZE = 100


def embed_kb_passages(
    client: genai.Client,
    documents: list[str],
    embedding_model: str,
) -> list[list[float]]:
    """Create Gemini embeddings for passage text using one pinned model in bounded batches."""
    embeddings: list[list[float]] = []
    for start in range(0, len(documents), EMBEDDING_BATCH_SIZE):
        batch = documents[start : start + EMBEDDING_BATCH_SIZE]
        response = client.models.embed_content(
            model=embedding_model,
            contents=batch,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=EMBEDDING_DIMENSIONS,
            ),
        )
        if not response.embeddings or len(response.embeddings) != len(batch):
            raise RuntimeError(
                f"Gemini embedding count mismatch: received "
                f"{len(response.embeddings) if response.embeddings else 0} vectors "
                f"for {len(batch)} passages."
            )
        for item in response.embeddings:
            values = item.values
            if not values or len(values) != EMBEDDING_DIMENSIONS:
                raise RuntimeError(
                    f"Expected {EMBEDDING_DIMENSIONS}-dimension embeddings; received {len(values) if values else 0}."
                )
            # Reduced-dimensional vectors must be manually L2-normalized.
            norm = math.sqrt(sum(float(x) * float(x) for x in values))
            if norm == 0.0 or not math.isfinite(norm):
                raise RuntimeError("Non-finite or zero L2 norm in embedding vector.")
            embeddings.append([float(x / norm) for x in values])

    if len(embeddings) != len(documents):
        raise RuntimeError(
            f"Gemini embedding count mismatch: received {len(embeddings)} vectors "
            f"for {len(documents)} passages."
        )
    dimensions = {len(embedding) for embedding in embeddings}
    if dimensions != {EMBEDDING_DIMENSIONS}:
        raise RuntimeError(
            f"Expected {EMBEDDING_DIMENSIONS}-dimension embeddings; received {sorted(dimensions)}."
        )
    return embeddings


def create_kb_collection(chroma_path: Path) -> Any:
    """Open the persistent cosine-distance collection used for KB passage retrieval."""
    chroma_path.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(chroma_path))
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata=COLLECTION_METADATA,
        embedding_function=None,
    )
    actual_metadata = collection.metadata or {}
    mismatches = {
        key: {"expected": value, "actual": actual_metadata.get(key)}
        for key, value in COLLECTION_METADATA.items()
        if actual_metadata.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Existing collection has incompatible settings: {mismatches}")
    return collection


def write_index_manifest(
    index_path: Path,
    source_path: Path,
    metadata_path: Path,
    source_sha256: str,
    metadata_manifest_sha256: str,
    embedding_model: str,
    article_count: int,
    passage_count: int,
    passage_records_hash: str,
) -> Path:
    """Record source, chunker, and embedding settings needed to reproduce this index."""
    manifest_path = index_path / "manifest.json"
    manifest = {
        "schema_version": 3,
        "collection": COLLECTION_NAME,
        "source_path": os.path.relpath(source_path, REPO_ROOT),
        "source_sha256": source_sha256,
        "metadata_manifest_path": os.path.relpath(metadata_path, REPO_ROOT),
        "metadata_manifest_sha256": metadata_manifest_sha256,
        "passage_records_sha256": passage_records_hash,
        "chunker_version": CHUNKER_VERSION,
        "embedding_provider": "gemini",
        "embedding_model": embedding_model,
        "embedding_dimensions": EMBEDDING_DIMENSIONS,
        "distance_metric": "cosine",
        "built_at_utc": datetime.now(UTC).isoformat(),
        "article_count": article_count,
        "passage_count": passage_count,
    }
    index_path.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def ingest_kb_articles(
    source_path: Path,
    index_path: Path,
    embedding_model: str,
    metadata_path: Path = DEFAULT_METADATA_PATH,
) -> None:
    """Embed and upsert every KB passage, remove stale records, and write the build manifest."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY or GOOGLE_API_KEY is not set. Add it to .env and run with `uv run --env-file .env`."
        )

    articles, source_sha256 = load_kb_articles(source_path)
    metadata_manifest, metadata_by_id, metadata_manifest_sha256 = (
        load_kb_metadata_manifest(metadata_path)
    )
    validate_kb_metadata_coverage(articles, metadata_by_id)
    records = build_passage_records(
        articles,
        metadata_by_id,
        metadata_owner=metadata_manifest["metadata_owner"],
        metadata_source=metadata_manifest["metadata_source"],
        review_interval_days=metadata_manifest["review_interval_days"],
    )
    records_hash = passage_records_sha256(records)
    documents = [record["document"] for record in records]
    client = genai.Client(api_key=api_key)
    embeddings = embed_kb_passages(client, documents, embedding_model)
    collection = create_kb_collection(index_path / "chroma")

    existing_ids = set(collection.get(include=[])["ids"])
    desired_ids = {record["id"] for record in records}
    collection.upsert(
        ids=[record["id"] for record in records],
        documents=documents,
        embeddings=embeddings,
        metadatas=[record["metadata"] for record in records],
    )
    stale_ids = sorted(existing_ids - desired_ids)
    if stale_ids:
        collection.delete(ids=stale_ids)

    manifest_path = write_index_manifest(
        index_path=index_path,
        source_path=source_path,
        metadata_path=metadata_path,
        source_sha256=source_sha256,
        metadata_manifest_sha256=metadata_manifest_sha256,
        embedding_model=embedding_model,
        article_count=len(articles),
        passage_count=len(records),
        passage_records_hash=records_hash,
    )
    print(f"Chroma version: {chromadb.__version__}")
    print(f"Collection: {collection.name}")
    print(f"Persistent path: {index_path / 'chroma'}")
    print(f"Embedding model: {embedding_model} ({EMBEDDING_DIMENSIONS} dimensions)")
    print(f"KB articles: {len(articles)}")
    print(f"Passage records: {collection.count()}")
    print(f"Stale passage records removed: {len(stale_ids)}")
    print(f"Manifest: {manifest_path}")
    print("Eligibility and citation metadata: capstone-owned metadata manifest.")


def main() -> None:
    """Run a reproducible passage ingestion from the packaged KB into local Chroma."""
    load_dotenv(REPO_ROOT / ".env")
    ingest_kb_articles(
        source_path=DEFAULT_SOURCE_PATH,
        index_path=DEFAULT_INDEX_PATH,
        metadata_path=DEFAULT_METADATA_PATH,
        embedding_model=os.environ.get(
            "GEMINI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL
        ),
    )


if __name__ == "__main__":
    main()
