"""Create the empty persistent Chroma collection used by the KB index."""

from pathlib import Path

import chromadb


REPO_ROOT = Path(__file__).resolve().parents[1]
CHROMA_PATH = REPO_ROOT / "data" / "index" / "chroma"
COLLECTION_NAME = "cloudserve_kb"
COLLECTION_METADATA = {
    "hnsw:space": "cosine",
    "schema_version": 1,
    "purpose": "CloudServe approved KB passage retrieval",
}


def main() -> None:
    """Create or inspect the collection without ingesting KB data."""
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata=COLLECTION_METADATA,
        # Keep embedding-model selection explicit in the future ingestion step.
        embedding_function=None,
    )

    actual_metadata = collection.metadata or {}
    mismatches = {
        key: {"expected": value, "actual": actual_metadata.get(key)}
        for key, value in COLLECTION_METADATA.items()
        if actual_metadata.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            f"Collection {COLLECTION_NAME!r} has incompatible metadata: {mismatches}"
        )

    print(f"Chroma version: {chromadb.__version__}")
    print(f"Persistent path: {CHROMA_PATH}")
    print(f"Collection: {collection.name}")
    print(f"Distance metric: {actual_metadata['hnsw:space']}")
    print(f"Passage records: {collection.count()}")
    print("No KB articles or embeddings were added.")


if __name__ == "__main__":
    main()
