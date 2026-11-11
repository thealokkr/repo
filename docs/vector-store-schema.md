# Vector Store Schema and Retrieval Design

## Decision

Use one persistent, local Chroma collection for the active CloudServe knowledge-base snapshot. Store one record per useful, citable article passage. Keep the original articles in `data/kb/documentation.json`; treat `data/index/` as a rebuildable search index, not the source of truth. Keep ticket decisions and audit history in `data/runtime/` as described in [`Agent.md`](../Agent.md).

The implementation uses OpenAI `text-embedding-3-small` (1,536 dimensions) for passage and query vectors. `gpt-6-luna` is the default low-cost model for optional structured ticket classification and a future drafting layer; KB retrieval is implemented, while reply drafting remains out of scope. Model names are configured through `OPENAI_EMBEDDING_MODEL` and `OPENAI_LLM_MODEL`. The index manifest records the source and metadata file hashes plus a canonical digest of the exact passage IDs, text, and metadata represented in Chroma.

Chroma is the current implementation choice because it provides local persistence, vector similarity, stored documents, stable record IDs, and metadata filters through one library. It does not require a separately operated vector server in local mode. This is an implementation convenience, not a claim that a vector database product is strictly necessary for a 29-article corpus.

## Why the system needs semantic passage retrieval

The retrieval layer has a product purpose beyond storing vectors: map a ticket written in a customer's own words to the approved support passage that actually answers it, then give later stages a source they can verify and cite. Without retrieval, generation would either search by brittle exact words or risk making unsupported claims. Retrieval also needs to return no result when it cannot establish a relevant source, so routing can use the existing fallback/escalation path.

This is required by the project evidence:

- The Stage 2 PRD marks **FR-02** as Must: return a relevant approved source and passage for a documented test ticket. **FR-03** requires drafts based only on approved sources. **FR-04** requires each draft to identify the article, passage, version, and review date. **NFR-03** requires answer support to be checked separately from routing accuracy; **NFR-08** bars expired or unapproved content.
- The Stage 4 sprint plan explicitly schedules **B-03** for chunking and embedding with traceable article ID, version, and review date, and **B-04** for the vector store and retrieval layer.
- The build specification's **A4** requires identifiable passages from the supplied corpus; **A6** requires citations to resolve to the passages actually retrieved.
- The packaged corpus at `data/kb/documentation.json` contains 29 articles with `doc_id`, `title`, `category`, `applies_to`, `content`, and `related_docs`. Articles use meaningful headings such as Symptoms, Common causes, Resolution, and Notes. These are useful retrieval facets and natural chunk boundaries.
- The packaged development and validation ticket files contain 500 and 80 tickets respectively. Ticket language can describe a symptom differently from the article heading, which is the reason to evaluate semantic retrieval and not rely on exact title lookup.

Therefore, the capability the system needs is **embedding-based passage retrieval with provenance and a relevance fallback**. Chroma is one way to deliver it; embeddings alone do not prove accuracy. The validation ticket set must show relevant passages are found, citations resolve, and unsupported/low-confidence cases fall back safely.

## Record model

Chroma's collection record has four parts: `id`, `document`, `embedding`, and `metadata`.

| Field | Type | Meaning and rule |
|---|---|---|
| `id` | string | Stable passage key: `{doc_id}:{content_sha256_prefix}:{chunk_index}`. A content change produces new passage IDs, avoiding accidental citation to stale text. |
| `document` | string | Exact source passage text, prefixed with the article title, applicability, and section heading for context. Preserve ordered resolution steps together; do not split a procedure in a way that loses its order or conditions. |
| `embedding` | vector of floats | Embedding of the stored `document`, created with the configured embedding model. Query text must use the same model and vector dimensions. Chroma can create embeddings or accept explicit vectors; the project must choose and pin one path. |
| `metadata` | object of scalar values | Filterable provenance and facets listed below. Keep canonical article text in `document`/source JSON rather than duplicating it in metadata. |

### Metadata per passage

| Key | Required? | Use |
|---|---|---|
| `doc_id` | yes | Stable source-article identity; returned in citations. |
| `title` | yes | Human-readable source title. |
| `category` | yes | Topic facet, such as `authentication` or `deployment`; use as a filter only when ticket classification is reliable, otherwise as a ranking hint. |
| `applies_to` | yes | Source applicability such as `Console, CLI, SDK`; only hard-filter when the ticket identifies a compatible product/surface. |
| `section` | yes | Section heading such as `Symptoms`, `Common causes`, `Resolution`, or `Notes`; supports diagnostics and explanation of what matched. |
| `chunk_index`, `chunk_count` | yes | Passage position and total count within the source article. |
| `content_sha256` | yes | Hash of the source article content used for this passage; enables stale-index checks and reproducible rebuilds. |
| `approval_status` | required for production eligibility | `approved`, `expired`, `draft`, `unapproved`, or `unknown`. Retrieval for answer drafting must include only currently approved content. |
| `approval_status_source` | yes | Provenance for the approval value, such as an authoritative KB manifest or an explicitly synthetic test fixture. Never treat synthetic fixture values as production approval evidence. |
| `metadata_owner`, `metadata_source` | yes | Identifies the capstone maintainer role and the project-owned metadata overlay that supplied eligibility and citation fields. |
| `article_version` | required for complete PRD citation | The capstone citation version from the project metadata manifest. It is not asserted to be a source-system version. |
| `reviewed_at` | required for complete PRD citation | The actual capstone review date in ISO-8601 format from the project metadata manifest. It does not assert a CloudServe production review date. |
| `review_interval_days` | yes | Capstone-only freshness interval applied to the manifest review date. It does not establish production freshness. |

The article text and project metadata have separate source files: `data/kb/documentation.json` contains the supplied corpus, while `data/kb/metadata_manifest.json` is the capstone-owned overlay. The manifest names the owner role, metadata source, authority scope, review procedure, update rule, article versions, ISO review dates, approval states, and review interval. Its values are authoritative only for this capstone demonstration. They do not represent CloudServe production approval, technical policy validation, or production freshness. Existing synthetic status fields and relative `last_reviewed_days_ago` values in the article JSON are ignored by ingestion; all eligibility and capstone review dates come from the project metadata manifest. The raw metadata-file hash records the build input; the passage-record digest determines whether indexed values changed. Runtime verifies the article source hash, the current derived passage-record digest, and every stored Chroma ID, document, and metadata row. Governance-only prose edits do not stale unchanged indexed records, while any article or indexed-metadata mismatch leaves the index unavailable. Candidate retrieval excludes non-approved, stale, incomplete, or mismatched records with an explicit reason.

Do not make `related_docs` an embedded vector, split the corpus into collections by category, or encode an `issue/action/escalation` class that the source data does not authoritatively provide. Keep article relationships in the source corpus and optionally use them to expand candidates after an initial relevant hit. Escalation decisions belong to routing and guardrails, not a guessed vector-store facet.

### Collection build manifest

Write a small `data/index/manifest.json` beside the persistent Chroma data. Record:

```json
{
  "schema_version": 3,
  "collection": "cloudserve_kb",
  "source_path": "data/kb/documentation.json",
  "source_sha256": "<hash of the complete source corpus>",
  "metadata_manifest_path": "data/kb/metadata_manifest.json",
  "metadata_manifest_sha256": "<hash of the project-owned metadata manifest>",
  "passage_records_sha256": "<canonical digest of sorted passage IDs, documents, and metadata>",
  "chunker_version": "<implementation version>",
  "embedding_provider": "<provider or local runtime>",
  "embedding_model": "<exact model identifier and revision>",
  "embedding_dimensions": "<integer>",
  "distance_metric": "cosine",
  "built_at_utc": "<ISO-8601 timestamp>",
  "article_count": 29,
  "passage_count": "<integer>"
}
```

Replace placeholders with real values during an index build; do not commit example values as runtime facts. A rebuild must use the same pinned embedding model for both corpus and queries. If the embedding model, chunking rules, article source, or indexed metadata changes, rebuild the index. Governance-only text changes update the raw build-input hash on the next rebuild but do not invalidate records when their canonical digest and stored Chroma values still match. Runtime treats a missing or mismatched passage-record digest, or any stored ID, text, or metadata difference, as unavailable.

## Chunking and facets

Use article sections as natural boundaries. Keep a short symptom list or cause list together where it forms one retrievable unit; keep a numbered resolution procedure intact when its steps depend on each other. Add the article title, `applies_to`, and section name to the text sent for embedding, while retaining the original passage text and metadata for the citation.

Use a single collection for the KB. Category and section are facets on records, not separate collections: an article can answer both a topical issue and a procedural question, and splitting collections would force the application to guess which one to search. Search broadly by default. Apply an applicability or approval filter only when its value is authoritative; category is generally a soft ranking hint. Keep protected-ticket routing independent from retrieval, so finding a relevant article never overrides a required escalation.

## Search and returned result

1. Use normalized ticket subject and body as the query. Do not add fixture labels, outcome history, or unrelated private context.
2. Generate its query embedding with the same configured model used at index time.
3. Retrieve the passages by cosine similarity and apply the frozen minimum similarity floor of `0.45`, a capstone heuristic pending development-only retrieval evaluation. Treat a candidate distance as valid only when it is a numeric, non-boolean cosine distance in the inclusive range `[0, 2]`; malformed or out-of-range distances are excluded rather than interpreted as relevance.
4. Apply the project manifest's approval and review-interval rules. Do not hard-filter on uncertain ticket categories.
5. Evaluate the retrieval floor on development operational data only, then freeze it before processing validation tickets. Validation measures the frozen threshold; it must not tune it. Return no eligible passage when evidence is weak or similarity metadata is invalid; let routing escalate or request more information instead of forcing a match.
6. Return the exact source-resolved passage and its metadata snapshot, not a citation invented by a language model. Runtime uses only normalized ticket subject and body; fixture labels and outcome history are never retrieval inputs.

Each result passed downstream includes `passage_id`, `doc_id`, `title`, `section`, exact `passage_text`, `category`, `applies_to`, `similarity`, `approval_status`, `approval_status_source`, `metadata_owner`, `metadata_source`, `article_version`, `reviewed_at`, source content hash, stable source reference, and metadata-manifest hash. Citation rendering must use these returned fields. A citation is valid only if its passage ID resolves back to the indexed content and source article. Each audit event stores the returned source snapshots in `sources_json`, so later index rebuilds do not change the citation evidence for that run.

## Chroma versus simpler local choices

| Choice | Operational footprint | Retrieval capability | Fit here |
|---|---|---|---|
| **Chroma PersistentClient** | Embedded/local library; no server to operate. More Python packages than the standard library and a persistent index directory to package or rebuild. | Ready-made document/vector/metadata collection model and vector similarity; supports metadata filtering. | **Recommended baseline.** Matches the planned B-03/B-04 vector-store work while keeping the application logic modest. Pin Chroma and the embedding implementation. |
| **SQLite FTS5 only** | Smallest operational footprint; uses one SQLite file and is commonly available through SQLite builds. | BM25/full-text term search, not semantic embeddings. | Good keyword component or fallback, but alone it does not implement the planned embedding-based retrieval. |
| **SQLite + exact cosine scan** | Can use one SQLite file with embeddings stored as blobs and calculate cosine similarity over all passages; no ANN extension/server. | Semantic search with exhaustive scoring; FTS5 can supply lexical candidates. | Potentially the lightest complete design at this corpus size, but requires custom vector serialization, dimension checks, scoring, filters, ranking/fusion, and maintenance code. There are only 29 source articles, so there is no scale need for approximate nearest-neighbor indexing. Consider this only if minimizing installed dependencies outweighs using a ready-made vector store. |
| **SQLite + `sqlite-vec`** | Single-file SQLite plus a loadable/native extension or package. | Vector KNN inside SQLite. | Viable, but the upstream project currently identifies itself as pre-v1, so extension compatibility and packaging add a portability/versioning concern. |

So Chroma is **not the absolute lightest option**. For this capstone it is the simplest complete vector-store implementation aligned to the written sprint plan, not a requirement imposed by the PRD. If the top priority changes to the smallest self-contained package, SQLite with exact cosine plus FTS5 is a credible alternative because the KB is tiny. That alternative should be benchmarked against the same validation tickets before replacing Chroma; lower package weight does not establish retrieval quality.

Embedding portability is separate from database portability. Pin the embedding provider/model and document whether first-time use needs network access, local model files, or an API credential. A copied Chroma index is only usable with a compatible embedding model; `data/kb/` plus a repeatable rebuild command remains the portable source of truth.

## Evidence and acceptance checks

The schema supports the requirements as follows:

| Requirement | Schema/retrieval support | Evidence to collect during implementation |
|---|---|---|
| PRD FR-02; Build Spec A4 | One record per meaningful passage with stable `passage_id` and source `doc_id`; semantic candidate retrieval. | On documented test tickets, report the returned relevant approved passage and its IDs; calculate retrieval hit rate on labeled data. |
| PRD FR-03; NFR-03; Build Spec A6 | Exact passage text is returned with citation metadata; generation is restricted to retrieved approved evidence; low-confidence search returns no hit. | Follow each generated citation to the returned passage and review claims for support. Track retrieval, answer, and route quality separately. |
| PRD FR-04 | Preserve article title, passage, capstone article version, and review date in the result. | Citation resolves to the article and passage and displays the project-owned metadata. This is complete for capstone provenance only; production authority remains outside the supplied corpus. |
| PRD NFR-08 | Approval/freshness metadata gates retrieval. | Demonstrate expired and unapproved articles are excluded; missing status is not silently treated as approved. |
| Stage 4 B-03/B-04 | Chunk and embed with traceable article identity; persist searchable records and rebuild metadata. | Rebuild from packaged KB, compare source hash/counts, and confirm relevant passage IDs resolve. |

Useful grounding examples from the supplied data:

- `DEV-0004` asks how to revert an earlier revision and labels `DOC-DEPLOY-002`; that article is titled “Rolling back a failed release.” A resolution-section passage is more useful and citable than returning only the article title.
- `DEV-0003` concerns audit access and labels `DOC-SEC-003` while its route is escalation/protected. It demonstrates why retrieval and route policy must be separate: relevant evidence can accompany an escalation without authorizing an automatic reply.
- The development ticket file has 500 records and the validation file has 80; these provide a way to measure retrieval behavior instead of assuming a schema or vector store is accurate.

The validation score, threshold, and exact embedding model are implementation/evaluation outputs, not claims established by this schema document. Do not assert the design meets retrieval accuracy until those checks are run.

## Technical references

- [Chroma architecture overview](https://docs.trychroma.com/reference/architecture/overview) — local embedded deployment mode and collection record model.
- [Chroma collection data](https://docs.trychroma.com/docs/collections/add-data) — unique IDs and document/embedding/metadata inputs.
- [Chroma metadata filtering](https://docs.trychroma.com/docs/querying-collections/metadata-filtering) — filter candidates by metadata.
- [Chroma introduction](https://docs.trychroma.com/docs/overview/introduction) — vector, keyword, hybrid retrieval, and metadata-filtering capabilities.
- [SQLite FTS5](https://sqlite.org/fts5.html) — full-text queries and BM25 ranking.
- [`sqlite-vec`](https://github.com/asg017/sqlite-vec) — SQLite vector-search extension; its upstream README calls it pre-v1.
