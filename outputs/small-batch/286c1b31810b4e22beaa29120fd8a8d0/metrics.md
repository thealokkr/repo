# Run metrics

- Run ID: `286c1b31810b4e22beaa29120fd8a8d0`
- Total tickets: 3
- Processed successfully: 0
- Processing failures: 3
- Tickets with provided labels: 3
- Elapsed run time: 10.790 seconds
  (from `processing_runs.started_at` through `completed_at`; not summed ticket latency)

| Label | Matched | Different | Not compared | Match rate |
| --- | ---: | ---: | ---: | ---: |
| intent | 0 | 0 | 3 | N/A |
| urgency | 0 | 0 | 3 | N/A |
| expected_route | 0 | 0 | 3 | N/A |
| answerable_from_docs | 0 | 0 | 3 | N/A |
| expected_doc_ids | 0 | 0 | 3 | N/A |
| must_not_auto_respond | 0 | 0 | 3 | N/A |

Label snapshots are evaluation-only and are read from SQLite after processing; they are never supplied to inference. `expected_route` maps the input-only `auto_respond` alias to the actual `draft_for_review` route; drafts remain unsent.
`answerable_from_docs` compares to the persisted evidence-readiness proxy, and `must_not_auto_respond` compares to the protected-case proxy; neither is a semantic ground-truth judgment. Document IDs compare as unordered sets.
Missing, invalid, or unavailable values are counted as not compared and are excluded from each field's match-rate denominator.
