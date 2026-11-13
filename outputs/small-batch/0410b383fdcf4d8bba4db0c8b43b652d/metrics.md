# Run metrics

- Run ID: `0410b383fdcf4d8bba4db0c8b43b652d`
- Total tickets: 3
- Processed successfully: 3
- Processing failures: 0
- Tickets with provided labels: 3
- Elapsed run time: 11.942 seconds
  (from `processing_runs.started_at` through `completed_at`; not summed ticket latency)

| Label | Matched | Different | Not compared | Match rate |
| --- | ---: | ---: | ---: | ---: |
| intent | 3 | 0 | 0 | 100.0% |
| urgency | 0 | 3 | 0 | 0.0% |
| expected_route | 2 | 1 | 0 | 66.7% |
| answerable_from_docs | 3 | 0 | 0 | 100.0% |
| expected_doc_ids | 2 | 1 | 0 | 66.7% |
| must_not_auto_respond | 3 | 0 | 0 | 100.0% |

Label snapshots are evaluation-only and are read from SQLite after processing; they are never supplied to inference. `expected_route` maps the input-only `auto_respond` alias to the actual `draft_for_review` route; drafts remain unsent.
`answerable_from_docs` compares to the persisted evidence-readiness proxy, and `must_not_auto_respond` compares to the protected-case proxy; neither is a semantic ground-truth judgment. Document IDs compare as unordered sets.
Missing, invalid, or unavailable values are counted as not compared and are excluded from each field's match-rate denominator.
