"""Deterministic unit tests for scripts/build_failed50.py sample selection."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_failed50 import (
    build_failed_sample,
    score_ticket_mismatches,
    select_failed_records,
)


def test_route_alias_normalization() -> None:
    """Check that expected auto_respond normalizes to draft_for_review."""
    matching_row = {
        "expected_route": "auto_respond",
        "predicted_route": "draft_for_review",
    }
    mismatch_count, breakdown = score_ticket_mismatches(matching_row)
    assert mismatch_count == 0
    assert breakdown["route"] is False

    differing_row = {
        "expected_route": "auto_respond",
        "predicted_route": "escalate",
    }
    mismatch_count, breakdown = score_ticket_mismatches(differing_row)
    assert mismatch_count == 1
    assert breakdown["route"] is True

    escalate_matching = {
        "expected_route": "escalate",
        "predicted_route": "escalate",
    }
    assert score_ticket_mismatches(escalate_matching)[0] == 0

    escalate_differing = {
        "expected_route": "escalate",
        "predicted_route": "draft_for_review",
    }
    assert score_ticket_mismatches(escalate_differing)[0] == 1


def test_unordered_doc_ids_comparison() -> None:
    """Check that expected document IDs compare as unordered sets."""
    matching_order = {
        "expected_doc_ids": ["DOC-AUTH-001", "DOC-API-002"],
        "predicted_doc_ids": ["DOC-API-002", "DOC-AUTH-001"],
    }
    mismatch_count, breakdown = score_ticket_mismatches(matching_order)
    assert mismatch_count == 0
    assert breakdown["expected_doc_ids"] is False

    empty_matching = {
        "expected_doc_ids": [],
        "predicted_doc_ids": [],
    }
    assert score_ticket_mismatches(empty_matching)[0] == 0

    differing_docs = {
        "expected_doc_ids": ["DOC-AUTH-001"],
        "predicted_doc_ids": ["DOC-AUTH-002"],
    }
    mismatch_count, breakdown = score_ticket_mismatches(differing_docs)
    assert mismatch_count == 1
    assert breakdown["expected_doc_ids"] is True


def test_missing_prediction_semantics() -> None:
    """Check that missing predictions count as mismatches only when expected labels exist."""
    missing_prediction = {
        "expected_intent": "api_key_issue",
        "predicted_intent": None,
        "expected_urgency": "high",
        "predicted_urgency": None,
    }
    mismatch_count, breakdown = score_ticket_mismatches(missing_prediction)
    assert mismatch_count == 2
    assert breakdown["intent"] is True
    assert breakdown["urgency"] is True

    missing_expected = {
        "expected_intent": None,
        "predicted_intent": "api_key_issue",
    }
    mismatch_count, breakdown = score_ticket_mismatches(missing_expected)
    assert mismatch_count == 0
    assert "intent" not in breakdown


def test_mismatch_ranking_latency_and_index_tie_breaks() -> None:
    """Check sort priority: mismatch_count desc, latency desc, input_index asc."""
    original_tickets = [
        {"ticket_id": f"TICKET-{i}", "body": f"Ticket body {i}"}
        for i in range(5)
    ]

    results_rows = [
        # Ticket 0: 2 mismatches, latency 1.5s
        {
            "input_index": 0,
            "ticket_id": "TICKET-0",
            "processing_latency_seconds": 1.5,
            "expected_intent": "a",
            "predicted_intent": "b",
            "expected_urgency": "high",
            "predicted_urgency": "low",
        },
        # Ticket 1: 3 mismatches, latency 0.8s
        {
            "input_index": 1,
            "ticket_id": "TICKET-1",
            "processing_latency_seconds": 0.8,
            "expected_intent": "a",
            "predicted_intent": "b",
            "expected_urgency": "high",
            "predicted_urgency": "low",
            "expected_must_not_auto_respond": True,
            "predicted_must_not_auto_respond": False,
        },
        # Ticket 2: 2 mismatches, latency 2.5s (should beat Ticket 0 on latency)
        {
            "input_index": 2,
            "ticket_id": "TICKET-2",
            "processing_latency_seconds": 2.5,
            "expected_intent": "a",
            "predicted_intent": "b",
            "expected_urgency": "high",
            "predicted_urgency": "low",
        },
        # Ticket 3: 2 mismatches, latency 2.5s (ties Ticket 2 on latency; input_index 2 beats 3)
        {
            "input_index": 3,
            "ticket_id": "TICKET-3",
            "processing_latency_seconds": 2.5,
            "expected_intent": "a",
            "predicted_intent": "b",
            "expected_urgency": "high",
            "predicted_urgency": "low",
        },
        # Ticket 4: 0 mismatches, latency 10.0s (qualifying only when mismatches > 0)
        {
            "input_index": 4,
            "ticket_id": "TICKET-4",
            "processing_latency_seconds": 10.0,
            "expected_intent": "a",
            "predicted_intent": "a",
        },
    ]

    selected_tickets, manifest_entries = select_failed_records(
        results_rows,
        original_tickets,
        max_tickets=10,
    )

    # Ticket 4 has 0 mismatches, so only 4 tickets qualify.
    assert len(selected_tickets) == 4
    # Expected order:
    # 1. TICKET-1 (3 mismatches)
    # 2. TICKET-2 (2 mismatches, latency 2.5, index 2)
    # 3. TICKET-3 (2 mismatches, latency 2.5, index 3)
    # 4. TICKET-0 (2 mismatches, latency 1.5, index 0)
    assert [t["ticket_id"] for t in selected_tickets] == [
        "TICKET-1",
        "TICKET-2",
        "TICKET-3",
        "TICKET-0",
    ]
    assert [m["mismatch_count"] for m in manifest_entries] == [3, 2, 2, 2]

    # Test max_tickets cap
    capped_tickets, capped_manifest = select_failed_records(
        results_rows,
        original_tickets,
        max_tickets=2,
    )
    assert len(capped_tickets) == 2
    assert [t["ticket_id"] for t in capped_tickets] == ["TICKET-1", "TICKET-2"]


def test_validation_of_indices_and_mapping() -> None:
    """Check that malformed, negative, or out-of-range indices are safely skipped."""
    original_tickets = [
        {"ticket_id": "TICKET-0", "body": "Zero"},
        {"ticket_id": "TICKET-1", "body": "One"},
    ]

    results_rows = [
        {"input_index": 0, "ticket_id": "TICKET-0", "expected_intent": "a", "predicted_intent": "b"},
        {"input_index": 999, "ticket_id": "OUT-OF-RANGE", "expected_intent": "a", "predicted_intent": "b"},
        {"input_index": None, "ticket_id": "NO-INDEX", "expected_intent": "a", "predicted_intent": "b"},
        {"input_index": -1, "ticket_id": "NEGATIVE", "expected_intent": "a", "predicted_intent": "b"},
    ]

    selected_tickets, manifest = select_failed_records(
        results_rows,
        original_tickets,
        max_tickets=10,
    )
    assert len(selected_tickets) == 1
    assert selected_tickets[0]["ticket_id"] == "TICKET-0"
    assert manifest[0]["input_index"] == 0


def test_build_failed_sample_end_to_end_and_determinism(tmp_path: Path) -> None:
    """Verify file output, manifest creation, and deterministic SHA-256 hash."""
    tickets = [
        {
            "ticket_id": f"VAL-{i:04d}",
            "channel": "email",
            "subject": f"Subject {i}",
            "body": f"Body text {i}",
            "labels": {"intent": "api_key_issue", "expected_route": "auto_respond"},
        }
        for i in range(5)
    ]
    results = [
        {
            "input_index": i,
            "ticket_id": f"VAL-{i:04d}",
            "expected_intent": "api_key_issue",
            "predicted_intent": "other" if i % 2 == 0 else "api_key_issue",
            "expected_route": "auto_respond",
            "predicted_route": "escalate" if i == 0 else "draft_for_review",
            "processing_latency_seconds": 1.0 + i * 0.1,
        }
        for i in range(5)
    ]

    tickets_file = tmp_path / "tickets.json"
    results_file = tmp_path / "results.jsonl"
    output_file = tmp_path / "output.json"
    manifest_file = tmp_path / "manifest.json"

    tickets_file.write_text(json.dumps(tickets, indent=2), encoding="utf-8")
    results_file.write_text(
        "\n".join(json.dumps(r) for r in results) + "\n", encoding="utf-8"
    )

    count_1, hash_1 = build_failed_sample(
        results_path=results_file,
        tickets_path=tickets_file,
        output_path=output_file,
        manifest_path=manifest_file,
        max_tickets=10,
    )

    assert count_1 == 3  # Indices 0 (2 mismatches), 2 (1 mismatch), 4 (1 mismatch)
    assert output_file.exists()
    assert manifest_file.exists()

    # Re-run and verify determinism
    count_2, hash_2 = build_failed_sample(
        results_path=results_file,
        tickets_path=tickets_file,
        output_path=output_file,
        manifest_path=manifest_file,
        max_tickets=10,
    )
    assert count_1 == count_2
    assert hash_1 == hash_2

    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert manifest["sample_size"] == 3
    assert manifest["sample_sha256"] == hash_1
    assert len(manifest["tickets"]) == 3


def test_deduplication_of_duplicate_input_index_or_ticket_id() -> None:
    """Ensure duplicate input_index or ticket_id rows are deduped to unique tickets."""
    original_tickets = [
        {"ticket_id": "TICKET-0", "body": "Zero"},
        {"ticket_id": "TICKET-1", "body": "One"},
    ]

    # Two rows with input_index 0 (TICKET-0), one with input_index 1 (TICKET-1)
    results_rows = [
        {
            "input_index": 0,
            "ticket_id": "TICKET-0",
            "processing_latency_seconds": 1.0,
            "expected_intent": "a",
            "predicted_intent": "b",
        },
        {
            "input_index": 0,
            "ticket_id": "TICKET-0",
            "processing_latency_seconds": 2.0,
            "expected_intent": "a",
            "predicted_intent": "b",
        },
        {
            "input_index": 1,
            "ticket_id": "TICKET-1",
            "processing_latency_seconds": 0.5,
            "expected_intent": "a",
            "predicted_intent": "b",
        },
    ]

    selected_tickets, manifest_entries = select_failed_records(
        results_rows,
        original_tickets,
        max_tickets=50,
    )

    # Must yield exactly 2 unique tickets, not 3
    assert len(selected_tickets) == 2
    assert [t["ticket_id"] for t in selected_tickets] == ["TICKET-0", "TICKET-1"]
    assert len(manifest_entries) == 2
    assert [m["ticket_id"] for m in manifest_entries] == ["TICKET-0", "TICKET-1"]


def test_max_tickets_validation_bounds() -> None:
    """Ensure max_tickets must be within 1..50 bounds."""
    original_tickets = [{"ticket_id": "T0", "body": "Test"}]
    results_rows = [{"input_index": 0, "ticket_id": "T0", "expected_intent": "a", "predicted_intent": "b"}]

    with pytest.raises(ValueError, match="max_tickets must be between 1 and 50"):
        select_failed_records(results_rows, original_tickets, max_tickets=0)

    with pytest.raises(ValueError, match="max_tickets must be between 1 and 50"):
        select_failed_records(results_rows, original_tickets, max_tickets=51)

    with pytest.raises(ValueError, match="max_tickets must be between 1 and 50"):
        select_failed_records(results_rows, original_tickets, max_tickets=-5)

    # Valid boundary values succeed
    selected_1, _ = select_failed_records(results_rows, original_tickets, max_tickets=1)
    assert len(selected_1) == 1

    selected_50, _ = select_failed_records(results_rows, original_tickets, max_tickets=50)
    assert len(selected_50) == 1

