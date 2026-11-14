"""Deterministically select top KPI-failing validation tickets for targeted remediation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_PATH = (
    REPO_ROOT
    / "outputs"
    / "gemini-validation"
    / "128ff352e13e439a8139eb18dab3852f"
    / "results.jsonl"
)
DEFAULT_TICKETS_PATH = REPO_ROOT / "data" / "tickets" / "validation_tickets.json"
DEFAULT_OUTPUT_PATH = (
    REPO_ROOT / "outputs" / "gemini-kpi-remediation" / "validation_failed50.json"
)
DEFAULT_MANIFEST_PATH = (
    REPO_ROOT / "outputs" / "gemini-kpi-remediation" / "sample_manifest.json"
)
DEFAULT_MAX_TICKETS = 50

EXPECTED_ROUTE_ALIASES = {
    "auto_respond": "draft_for_review",
    "escalate": "escalate",
    "draft_for_review": "draft_for_review",
    "degrade": "degrade",
}


def score_ticket_mismatches(result_row: dict[str, Any]) -> tuple[int, dict[str, bool]]:
    """Count KPI mismatches using run_metrics.py evaluation semantics.

    Checks:
    - intent: expected != predicted (when expected is provided)
    - urgency: expected != predicted (when expected is provided)
    - route: expected (normalized) != predicted (when expected is provided)
    - answerability: expected != predicted (when expected is provided)
    - expected_doc_ids: unordered set comparison (when expected is provided)
    - must_not_auto_respond: expected != predicted (when expected is provided)
    """
    mismatches: dict[str, bool] = {}

    expected_intent = result_row.get("expected_intent")
    predicted_intent = result_row.get("predicted_intent")
    if expected_intent is not None:
        mismatches["intent"] = expected_intent != predicted_intent

    expected_urgency = result_row.get("expected_urgency")
    predicted_urgency = result_row.get("predicted_urgency")
    if expected_urgency is not None:
        mismatches["urgency"] = expected_urgency != predicted_urgency

    expected_route = result_row.get("expected_route")
    predicted_route = result_row.get("predicted_route")
    if expected_route is not None:
        normalized_expected_route = EXPECTED_ROUTE_ALIASES.get(
            expected_route, expected_route
        )
        mismatches["route"] = normalized_expected_route != predicted_route

    expected_answerable = result_row.get("expected_answerable_from_docs")
    predicted_answerable = result_row.get("predicted_answerable_from_docs")
    if expected_answerable is not None:
        mismatches["answerability"] = expected_answerable != predicted_answerable

    expected_docs = result_row.get("expected_doc_ids")
    predicted_docs = result_row.get("predicted_doc_ids")
    if expected_docs is not None:
        expected_set = set(expected_docs) if isinstance(expected_docs, list) else set()
        predicted_set = (
            set(predicted_docs) if isinstance(predicted_docs, list) else set()
        )
        mismatches["expected_doc_ids"] = expected_set != predicted_set

    expected_protected = result_row.get("expected_must_not_auto_respond")
    predicted_protected = result_row.get("predicted_must_not_auto_respond")
    if expected_protected is not None:
        mismatches["must_not_auto_respond"] = expected_protected != predicted_protected

    total_mismatches = sum(1 for is_mismatch in mismatches.values() if is_mismatch)
    return total_mismatches, mismatches


def select_failed_records(
    results_rows: list[dict[str, Any]],
    original_tickets: list[dict[str, Any]],
    *,
    max_tickets: int = DEFAULT_MAX_TICKETS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Rank tickets by mismatch count desc, latency desc, input_index asc."""
    if max_tickets <= 0 or max_tickets > 50:
        raise ValueError(
            f"max_tickets must be between 1 and 50 (got {max_tickets})"
        )

    scored_records: list[dict[str, Any]] = []
    ticket_by_index = {
        index: ticket for index, ticket in enumerate(original_tickets)
    }

    for row in results_rows:
        input_index = row.get("input_index")
        if input_index is None or input_index not in ticket_by_index:
            continue
        ticket_id = row.get("ticket_id", f"INDEX-{input_index}")
        latency = float(row.get("processing_latency_seconds", 0.0))
        mismatch_count, mismatches_breakdown = score_ticket_mismatches(row)

        scored_records.append(
            {
                "input_index": input_index,
                "ticket_id": ticket_id,
                "mismatch_count": mismatch_count,
                "processing_latency_seconds": latency,
                "mismatches": mismatches_breakdown,
                "original_ticket": ticket_by_index[input_index],
            }
        )

    # Sort deterministically:
    # 1. Mismatch count descending (-x["mismatch_count"])
    # 2. Processing latency descending (-x["processing_latency_seconds"])
    # 3. Input index ascending (x["input_index"])
    scored_records.sort(
        key=lambda x: (
            -x["mismatch_count"],
            -x["processing_latency_seconds"],
            x["input_index"],
        )
    )

    selected_scored: list[dict[str, Any]] = []
    seen_ticket_ids: set[str] = set()
    for rec in scored_records:
        if rec["mismatch_count"] <= 0:
            continue
        tid = rec["ticket_id"]
        if tid in seen_ticket_ids:
            continue
        seen_ticket_ids.add(tid)
        selected_scored.append(rec)
        if len(selected_scored) >= max_tickets:
            break

    selected_tickets = [rec["original_ticket"] for rec in selected_scored]
    manifest_entries = [
        {
            "rank": rank + 1,
            "input_index": rec["input_index"],
            "ticket_id": rec["ticket_id"],
            "mismatch_count": rec["mismatch_count"],
            "processing_latency_seconds": rec["processing_latency_seconds"],
            "mismatches": rec["mismatches"],
        }
        for rank, rec in enumerate(selected_scored)
    ]
    return selected_tickets, manifest_entries


def build_failed_sample(
    *,
    results_path: Path = DEFAULT_RESULTS_PATH,
    tickets_path: Path = DEFAULT_TICKETS_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    max_tickets: int = DEFAULT_MAX_TICKETS,
) -> tuple[int, str]:
    """Select the failure sample and write output JSON and manifest."""
    with open(results_path, encoding="utf-8") as file:
        results_rows = [
            json.loads(line) for line in file if line.strip()
        ]

    with open(tickets_path, encoding="utf-8") as file:
        original_tickets = json.load(file)

    selected_tickets, manifest_entries = select_failed_records(
        results_rows,
        original_tickets,
        max_tickets=max_tickets,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized_tickets = json.dumps(selected_tickets, indent=2) + "\n"
    output_path.write_text(serialized_tickets, encoding="utf-8")
    sample_sha256 = hashlib.sha256(serialized_tickets.encode("utf-8")).hexdigest()

    manifest_data = {
        "sample_size": len(selected_tickets),
        "sample_sha256": sample_sha256,
        "source_results": str(results_path),
        "source_tickets": str(tickets_path),
        "selection_rule": (
            "mismatch_count descending, then processing_latency_seconds descending, "
            "then input_index ascending"
        ),
        "tickets": manifest_entries,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest_data, indent=2) + "\n", encoding="utf-8")

    return len(selected_tickets), sample_sha256


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Select the 50 highest-priority KPI-failing validation tickets."
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
        help=f"Validation results.jsonl path (default: {DEFAULT_RESULTS_PATH})",
    )
    parser.add_argument(
        "--tickets",
        type=Path,
        default=DEFAULT_TICKETS_PATH,
        help=f"Validation tickets input path (default: {DEFAULT_TICKETS_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output sample path (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help=f"Output manifest path (default: {DEFAULT_MANIFEST_PATH})",
    )
    parser.add_argument(
        "--max-tickets",
        type=int,
        default=DEFAULT_MAX_TICKETS,
        help=f"Maximum tickets to select (1..50, default: {DEFAULT_MAX_TICKETS})",
    )
    args = parser.parse_args()
    if args.max_tickets <= 0 or args.max_tickets > 50:
        parser.error(f"--max-tickets must be between 1 and 50 (got {args.max_tickets})")
    return args


def main() -> None:
    """Run selection and report sample statistics without customer text or labels."""
    args = parse_args()
    count, sample_sha256 = build_failed_sample(
        results_path=args.results,
        tickets_path=args.tickets,
        output_path=args.output,
        manifest_path=args.manifest,
        max_tickets=args.max_tickets,
    )
    print(f"Selected {count} failing tickets (max: {args.max_tickets})")
    print(f"Output file: {args.output}")
    print(f"Manifest file: {args.manifest}")
    print(f"Sample SHA-256: {sample_sha256}")


if __name__ == "__main__":
    main()
