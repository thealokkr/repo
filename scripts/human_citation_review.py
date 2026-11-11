"""Prepare blinded human review packets and score completed independent ratings."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cloudserve_support.guardrails.draft_safety import sanitize_handoff_content


MINIMUM_SAMPLE_SIZE = 50
SUPPORT_RATINGS = frozenset({"supported", "unsupported", "uncertain"})
YES_NO_UNCERTAIN = frozenset({"yes", "no", "uncertain"})
CORRECTNESS_RATINGS = frozenset({"correct", "partly_correct", "incorrect", "uncertain"})
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")
GENERATED_REVIEW_ARTIFACTS = (
    "readiness.json",
    "assessor-1.json",
    "assessor-2.json",
    "assessor-1-ratings.json",
    "assessor-2-ratings.json",
    "reference-key.json",
    "review-instructions.md",
    "scorecard.json",
)


def read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    """Read persisted run records for post-run review without touching runtime state."""
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object in {path} line {line_number}.")
            records.append(value)
    return records


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest used to reproduce a review sample's inputs."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_review_text(value: object) -> str:
    """Remove common private data before placing ticket or draft text in a review copy."""
    if not isinstance(value, str):
        return ""
    sanitized, _, _ = sanitize_handoff_content(value)
    return sanitized if isinstance(sanitized, str) else ""


def split_claim_sentences(claim_text: str) -> list[str]:
    """Split draft claims into reviewable sentences while retaining bullet content."""
    return [
        sentence.strip()
        for sentence in SENTENCE_BOUNDARY.split(claim_text.strip())
        if sentence.strip()
    ]


def eligible_development_drafts(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep completed generated drafts from the operational development run."""
    eligible = []
    for record in records:
        draft = record.get("draft")
        generation = record.get("generation")
        if (
            record.get("status") == "completed"
            and record.get("route") == "draft_for_review"
            and isinstance(record.get("draft_response"), str)
            and record["draft_response"].strip()
            and isinstance(draft, dict)
            and isinstance(draft.get("claims"), list)
            and bool(draft["claims"])
            and isinstance(generation, dict)
            and generation.get("status") == "generated"
        ):
            eligible.append(record)
    return eligible


def build_blinded_item(
    record: dict[str, Any],
    ticket: dict[str, Any],
    blinded_item_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build one label-free assessor item and its separate identity/reference key."""
    source_by_id = {
        source.get("passage_id"): source
        for source in record.get("sources", [])
        if isinstance(source, dict) and isinstance(source.get("passage_id"), str)
    }
    sentences: list[dict[str, Any]] = []
    citation_index = 0
    for claim in record["draft"]["claims"]:
        if not isinstance(claim, dict) or not isinstance(claim.get("text"), str):
            continue
        claim_citations = claim.get("citations")
        if not isinstance(claim_citations, list):
            claim_citations = []
        citations_by_id = {
            citation.get("passage_id"): citation
            for citation in claim_citations
            if isinstance(citation, dict) and isinstance(citation.get("passage_id"), str)
        }
        source_ids = claim.get("source_ids")
        if not isinstance(source_ids, list):
            source_ids = list(citations_by_id)
        claim_sentence_texts = split_claim_sentences(clean_review_text(claim["text"]))
        for sentence_text in claim_sentence_texts:
            sentence_citations = []
            for source_id in source_ids:
                if not isinstance(source_id, str):
                    continue
                citation_index += 1
                citation = citations_by_id.get(source_id, {})
                source = source_by_id.get(source_id)
                sentence_citations.append(
                    {
                        "citation_id": f"C{citation_index:03d}",
                        "passage_id": source_id,
                        "source_ref": citation.get("source_ref"),
                        "resolved": isinstance(source, dict),
                        "passage": (
                            {
                                "document": source.get("doc_id"),
                                "title": clean_review_text(source.get("title")),
                                "section": clean_review_text(source.get("section")),
                                "text": clean_review_text(source.get("passage_text")),
                            }
                            if isinstance(source, dict)
                            else None
                        ),
                    }
                )
            sentences.append(
                {
                    "sentence_id": f"S{len(sentences) + 1:03d}",
                    "text": sentence_text,
                    "citations": sentence_citations,
                    "citation_missing": not sentence_citations,
                }
            )

    item = {
        "item_id": blinded_item_id,
        "ticket_context": {
            "channel": record.get("channel"),
            "subject": clean_review_text(ticket.get("subject")),
            "message": clean_review_text(ticket.get("body")),
        },
        "draft_response": clean_review_text(record.get("draft_response")),
        "sentences": sentences,
    }
    labels = ticket.get("labels") if isinstance(ticket.get("labels"), dict) else {}
    reference = {
        "item_id": blinded_item_id,
        "ticket_id": ticket.get("ticket_id"),
        "input_index": record.get("input_index"),
        "expected_document_ids": labels.get("expected_doc_ids", []),
        "reference_response": "",
        "must_mention": [],
        "must_not_claim": [],
    }
    return item, reference


def reuse_matching_review_sample(
    output_dir: Path,
    readiness: dict[str, Any],
) -> dict[str, Any] | None:
    """Keep an existing ready sample and completed human work when its inputs match exactly."""
    readiness_path = output_dir / "readiness.json"
    if not readiness_path.is_file():
        return None
    try:
        existing = json.loads(readiness_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if any(
        existing.get(field) != readiness[field]
        for field in ("run_id", "sample_size_requested", "seed", "source_sha256")
    ) or existing.get("status") != "ready":
        return None
    if all((output_dir / filename).is_file() for filename in existing.get("assessor_forms", [])) and (
        output_dir / "reference-key.json"
    ).is_file() and (output_dir / "review-instructions.md").is_file():
        return existing
    return None


def archive_superseded_review_artifacts(output_dir: Path, run_id: str | None) -> str | None:
    """Move only this tool's known sample files aside before replacing or blocking a sample."""
    existing_files = [output_dir / filename for filename in GENERATED_REVIEW_ARTIFACTS]
    existing_files = [path for path in existing_files if path.is_file()]
    if not existing_files:
        return None
    archive_name = run_id or "unknown-run"
    archive_dir = output_dir / "superseded-samples" / archive_name
    if archive_dir.exists():
        suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        archive_dir = archive_dir.with_name(f"{archive_name}-{suffix}")
    archive_dir.mkdir(parents=True, exist_ok=False)
    for source_path in existing_files:
        shutil.move(str(source_path), str(archive_dir / source_path.name))
    return str(archive_dir.relative_to(output_dir))


def prepare_review_packet(arguments: argparse.Namespace) -> int:
    """Create two independent blinded forms, or record why a 50-draft sample is unavailable."""
    if arguments.sample_size < MINIMUM_SAMPLE_SIZE:
        raise ValueError(f"The sample size must be at least {MINIMUM_SAMPLE_SIZE}.")
    records = read_jsonl_records(arguments.run_dir / "results.jsonl")
    tickets = json.loads(arguments.input.read_text(encoding="utf-8"))
    truth_records = json.loads(arguments.ground_truth.read_text(encoding="utf-8"))
    if not isinstance(tickets, list) or not isinstance(truth_records, list):
        raise ValueError("Development input and ground-truth files must contain JSON arrays.")
    ticket_by_id = {
        ticket.get("ticket_id"): ticket
        for ticket in tickets
        if isinstance(ticket, dict) and isinstance(ticket.get("ticket_id"), str)
    }
    truth_by_id = {
        answer.get("ticket_id"): answer
        for answer in truth_records
        if isinstance(answer, dict) and isinstance(answer.get("ticket_id"), str)
    }
    eligible = eligible_development_drafts(records)
    complete_candidates = [
        record
        for record in eligible
        if record.get("ticket_id") in ticket_by_id
        and record.get("ticket_id") in truth_by_id
    ]
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    readiness = {
        "status": "preparing",
        "sample_active": False,
        "run_id": records[0].get("run_id") if records else None,
        "sample_size_requested": arguments.sample_size,
        "minimum_sample_size": MINIMUM_SAMPLE_SIZE,
        "seed": arguments.seed,
        "generated_drafts_available": len(eligible),
        "candidates_with_input_and_reference": len(complete_candidates),
        "source_sha256": {
            "results.jsonl": file_sha256(arguments.run_dir / "results.jsonl"),
            "development_input": file_sha256(arguments.input),
            "ground_truth": file_sha256(arguments.ground_truth),
        },
        "block_reason": None,
    }
    existing_sample = (
        reuse_matching_review_sample(arguments.output_dir, readiness)
        if len(complete_candidates) >= arguments.sample_size
        else None
    )
    if existing_sample:
        existing_sample["reused_existing_sample"] = True
        print(json.dumps(existing_sample, ensure_ascii=False, sort_keys=True))
        return 0
    archived_path = archive_superseded_review_artifacts(
        arguments.output_dir, readiness["run_id"]
    )
    if archived_path:
        readiness["superseded_artifacts"] = archived_path
    (arguments.output_dir / "readiness.json").write_text(
        json.dumps(readiness, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if len(complete_candidates) < arguments.sample_size:
        readiness["status"] = "blocked"
        readiness["block_reason"] = (
            "The selected run has fewer generated development drafts than the requested sample size."
        )
        (arguments.output_dir / "readiness.json").write_text(
            json.dumps(readiness, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(readiness, ensure_ascii=False, sort_keys=True))
        return 2

    ordered_candidates = sorted(
        complete_candidates,
        key=lambda row: (str(row.get("ticket_id", "")), int(row.get("input_index", -1))),
    )
    selected = random.Random(arguments.seed).sample(ordered_candidates, arguments.sample_size)
    reviewer_one_items = []
    reviewer_two_items = []
    reference_key = []
    for index, record in enumerate(selected, 1):
        ticket_id = record["ticket_id"]
        item_id = f"R{index:03d}"
        item, reference = build_blinded_item(record, ticket_by_id[ticket_id], item_id)
        answer = truth_by_id[ticket_id]
        reference["reference_response"] = clean_review_text(answer.get("reference_response"))
        reference["must_mention"] = answer.get("must_mention", [])
        reference["must_not_claim"] = answer.get("must_not_claim", [])
        reviewer_one_items.append(item)
        reviewer_two_items.append(item)
        reference_key.append(reference)

    reviewer_one_items = random.Random(arguments.seed + 101).sample(
        reviewer_one_items, len(reviewer_one_items)
    )
    reviewer_two_items = random.Random(arguments.seed + 202).sample(
        reviewer_two_items, len(reviewer_two_items)
    )
    packet = {
        "schema_version": 1,
        "run_id": readiness["run_id"],
        "sample_size": arguments.sample_size,
        "seed": arguments.seed,
        "sample_source_sha256": readiness["source_sha256"],
    }
    assessor_one_packet = {**packet, "items": reviewer_one_items}
    assessor_two_packet = {**packet, "items": reviewer_two_items}
    (arguments.output_dir / "assessor-1.json").write_text(
        json.dumps(assessor_one_packet, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (arguments.output_dir / "assessor-2.json").write_text(
        json.dumps(assessor_two_packet, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (arguments.output_dir / "assessor-1-ratings.json").write_text(
        json.dumps(build_assessor_ratings(assessor_one_packet, "", ""), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (arguments.output_dir / "assessor-2-ratings.json").write_text(
        json.dumps(build_assessor_ratings(assessor_two_packet, "", ""), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (arguments.output_dir / "reference-key.json").write_text(
        json.dumps({"items": reference_key}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_review_instructions(arguments.output_dir / "review-instructions.md")
    readiness["status"] = "ready"
    readiness["sample_active"] = True
    readiness["sample_item_count"] = len(selected)
    readiness["assessor_forms"] = [
        "assessor-1.json",
        "assessor-2.json",
        "assessor-1-ratings.json",
        "assessor-2-ratings.json",
    ]
    (arguments.output_dir / "readiness.json").write_text(
        json.dumps(readiness, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(readiness, ensure_ascii=False, sort_keys=True))
    return 0


def blank_rating_for_item(item: dict[str, Any]) -> dict[str, Any]:
    """Create the independent assessor fields for one blinded response."""
    return {
        "item_id": item["item_id"],
        "response_correctness": None,
        "hallucinated": None,
        "sentences": [
            {
                "sentence_id": sentence["sentence_id"],
                "unsupported_factual_claim": None,
                "prohibited_claim_present": None,
                "citations": [
                    {"citation_id": citation["citation_id"], "support": None, "note": ""}
                    for citation in sentence["citations"]
                ],
                "note": "",
            }
            for sentence in item["sentences"]
        ],
        "note": "",
    }


def build_assessor_ratings(packet: dict[str, Any], reviewer_id: str, reviewer_role: str) -> dict[str, Any]:
    """Build a blank rating form that preserves every sample item and citation denominator."""
    return {
        "schema_version": 1,
        "assessor_id": reviewer_id,
        "assessor_role": reviewer_role,
        "completed_at_utc": None,
        "items": [blank_rating_for_item(item) for item in packet["items"]],
    }


def write_review_instructions(path: Path) -> None:
    """Write a concise rubric that keeps independent reviewers on the same definitions."""
    path.write_text(
        "# Independent response review\n\n"
        "Each assessor completes their own JSON form without discussing items or seeing the other form. "
        "Do not open `reference-key.json`; it contains evaluation labels for the post-review coordinator.\n\n"
        "For each sentence, set `unsupported_factual_claim` to `yes`, `no`, or `uncertain`. Set "
        "`prohibited_claim_present` to the same values based on explicit applicable policy or a prohibited "
        "claim you can identify from the response and cited evidence. For each listed citation, set "
        "`support` to `supported`, `unsupported`, or `uncertain`: supported means the cited passage "
        "substantively supports that sentence. A resolvable citation ID alone is not support. Mark "
        "uncited sentences and unresolved citations in your notes.\n\n"
        "For each whole response, rate `response_correctness` as `correct`, `partly_correct`, `incorrect`, "
        "or `uncertain`; set `hallucinated` to `yes`, `no`, or `uncertain`. A response is hallucinated "
        "when at least one factual claim is unsupported by its cited evidence or contradicts that evidence. "
        "Do not treat missing evidence as proof that a claim is true. Leave no rating blank; use `uncertain` "
        "when the evidence does not support a confident judgement. Include assessor ID, role, and completion "
        "time in UTC.\n\n"
        "Citation accuracy denominator: every sentence-citation pair in the sample, including unresolved "
        "citations. The numerator counts `supported`; `unsupported` and `uncertain` remain in the denominator. "
        "Hallucination denominator: every reviewed response. Report assessor-specific counts and agreement. "
        "The coordinator records a reasoned adjudication for every disagreement before declaring final metrics.\n",
        encoding="utf-8",
    )


def validate_rating_form(packet: dict[str, Any], ratings: dict[str, Any], name: str) -> dict[str, dict[str, Any]]:
    """Check a completed independent form against the exact blinded packet structure."""
    if not isinstance(ratings.get("assessor_id"), str) or not ratings["assessor_id"].strip():
        raise ValueError(f"{name} must contain an assessor_id.")
    if not isinstance(ratings.get("assessor_role"), str) or not ratings["assessor_role"].strip():
        raise ValueError(f"{name} must contain an assessor_role.")
    if not isinstance(ratings.get("completed_at_utc"), str) or not ratings["completed_at_utc"].strip():
        raise ValueError(f"{name} must contain completed_at_utc.")
    expected = {item["item_id"]: item for item in packet["items"]}
    actual_rows = ratings.get("items")
    if not isinstance(actual_rows, list):
        raise ValueError(f"{name} must contain an items array.")
    actual = {row.get("item_id"): row for row in actual_rows if isinstance(row, dict)}
    if set(actual) != set(expected) or len(actual) != len(actual_rows):
        raise ValueError(f"{name} items do not match the blinded packet exactly.")
    for item_id, row in actual.items():
        if row.get("response_correctness") not in CORRECTNESS_RATINGS:
            raise ValueError(f"{name} item {item_id} needs a valid response_correctness rating.")
        if row.get("hallucinated") not in YES_NO_UNCERTAIN:
            raise ValueError(f"{name} item {item_id} needs a valid hallucinated rating.")
        expected_sentences = {
            sentence["sentence_id"]: sentence for sentence in expected[item_id]["sentences"]
        }
        actual_sentences = {
            sentence.get("sentence_id"): sentence
            for sentence in row.get("sentences", [])
            if isinstance(sentence, dict)
        }
        if set(actual_sentences) != set(expected_sentences):
            raise ValueError(f"{name} item {item_id} sentence ratings do not match the packet.")
        for sentence_id, sentence_rating in actual_sentences.items():
            if sentence_rating.get("unsupported_factual_claim") not in YES_NO_UNCERTAIN:
                raise ValueError(f"{name} item {item_id} sentence {sentence_id} needs an unsupported claim rating.")
            if sentence_rating.get("prohibited_claim_present") not in YES_NO_UNCERTAIN:
                raise ValueError(f"{name} item {item_id} sentence {sentence_id} needs a prohibited claim rating.")
            expected_citations = {
                citation["citation_id"]
                for citation in expected_sentences[sentence_id]["citations"]
            }
            actual_citations = {
                citation.get("citation_id"): citation
                for citation in sentence_rating.get("citations", [])
                if isinstance(citation, dict)
            }
            if set(actual_citations) != expected_citations:
                raise ValueError(f"{name} item {item_id} sentence {sentence_id} citations do not match the packet.")
            if any(
                citation.get("support") not in SUPPORT_RATINGS
                for citation in actual_citations.values()
            ):
                raise ValueError(f"{name} item {item_id} sentence {sentence_id} has an incomplete citation rating.")
    return actual


def validate_matching_assessor_packets(
    packet_one: dict[str, Any],
    packet_two: dict[str, Any],
) -> None:
    """Require both independent assessors to rate identical evidence in a different order."""
    metadata_one = {key: value for key, value in packet_one.items() if key != "items"}
    metadata_two = {key: value for key, value in packet_two.items() if key != "items"}
    if metadata_one != metadata_two:
        raise ValueError("Assessor packets have different sample metadata or source hashes.")
    items_one = packet_one.get("items")
    items_two = packet_two.get("items")
    if not isinstance(items_one, list) or not isinstance(items_two, list):
        raise ValueError("Both assessor packets must contain an items array.")
    by_id_one = {
        item.get("item_id"): item for item in items_one if isinstance(item, dict)
    }
    by_id_two = {
        item.get("item_id"): item for item in items_two if isinstance(item, dict)
    }
    if (
        len(by_id_one) != len(items_one)
        or len(by_id_two) != len(items_two)
        or set(by_id_one) != set(by_id_two)
    ):
        raise ValueError("Assessor packets do not contain the same unique blinded items.")
    if any(by_id_one[item_id] != by_id_two[item_id] for item_id in by_id_one):
        raise ValueError("Assessor packets contain different item text, sources, sentences, or citations.")


def citation_rating_rows(
    packet: dict[str, Any],
    ratings: dict[str, dict[str, Any]],
) -> dict[tuple[str, str, str], str]:
    """Map each sentence-citation pair to one assessor's support rating."""
    return {
        (item["item_id"], sentence["sentence_id"], citation["citation_id"]): citation["support"]
        for item in packet["items"]
        for sentence in item["sentences"]
        for citation in next(
            rating_sentence["citations"]
            for rating_sentence in ratings[item["item_id"]]["sentences"]
            if rating_sentence["sentence_id"] == sentence["sentence_id"]
        )
    }


def fraction_summary(numerator: int, denominator: int) -> dict[str, Any]:
    """Return explicit numerator, denominator, fraction, and zero-denominator status."""
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": numerator / denominator if denominator else None,
        "status": "available" if denominator else "not_applicable",
    }


def score_review_packet(arguments: argparse.Namespace) -> int:
    """Summarize independent ratings, disagreement, and target status after human review."""
    packet_one = json.loads(arguments.packet_one.read_text(encoding="utf-8"))
    packet_two = json.loads(arguments.packet_two.read_text(encoding="utf-8"))
    ratings_one_raw = json.loads(arguments.assessor_one.read_text(encoding="utf-8"))
    ratings_two_raw = json.loads(arguments.assessor_two.read_text(encoding="utf-8"))
    validate_matching_assessor_packets(packet_one, packet_two)
    if packet_one["sample_size"] != len(packet_one.get("items", [])) or packet_two["sample_size"] != len(packet_two.get("items", [])):
        raise ValueError("Assessor packet item counts do not match the declared sample size.")
    ratings_one = validate_rating_form(packet_one, ratings_one_raw, "assessor_one")
    ratings_two = validate_rating_form(packet_two, ratings_two_raw, "assessor_two")
    if ratings_one_raw["assessor_id"] == ratings_two_raw["assessor_id"]:
        raise ValueError("The two independent forms must have different assessor IDs.")
    first_citations = citation_rating_rows(packet_one, ratings_one)
    second_citations = citation_rating_rows(packet_two, ratings_two)
    if set(first_citations) != set(second_citations):
        raise ValueError("Assessor forms do not contain the same sentence-citation pairs.")

    citation_disagreements = [
        {"key": f"{item}|support|{sentence}|{citation}",
         "item_id": item, "sentence_id": sentence, "citation_id": citation,
         "assessor_1": first_citations[(item, sentence, citation)],
         "assessor_2": second_citations[(item, sentence, citation)]}
        for item, sentence, citation in sorted(first_citations)
        if first_citations[(item, sentence, citation)] != second_citations[(item, sentence, citation)]
    ]
    sentence_rating_rows = {
        (item_id, sentence["sentence_id"], field): sentence[field]
        for item_id, item_rating in ratings_one.items()
        for sentence in item_rating["sentences"]
        for field in ("unsupported_factual_claim", "prohibited_claim_present")
    }
    sentence_rating_rows_two = {
        (item_id, sentence["sentence_id"], field): sentence[field]
        for item_id, item_rating in ratings_two.items()
        for sentence in item_rating["sentences"]
        for field in ("unsupported_factual_claim", "prohibited_claim_present")
    }
    sentence_disagreements = [
        {"key": f"{item}|{field}|{sentence}",
         "item_id": item, "sentence_id": sentence, "field": field,
         "assessor_1": sentence_rating_rows[(item, sentence, field)],
         "assessor_2": sentence_rating_rows_two[(item, sentence, field)]}
        for item, sentence, field in sorted(sentence_rating_rows)
        if sentence_rating_rows[(item, sentence, field)] != sentence_rating_rows_two[(item, sentence, field)]
    ]
    response_disagreements = []
    response_agreement_counts = Counter()
    response_rows = []
    for item_id in sorted(ratings_one):
        first = ratings_one[item_id]
        second = ratings_two[item_id]
        response_row = {"item_id": item_id, "assessor_1": {}, "assessor_2": {}}
        for field in ("response_correctness", "hallucinated"):
            first_rating = first[field]
            second_rating = second[field]
            response_row["assessor_1"][field] = first_rating
            response_row["assessor_2"][field] = second_rating
            same = first_rating == second_rating
            response_agreement_counts[field] += int(same)
            if not same:
                response_disagreements.append(
                    {"key": f"{item_id}|{field}",
                     "item_id": item_id, "field": field, "assessor_1": first_rating,
                     "assessor_2": second_rating}
                )
        response_rows.append(response_row)

    disagreement_keys: dict[str, tuple[str, str, str | None]] = {}
    for row in citation_disagreements:
        key = f"{row['item_id']}|support|{row['sentence_id']}|{row['citation_id']}"
        disagreement_keys[key] = ("support", row["item_id"], row["sentence_id"])
    for row in sentence_disagreements:
        key = f"{row['item_id']}|{row['field']}|{row['sentence_id']}"
        disagreement_keys[key] = (row["field"], row["item_id"], row["sentence_id"])
    for row in response_disagreements:
        key = f"{row['item_id']}|{row['field']}"
        disagreement_keys[key] = (row["field"], row["item_id"], None)
    resolution_values: dict[str, str] = {}
    adjudication_record: dict[str, Any]
    if arguments.adjudications:
        adjudication_record = json.loads(arguments.adjudications.read_text(encoding="utf-8"))
        if not isinstance(adjudication_record.get("adjudicator_id"), str) or not adjudication_record["adjudicator_id"].strip():
            raise ValueError("The adjudications file must contain an adjudicator_id.")
        if not isinstance(adjudication_record.get("completed_at_utc"), str) or not adjudication_record["completed_at_utc"].strip():
            raise ValueError("The adjudications file must contain completed_at_utc.")
        resolutions = adjudication_record.get("resolutions")
        if not isinstance(resolutions, list):
            raise ValueError("The adjudications file must contain a resolutions array.")
        for resolution in resolutions:
            if not isinstance(resolution, dict):
                raise ValueError("Each adjudication must be a JSON object.")
            key = resolution.get("key")
            decision = resolution.get("decision")
            rationale = resolution.get("rationale")
            if key not in disagreement_keys or key in resolution_values:
                raise ValueError("Adjudication keys must match each disagreement exactly once.")
            field = disagreement_keys[key][0]
            allowed = SUPPORT_RATINGS if field == "support" else CORRECTNESS_RATINGS if field == "response_correctness" else YES_NO_UNCERTAIN
            if decision not in allowed or not isinstance(rationale, str) or not rationale.strip():
                raise ValueError(f"Adjudication {key} needs an allowed decision and a written rationale.")
            resolution_values[key] = decision
        if set(resolution_values) != set(disagreement_keys):
            raise ValueError("Every assessor disagreement must have one documented adjudication.")
        adjudication_record["status"] = "resolved"
    else:
        adjudication_record = {
            "status": "pending" if disagreement_keys else "not_required",
            "unresolved_keys": sorted(disagreement_keys),
            "resolutions": [],
        }

    final_citation_ratings = dict(first_citations)
    final_hallucination_ratings = {
        item_id: rating["hallucinated"] for item_id, rating in ratings_one.items()
    }
    for key, decision in resolution_values.items():
        parts = key.split("|")
        if len(parts) == 4 and parts[1] == "support":
            final_citation_ratings[(parts[0], parts[2], parts[3])] = decision
        elif len(parts) == 2 and parts[1] == "hallucinated":
            final_hallucination_ratings[parts[0]] = decision

    cited_sentence_count = sum(
        bool(sentence["citations"])
        for item in packet_one["items"]
        for sentence in item["sentences"]
    )
    uncited_sentence_count = sum(
        sentence["citation_missing"]
        for item in packet_one["items"]
        for sentence in item["sentences"]
    )
    unresolved_citation_count = sum(
        not citation["resolved"]
        for item in packet_one["items"]
        for sentence in item["sentences"]
        for citation in sentence["citations"]
    )

    per_assessor = {}
    for assessor_name, citation_ratings, response_ratings, raw in (
        ("assessor_1", first_citations, ratings_one, ratings_one_raw),
        ("assessor_2", second_citations, ratings_two, ratings_two_raw),
    ):
        supported = sum(value == "supported" for value in citation_ratings.values())
        uncertain_citations = sum(value == "uncertain" for value in citation_ratings.values())
        hallucinated = sum(row["hallucinated"] == "yes" for row in response_ratings.values())
        uncertain_hallucinations = sum(row["hallucinated"] == "uncertain" for row in response_ratings.values())
        sentence_rows = [sentence for row in response_ratings.values() for sentence in row["sentences"]]
        unsupported_claim_count = sum(
            sentence["unsupported_factual_claim"] == "yes" for sentence in sentence_rows
        )
        prohibited_claim_count = sum(
            sentence["prohibited_claim_present"] == "yes" for sentence in sentence_rows
        )
        per_assessor[assessor_name] = {
            "assessor_id": raw["assessor_id"],
            "assessor_role": raw["assessor_role"],
            "citation_accuracy": fraction_summary(supported, len(citation_ratings)),
            "uncertain_citation_judgements": uncertain_citations,
            "hallucination_rate": fraction_summary(hallucinated, len(response_ratings)),
            "uncertain_hallucination_responses": uncertain_hallucinations,
            "unsupported_claims": fraction_summary(unsupported_claim_count, len(sentence_rows)),
            "prohibited_claims": fraction_summary(prohibited_claim_count, len(sentence_rows)),
        }
        citation_accuracy = per_assessor[assessor_name]["citation_accuracy"]["value"]
        hallucination_rate = per_assessor[assessor_name]["hallucination_rate"]["value"]
        per_assessor[assessor_name]["citation_accuracy_target_status"] = (
            "incomplete" if uncertain_citations
            else "not_applicable" if citation_accuracy is None
            else "met" if citation_accuracy >= 0.95
            else "below_target"
        )
        per_assessor[assessor_name]["hallucination_target_status"] = (
            "incomplete" if uncertain_hallucinations
            else "not_applicable" if hallucination_rate is None
            else "met" if hallucination_rate <= 0.05
            else "below_target"
        )

    citation_agreement = sum(
        first_citations[key] == second_citations[key] for key in first_citations
    )
    agreement = {
        "citation_support": fraction_summary(citation_agreement, len(first_citations)),
        "hallucination": fraction_summary(
            response_agreement_counts["hallucinated"], len(response_rows)
        ),
        "response_correctness": fraction_summary(
            response_agreement_counts["response_correctness"], len(response_rows)
        ),
        "unsupported_factual_claim": fraction_summary(
            sum(
                sentence_rating_rows[key] == sentence_rating_rows_two[key]
                for key in sentence_rating_rows
                if key[2] == "unsupported_factual_claim"
            ),
            sum(key[2] == "unsupported_factual_claim" for key in sentence_rating_rows),
        ),
        "prohibited_claim_present": fraction_summary(
            sum(
                sentence_rating_rows[key] == sentence_rating_rows_two[key]
                for key in sentence_rating_rows
                if key[2] == "prohibited_claim_present"
            ),
            sum(key[2] == "prohibited_claim_present" for key in sentence_rating_rows),
        ),
    }
    final_citation_supported = sum(value == "supported" for value in final_citation_ratings.values())
    final_citation_uncertain = sum(value == "uncertain" for value in final_citation_ratings.values())
    final_hallucinated = sum(value == "yes" for value in final_hallucination_ratings.values())
    final_hallucination_uncertain = sum(
        value == "uncertain" for value in final_hallucination_ratings.values()
    )
    citation_disagreement_keys = [key for key, (field, _, _) in disagreement_keys.items() if field == "support"]
    hallucination_disagreement_keys = [key for key, (field, _, _) in disagreement_keys.items() if field == "hallucinated"]
    citation_resolution_pending = any(key not in resolution_values for key in citation_disagreement_keys)
    hallucination_resolution_pending = any(key not in resolution_values for key in hallucination_disagreement_keys)
    citation_target_status = (
        "pending_disagreement_resolution" if citation_resolution_pending
        else "incomplete" if final_citation_uncertain or not final_citation_ratings
        else "met" if final_citation_supported / len(final_citation_ratings) >= 0.95
        else "below_target"
    )
    hallucination_target_status = (
        "pending_disagreement_resolution" if hallucination_resolution_pending
        else "incomplete" if final_hallucination_uncertain or not final_hallucination_ratings
        else "met" if final_hallucinated / len(final_hallucination_ratings) <= 0.05
        else "below_target"
    )
    results = {
        "schema_version": 1,
        "run_id": packet_one["run_id"],
        "sample_size": len(packet_one["items"]),
        "assessors": per_assessor,
        "denominators": {
            "reviewed_responses": len(response_rows),
            "sentence_count": sum(len(item["sentences"]) for item in packet_one["items"]),
            "cited_sentence_count": cited_sentence_count,
            "uncited_sentence_count": uncited_sentence_count,
            "sentence_citation_pairs": len(first_citations),
            "unresolved_citation_count": unresolved_citation_count,
        },
        "agreement": agreement,
        "citation_disagreements": citation_disagreements,
        "sentence_disagreements": sentence_disagreements,
        "response_disagreements": response_disagreements,
        "adjudications": adjudication_record,
        "targets": {
            "citation_accuracy": {
                "threshold": 0.95,
                "assessor_1": per_assessor["assessor_1"]["citation_accuracy_target_status"],
                "assessor_2": per_assessor["assessor_2"]["citation_accuracy_target_status"],
                "final_rate": fraction_summary(final_citation_supported, len(final_citation_ratings)),
                "status": citation_target_status,
            },
            "hallucination_rate": {
                "threshold": 0.05,
                "assessor_1": per_assessor["assessor_1"]["hallucination_target_status"],
                "assessor_2": per_assessor["assessor_2"]["hallucination_target_status"],
                "final_rate": fraction_summary(final_hallucinated, len(final_hallucination_ratings)),
                "status": hallucination_target_status,
            },
        },
        "assessment_status": "incomplete" if len(response_rows) < MINIMUM_SAMPLE_SIZE else "two_assessors_recorded",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    arguments.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(arguments.output), "assessment_status": results["assessment_status"],
                      "responses": len(response_rows), "citation_pairs": len(first_citations),
                      "citation_disagreements": len(citation_disagreements),
                      "response_disagreements": len(response_disagreements)}, sort_keys=True))
    return 0


def create_argument_parser() -> argparse.ArgumentParser:
    """Expose post-run human-review preparation and scoring commands."""
    parser = argparse.ArgumentParser(description="Prepare blinded human review of generated development drafts.")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare", help="sample drafts and create two independent blinded forms")
    prepare_parser.add_argument("--run-dir", required=True, type=Path)
    prepare_parser.add_argument("--input", required=True, type=Path)
    prepare_parser.add_argument("--ground-truth", required=True, type=Path)
    prepare_parser.add_argument("--output-dir", required=True, type=Path)
    prepare_parser.add_argument("--sample-size", type=int, default=MINIMUM_SAMPLE_SIZE)
    prepare_parser.add_argument("--seed", type=int, default=24024)
    prepare_parser.set_defaults(handler=prepare_review_packet)
    score_parser = commands.add_parser("score", help="score completed independent forms and preserve disagreement")
    score_parser.add_argument("--packet-one", required=True, type=Path)
    score_parser.add_argument("--packet-two", required=True, type=Path)
    score_parser.add_argument("--assessor-one", required=True, type=Path)
    score_parser.add_argument("--assessor-two", required=True, type=Path)
    score_parser.add_argument("--adjudications", type=Path, help="documented human resolutions for all disagreements")
    score_parser.add_argument("--output", required=True, type=Path)
    score_parser.set_defaults(handler=score_review_packet)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run a review subcommand and report input or evidence failures clearly."""
    parser = create_argument_parser()
    arguments = parser.parse_args(argv)
    try:
        return arguments.handler(arguments)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        print(f"human citation review: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
