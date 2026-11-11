"""Command-line entry point for CloudServe Support Automation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from cloudserve_support.processing.run_tickets import (
    DEFAULT_DATABASE_PATH,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_ROUTING_POLICY_CONFIG_PATH,
    run_ticket_batch,
)
from cloudserve_support.errors import CloudServeError
from cloudserve_support.evaluation.quality_judge import (
    QUALITY_JUDGE_SAMPLE_SEED,
    evaluate_quality_run,
)
from cloudserve_support.config import Settings
from cloudserve_support.reporting.terminal_summary import (
    build_terminal_summary,
    format_terminal_summary,
)
from cloudserve_support.routing.policy_calibration import (
    calibrate_development_routing_policy,
    write_frozen_routing_policy,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    """Create the CloudServe CLI parser with the public ticket batch command."""
    parser = argparse.ArgumentParser(
        prog="cloudserve",
        description="CloudServe Support Automation command-line tools.",
    )
    commands = parser.add_subparsers(dest="command")
    run_parser = commands.add_parser(
        "run",
        help="process a JSON ticket batch and write one isolated result run",
        description="Persist one route and audit event for each record in a JSON ticket array.",
    )
    run_parser.add_argument("--input", required=True, type=Path, help="Path to a JSON ticket array")
    run_parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Directory for generated run folders (default: {DEFAULT_OUTPUT_ROOT})",
    )
    run_parser.add_argument(
        "--database",
        type=Path,
        default=DEFAULT_DATABASE_PATH,
        help=f"SQLite runtime database path (default: {DEFAULT_DATABASE_PATH})",
    )
    run_parser.add_argument(
        "--policy",
        type=Path,
        default=DEFAULT_ROUTING_POLICY_CONFIG_PATH,
        help=(
            "Frozen development-calibrated routing policy "
            f"(default: {DEFAULT_ROUTING_POLICY_CONFIG_PATH})"
        ),
    )
    run_parser.add_argument("--label", help="Optional human-readable label stored with this run")

    calibrate_parser = commands.add_parser(
        "calibrate",
        help="freeze a routing confidence threshold from development tickets",
        description="Calibrate once from development operational fields; validation runs reuse the frozen file.",
    )
    calibrate_parser.add_argument(
        "--development-input",
        required=True,
        type=Path,
        help="Development ticket JSON array used only for operational confidence distribution",
    )
    calibrate_parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ROUTING_POLICY_CONFIG_PATH,
        help=f"Path for the frozen policy JSON (default: {DEFAULT_ROUTING_POLICY_CONFIG_PATH})",
    )
    quality_parser = commands.add_parser(
        "evaluate-quality",
        help="judge a reproducible 20-draft sample after a completed run",
        description=(
            "Use GPT-6 Luna after processing to score response quality, citation support, "
            "and unsupported factual claims without changing ticket results or audits."
        ),
    )
    quality_parser.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help="Completed run directory containing metrics, results, and decision log exports",
    )
    quality_parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Exact ticket input file used by the completed run",
    )
    quality_parser.add_argument(
        "--ground-truth",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "evaluation" / "ground_truth_responses.json",
        help="Optional matching reference responses and rubric criteria",
    )
    quality_parser.add_argument(
        "--sample-seed",
        default=QUALITY_JUDGE_SAMPLE_SEED,
        help="Stable sample seed recorded in metrics (default: fixed rubric seed)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selected CLI command and return a shell-compatible exit code."""
    parser = build_parser()
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        parser.print_help()
        return 0
    parsed = parser.parse_args(arguments)
    if parsed.command == "calibrate":
        try:
            policy = calibrate_development_routing_policy(parsed.development_input)
            write_frozen_routing_policy(policy, parsed.output.expanduser().resolve())
        except (CloudServeError, OSError, RuntimeError, ValueError) as error:
            print(f"cloudserve: error: {error}", file=sys.stderr)
            return 2
        print(f"Frozen policy: {parsed.output.expanduser().resolve()}")
        print(
            json.dumps(
                {
                    "policy_version": policy["policy_version"],
                    "classification_confidence_threshold": policy[
                        "classification_confidence_threshold"
                    ],
                    "calibration_source": policy["calibration_source"],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    if parsed.command == "evaluate-quality":
        judge_started = time.perf_counter()
        try:
            scorecard = evaluate_quality_run(
                run_directory=parsed.run_dir,
                input_path=parsed.input,
                ground_truth_path=parsed.ground_truth,
                sample_seed=parsed.sample_seed,
                settings=Settings.from_quality_judge_env(),
            )
        except (CloudServeError, OSError, RuntimeError, ValueError) as error:
            print(f"cloudserve: error: {error}", file=sys.stderr)
            return 2
        print(f"Run ID: {scorecard['run_id']}")
        print(f"Quality judge status: {scorecard['status']}")
        print(
            f"Sample: {scorecard['evaluated_count']}/{scorecard['target_sample_size']} "
            f"evaluated; {scorecard['failed_count']} failed; "
            f"{scorecard['missing_count']} missing; "
            f"{scorecard['reference_available_count']} references available."
        )
        for name, criterion in (
            ("Response quality", scorecard["response_quality_proxy"]),
            ("Citation support", scorecard["citation_support_rate"]),
            ("Unsupported-response rate", scorecard["hallucination_estimate"]),
        ):
            value = criterion["value"]
            rendered = "not measurable" if value is None else f"{value:.4f}"
            print(
                f"{name}: {rendered} — {criterion['status']} "
                f"({criterion['numerator']}/{criterion['denominator']}; "
                f"target {criterion['target']})."
            )
        print(f"Judge artifact: {parsed.run_dir.expanduser().resolve() / 'quality_judgments.jsonl'}")
        print(
            "Post-run judge duration: "
            f"{time.perf_counter() - judge_started:.3f}s "
            "(excluded from production ticket latency)."
        )
        return 0 if scorecard["status"] == "complete" else 1
    if parsed.command != "run":
        parser.print_help()
        return 0

    try:
        run_id, output_directory, metrics = run_ticket_batch(
            input_path=parsed.input,
            output_root=parsed.output_root,
            database_path=parsed.database,
            policy_path=parsed.policy,
            label=parsed.label,
        )
    except Exception as error:
        print(f"cloudserve: error: {error}", file=sys.stderr)
        return 1
    print(f"Run ID: {run_id}")
    print(f"Output directory: {output_directory}")
    print(
        "Batch export and finalization duration: "
        f"{metrics['batch_export_duration_seconds']:.3f}s "
        "(reported separately from per-ticket processing)."
    )
    print(json.dumps(metrics, ensure_ascii=False, sort_keys=True))
    try:
        input_records = json.loads(parsed.input.read_text(encoding="utf-8"))
        results = _read_jsonl(Path(output_directory) / "results.jsonl")
        summary = build_terminal_summary(
            input_records=input_records if isinstance(input_records, list) else [],
            results=results,
            metrics=metrics,
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(f"cloudserve: could not build post-run terminal summary: {error}", file=sys.stderr)
        return 1
    print(format_terminal_summary(summary))
    return 0


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    """Read persisted result rows for a post-run summary after processing ends."""
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows
