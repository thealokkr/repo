"""Command-line entry point for CloudServe Support Automation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from dotenv import load_dotenv

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
    run_parser.add_argument(
        "--label",
        help="Optional run label stored with the run and used to name its output folder",
    )
    run_parser.add_argument(
        "--skip-draft",
        action="store_true",
        help="Escalate draft-eligible tickets without invoking draft generation",
    )
    run_parser.add_argument(
        "--skip-draft-review",
        action="store_true",
        help="Emit generated drafts without in-run review checks (capstone testing only)",
    )

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
            "Use Gemini 3.8 Flash after processing to score response quality, citation support, "
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
    load_dotenv(REPOSITORY_ROOT / ".env", override=False)
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
        _run_id, output_directory, _metrics = run_ticket_batch(
            input_path=parsed.input,
            output_root=parsed.output_root,
            database_path=parsed.database,
            policy_path=parsed.policy,
            label=parsed.label,
            skip_draft=parsed.skip_draft,
            skip_draft_review=parsed.skip_draft_review,
            progress_callback=_print_run_progress,
        )
    except Exception as error:
        print(f"cloudserve: error: {error}", file=sys.stderr)
        return 1
    metrics_path = Path(output_directory) / "metrics.md"
    try:
        metrics_output = metrics_path.read_text(encoding="utf-8").strip()
    except OSError as error:
        print(f"cloudserve: could not read metrics output: {error}", file=sys.stderr)
        return 1
    print(f"Output folder: {output_directory}")
    print("Metrics (metrics.md):")
    print(metrics_output)
    return 0


def _print_run_progress(stage: str, status: str) -> None:
    """Print run-stage transitions and grouped per-ticket processing steps."""
    if status == "ticket":
        print(f"{stage}:", flush=True)
    elif status == "ticket_step":
        if stage.startswith("  "):
            print(f"  {stage}", flush=True)
        else:
            print(f"  - {stage}", flush=True)
    elif status == "ticket_separator":
        print(flush=True)
    else:
        message = "in progress..." if status == "in_progress" else "done."
        print(f"{stage} {message}", flush=True)
