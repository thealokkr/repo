"""CLI process status contracts."""

from __future__ import annotations

import re
from pathlib import Path

from cloudserve_support import cli


def test_run_fails_immediately_when_gemini_key_is_missing_from_env(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    called = False

    def fail_if_called(**kwargs):
        nonlocal called
        called = True
        raise AssertionError("run_ticket_batch should not be called without a Gemini key")

    monkeypatch.setattr(cli, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(cli, "run_ticket_batch", fail_if_called)

    assert cli.main(["run", "--input", "data/tickets/validation_tickets.json"]) == 1
    assert called is False
    stderr = capsys.readouterr().err
    assert "GEMINI_API_KEY or GOOGLE_API_KEY is required" in stderr
    assert ".env" in stderr

    (tmp_path / ".env").write_text("GEMINI_API_KEY=\nGOOGLE_API_KEY=\n", encoding="utf-8")
    assert cli.main(["run", "--input", "data/tickets/validation_tickets.json"]) == 1
    assert called is False
    stderr = capsys.readouterr().err
    assert "GEMINI_API_KEY or GOOGLE_API_KEY is required" in stderr
    assert ".env" in stderr


def test_run_startup_failure_returns_exit_status_one(monkeypatch, capsys) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    def fail_before_processing(**kwargs):
        del kwargs
        raise OSError("input path is unavailable")

    monkeypatch.setattr(cli, "run_ticket_batch", fail_before_processing)

    assert cli.main(["run", "--input", "unavailable.json"]) == 1
    assert "input path is unavailable" in capsys.readouterr().err


def test_run_prints_stage_progress_output_folder_and_metrics_table(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    output_directory = tmp_path / "run-123"
    output_directory.mkdir()
    metrics_content = (
        "# Run metrics\n\n"
        "| Label | Matched | Different | Not compared | Match rate |\n"
        "| --- | ---: | ---: | ---: | ---: |\n"
        "| intent | 1 | 0 | 0 | 100.0% |"
    )
    (output_directory / "metrics.md").write_text(metrics_content, encoding="utf-8")

    def fake_run_ticket_batch(**kwargs):
        progress_callback = kwargs["progress_callback"]
        progress_callback("Classification", "in_progress")
        progress_callback("Classification", "done")
        progress_callback("Retrieval", "in_progress")
        progress_callback("Retrieval", "done")
        progress_callback("Routing", "in_progress")
        routing_label = "Routing ticket 1/2 ('DEV-0001')"
        progress_callback(routing_label, "routing_ticket_started")
        progress_callback(
            "Gemini claim-grounding request took 1.23s", "routing_ticket_detail"
        )
        progress_callback(
            "Routing decision: draft_for_review (eligible)", "routing_ticket_detail"
        )
        progress_callback(routing_label, "routing_ticket_finished")
        progress_callback("Routing", "done")
        progress_callback("Ticket processing", "in_progress")
        progress_callback("Ticket 1/2 ('DEV-0001')", "ticket")
        progress_callback(
            "Pre-draft check results (computed during routing):", "ticket_step"
        )
        progress_callback("  - Data checks:", "ticket_step")
        progress_callback(
            "    - classification_confidence_check [passed]: confidence is sufficient",
            "ticket_step",
        )
        progress_callback("Draft eligibility: eligible", "ticket_step")
        progress_callback("Generating draft", "ticket_step")
        progress_callback("Draft generation provider request took 0.45s", "ticket_step")
        progress_callback("Post-draft checks:", "ticket_step")
        progress_callback(
            "  - citation_grounding [passed]: claims are source-grounded",
            "ticket_step",
        )
        progress_callback("Ticket result persisted", "ticket_step")
        progress_callback("", "ticket_separator")
        progress_callback("Ticket 2/2 ('DEV-0002')", "ticket")
        progress_callback("Draft generation skipped (route: escalate)", "ticket_step")
        progress_callback("", "ticket_separator")
        progress_callback("Ticket processing", "done")
        progress_callback("Export finalization", "in_progress")
        progress_callback("Export finalization", "done")
        return "run-123", output_directory, {"internal_metric_dump": "hidden"}

    monkeypatch.setattr(cli, "run_ticket_batch", fake_run_ticket_batch)

    assert cli.main(
        [
            "run",
            "--input",
            str(tmp_path / "unused.json"),
            "--output-root",
            str(tmp_path / "outputs"),
            "--label",
            "quick acceptance",
        ]
    ) == 0

    output = capsys.readouterr().out
    assert "Classification in progress..." in output
    assert re.search(r"Classification done in [0-9]+\.[0-9]{2}s\.", output)
    assert re.search(r"Retrieval done in [0-9]+\.[0-9]{2}s\.", output)
    assert "Routing ticket 1/2 ('DEV-0001') in progress..." in output
    assert re.search(r"Routing ticket 1/2 .* done in [0-9]+\.[0-9]{2}s\.", output)
    assert "Gemini claim-grounding request took 1.23s" in output
    assert re.search(r"Routing done in [0-9]+\.[0-9]{2}s\.", output)
    assert (
        "Ticket processing in progress...\n"
        "Ticket 1/2 ('DEV-0001'):\n"
        "  - Pre-draft check results (computed during routing):\n"
        "    - Data checks:\n"
        "      - classification_confidence_check [passed]: confidence is sufficient\n"
        "  - Draft eligibility: eligible\n"
        "  - Generating draft\n"
        "  - Draft generation provider request took 0.45s\n"
        "  - Post-draft checks:\n"
        "    - citation_grounding [passed]: claims are source-grounded\n"
        "  - Ticket result persisted\n"
        "  - Ticket 1/2 ('DEV-0001') completed in "
    ) in output
    assert "Ticket 2/2 ('DEV-0002'):\n  - Draft generation skipped (route: escalate)" in output
    assert "Ticket 2/2 ('DEV-0002') completed in " in output
    assert re.search(r"Ticket processing done in [0-9]+\.[0-9]{2}s\.", output)
    assert re.search(r"Export finalization done in [0-9]+\.[0-9]{2}s\.", output)
    assert f"Output folder: {output_directory}" in output
    assert "Metrics (metrics.md):" in output
    assert metrics_content in output
    assert "internal_metric_dump" not in output
    assert "Development evaluation summary:" not in output


def test_quality_judge_duration_is_reported_outside_production_latency(
    monkeypatch, capsys
) -> None:
    scorecard = {
        "run_id": "a" * 32,
        "status": "complete",
        "evaluated_count": 20,
        "target_sample_size": 20,
        "failed_count": 0,
        "missing_count": 0,
        "reference_available_count": 20,
        "response_quality_proxy": {
            "value": 4.5,
            "status": "pass",
            "numerator": 90,
            "denominator": 20,
            "target": ">=4.0/5",
        },
        "citation_support_rate": {
            "value": 1.0,
            "status": "pass",
            "numerator": 20,
            "denominator": 20,
            "target": ">=95%",
        },
        "hallucination_estimate": {
            "value": 0.0,
            "status": "pass",
            "numerator": 0,
            "denominator": 20,
            "target": "<=5%",
        },
    }
    clock_values = iter((1.0, 3.25))
    monkeypatch.setattr(cli, "evaluate_quality_run", lambda **_: scorecard)
    monkeypatch.setattr(cli.time, "perf_counter", lambda: next(clock_values))

    assert cli.main(
        ["evaluate-quality", "--run-dir", "outputs/run", "--input", "development.json"]
    ) == 0

    output = capsys.readouterr().out
    assert "Post-run judge duration: 2.250s" in output
    assert "excluded from production ticket latency" in output
