"""CLI process status contracts."""

from __future__ import annotations

from pathlib import Path

from cloudserve_support import cli


def test_run_startup_failure_returns_exit_status_one(monkeypatch, capsys) -> None:
    def fail_before_processing(**kwargs):
        del kwargs
        raise OSError("input path is unavailable")

    monkeypatch.setattr(cli, "run_ticket_batch", fail_before_processing)

    assert cli.main(["run", "--input", "unavailable.json"]) == 1
    assert "input path is unavailable" in capsys.readouterr().err


def test_run_prints_stage_progress_and_only_run_name_and_metrics(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    output_directory = tmp_path / "run-123"
    output_directory.mkdir()
    metrics_content = "# Run metrics\n\n- Total tickets: 1"
    (output_directory / "metrics.md").write_text(metrics_content, encoding="utf-8")

    def fake_run_ticket_batch(**kwargs):
        progress_callback = kwargs["progress_callback"]
        progress_callback("Classification", "in_progress")
        progress_callback("Classification", "done")
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
    assert "Classification done." in output
    assert "Run name: quick acceptance" in output
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
