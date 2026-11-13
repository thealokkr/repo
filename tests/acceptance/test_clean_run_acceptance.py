"""Credential-free acceptance checks for clean setup guidance."""

from __future__ import annotations

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_clean_checkout_configuration_is_pinned_and_secret_safe() -> None:
    pyproject = (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    lockfile = REPOSITORY_ROOT / "uv.lock"
    example = (REPOSITORY_ROOT / ".env.example").read_text(encoding="utf-8")
    readme = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
    ignore_rules = (REPOSITORY_ROOT / ".gitignore").read_text(encoding="utf-8")

    assert 'requires-python = ">=3.13,<3.15"' in pyproject
    assert lockfile.is_file()
    assert "GEMINI_API_KEY=" in example
    assert "GOOGLE_API_KEY=" in example
    assert "GEMINI_API_KEY=AIza" not in example
    assert "GOOGLE_API_KEY=AIza" not in example
    gate_instructions = readme.split("### Reproducible development and validation gates", 1)[1]
    gate_instructions = gate_instructions.split("\n### ", 1)[0]
    assert gate_instructions.count("uv run cloudserve run") == 2
    assert "GEMINI_API_KEY=" not in gate_instructions
    assert "GOOGLE_API_KEY=" not in gate_instructions
    assert "cannot establish whether retrieval and citation-quality criteria pass" in gate_instructions
    assert "may incur provider charges" in gate_instructions
    assert "does not bypass packaged-index freshness checks" in gate_instructions
    assert "Label comparisons remain visible diagnostics and do not require exact equality" in gate_instructions
    assert "Only after the final development criteria gate has passed" in gate_instructions
    assert ".env" in ignore_rules
    assert "data/runtime/*" in ignore_rules
    assert "outputs/*" in ignore_rules
