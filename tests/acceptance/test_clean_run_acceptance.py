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
    setup_script = (REPOSITORY_ROOT / "setup" / "setup.sh").read_text(encoding="utf-8")

    assert 'requires-python = ">=3.13,<3.15"' in pyproject
    assert lockfile.is_file()
    assert "GEMINI_API_KEY=" in example
    assert "GOOGLE_API_KEY=" in example
    assert "GEMINI_API_KEY=AIza" not in example
    assert "GOOGLE_API_KEY=AIza" not in example
    assert "GEMINI_API_KEY=AIza" not in readme
    assert "GOOGLE_API_KEY=AIza" not in readme
    assert "uv run cloudserve run --input data/tickets/validation_tickets.json --skip-draft --skip-draft-review" in readme
    assert "uv run cloudserve run --input data/tickets/validation_tickets.json --skip-draft --skip-draft-review" in setup_script
    assert "--database" not in readme
    assert "may incur provider charges" in readme
    assert ".env" in ignore_rules
    assert "data/runtime/*" in ignore_rules
    assert "outputs/*" in ignore_rules
