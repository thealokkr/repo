"""Verify the portable distribution includes required data and excludes local state."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PACKAGING_SCRIPT = REPOSITORY_ROOT / "scripts/package_source.py"


def _create_archive(repository_root: Path, archive_path: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(PACKAGING_SCRIPT),
            "--repository-root",
            str(repository_root),
            "--output",
            str(archive_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_source_archive_contains_project_inputs_and_excludes_generated_state(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "cloudserve-support.zip"
    _create_archive(REPOSITORY_ROOT, archive_path)

    with ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        assert archive.testzip() is None

    required = {
        "cloudserve-support/README.md",
        "cloudserve-support/Agent.md",
        "cloudserve-support/pyproject.toml",
        "cloudserve-support/uv.lock",
        "cloudserve-support/.env.example",
        "cloudserve-support/data/kb/documentation.json",
        "cloudserve-support/data/kb/metadata_manifest.json",
        "cloudserve-support/data/index/manifest.json",
        "cloudserve-support/data/tickets/development_tickets.json",
        "cloudserve-support/data/tickets/validation_tickets.json",
        "cloudserve-support/data/policy/frozen_routing_policy.json",
        "cloudserve-support/src/cloudserve_support/cli.py",
    }
    assert required <= names
    assert "cloudserve-support/.env" not in names
    assert not any(
        name.startswith("cloudserve-support/.env.")
        and name != "cloudserve-support/.env.example"
        for name in names
    )
    assert "cloudserve-support/.env.example" in names
    assert not any("/.venv/" in name for name in names)
    assert not any("/__pycache__/" in name for name in names)
    assert not any("/.pytest_cache/" in name for name in names)
    assert not any(name.startswith("cloudserve-support/.agents/") for name in names)
    assert not any(name.startswith("cloudserve-support/.workarea/") for name in names)
    assert not any(name.startswith("cloudserve-support/.codex/") for name in names)
    assert not any(
        name.startswith("cloudserve-support/data/runtime/")
        and name != "cloudserve-support/data/runtime/.gitkeep"
        for name in names
    )
    assert not any(
        name.startswith("cloudserve-support/outputs/")
        and not name.endswith("/outputs/.gitkeep")
        for name in names
    )


def test_source_archive_filter_excludes_all_local_env_and_generated_files(
    tmp_path: Path,
) -> None:
    repository_root = tmp_path / "fixture-repo"
    for relative_path in (
        "README.md",
        ".env.example",
        ".env",
        ".env.local",
        ".venv/bin/python",
        ".agents/skills/implement-task/SKILL.md",
        ".workarea/feature-log/CS-000.md",
        ".codex/development-settings.json",
        "src/module.py",
        "src/__pycache__/module.pyc",
        "data/runtime/cloudserve.sqlite3",
        "data/runtime/.gitkeep",
        "data/runtime/old/.gitkeep",
        "outputs/run/results.jsonl",
        "outputs/run/.gitkeep",
        "outputs/.gitkeep",
        "data/kb/documentation.json",
    ):
        target = repository_root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("fixture", encoding="utf-8")

    archive_path = tmp_path / "fixture.zip"
    _create_archive(repository_root, archive_path)

    with ZipFile(archive_path) as archive:
        names = set(archive.namelist())
    assert names == {
        "cloudserve-support/README.md",
        "cloudserve-support/.env.example",
        "cloudserve-support/src/module.py",
        "cloudserve-support/data/runtime/.gitkeep",
        "cloudserve-support/outputs/.gitkeep",
        "cloudserve-support/data/kb/documentation.json",
    }
    assert not any(name.startswith("cloudserve-support/.agents/") for name in names)
    assert not any(name.startswith("cloudserve-support/.workarea/") for name in names)
    assert not any(name.startswith("cloudserve-support/.codex/") for name in names)
