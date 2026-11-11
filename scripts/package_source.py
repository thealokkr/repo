"""Create a portable source ZIP without secrets or generated local state."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ARCHIVE_ROOT = "cloudserve-support"
EXCLUDED_DIRECTORIES = {
    ".git",
    ".agents",
    ".workarea",
    ".codex",
    ".venv",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "__pycache__",
    "build",
    "dist",
}


def create_source_archive(repository_root: Path, archive_path: Path) -> list[str]:
    """Write a ZIP containing project files but no secrets or generated state."""
    root = repository_root.expanduser().resolve()
    destination = archive_path.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Repository root does not exist: {root}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    included_files = _portable_source_files(root, destination)
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = temporary_file.name
        with ZipFile(temporary_path, "w", compression=ZIP_DEFLATED) as archive:
            for source_path in included_files:
                relative_path = source_path.relative_to(root).as_posix()
                archive.write(source_path, f"{ARCHIVE_ROOT}/{relative_path}")
        with ZipFile(temporary_path) as archive:
            corrupt_entry = archive.testzip()
            if corrupt_entry is not None:
                raise RuntimeError(f"Archive verification failed for {corrupt_entry}")
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)
    return [f"{ARCHIVE_ROOT}/{path.relative_to(root).as_posix()}" for path in included_files]


def _portable_source_files(repository_root: Path, archive_path: Path) -> list[Path]:
    """Select sorted source and fixture files while excluding local state."""
    selected: list[Path] = []
    for path in sorted(repository_root.rglob("*")):
        if path.is_dir() or path.is_symlink():
            continue
        relative_path = path.relative_to(repository_root)
        if path.resolve() == archive_path:
            continue
        if any(part in EXCLUDED_DIRECTORIES for part in relative_path.parts):
            continue
        if relative_path.parts[:2] == ("data", "runtime") and relative_path.parts != (
            "data",
            "runtime",
            ".gitkeep",
        ):
            continue
        if relative_path.parts[:1] == ("outputs",) and relative_path.parts != (
            "outputs",
            ".gitkeep",
        ):
            continue
        if path.name == ".env" or (
            path.name.startswith(".env.") and path.name != ".env.example"
        ):
            continue
        if path.name == ".DS_Store" or path.suffix in {".pyc", ".pyo"}:
            continue
        selected.append(path)
    return selected


def main(argv: list[str] | None = None) -> int:
    """Build a verified source ZIP at the requested destination."""
    parser = argparse.ArgumentParser(
        description="Package source, fixtures, KB, and index without local secrets/state."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dist/cloudserve-support.zip"),
        help="Destination ZIP path (default: dist/cloudserve-support.zip)",
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root (defaults to this script's parent repository)",
    )
    args = parser.parse_args(argv)
    entries = create_source_archive(args.repository_root, args.output)
    print(f"Source archive: {args.output.expanduser().resolve()}")
    print(f"Files included: {len(entries)}")
    print("Archive integrity: verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
