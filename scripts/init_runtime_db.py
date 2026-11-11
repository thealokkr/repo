"""Create the local SQLite tables used for ticket runs and their audit trail."""

from __future__ import annotations

import argparse
from pathlib import Path

from cloudserve_support.persistence.runtime_database import (
    DATABASE_SCHEMA_VERSION,
    initialize_runtime_database,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE_PATH = REPO_ROOT / "data" / "runtime" / "cloudserve.sqlite3"


def parse_database_args() -> argparse.Namespace:
    """Parse the optional SQLite database path for runtime table initialization."""
    parser = argparse.ArgumentParser(
        description="Create the CloudServe ticket-run and audit tables in SQLite."
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=DEFAULT_DATABASE_PATH,
        help=f"SQLite file path (default: {DEFAULT_DATABASE_PATH})",
    )
    return parser.parse_args()


def main() -> None:
    """Initialize or safely recheck the local SQLite runtime database."""
    args = parse_database_args()
    database_path = args.database.expanduser().resolve()
    initialize_runtime_database(database_path)
    print(f"SQLite database: {database_path}")
    print(f"Schema version: {DATABASE_SCHEMA_VERSION}")
    print("Tables created or verified: processing_runs, ticket_runs, audit_events")
    print("Existing records were preserved; no ticket data was inserted.")


if __name__ == "__main__":
    main()
