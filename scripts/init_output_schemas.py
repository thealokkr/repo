"""Create or verify the unversioned JSON contracts for per-run exports."""

from __future__ import annotations

import argparse
from pathlib import Path

from cloudserve_support.reporting.output_schemas import (
    OUTPUT_SCHEMA_DIRECTORY,
    initialize_output_schemas,
)


def parse_schema_directory_args() -> argparse.Namespace:
    """Parse an optional directory for the output JSON Schema contracts."""
    parser = argparse.ArgumentParser(
        description="Create or verify CloudServe per-run output JSON Schemas."
    )
    parser.add_argument(
        "--schema-directory",
        type=Path,
        default=OUTPUT_SCHEMA_DIRECTORY,
        help=f"Output schema directory (default: {OUTPUT_SCHEMA_DIRECTORY})",
    )
    return parser.parse_args()


def main() -> None:
    """Create missing export contracts or check existing contracts without edits."""
    args = parse_schema_directory_args()
    schema_directory = args.schema_directory.expanduser().resolve()
    initialize_output_schemas(schema_directory)
    print(f"Output schemas: {schema_directory}")
    print(
        "Contracts created or verified: result, decision-log event, metrics, "
        "quality judgment, exception"
    )


if __name__ == "__main__":
    main()
