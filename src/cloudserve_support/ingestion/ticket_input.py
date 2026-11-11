"""Read JSON ticket batches without interpreting evaluation-only fields."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_ticket_records(input_path: Path) -> list[Any]:
    """Read a top-level JSON array; each array item is retained for one result row."""
    resolved_path = input_path.expanduser().resolve()
    if not resolved_path.is_file():
        raise ValueError(f"Ticket input path is not a file: {input_path}")
    try:
        document = json.loads(resolved_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Ticket input could not be read as JSON: {input_path}: {error}") from error
    if not isinstance(document, list):
        raise ValueError(f"Ticket input must be a top-level JSON array: {input_path}")
    return document
