"""Fingerprint the frozen policy and classifier used by a ticket run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def fingerprint_policy_file(policy_path: Path) -> str:
    """Return the exact SHA-256 fingerprint of the frozen policy file used by a run."""
    return hashlib.sha256(policy_path.read_bytes()).hexdigest()


def fingerprint_policy_threshold(policy: dict[str, Any]) -> str:
    """Return a stable fingerprint for the frozen routing threshold and its calibration."""
    threshold_contract = {
        "classification_confidence_threshold": policy["classification_confidence_threshold"],
        "threshold_source": policy["threshold_source"],
        "calibration_method": policy["calibration_method"],
        "calibration_source": policy["calibration_source"],
    }
    canonical = json.dumps(
        threshold_contract, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def fingerprint_classifier_implementation() -> str:
    """Return a source fingerprint for classification rules and their safety predicates."""
    package_root = Path(__file__).resolve().parents[1]
    source_paths = (
        "classification/ticket_classification.py",
        "guardrails/operational_safety.py",
    )
    source_hashes = {
        source_path: hashlib.sha256((package_root / source_path).read_bytes()).hexdigest()
        for source_path in source_paths
    }
    canonical = json.dumps(
        source_hashes, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def fingerprint_classifier_model(provider: str, model_name: str) -> str:
    """Return a stable fingerprint for the classifier provider and configured model name."""
    model_identity = json.dumps(
        {"provider": provider, "model_name": model_name},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(model_identity).hexdigest()
