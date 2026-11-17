"""Define and load the unversioned public contracts for run export files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from cloudserve_support.classification.ticket_classification import INTENT_VALUES

OUTPUT_SCHEMA_DIRECTORY = Path(__file__).resolve().parents[3] / "schemas" / "outputs"
# Accept legacy 32-char hex UUIDs and filename-safe output-folder basenames without separators or whitespace.
RUN_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]*$"
INTENT_SCHEMA = {"enum": sorted(INTENT_VALUES)}
URGENCY_SCHEMA = {"enum": ["high", "medium", "low", "unknown"]}
EXPECTED_ROUTE_SCHEMA = {"enum": ["auto_respond", "draft_for_review", "escalate", "degrade"]}
PREDICTED_ROUTE_SCHEMA = {"enum": ["draft_for_review", "escalate", "degrade"]}
DOC_IDS_SCHEMA = {
    "type": "array",
    "items": {"type": "string", "minLength": 1, "pattern": r"\S"},
}
CALIBRATION_SOURCE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "source_set", "input_ref", "operational_sha256", "total_records",
        "valid_operational_records", "confidence_sample_count", "excluded_invalid_records",
        "failed_classifications", "evaluation_fields_used",
    ],
    "properties": {
        "source_set": {"const": "development"},
        "input_ref": {"type": "string", "minLength": 1},
        "operational_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "total_records": {"type": "integer", "minimum": 0},
        "valid_operational_records": {"type": "integer", "minimum": 0},
        "confidence_sample_count": {"type": "integer", "minimum": 1},
        "excluded_invalid_records": {"type": "integer", "minimum": 0},
        "failed_classifications": {"type": "integer", "minimum": 0},
        "evaluation_fields_used": {"const": False},
    },
}
POLICY_PROVENANCE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "policy_version", "classification_confidence_threshold", "threshold_source",
        "calibration_method", "calibration_source", "frozen", "policy_file_sha256",
        "threshold_sha256", "classifier_implementation_sha256", "classifier_model_sha256",
    ],
    "properties": {
        "policy_version": {"type": "string", "minLength": 1},
        "classification_confidence_threshold": {"type": "number", "minimum": 0, "maximum": 1},
        "threshold_source": {"type": "string", "minLength": 1},
        "calibration_method": {"type": "string", "minLength": 1},
        "calibration_source": CALIBRATION_SOURCE_SCHEMA,
        "frozen": {"const": True},
        "policy_file_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "threshold_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "classifier_implementation_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "classifier_model_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
    },
}
EVALUATION_METRIC_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "value", "status", "numerator", "denominator", "sample_size",
        "missing_count", "note",
    ],
    "properties": {
        "value": {"type": ["number", "null"]},
        "status": {"enum": ["available", "unavailable", "not_applicable"]},
        "numerator": {"type": ["number", "null"]},
        "denominator": {"type": "integer", "minimum": 0},
        "sample_size": {"type": "integer", "minimum": 0},
        "missing_count": {"type": "integer", "minimum": 0},
        "note": {"type": ["string", "null"]},
    },
}
DURATION_DISTRIBUTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["median_seconds", "p95_seconds"],
    "properties": {
        "median_seconds": EVALUATION_METRIC_SCHEMA,
        "p95_seconds": EVALUATION_METRIC_SCHEMA,
    },
}
CITATION_SOURCE_INTEGRITY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "checked_at_utc", "run_result_count", "generated_draft_count", "claim_count",
        "citation_count", "citation_resolution", "eligible_source_provenance",
        "claim_excerpt_match", "claim_traceability", "claims_missing_citations",
        "claims_with_source_id_mismatch", "failure_counts", "expected_document_coverage",
        "expected_document_count", "expected_document_hit_count",
        "missing_expected_document_label_count", "semantic_quality", "note",
    ],
    "properties": {
        "checked_at_utc": {"type": "string", "format": "date-time"},
        "run_result_count": {"type": "integer", "minimum": 0},
        "generated_draft_count": {"type": "integer", "minimum": 0},
        "claim_count": {"type": "integer", "minimum": 0},
        "citation_count": {"type": "integer", "minimum": 0},
        "citation_resolution": EVALUATION_METRIC_SCHEMA,
        "eligible_source_provenance": EVALUATION_METRIC_SCHEMA,
        "claim_excerpt_match": EVALUATION_METRIC_SCHEMA,
        "claim_traceability": EVALUATION_METRIC_SCHEMA,
        "claims_missing_citations": {"type": "integer", "minimum": 0},
        "claims_with_source_id_mismatch": {"type": "integer", "minimum": 0},
        "failure_counts": {
            "type": "object",
            "additionalProperties": {"type": "integer", "minimum": 0},
        },
        "expected_document_coverage": EVALUATION_METRIC_SCHEMA,
        "expected_document_count": {"type": "integer", "minimum": 0},
        "expected_document_hit_count": {"type": "integer", "minimum": 0},
        "missing_expected_document_label_count": {"type": "integer", "minimum": 0},
        "semantic_quality": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "semantic_citation_support", "whole_answer_correctness",
                "human_rated_citation_accuracy", "hallucination",
            ],
            "properties": {
                name: {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["status", "reason"],
                    "properties": {
                        "status": {"const": "not_measured"},
                        "reason": {"type": "string", "minLength": 1},
                    },
                }
                for name in (
                    "semantic_citation_support", "whole_answer_correctness",
                    "human_rated_citation_accuracy", "hallucination",
                )
            },
        },
        "note": {"type": "string", "minLength": 1},
    },
}
HISTORICAL_BASELINE_METRICS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "first_contact_resolution", "escalation_rate", "repeat_contact_rate",
        "csat_mean", "resolution_time_minutes_mean",
        "resolution_time_minutes_median", "first_response_time_minutes",
    ],
    "properties": {
        "first_contact_resolution": EVALUATION_METRIC_SCHEMA,
        "escalation_rate": EVALUATION_METRIC_SCHEMA,
        "repeat_contact_rate": EVALUATION_METRIC_SCHEMA,
        "csat_mean": EVALUATION_METRIC_SCHEMA,
        "resolution_time_minutes_mean": EVALUATION_METRIC_SCHEMA,
        "resolution_time_minutes_median": EVALUATION_METRIC_SCHEMA,
        "first_response_time_minutes": EVALUATION_METRIC_SCHEMA,
    },
}
HISTORICAL_BASELINE_SEGMENT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["sample_size", "metrics"],
    "properties": {
        "sample_size": {"type": "integer", "minimum": 0},
        "metrics": HISTORICAL_BASELINE_METRICS_SCHEMA,
    },
}
HISTORICAL_BASELINE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "source", "sample_size", "input_sample_size", "causal_claim_about_this_run",
        "overall", "segments",
    ],
    "properties": {
        "source": {"const": "input.history"},
        "sample_size": {"type": "integer", "minimum": 0},
        "input_sample_size": {"type": "integer", "minimum": 0},
        "causal_claim_about_this_run": {"const": False},
        "overall": HISTORICAL_BASELINE_METRICS_SCHEMA,
        "segments": {
            "type": "object",
            "additionalProperties": False,
            "required": ["language_fluency", "customer_tier"],
            "properties": {
                "language_fluency": {
                    "type": "object",
                    "additionalProperties": HISTORICAL_BASELINE_SEGMENT_SCHEMA,
                },
                "customer_tier": {
                    "type": "object",
                    "additionalProperties": HISTORICAL_BASELINE_SEGMENT_SCHEMA,
                },
            },
        },
    },
}
CRITERION_STATUS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "status", "target", "value", "numerator", "denominator",
        "sample_size", "missing_count", "note",
    ],
    "properties": {
        "status": {"enum": ["pass", "fail", "not_measurable", "excluded", "non_gating"]},
        "target": {"type": ["string", "null"]},
        "value": {"type": ["number", "null"]},
        "numerator": {"type": ["number", "null"]},
        "denominator": {"type": "integer", "minimum": 0},
        "sample_size": {"type": "integer", "minimum": 0},
        "missing_count": {"type": "integer", "minimum": 0},
        "note": {"type": "string", "minLength": 1},
        "populated_bands": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "lower_bound", "upper_bound", "mean_confidence",
                    "observed_accuracy", "absolute_error", "sample_size",
                ],
                "properties": {
                    "lower_bound": {"type": "number", "minimum": 0, "maximum": 1},
                    "upper_bound": {"type": "number", "minimum": 0, "maximum": 1},
                    "mean_confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "observed_accuracy": {"type": "number", "minimum": 0, "maximum": 1},
                    "absolute_error": {"type": "number", "minimum": 0, "maximum": 1},
                    "sample_size": {"type": "integer", "minimum": 1},
                },
            },
        },
        "dimensions": {
            "type": "object",
            "additionalProperties": False,
            "required": ["language_fluency", "customer_tier"],
            "properties": {
                dimension: {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["sample_size", "missing_segment_count", "measures"],
                    "properties": {
                        "sample_size": {"type": "integer", "minimum": 0},
                        "missing_segment_count": {"type": "integer", "minimum": 0},
                        "measures": {
                            "type": "object",
                            "additionalProperties": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": [
                                    "status", "value", "denominator", "sample_size",
                                    "missing_count", "segments",
                                ],
                                "properties": {
                                    "status": {"enum": ["pass", "fail", "not_measurable"]},
                                    "value": {"type": ["number", "null"]},
                                    "denominator": {"type": "integer", "minimum": 0},
                                    "sample_size": {"type": "integer", "minimum": 0},
                                    "missing_count": {"type": "integer", "minimum": 0},
                                    "segments": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "additionalProperties": False,
                                            "required": [
                                                "group", "value", "numerator", "denominator",
                                                "sample_size", "missing_count",
                                            ],
                                            "properties": {
                                                "group": {"type": "string", "minLength": 1},
                                                "value": {"type": ["number", "null"]},
                                                "numerator": {"type": ["integer", "null"]},
                                                "denominator": {"type": "integer", "minimum": 0},
                                                "sample_size": {"type": "integer", "minimum": 0},
                                                "missing_count": {"type": "integer", "minimum": 0},
                                            },
                                        },
                                    },
                                },
                            },
                        },
                    },
                }
                for dimension in ("language_fluency", "customer_tier")
            },
        },
    },
}
QUALITY_JUDGE_METRIC_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "status", "label", "target", "value", "numerator", "denominator",
        "sample_size", "missing_count", "note",
    ],
    "properties": {
        "status": {"enum": ["pass", "fail", "incomplete", "not_measurable"]},
        "label": {"type": "string", "minLength": 1},
        "target": {"type": "string", "minLength": 1},
        "value": {"type": ["number", "null"]},
        "numerator": {"type": ["number", "null"]},
        "denominator": {"type": "integer", "minimum": 0},
        "sample_size": {"type": "integer", "minimum": 0},
        "missing_count": {"type": "integer", "minimum": 0},
        "note": {"type": ["string", "null"]},
    },
}
QUALITY_JUDGE_SCORECARD_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "status", "label", "run_id", "source_set", "judge_model", "rubric_version",
        "prompt_version", "target_sample_size", "available_draft_count",
        "selected_count", "evaluated_count", "failed_count", "missing_count",
        "sample_seed", "input_identity_sha256", "ground_truth_sha256",
        "selected_ticket_ids", "reference_available_count",
        "citation_occurrence_count", "citation_supported_count",
        "citation_checked_count", "hallucinated_response_count",
        "unsupported_claim_count", "failure_counts", "evaluated_at_utc",
        "operational_artifact_sha256", "response_quality_proxy",
        "citation_support_rate", "hallucination_estimate",
    ],
    "properties": {
        "status": {"enum": ["complete", "incomplete"]},
        "label": {"type": "string", "minLength": 1},
        "run_id": {"type": "string", "pattern": RUN_ID_PATTERN},
        "source_set": {"enum": ["development", "validation", "external"]},
        "judge_model": {"type": "string", "minLength": 1},
        "rubric_version": {"type": "string", "minLength": 1},
        "prompt_version": {"type": "string", "minLength": 1},
        "target_sample_size": {"const": 20},
        "available_draft_count": {"type": "integer", "minimum": 0},
        "selected_count": {"type": "integer", "minimum": 0, "maximum": 20},
        "evaluated_count": {"type": "integer", "minimum": 0, "maximum": 20},
        "failed_count": {"type": "integer", "minimum": 0},
        "missing_count": {"type": "integer", "minimum": 0},
        "sample_seed": {"type": "string", "minLength": 1},
        "input_identity_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "ground_truth_sha256": {
            "type": ["string", "null"],
            "pattern": "^[0-9a-f]{64}$",
        },
        "selected_ticket_ids": {
            "type": "array",
            "uniqueItems": True,
            "items": {"type": "string", "minLength": 1},
        },
        "reference_available_count": {"type": "integer", "minimum": 0},
        "citation_occurrence_count": {"type": "integer", "minimum": 0},
        "citation_supported_count": {"type": "integer", "minimum": 0},
        "citation_checked_count": {"type": "integer", "minimum": 0},
        "hallucinated_response_count": {"type": "integer", "minimum": 0},
        "unsupported_claim_count": {"type": "integer", "minimum": 0},
        "failure_counts": {
            "type": "object",
            "additionalProperties": {"type": "integer", "minimum": 0},
        },
        "evaluated_at_utc": {"type": "string", "format": "date-time"},
        "operational_artifact_sha256": {
            "type": "object",
            "additionalProperties": False,
            "required": ["results.jsonl", "decision_log.jsonl"],
            "properties": {
                "results.jsonl": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                "decision_log.jsonl": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            },
        },
        "response_quality_proxy": QUALITY_JUDGE_METRIC_SCHEMA,
        "citation_support_rate": QUALITY_JUDGE_METRIC_SCHEMA,
        "hallucination_estimate": QUALITY_JUDGE_METRIC_SCHEMA,
    },
}
QUALITY_JUDGMENT_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "CloudServe per-ticket post-run quality judgment",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "run_id", "ticket_id", "input_index", "sample_index", "judge_model",
        "rubric_version", "prompt_version", "sample_seed",
        "input_identity_sha256", "reference_available", "status",
        "dimension_scores", "overall_quality_score", "citation_judgments",
        "unsupported_factual_claims", "error",
    ],
    "properties": {
        "run_id": {"type": "string", "pattern": RUN_ID_PATTERN},
        "ticket_id": {"type": "string", "minLength": 1},
        "input_index": {"type": ["integer", "null"], "minimum": 0},
        "sample_index": {"type": "integer", "minimum": 0, "maximum": 19},
        "judge_model": {"type": "string", "minLength": 1},
        "rubric_version": {"type": "string", "minLength": 1},
        "prompt_version": {"type": "string", "minLength": 1},
        "sample_seed": {"type": "string", "minLength": 1},
        "input_identity_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "reference_available": {"type": "boolean"},
        "status": {"enum": ["evaluated", "failed"]},
        "dimension_scores": {
            "type": ["object", "null"],
            "additionalProperties": False,
            "required": [
                "relevance", "factual_correctness",
                "completeness_actionability", "clarity",
            ],
            "properties": {
                dimension: {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["score", "rationale"],
                    "properties": {
                        "score": {"type": "integer", "minimum": 1, "maximum": 5},
                        "rationale": {"type": "string", "minLength": 1},
                    },
                }
                for dimension in (
                    "relevance", "factual_correctness",
                    "completeness_actionability", "clarity",
                )
            },
        },
        "overall_quality_score": {"type": ["number", "null"], "minimum": 1, "maximum": 5},
        "citation_judgments": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "claim_index", "citation_index", "passage_id", "source_ref",
                    "citation_id", "supported", "rationale",
                ],
                "properties": {
                    "claim_index": {"type": "integer", "minimum": 0},
                    "citation_index": {"type": "integer", "minimum": 0},
                    "passage_id": {"type": "string", "minLength": 1},
                    "source_ref": {"type": "string", "minLength": 1},
                    "citation_id": {"type": "string", "minLength": 1},
                    "supported": {"type": "boolean"},
                    "rationale": {"type": "string", "minLength": 1},
                },
            },
        },
        "unsupported_factual_claims": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["claim", "reviewed_passage_ids", "rationale"],
                "properties": {
                    "claim": {"type": "string", "minLength": 1},
                    "reviewed_passage_ids": {
                        "type": "array", "items": {"type": "string", "minLength": 1},
                    },
                    "rationale": {"type": "string", "minLength": 1},
                },
            },
        },
        "error": {
            "type": ["object", "null"],
            "additionalProperties": False,
            "required": ["code", "message"],
            "properties": {
                "code": {"type": "string", "minLength": 1},
                "message": {"type": "string", "minLength": 1},
            },
        },
    },
}
CRITERIA_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "runtime_latency", "availability", "private_data_output",
        "decision_audit_coverage", "confidence_calibration",
        "group_quality_variation", "business_outcomes",
    ],
    "properties": {
        "runtime_latency": CRITERION_STATUS_SCHEMA,
        "availability": CRITERION_STATUS_SCHEMA,
        "private_data_output": CRITERION_STATUS_SCHEMA,
        "decision_audit_coverage": CRITERION_STATUS_SCHEMA,
        "confidence_calibration": CRITERION_STATUS_SCHEMA,
        "group_quality_variation": CRITERION_STATUS_SCHEMA,
        "business_outcomes": CRITERION_STATUS_SCHEMA,
        "llm_judge": QUALITY_JUDGE_SCORECARD_SCHEMA,
    },
}
EVALUATION_REPORT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "report_version", "run_id", "source_set", "input_ref", "input_sha256",
        "evaluated_at_utc", "sample_count", "evaluation_data", "integrity",
        "policy_provenance", "run_observed", "criteria", "historical_baseline",
        "limitations",
    ],
    "properties": {
        "report_version": {"const": 1},
        "run_id": {"type": "string", "pattern": RUN_ID_PATTERN},
        "source_set": {"enum": ["development", "validation", "external"]},
        "input_ref": {"type": "string", "minLength": 1},
        "input_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "evaluated_at_utc": {"type": "string", "format": "date-time"},
        "sample_count": {"type": "integer", "minimum": 0},
        "evaluation_data": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "labels_record_count", "history_record_count",
                "labels_used_after_processing", "history_used_after_processing",
            ],
            "properties": {
                "labels_record_count": {"type": "integer", "minimum": 0},
                "history_record_count": {"type": "integer", "minimum": 0},
                "labels_used_after_processing": {"const": True},
                "history_used_after_processing": {"const": True},
            },
        },
        "integrity": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "status", "expected_count", "result_count", "decision_count",
                "missing_result_indices", "mismatched_indices",
                "duplicate_ticket_ids", "errors",
            ],
            "properties": {
                "status": {"enum": ["complete", "mismatch"]},
                "expected_count": {"type": "integer", "minimum": 0},
                "result_count": {"type": "integer", "minimum": 0},
                "decision_count": {"type": "integer", "minimum": 0},
                "missing_result_indices": {"type": "array", "items": {"type": "integer", "minimum": 0}},
                "mismatched_indices": {"type": "array", "items": {"type": "integer", "minimum": 0}},
                "duplicate_ticket_ids": {"type": "array", "items": {"type": "string"}},
                "errors": {"type": "array", "items": {"type": "string"}},
            },
        },
        "policy_provenance": {"type": "object"},
        "run_observed": {
            "type": "object",
            "additionalProperties": False,
            "required": ["volume", "business", "technical", "governance"],
            "properties": {
                "volume": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "tickets_processed", "drafts_for_agent_review", "escalated",
                        "degraded", "blocked_by_guardrails",
                    ],
                    "properties": {
                        "tickets_processed": {"type": "integer", "minimum": 0},
                        "drafts_for_agent_review": {"type": "integer", "minimum": 0},
                        "escalated": {"type": "integer", "minimum": 0},
                        "degraded": {"type": "integer", "minimum": 0},
                        "blocked_by_guardrails": {"type": "integer", "minimum": 0},
                    },
                },
                "business": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "first_contact_resolution", "first_response_time_minutes_mean",
                        "first_response_time_minutes_median", "escalation_rate",
                        "historical_resolution_time_minutes_mean",
                        "historical_resolution_time_minutes_median",
                    ],
                    "properties": {
                        "first_contact_resolution": EVALUATION_METRIC_SCHEMA,
                        "first_response_time_minutes_mean": EVALUATION_METRIC_SCHEMA,
                        "first_response_time_minutes_median": EVALUATION_METRIC_SCHEMA,
                        "escalation_rate": EVALUATION_METRIC_SCHEMA,
                        "historical_resolution_time_minutes_mean": EVALUATION_METRIC_SCHEMA,
                        "historical_resolution_time_minutes_median": EVALUATION_METRIC_SCHEMA,
                    },
                },
                "technical": {
                    "type": "object",
                    "required": [
                        "classification", "route_accuracy", "route_quality", "answerability",
                        "retrieval_hit_rate",
                        "citation_resolution", "citation_source_integrity", "coverage",
                        "availability", "latency", "generation_inclusive_processing_duration",
                        "draft_generation_request_duration", "decision_log_reconciliation",
                    ],
                    "properties": {
                        "classification": {"type": "object"},
                        "route_accuracy": EVALUATION_METRIC_SCHEMA,
                        "route_quality": {"type": "object"},
                        "answerability": {"type": "object"},
                        "retrieval_hit_rate": EVALUATION_METRIC_SCHEMA,
                        "citation_resolution": EVALUATION_METRIC_SCHEMA,
                        "citation_source_integrity": CITATION_SOURCE_INTEGRITY_SCHEMA,
                        "coverage": EVALUATION_METRIC_SCHEMA,
                        "availability": EVALUATION_METRIC_SCHEMA,
                        "latency": DURATION_DISTRIBUTION_SCHEMA,
                        "generation_inclusive_processing_duration": DURATION_DISTRIBUTION_SCHEMA,
                        "draft_generation_request_duration": DURATION_DISTRIBUTION_SCHEMA,
                        "decision_log_reconciliation": EVALUATION_METRIC_SCHEMA,
                    },
                },
                "governance": {
                    "type": "object",
                    "properties": {"protected_case_safety": {"type": "object"}},
                    "required": ["protected_case_safety"],
                },
            },
        },
        "criteria": CRITERIA_SCHEMA,
        "historical_baseline": HISTORICAL_BASELINE_SCHEMA,
        "limitations": {"type": "array", "items": {"type": "string"}},
    },
}
RETRIEVAL_DIAGNOSTICS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "status", "reason_code", "reason", "eligible_candidate_count",
        "excluded_candidate_counts", "minimum_similarity", "embedding_model",
        "source_sha256", "metadata_manifest_sha256",
    ],
    "properties": {
        "status": {"type": "string", "minLength": 1},
        "reason_code": {"type": "string", "minLength": 1},
        "reason": {"type": "string", "minLength": 1},
        "eligible_candidate_count": {"type": "integer", "minimum": 0},
        "excluded_candidate_counts": {
            "type": "object",
            "additionalProperties": {"type": "integer", "minimum": 0},
        },
        "minimum_similarity": {"type": "number", "minimum": 0, "maximum": 1},
        "embedding_model": {"type": "string", "minLength": 1},
        "source_sha256": {"type": ["string", "null"]},
        "metadata_manifest_sha256": {"type": ["string", "null"]},
    },
}
POLICY_CHECK_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["name", "status", "confidence", "reason"],
    "properties": {
        "name": {
            "enum": [
                "protected_rule_check",
                "classification_confidence_check",
                "source_presence_check",
                "claim_grounding_check",
            ]
        },
        "status": {"enum": ["passed", "failed", "skipped"]},
        "confidence": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
        "reason": {"type": "string", "minLength": 1},
        "matched_rule": {"type": ["string", "null"], "minLength": 1},
        "protected_case": {"type": "boolean"},
        "threshold": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
        "retrieved_doc_ids": DOC_IDS_SCHEMA,
        "related_doc_ids": DOC_IDS_SCHEMA,
        "excluded_source_count": {"type": "integer", "minimum": 0},
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["text", "source_ids"],
                "properties": {
                    "text": {"type": "string", "minLength": 1},
                    "source_ids": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"type": "string", "minLength": 1},
                    },
                },
            },
        },
        "unsupported_points": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
        },
        "conflicting_points": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
        },
        "error_code": {"type": ["string", "null"], "minLength": 1},
    },
}
POLICY_FINAL_DECISION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "route", "matched_rule", "reason", "failed_check",
        "protected_case", "evidence_ready",
    ],
    "properties": {
        "route": PREDICTED_ROUTE_SCHEMA,
        "matched_rule": {"type": "string", "minLength": 1},
        "reason": {"type": "string", "minLength": 1},
        "failed_check": {
            "enum": [
                "protected_rule_check",
                "classification_confidence_check",
                "source_presence_check",
                "claim_grounding_check",
                None,
            ]
        },
        "protected_case": {"type": "boolean"},
        "evidence_ready": {"type": ["boolean", "null"]},
    },
}
ROUTING_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "policy_version", "matched_rule",
        "confidence_threshold", "threshold_source", "calibration_method",
        "calibration_source", "policy_frozen", "policy_provenance", "retrieval",
        "checks", "final_decision",
    ],
    "properties": {
        "policy_version": {"type": "string", "minLength": 1},
        "matched_rule": {"type": "string", "minLength": 1},
        "confidence_threshold": {"type": "number", "minimum": 0, "maximum": 1},
        "threshold_source": {"type": "string", "minLength": 1},
        "calibration_method": {"type": "string", "minLength": 1},
        "calibration_source": CALIBRATION_SOURCE_SCHEMA,
        "policy_frozen": {"const": True},
        "policy_provenance": POLICY_PROVENANCE_SCHEMA,
        "retrieval": RETRIEVAL_DIAGNOSTICS_SCHEMA,
        "checks": {
            "type": "array",
            "items": POLICY_CHECK_SCHEMA,
        },
        "final_decision": POLICY_FINAL_DECISION_SCHEMA,
    },
}
SOURCE_EVIDENCE_SCHEMA = {
    "type": "object",
    "required": [
        "passage_id", "doc_id", "similarity", "minimum_similarity", "source_ref",
    ],
    "properties": {
        "passage_id": {"type": "string", "minLength": 1},
        "doc_id": {"type": "string", "minLength": 1},
        "similarity": {"type": "number", "minimum": 0, "maximum": 1},
        "minimum_similarity": {"type": "number", "minimum": 0, "maximum": 1},
        "source_ref": {"type": "string", "minLength": 1},
        "source_kind": {"enum": ["primary", "related"]},
    },
}
DRAFT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "plain_text", "claims", "unsupported_points", "conflicting_points",
        "approval_required", "send_status",
    ],
    "properties": {
        "plain_text": {"type": "string", "minLength": 1},
        "claims": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["text", "source_ids", "citations"],
                "properties": {
                    "text": {"type": "string", "minLength": 1},
                    "source_ids": {"type": "array", "minItems": 1, "items": {"type": "string"}},
                    "citations": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["passage_id", "source_ref"],
                            "properties": {
                                "passage_id": {"type": "string", "minLength": 1},
                                "source_ref": {"type": "string", "minLength": 1},
                            },
                        },
                    },
                },
            },
        },
        "unsupported_points": {"type": "array", "items": {"type": "string"}},
        "conflicting_points": {"type": "array", "items": {"type": "string"}},
        "approval_required": {"const": True},
        "send_status": {"const": "not_sent"},
    },
}
HANDOFF_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "summary", "issue", "attempted_steps", "uncertainties",
        "owner", "next_action",
    ],
    "properties": {
        "summary": {"type": "string", "minLength": 1},
        "issue": {"type": "string", "minLength": 1},
        "attempted_steps": {"type": "array", "items": {"type": "string"}},
        "uncertainties": {"type": "array", "minItems": 1, "items": {"type": "string"}},
        "owner": {"type": "string", "minLength": 1},
        "next_action": {"type": "string", "minLength": 1},
    },
}
GUARDRAILS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["checks"],
    "properties": {
        "checks": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["check", "passed", "blocked", "evidence", "action"],
                "properties": {
                    "check": {"type": "string", "minLength": 1},
                    "passed": {"type": "boolean"},
                    "blocked": {"type": "boolean"},
                    "evidence": {"type": "string", "minLength": 1},
                    "action": {"type": "string", "minLength": 1},
                },
            },
        }
    },
}
PRE_ROUTING_CHECKS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["data_checks", "safety_checks"],
    "properties": {
        "data_checks": {
            "type": "array",
            "items": {"type": "object"},
        },
        "safety_checks": {
            "type": "array",
            "items": {"type": "object"},
        },
    },
}
GENERATION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["provider", "model_name", "reasoning_effort", "status", "error_code"],
    "properties": {
        "provider": {"enum": ["offline", "gemini"]},
        "model_name": {"type": "string", "minLength": 1},
        "reasoning_effort": {"type": ["string", "null"]},
        "status": {"enum": ["not_attempted", "generated", "failed"]},
        "error_code": {"type": ["string", "null"]},
    },
}
OUTPUT_SCHEMA_DEFINITIONS: dict[str, dict[str, Any]] = {
    "result.schema.json": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "CloudServe per-ticket result",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "run_id", "input_index", "ticket_id", "channel", "subject", "body",
            "expected_intent", "predicted_intent",
            "expected_urgency", "predicted_urgency",
            "expected_route", "predicted_route",
            "expected_answerable_from_docs", "predicted_answerable_from_docs",
            "expected_doc_ids", "predicted_doc_ids",
            "expected_must_not_auto_respond", "predicted_must_not_auto_respond",
            "status", "reason",
            "classifier_confidence", "classifier_provider", "classifier_model", "classifier_status",
            "routing", "sources", "draft", "handoff", "guardrails", "generation",
            "processing_duration_seconds", "draft_generation_request_duration_seconds",
            "processing_latency_seconds",
        ],
        "properties": {
            "run_id": {"type": "string", "pattern": RUN_ID_PATTERN},
            "input_index": {"type": "integer", "minimum": 0},
            "ticket_id": {"type": ["string", "null"], "minLength": 1},
            "channel": {"type": ["string", "null"], "minLength": 1},
            "subject": {"type": ["string", "null"]},
            "body": {"type": ["string", "null"]},
            "received_at": {"type": ["string", "null"], "minLength": 1},
            "customer_tier": {"type": ["string", "null"], "minLength": 1},
            "language_fluency": {"type": ["string", "null"], "minLength": 1},
            "expected_intent": {"oneOf": [INTENT_SCHEMA, {"type": "null"}]},
            "predicted_intent": {"oneOf": [INTENT_SCHEMA, {"type": "null"}]},
            "expected_urgency": {"oneOf": [URGENCY_SCHEMA, {"type": "null"}]},
            "predicted_urgency": {"oneOf": [URGENCY_SCHEMA, {"type": "null"}]},
            "expected_route": {"oneOf": [EXPECTED_ROUTE_SCHEMA, {"type": "null"}]},
            "predicted_route": {"oneOf": [PREDICTED_ROUTE_SCHEMA, {"type": "null"}]},
            "expected_answerable_from_docs": {"type": ["boolean", "null"]},
            "predicted_answerable_from_docs": {"type": ["boolean", "null"]},
            "expected_doc_ids": {"oneOf": [DOC_IDS_SCHEMA, {"type": "null"}]},
            "predicted_doc_ids": {"oneOf": [DOC_IDS_SCHEMA, {"type": "null"}]},
            "expected_must_not_auto_respond": {"type": ["boolean", "null"]},
            "predicted_must_not_auto_respond": {"type": ["boolean", "null"]},
            "status": {"enum": ["completed", "failed"]},
            "reason": {"type": "string", "minLength": 1},
            "error": {
                "type": "object",
                "additionalProperties": False,
                "required": ["code", "message"],
                "properties": {
                    "code": {"type": "string", "minLength": 1},
                    "message": {"type": "string", "minLength": 1},
                },
            },
            "classifier_confidence": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
            "classifier_provider": {"enum": ["gemini", "not_attempted"]},
            "classifier_model": {"type": "string", "minLength": 1},
            "classifier_status": {"enum": ["classified", "fallback"]},
            "routing": ROUTING_SCHEMA,
            "sources": {"type": "array", "items": SOURCE_EVIDENCE_SCHEMA},
            "draft": {"oneOf": [DRAFT_SCHEMA, {"type": "null"}]},
            "handoff": {"oneOf": [HANDOFF_SCHEMA, {"type": "null"}]},
            "guardrails": GUARDRAILS_SCHEMA,
            "pre_routing_checks": PRE_ROUTING_CHECKS_SCHEMA,
            "generation": GENERATION_SCHEMA,
            "processing_duration_seconds": {"type": "number", "minimum": 0},
            "draft_generation_request_duration_seconds": {"type": "number", "minimum": 0},
            "processing_latency_seconds": {"type": "number", "minimum": 0},
        },
        "allOf": [
            {
                "if": {
                    "required": ["status"],
                    "properties": {"status": {"const": "failed"}},
                },
                "then": {
                    "required": ["error"],
                    "properties": {
                        "predicted_intent": {"type": "null"},
                        "predicted_urgency": {"type": "null"},
                        "predicted_route": {"type": "null"},
                        "predicted_answerable_from_docs": {"type": "null"},
                        "predicted_doc_ids": {"type": "null"},
                        "predicted_must_not_auto_respond": {"type": "null"},
                        "draft": {"type": "null"},
                        "handoff": HANDOFF_SCHEMA,
                    },
                },
                "else": {
                    "not": {"required": ["error"]},
                },
            },
            {
                "if": {
                    "required": ["classifier_status"],
                    "properties": {"classifier_status": {"const": "classified"}},
                },
                "then": {
                    "properties": {
                        "classifier_confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                },
                "else": {
                    "properties": {
                        "classifier_confidence": {"type": "null"},
                    },
                },
            },
            {
                "if": {
                    "required": ["predicted_intent"],
                    "properties": {"predicted_intent": {"type": "null"}},
                },
                "then": {
                    "properties": {
                        "predicted_urgency": {"type": "null"},
                        "predicted_route": {"type": "null"},
                        "predicted_answerable_from_docs": {"type": "null"},
                        "predicted_doc_ids": {"type": "null"},
                        "predicted_must_not_auto_respond": {"type": "null"},
                    },
                },
                "else": {
                    "properties": {
                        "predicted_intent": INTENT_SCHEMA,
                        "predicted_urgency": URGENCY_SCHEMA,
                        "predicted_route": PREDICTED_ROUTE_SCHEMA,
                        "predicted_answerable_from_docs": {"type": "boolean"},
                        "predicted_doc_ids": DOC_IDS_SCHEMA,
                        "predicted_must_not_auto_respond": {"type": "boolean"},
                    },
                },
            },
            {
                "if": {
                    "required": ["predicted_route"],
                    "properties": {"predicted_route": {"const": "draft_for_review"}},
                },
                "then": {
                    "properties": {
                        "draft": DRAFT_SCHEMA,
                        "handoff": {"type": "null"},
                    },
                },
            },
            {
                "if": {
                    "required": ["predicted_route"],
                    "properties": {"predicted_route": {"enum": ["escalate", "degrade"]}},
                },
                "then": {
                    "properties": {
                        "draft": {"type": "null"},
                        "handoff": HANDOFF_SCHEMA,
                    },
                },
            },
            {
                "oneOf": [
                    {
                        "properties": {
                            "draft": DRAFT_SCHEMA,
                            "handoff": {"type": "null"},
                        },
                    },
                    {
                        "properties": {
                            "draft": {"type": "null"},
                            "handoff": HANDOFF_SCHEMA,
                        },
                    },
                ],
            },
        ],
    },
    "decision-log.schema.json": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "CloudServe decision log event",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "event_id", "run_id", "ticket_occurrence_id", "input_index", "ticket_id",
            "occurred_at", "event_type", "actor_type", "terminal",
        ],
        "properties": {
            "event_id": {"type": "string", "pattern": "^[0-9a-f]{32}$"},
            "run_id": {"type": "string", "pattern": RUN_ID_PATTERN},
            "ticket_occurrence_id": {"type": "string", "minLength": 1},
            "input_index": {"type": "integer", "minimum": 0},
            "ticket_id": {"type": "string", "minLength": 1},
            "occurred_at": {"type": "string", "format": "date-time"},
            "event_type": {"enum": ["classification", "retrieval", "routing", "generation", "ticket_completed", "ticket_failed"]},
            "actor_type": {"const": "system"},
            "terminal": {"type": "boolean"},
            "action_taken": {"enum": ["draft_for_review", "escalate", "degrade"]},
            "reason": {"type": "string", "minLength": 1},
            "stage_result": {
                "type": "object",
                "additionalProperties": False,
                "required": ["stage", "status", "value", "reason"],
                "properties": {
                    "stage": {"enum": ["classification", "retrieval", "routing", "generation"]},
                    "status": {"type": "string", "minLength": 1},
                    "value": {"type": "object"},
                    "reason": {"type": "string", "minLength": 1},
                },
            },
            "result": {
                "type": "object",
                "required": [
                    "run_id", "input_index", "ticket_id", "channel", "route",
                    "status", "reason", "draft_response", "sources", "classification", "routing",
                    "processing_duration_seconds", "draft_generation_request_duration_seconds",
                    "processing_latency_seconds", "draft", "handoff", "guardrails", "generation",
                ],
            },
        },
        "oneOf": [
            {
                "properties": {
                    "terminal": {"const": False},
                    "event_type": {"enum": ["classification", "retrieval", "routing", "generation"]},
                },
                "required": ["stage_result"],
                "not": {"anyOf": [{"required": ["action_taken"]}, {"required": ["result"]}]},
            },
            {
                "properties": {
                    "terminal": {"const": True},
                    "event_type": {"enum": ["ticket_completed", "ticket_failed"]},
                },
                "required": ["action_taken", "reason", "result"],
                "not": {"required": ["stage_result"]},
            },
        ],
    },
    "metrics.schema.json": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "CloudServe per-run metrics",
        "type": "object",
        "additionalProperties": False,
        "required": ["run_id", "total_count", "completed_count", "failed_count", "exception_count", "route_counts", "batch_export_duration_seconds", "evaluation"],
        "properties": {
            "run_id": {"type": "string", "pattern": RUN_ID_PATTERN},
            "total_count": {"type": "integer", "minimum": 0},
            "completed_count": {"type": "integer", "minimum": 0},
            "failed_count": {"type": "integer", "minimum": 0},
            "exception_count": {"type": "integer", "minimum": 0},
            "batch_export_duration_seconds": {
                "type": "number",
                "minimum": 0,
                "description": "Elapsed time for writing run exports and finalizing the processing-run record, excluding the subsequent metrics.json timing-field rewrite.",
            },
            "route_counts": {
                "type": "object",
                "additionalProperties": False,
                "required": ["draft_for_review", "escalate", "degrade"],
                "properties": {
                    "draft_for_review": {"type": "integer", "minimum": 0},
                    "escalate": {"type": "integer", "minimum": 0},
                    "degrade": {"type": "integer", "minimum": 0},
                },
            },
            "evaluation": EVALUATION_REPORT_SCHEMA,
        },
    },
    "quality-judgment.schema.json": QUALITY_JUDGMENT_SCHEMA,
    "exception.schema.json": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "CloudServe per-ticket exception",
        "type": "object",
        "additionalProperties": False,
        "required": ["run_id", "input_index", "ticket_id", "occurred_at", "error_code", "message"],
        "properties": {
            "run_id": {"type": "string", "pattern": RUN_ID_PATTERN},
            "input_index": {"type": "integer", "minimum": 0},
            "ticket_id": {"type": "string", "minLength": 1},
            "occurred_at": {"type": "string", "format": "date-time"},
            "error_code": {"type": "string", "minLength": 1},
            "message": {"type": "string", "minLength": 1},
        },
    },
}


def load_output_schemas(schema_directory: Path | None = None) -> dict[str, dict[str, Any]]:
    """Load the per-run output contracts and verify each is valid JSON Schema."""
    schema_directory = schema_directory or OUTPUT_SCHEMA_DIRECTORY
    loaded: dict[str, dict[str, Any]] = {}
    for filename in OUTPUT_SCHEMA_DEFINITIONS:
        schema_path = schema_directory / filename
        try:
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise RuntimeError(
                f"Output schema is missing: {schema_path}. Run scripts/init_output_schemas.py first."
            ) from error
        Draft202012Validator.check_schema(schema)
        loaded[filename] = schema
    return loaded


def initialize_output_schemas(schema_directory: Path) -> None:
    """Create missing output schemas and refuse to overwrite a different contract."""
    schema_directory.mkdir(parents=True, exist_ok=True)
    for filename, expected_schema in OUTPUT_SCHEMA_DEFINITIONS.items():
        schema_path = schema_directory / filename
        if schema_path.exists():
            actual_schema = json.loads(schema_path.read_text(encoding="utf-8"))
            if actual_schema != expected_schema:
                raise RuntimeError(
                    f"Output schema contract differs from the initializer: {schema_path}"
                )
            continue
        schema_path.write_text(
            json.dumps(expected_schema, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


def validate_output_record(
    record: dict[str, Any],
    *,
    schema_filename: str,
    schemas: dict[str, dict[str, Any]],
) -> None:
    """Validate one exported JSON object against its named output schema contract."""
    schema = schemas[schema_filename]
    Draft202012Validator(
        schema,
        format_checker=Draft202012Validator.FORMAT_CHECKER,
    ).validate(record)
