"""Canonical ticket classification values shared by runtime and evaluation."""

# Supported intent labels for ticket classification and generated-label comparison.
TICKET_INTENT_VALUES = frozenset(
    {
        "account",
        "account_compromise",
        "account_access",
        "api",
        "api_key_issue",
        "api_usage_question",
        "authentication",
        "authentication_failure",
        "billing",
        "billing_dispute",
        "billing_query",
        "compliance",
        "compliance_request",
        "configuration_help",
        "data",
        "data_location",
        "data_export",
        "data_residency",
        "database_issue",
        "deployment",
        "deployment_failure",
        "feature_request",
        "integration",
        "integration_help",
        "onboarding",
        "other",
        "performance",
        "performance_degradation",
        "quota_or_overage",
        "rate_limit",
        "rollback_request",
        "roadmap_request",
        "security",
        "security_incident",
        "sso_configuration",
        "unclear_request",
        "webhook_issue",
    }
)

# Supported urgency labels for ticket classification and generated-label comparison.
TICKET_URGENCY_VALUES = frozenset({"high", "medium", "low", "unknown"})

# Active 22 dataset intent choices sent to the Decisions classifier.
CLASSIFIER_INTENT_CHOICES = frozenset(
    {
        "account_access",
        "api_key_issue",
        "api_usage_question",
        "authentication_failure",
        "billing_query",
        "compliance_request",
        "configuration_help",
        "data_export",
        "data_residency",
        "database_issue",
        "deployment_failure",
        "feature_request",
        "integration_help",
        "onboarding",
        "performance_degradation",
        "quota_or_overage",
        "rate_limit",
        "rollback_request",
        "security_incident",
        "sso_configuration",
        "unclear_request",
        "webhook_issue",
    }
)

# Active 3 urgency choices sent to the Decisions classifier (excluding "unknown").
CLASSIFIER_URGENCY_CHOICES = frozenset({"high", "medium", "low"})

assert CLASSIFIER_INTENT_CHOICES <= TICKET_INTENT_VALUES
assert len(CLASSIFIER_INTENT_CHOICES) == 22
assert CLASSIFIER_URGENCY_CHOICES <= TICKET_URGENCY_VALUES
assert len(CLASSIFIER_URGENCY_CHOICES) == 3
