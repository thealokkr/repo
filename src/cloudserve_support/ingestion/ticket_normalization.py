"""Normalize supported channel records into the operational ticket contract."""

from __future__ import annotations

from cloudserve_support.domain.ticket import CanonicalTicket


SUPPORTED_TICKET_CHANNELS = frozenset({"chat", "docs_comment", "email", "forum"})
EVALUATION_ONLY_FIELDS = frozenset({"labels", "history"})
OPTIONAL_STRING_FIELDS = (
    "received_at",
    "customer_tier",
    "language_fluency",
)


def normalize_ticket_record(raw_record: object, *, input_index: int) -> CanonicalTicket:
    """Normalize one JSON value while keeping malformed records reconcilable."""
    record = raw_record if isinstance(raw_record, dict) else {}
    errors: list[str] = []

    raw_ticket_id = record.get("ticket_id")
    if isinstance(raw_ticket_id, str):
        raw_ticket_id, ticket_id_is_utf8 = _sanitize_unicode_text(raw_ticket_id)
    else:
        ticket_id_is_utf8 = True
    has_ticket_id = (
        isinstance(raw_ticket_id, str)
        and ticket_id_is_utf8
        and bool(raw_ticket_id.strip())
    )
    ticket_id = raw_ticket_id.strip() if has_ticket_id else f"invalid-{input_index + 1:06d}"
    if not has_ticket_id:
        if ticket_id_is_utf8:
            errors.append("ticket_id must be a non-empty string")
        else:
            errors.append("ticket_id must contain valid Unicode text")

    raw_channel = record.get("channel")
    if isinstance(raw_channel, str):
        channel, channel_is_utf8 = _sanitize_unicode_text(raw_channel)
        if not channel_is_utf8:
            errors.append("channel must contain valid Unicode text")
    else:
        channel = ""
    if channel not in SUPPORTED_TICKET_CHANNELS:
        errors.append("channel must be chat, docs_comment, email, or forum")
        channel = channel or "unknown"

    raw_subject = record.get("subject", "")
    if isinstance(raw_subject, str):
        subject, subject_is_utf8 = _sanitize_unicode_text(raw_subject)
        if not subject_is_utf8:
            errors.append("subject must contain valid Unicode text")
    else:
        subject = ""
        errors.append("subject must be a string")

    raw_body = record.get("body")
    if isinstance(raw_body, str):
        body, body_is_utf8 = _sanitize_unicode_text(raw_body)
        if not body_is_utf8:
            errors.append("body must contain valid Unicode text")
    else:
        body = ""
    if not isinstance(raw_body, str) or not body.strip():
        body = ""
        errors.append("body must be a non-empty string")

    optional_values: dict[str, str | None] = {}
    for field_name in OPTIONAL_STRING_FIELDS:
        raw_value = record.get(field_name)
        if raw_value is None:
            optional_values[field_name] = None
        elif isinstance(raw_value, str):
            safe_value, value_is_utf8 = _sanitize_unicode_text(raw_value)
            if not value_is_utf8:
                errors.append(f"{field_name} must contain valid Unicode text")
            optional_values[field_name] = safe_value.strip() or None
        else:
            optional_values[field_name] = None
            errors.append(f"{field_name} must be a string when supplied")

    # Keep channel and customer context for operational use, but never expose
    # fixture labels or evaluation outcome history to ticket processors.
    context: dict[str, object] = {
        key: value
        for key, value in record.items()
        if key not in EVALUATION_ONLY_FIELDS
    }
    context.update({"ticket_id": ticket_id, "channel": channel, "subject": subject, "body": body})
    for field_name, value in optional_values.items():
        if field_name in record:
            context[field_name] = value
    context = _sanitize_operational_context(context, errors=errors)

    return CanonicalTicket(
        ticket_id=ticket_id,
        has_ticket_id=has_ticket_id,
        channel=channel,
        subject=subject,
        body=body,
        received_at=optional_values["received_at"],
        customer_tier=optional_values["customer_tier"],
        language_fluency=optional_values["language_fluency"],
        context=context,
        validation_errors=tuple(errors),
    )


def _sanitize_unicode_text(value: str) -> tuple[str, bool]:
    """Replace unpaired surrogate code points before operational persistence."""
    sanitized = "".join(
        "\ufffd" if 0xD800 <= ord(character) <= 0xDFFF else character
        for character in value
    )
    return sanitized, sanitized == value


def _sanitize_operational_context(
    context: dict[str, object],
    *,
    errors: list[str],
) -> dict[str, object]:
    """Sanitize operational context values and keys while preserving valid Unicode."""
    sanitized_context: dict[str, object] = {}
    for key, value in context.items():
        safe_key, key_is_utf8 = _sanitize_unicode_text(key)
        if not key_is_utf8:
            errors.append("operational context keys must contain valid Unicode text")
        sanitized_context[safe_key] = _sanitize_context_value(
            value,
            field_path=f"context.{safe_key}",
            errors=errors,
        )
    return sanitized_context


def _sanitize_context_value(
    value: object,
    *,
    field_path: str,
    errors: list[str],
) -> object:
    """Replace invalid Unicode in nested operational context and report its path."""
    if isinstance(value, str):
        sanitized, is_utf8 = _sanitize_unicode_text(value)
        if not is_utf8:
            errors.append(f"{field_path} must contain valid Unicode text")
        return sanitized
    if isinstance(value, list):
        return [
            _sanitize_context_value(
                item,
                field_path=f"{field_path}[{index}]",
                errors=errors,
            )
            for index, item in enumerate(value)
        ]
    if isinstance(value, dict):
        nested_context: dict[str, object] = {}
        for key, nested_value in value.items():
            if isinstance(key, str):
                safe_key, key_is_utf8 = _sanitize_unicode_text(key)
                if not key_is_utf8:
                    errors.append(
                        f"{field_path} keys must contain valid Unicode text"
                    )
            else:
                safe_key = str(key)
            nested_context[safe_key] = _sanitize_context_value(
                nested_value,
                field_path=f"{field_path}.{safe_key}",
                errors=errors,
            )
        return nested_context
    return value
