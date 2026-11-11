"""Structured draft provider and parser behavior."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import openai
import pytest

from cloudserve_support.config import Settings
from cloudserve_support.domain.ticket import CanonicalTicket
from cloudserve_support.errors import ConfigurationError
from cloudserve_support.guardrails.draft_safety import (
    check_claim_grounding,
    compose_cited_draft,
)
from cloudserve_support.generation.draft_generation import (
    DraftGenerationError,
    OfflineDraftGenerationProvider,
    OPENAI_DRAFT_RESPONSE_FORMAT,
    OpenAIDraftGenerationProvider,
    create_draft_generation_provider,
    parse_generated_claims,
)


def _ticket() -> CanonicalTicket:
    return CanonicalTicket(
        ticket_id="T-GENERATION-UNIT",
        has_ticket_id=True,
        channel="chat",
        subject="Dependency resolution fails",
        body="Builds fail dependency resolution.",
        received_at=None,
        customer_tier=None,
        language_fluency=None,
        context={
            "labels": {"expected_route": "LABEL_SECRET"},
            "history": {"prior_outcome": "HISTORY_SECRET"},
        },
    )


def test_offline_generation_returns_a_deterministic_source_excerpt() -> None:
    source = {
        "passage_id": "DOC-1:hash:0",
        "passage_text": "Refresh the package cache before retrying.",
        "title": "Package cache",
    }
    provider = OfflineDraftGenerationProvider()

    first = provider.generate_claims(_ticket(), sources=(source,), model_name="ignored")
    second = provider.generate_claims(_ticket(), sources=(source,), model_name="another")

    assert first == second
    assert json.loads(first)["claims"] == [
        {
            "text": source["passage_text"],
            "source_ids": [source["passage_id"]],
        }
    ]
    assert json.loads(first)["clarification_kind"] == "error_details"


def test_offline_generation_skips_standalone_numbered_list_marker() -> None:
    source = {
        "passage_id": "DOC-1:hash:0",
        "passage_text": "1. Refresh the package cache before retrying. 2. Restart the service.",
        "title": "Package cache",
    }

    generated = json.loads(
        OfflineDraftGenerationProvider().generate_claims(
            _ticket(), sources=(source,), model_name="ignored"
        )
    )

    assert generated["claims"][0]["text"] == "Refresh the package cache before retrying."


def test_openai_generation_uses_configured_model_and_strict_json_schema(monkeypatch) -> None:
    calls = {}

    class FakeOpenAIClient:
        def __init__(self, **kwargs):
            calls["client_options"] = kwargs
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self.create_completion)
            )

        @staticmethod
        def create_completion(**kwargs):
            calls["completion"] = kwargs
            time.sleep(0.02)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=json.dumps(
                                {
                                    "claims": [
                                        {
                                            "text": "Refresh the package cache.",
                                            "source_ids": ["DOC-1:hash:0"],
                                        }
                                    ],
                                    "clarification_kind": "error_details",
                                    "unsupported_points": [],
                                    "conflicting_points": [],
                                }
                            )
                        )
                    )
                ]
            )

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAIClient)
    settings = Settings(
        provider="OFFLINE",
        draft_generation_provider="OPENAI",
        openai_api_key="test-key",
        embedding_model="test-embeddings",
        llm_model="support-draft-model",
    )
    provider = create_draft_generation_provider(settings)
    source = {
        "passage_id": "DOC-1:hash:0",
        "passage_text": "Refresh the package cache.",
        "title": "Package cache",
    }

    content = provider.generate_claims(_ticket(), sources=(source,), model_name=settings.llm_model)

    assert provider.name == "openai"
    assert settings.provider == "OFFLINE"
    assert json.loads(content)["claims"][0]["source_ids"] == ["DOC-1:hash:0"]
    assert calls["completion"]["model"] == "support-draft-model"
    assert calls["completion"]["reasoning_effort"] == "none"
    assert calls["completion"]["response_format"] == OPENAI_DRAFT_RESPONSE_FORMAT
    assert calls["completion"]["response_format"]["json_schema"]["strict"] is True
    assert provider.last_request_duration_seconds >= 0.015
    schema = calls["completion"]["response_format"]["json_schema"]["schema"]
    assert schema["required"] == [
        "claims",
        "clarification_kind",
        "unsupported_points",
        "conflicting_points",
    ]
    assert schema["properties"]["clarification_kind"]["enum"] == [
        "error_details",
        "steps_tried",
        "expected_observed",
        "environment_version",
        "occurrence_time",
        "request_identifier",
        "none",
    ]
    assert schema["additionalProperties"] is False
    assert calls["client_options"]["timeout"] == 15.0
    assert calls["client_options"]["max_retries"] == 0
    provider_input = calls["completion"]["messages"][1]["content"]
    assert "LABEL_SECRET" not in provider_input
    assert "HISTORY_SECRET" not in provider_input


@pytest.mark.parametrize(
    "raw_json",
    [
        "{not json",
        json.dumps({"claims": [], "unsupported_points": []}),
        json.dumps(
            {
                "claims": [{"text": "No citation"}],
                "unsupported_points": [],
                "conflicting_points": [],
            }
        ),
    ],
)
def test_generated_claim_parser_rejects_bad_provider_json(raw_json: str) -> None:
    with pytest.raises(DraftGenerationError):
        parse_generated_claims(raw_json)


def test_generated_claim_parser_enforces_limits_not_encoded_in_json_schema() -> None:
    too_many_claims = [
        {"text": "Supported excerpt.", "source_ids": ["DOC-1:hash:0"]}
        for _ in range(4)
    ]
    missing_source_ids = [
        {"text": "Supported excerpt.", "source_ids": []}
    ]

    for claims in (too_many_claims, missing_source_ids):
        with pytest.raises(DraftGenerationError):
            parse_generated_claims(
                json.dumps(
                    {
                        "claims": claims,
                        "clarification_kind": "error_details",
                        "unsupported_points": [],
                        "conflicting_points": [],
                    }
                )
            )


def test_openai_provider_failure_is_returned_as_a_safe_structured_error(monkeypatch) -> None:
    class FakeOpenAIClient:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self.create_completion)
            )

        @staticmethod
        def create_completion(**kwargs):
            del kwargs
            time.sleep(0.02)
            raise TimeoutError("private provider detail")

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAIClient)
    provider = OpenAIDraftGenerationProvider(_settings())

    with pytest.raises(DraftGenerationError) as raised:
        provider.generate_claims(
            _ticket(),
            sources=(_source("DOC-1:hash:0", "Refresh the package cache."),),
            model_name="support-draft-model",
        )

    assert raised.value.code == "generation_timeout"
    assert "private provider detail" not in str(raised.value)
    assert provider.last_request_duration_seconds >= 0.015


def test_openai_provider_rejects_empty_evidence_before_client_creation(monkeypatch) -> None:
    monkeypatch.setattr(
        openai,
        "OpenAI",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("provider client must not be created without evidence")
        ),
    )
    provider = OpenAIDraftGenerationProvider(_settings())

    with pytest.raises(DraftGenerationError) as raised:
        provider.generate_claims(_ticket(), sources=(), model_name="support-draft-model")

    assert raised.value.code == "generation_no_sources"
    assert provider.last_request_duration_seconds == 0


def test_claim_citations_must_each_support_the_claim() -> None:
    first = _source("DOC-1:hash:0", "Refresh the package cache before retrying.")
    unrelated = _source("DOC-2:hash:0", "Check the configured runtime version.")
    claims = [
        {
            "text": "Refresh the package cache before retrying.",
            "source_ids": [first["passage_id"], unrelated["passage_id"]],
        }
    ]

    grounded, _ = check_claim_grounding(claims, (first, unrelated))

    assert grounded is False


def test_draft_provider_can_use_openai_without_switching_classifier_provider() -> None:
    settings = Settings.from_env(
        {
            "LLM_PROVIDER": "OFFLINE",
            "DRAFT_GENERATION_PROVIDER": "OPENAI",
            "OPENAI_API_KEY": "test-key",
        }
    )

    assert settings.provider == "OFFLINE"
    assert settings.draft_generation_provider == "OPENAI"
    assert create_draft_generation_provider(settings).name == "openai"
    assert settings.require_draft_generation_api_key() == "test-key"
    with pytest.raises(ConfigurationError):
        settings.require_openai_api_key()


def test_draft_generation_provider_setting_rejects_unsupported_value() -> None:
    with pytest.raises(ConfigurationError):
        Settings.from_env(
            {
                "LLM_PROVIDER": "OFFLINE",
                "DRAFT_GENERATION_PROVIDER": "NOT_A_PROVIDER",
            }
        )


def test_composition_qualifies_general_causes_and_renders_safe_question() -> None:
    source = {
        **_source(
            "DOC-1:hash:0",
            "Common causes include a service-level override and a stale project setting.",
        ),
        "section": "Common causes",
        "source_ref": "data/kb/documentation.json#DOC-1#Common causes",
    }
    draft = compose_cited_draft(
        {
            "claims": [
                {
                    "text": source["passage_text"],
                    "source_ids": [source["passage_id"]],
                }
            ],
            "clarification_kind": "error_details",
            "unsupported_points": [],
            "conflicting_points": [],
        },
        (source,),
        ticket_subject="Build variable is not read",
        ticket_text="Build variable is not read. No error message or code was shared.",
    )

    assert "possible explanations" in draft["plain_text"].casefold()
    assert "not confirmation that any one cause applies to your case" in draft["plain_text"]
    assert "What exact error message or code do you see?" in draft["plain_text"]
    assert f"[citation:{source['passage_id']}]" in draft["plain_text"]
    assert draft["claims"][0]["text"] == source["passage_text"]


def test_composition_omits_question_when_requested_detail_is_already_present() -> None:
    source = _source("DOC-1:hash:0", "Refresh the package cache before retrying.")
    draft = compose_cited_draft(
        {
            "claims": [
                {"text": source["passage_text"], "source_ids": [source["passage_id"]]}
            ],
            "clarification_kind": "error_details",
            "unsupported_points": [],
            "conflicting_points": [],
        },
        (source,),
        ticket_subject="Builds fail dependency resolution",
        ticket_text="Builds fail dependency resolution with HTTP 403.",
    )

    assert "What exact error message or code do you see?" not in draft["plain_text"]
    assert draft["plain_text"].endswith("specific environment.")


def _settings() -> Settings:
    return Settings(
        provider="OFFLINE",
        draft_generation_provider="OPENAI",
        openai_api_key="test-key",
        embedding_model="test-embeddings",
        llm_model="support-draft-model",
    )


def _source(passage_id: str, passage_text: str) -> dict[str, str]:
    return {
        "passage_id": passage_id,
        "passage_text": passage_text,
        "title": "Package cache",
        "section": "Steps",
        "source_ref": "data/kb/documentation.json#DOC-1#Steps",
    }
