"""Deterministic checks for source-grounded drafting and provider isolation."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

from google import genai
import pytest

from cloudserve_support.config import Settings
from cloudserve_support.domain.ticket import CanonicalTicket
from cloudserve_support.errors import ConfigurationError
from cloudserve_support.generation.draft_generation import (
    GEMINI_DRAFT_RESPONSE_SCHEMA,
    DraftGenerationError,
    GeminiDraftGenerationProvider,
    OfflineDraftGenerationProvider,
    create_draft_generation_provider,
    parse_generated_claims,
)
from cloudserve_support.guardrails.draft_safety import (
    check_claim_grounding,
    compose_cited_draft,
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


def _source(passage_id: str, passage_text: str) -> dict[str, str]:
    return {
        "passage_id": passage_id,
        "passage_text": passage_text,
        "title": "Package cache",
        "section": "Steps",
        "source_ref": "data/kb/documentation.json#DOC-1#Steps",
    }


def test_offline_generation_returns_a_deterministic_source_excerpt() -> None:
    source = _source("DOC-1:hash:0", "Refresh the package cache before retrying.")
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
    source = _source("DOC-1:hash:0", "1. Refresh the package cache before retrying. 2. Restart the service.")
    provider = OfflineDraftGenerationProvider()
    generated = json.loads(
        provider.generate_claims(
            _ticket(), sources=(source,), model_name="ignored"
        )
    )

    assert generated["claims"][0]["text"] == "Refresh the package cache before retrying."


def test_gemini_generation_uses_configured_model_and_strict_json_schema(monkeypatch) -> None:
    calls = {}

    class FakeModels:
        @staticmethod
        def generate_content(*, model: str, contents: str, config: object) -> object:
            calls["model"] = model
            calls["contents"] = contents
            calls["config"] = config
            time.sleep(0.02)
            return SimpleNamespace(
                text=json.dumps(
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

    class FakeGenAIClient:
        def __init__(self, **kwargs):
            calls["client_options"] = kwargs
            self.models = FakeModels()

    monkeypatch.setattr(genai, "Client", FakeGenAIClient)
    settings = Settings(
        draft_generation_provider="GEMINI",
        gemini_api_key="test-key",
        embedding_model="test-embeddings",
        llm_model="gemini-3.8-flash",
    )
    provider = create_draft_generation_provider(settings)
    source = {
        "passage_id": "DOC-1:hash:0",
        "passage_text": "Refresh the package cache.",
        "title": "Package cache",
    }

    content = provider.generate_claims(_ticket(), sources=(source,), model_name=settings.llm_model)

    assert provider.name == "gemini"
    assert json.loads(content)["claims"][0]["source_ids"] == ["DOC-1:hash:0"]
    assert calls["model"] == "gemini-3.8-flash"
    assert provider.last_request_duration_seconds >= 0.015
    config = calls["config"]
    assert config.response_mime_type == "application/json"
    assert config.response_schema == GEMINI_DRAFT_RESPONSE_SCHEMA
    schema = config.response_schema
    assert schema["required"] == [
        "claims",
        "clarification_kind",
        "unsupported_points",
        "conflicting_points",
    ]
    provider_input = calls["contents"]
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


def test_gemini_provider_failure_is_returned_as_a_safe_structured_error(monkeypatch) -> None:
    class FakeModels:
        @staticmethod
        def generate_content(*args, **kwargs):
            del args, kwargs
            time.sleep(0.02)
            raise TimeoutError("private provider detail")

    class FakeGenAIClient:
        def __init__(self, **kwargs):
            self.models = FakeModels()

    monkeypatch.setattr(genai, "Client", FakeGenAIClient)
    provider = GeminiDraftGenerationProvider(_settings())

    with pytest.raises(DraftGenerationError) as raised:
        provider.generate_claims(
            _ticket(),
            sources=(_source("DOC-1:hash:0", "Refresh the package cache."),),
            model_name="gemini-3.8-flash",
        )

    assert raised.value.code == "generation_timeout"
    assert "private provider detail" not in str(raised.value)
    assert provider.last_request_duration_seconds >= 0.015


def test_gemini_provider_rejects_empty_evidence_before_client_creation(monkeypatch) -> None:
    monkeypatch.setattr(
        genai,
        "Client",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("provider client must not be created without evidence")
        ),
    )
    provider = GeminiDraftGenerationProvider(_settings())

    with pytest.raises(DraftGenerationError) as raised:
        provider.generate_claims(_ticket(), sources=(), model_name="gemini-3.8-flash")

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


def test_draft_provider_defaults_to_gemini() -> None:
    settings = Settings.from_env(
        {
            "GEMINI_API_KEY": "test-key",
        }
    )

    assert settings.draft_generation_provider == "GEMINI"
    assert create_draft_generation_provider(settings).name == "gemini"
    assert settings.require_draft_generation_api_key() == "test-key"
    assert settings.require_gemini_api_key() == "test-key"


def test_draft_generation_provider_setting_rejects_unsupported_value() -> None:
    with pytest.raises(ConfigurationError):
        Settings.from_env(
            {
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
        draft_generation_provider="GEMINI",
        gemini_api_key="test-key",
        embedding_model="test-embeddings",
        llm_model="gemini-3.8-flash",
    )
