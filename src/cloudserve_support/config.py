"""Environment-backed application settings."""

import os
from dataclasses import dataclass, field
from typing import Mapping

from cloudserve_support.errors import ConfigurationError


@dataclass(frozen=True)
class Settings:
    """Minimal settings shared by future CLI commands."""

    provider: str = "OFFLINE"
    draft_generation_provider: str | None = None
    openai_api_key: str | None = field(default=None, repr=False)
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-6-luna"
    quality_judge_model: str = "gpt-6-luna"

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "Settings":
        """Load and validate settings without requiring provider credentials."""
        values = os.environ if environ is None else environ
        provider = values.get("LLM_PROVIDER", "OFFLINE").strip().upper()
        if provider not in {"OFFLINE", "OPENAI"}:
            raise ConfigurationError(
                "unsupported_provider",
                "LLM_PROVIDER must be OFFLINE or OPENAI; update the setting in your environment or .env file.",
            )

        draft_generation_provider = (
            values.get("DRAFT_GENERATION_PROVIDER", "").strip().upper() or provider
        )
        if draft_generation_provider not in {"OFFLINE", "OPENAI"}:
            raise ConfigurationError(
                "unsupported_provider",
                "DRAFT_GENERATION_PROVIDER must be OFFLINE or OPENAI; update the setting in your environment or .env file.",
            )

        embedding_model = values.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small").strip()
        if not embedding_model:
            raise ConfigurationError(
                "invalid_model_setting",
                "OPENAI_EMBEDDING_MODEL must not be empty. Set a model name in your environment or .env file.",
            )

        llm_model = values.get("OPENAI_LLM_MODEL", "gpt-6-luna").strip()
        if not llm_model:
            raise ConfigurationError(
                "invalid_model_setting",
                "OPENAI_LLM_MODEL must not be empty. Set a model name in your environment or .env file.",
            )

        quality_judge_model = values.get(
            "OPENAI_QUALITY_JUDGE_MODEL", "gpt-6-luna"
        ).strip()
        if not quality_judge_model:
            raise ConfigurationError(
                "invalid_model_setting",
                "OPENAI_QUALITY_JUDGE_MODEL must not be empty. Set a model name in your environment or .env file.",
            )

        return cls(
            provider=provider,
            draft_generation_provider=draft_generation_provider,
            openai_api_key=values.get("OPENAI_API_KEY", "").strip() or None,
            embedding_model=embedding_model,
            llm_model=llm_model,
            quality_judge_model=quality_judge_model,
        )

    @classmethod
    def from_quality_judge_env(
        cls, environ: Mapping[str, str] | None = None
    ) -> "Settings":
        """Read only the post-run judge model and key, independent of inference settings."""
        values = os.environ if environ is None else environ
        quality_judge_model = values.get(
            "OPENAI_QUALITY_JUDGE_MODEL", "gpt-6-luna"
        ).strip()
        if not quality_judge_model:
            raise ConfigurationError(
                "invalid_model_setting",
                "OPENAI_QUALITY_JUDGE_MODEL must not be empty. Set a model name in your environment or .env file.",
            )
        return cls(
            provider="OFFLINE",
            openai_api_key=values.get("OPENAI_API_KEY", "").strip() or None,
            quality_judge_model=quality_judge_model,
        )

    def require_openai_api_key(self) -> str:
        """Return the key for provider-backed work or raise a safe error."""
        if self.provider == "OPENAI" and self.openai_api_key:
            return self.openai_api_key
        raise ConfigurationError(
            "missing_openai_api_key",
            "OPENAI_API_KEY is required for provider-backed commands. Set it in your environment or .env file.",
        )

    def require_draft_generation_api_key(self) -> str:
        """Return the key only when OpenAI drafting was explicitly configured."""
        if (self.draft_generation_provider or self.provider) == "OPENAI" and self.openai_api_key:
            return self.openai_api_key
        raise ConfigurationError(
            "missing_openai_api_key",
            "OPENAI_API_KEY is required for OpenAI draft generation. Set it in your environment or .env file.",
        )

    def require_quality_judge_api_key(self) -> str:
        """Return the API key for post-run judging without changing inference provider selection."""
        if self.openai_api_key:
            return self.openai_api_key
        raise ConfigurationError(
            "missing_openai_api_key",
            "OPENAI_API_KEY is required for the quality judge. Set it in your environment or .env file.",
        )
