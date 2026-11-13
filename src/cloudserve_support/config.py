"""Environment-backed application settings."""

import os
from dataclasses import dataclass, field
from typing import Mapping

from cloudserve_support.errors import ConfigurationError


@dataclass(frozen=True)
class Settings:
    """Minimal settings shared by CLI commands."""

    draft_generation_provider: str = "GEMINI"
    gemini_api_key: str | None = field(default=None, repr=False)
    embedding_model: str = "gemini-embedding-001"
    embedding_dimensions: int = 768
    llm_model: str = "gemini-3.8-flash"
    quality_judge_model: str = "gemini-3.8-flash"

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "Settings":
        """Load and validate settings without requiring provider credentials."""
        values = os.environ if environ is None else environ
        draft_generation_provider = (
            values.get("DRAFT_GENERATION_PROVIDER", "GEMINI").strip().upper()
        )
        if draft_generation_provider not in {"OFFLINE", "GEMINI"}:
            raise ConfigurationError(
                "unsupported_provider",
                "DRAFT_GENERATION_PROVIDER must be OFFLINE or GEMINI; update the setting in your environment or .env file.",
            )

        embedding_model = values.get("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001").strip()
        if not embedding_model:
            raise ConfigurationError(
                "invalid_model_setting",
                "GEMINI_EMBEDDING_MODEL must not be empty. Set a model name in your environment or .env file.",
            )

        llm_model = values.get("GEMINI_LLM_MODEL", "gemini-3.8-flash").strip()
        if not llm_model:
            raise ConfigurationError(
                "invalid_model_setting",
                "GEMINI_LLM_MODEL must not be empty. Set a model name in your environment or .env file.",
            )

        quality_judge_model = values.get(
            "GEMINI_QUALITY_JUDGE_MODEL", "gemini-3.8-flash"
        ).strip()
        if not quality_judge_model:
            raise ConfigurationError(
                "invalid_model_setting",
                "GEMINI_QUALITY_JUDGE_MODEL must not be empty. Set a model name in your environment or .env file.",
            )

        api_key = (
            values.get("GEMINI_API_KEY", "").strip()
            or values.get("GOOGLE_API_KEY", "").strip()
            or None
        )

        return cls(
            draft_generation_provider=draft_generation_provider,
            gemini_api_key=api_key,
            embedding_model=embedding_model,
            embedding_dimensions=768,
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
            "GEMINI_QUALITY_JUDGE_MODEL", "gemini-3.8-flash"
        ).strip()
        if not quality_judge_model:
            raise ConfigurationError(
                "invalid_model_setting",
                "GEMINI_QUALITY_JUDGE_MODEL must not be empty. Set a model name in your environment or .env file.",
            )
        api_key = (
            values.get("GEMINI_API_KEY", "").strip()
            or values.get("GOOGLE_API_KEY", "").strip()
            or None
        )
        return cls(
            gemini_api_key=api_key,
            quality_judge_model=quality_judge_model,
        )

    def require_gemini_api_key(self) -> str:
        """Return the key for provider-backed work or raise a safe error."""
        if self.gemini_api_key:
            return self.gemini_api_key
        raise ConfigurationError(
            "missing_gemini_api_key",
            "GEMINI_API_KEY or GOOGLE_API_KEY is required for provider-backed commands. Set it in your environment or .env file.",
        )

    def require_draft_generation_api_key(self) -> str:
        """Return the key when Gemini drafting is selected."""
        if self.draft_generation_provider == "GEMINI" and self.gemini_api_key:
            return self.gemini_api_key
        raise ConfigurationError(
            "missing_gemini_api_key",
            "GEMINI_API_KEY or GOOGLE_API_KEY is required for Gemini draft generation. Set it in your environment or .env file.",
        )

    def require_quality_judge_api_key(self) -> str:
        """Return the API key for post-run judging without changing inference provider selection."""
        if self.gemini_api_key:
            return self.gemini_api_key
        raise ConfigurationError(
            "missing_gemini_api_key",
            "GEMINI_API_KEY or GOOGLE_API_KEY is required for the quality judge. Set it in your environment or .env file.",
        )
