"""Smoke tests for the installable package and configuration boundary."""

import pytest

from cloudserve_support import __version__
from cloudserve_support.cli import main
from cloudserve_support.config import Settings
from cloudserve_support.errors import ConfigurationError


def test_package_import_exposes_version() -> None:
    assert __version__ == "0.1.0"


def test_cli_help_does_not_require_provider_credentials(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as result:
        main(["--help"])

    assert result.value.code == 0
    assert "usage: cloudserve" in capsys.readouterr().out


def test_cli_without_arguments_shows_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert "usage: cloudserve" in capsys.readouterr().out


def test_settings_can_load_without_api_credentials() -> None:
    settings = Settings.from_env({})

    assert settings.gemini_api_key is None
    assert settings.draft_generation_provider == "GEMINI"
    assert settings.embedding_model == "gemini-embedding-001"
    assert settings.quality_judge_model == "gemini-3.8-flash"


def test_settings_repr_does_not_expose_provider_credentials() -> None:
    settings = Settings.from_env({"GEMINI_API_KEY": "never-log-this-secret"})
    assert "never-log-this-secret" not in repr(settings)

    settings_google = Settings.from_env({"GOOGLE_API_KEY": "never-log-this-secret-2"})
    assert "never-log-this-secret-2" not in repr(settings_google)
    assert settings_google.gemini_api_key == "never-log-this-secret-2"


def test_missing_provider_key_has_actionable_secret_safe_error() -> None:
    settings = Settings.from_env({})

    with pytest.raises(ConfigurationError) as result:
        settings.require_gemini_api_key()

    assert result.value.code == "missing_gemini_api_key"
    assert "GEMINI_API_KEY" in result.value.message
    assert result.value.as_dict() == {
        "code": "missing_gemini_api_key",
        "message": result.value.message,
    }


@pytest.mark.parametrize(
    "environment_variable",
    ["GEMINI_EMBEDDING_MODEL", "GEMINI_LLM_MODEL", "GEMINI_QUALITY_JUDGE_MODEL"],
)
@pytest.mark.parametrize("value", ["", "   "])
def test_empty_model_setting_has_structured_actionable_error(
    environment_variable: str,
    value: str,
) -> None:
    with pytest.raises(ConfigurationError) as result:
        Settings.from_env({environment_variable: value})

    assert result.value.code == "invalid_model_setting"
    assert environment_variable in result.value.message
