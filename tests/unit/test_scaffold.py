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

    assert settings.openai_api_key is None
    assert settings.draft_generation_provider == "OPENAI"
    assert settings.embedding_model == "text-embedding-3-small"
    assert settings.quality_judge_model == "gpt-6-luna"


def test_settings_repr_does_not_expose_provider_credentials() -> None:
    settings = Settings.from_env({"OPENAI_API_KEY": "never-log-this-secret"})

    assert "never-log-this-secret" not in repr(settings)


def test_missing_provider_key_has_actionable_secret_safe_error() -> None:
    settings = Settings.from_env({})

    with pytest.raises(ConfigurationError) as result:
        settings.require_openai_api_key()

    assert result.value.code == "missing_openai_api_key"
    assert "OPENAI_API_KEY" in result.value.message
    assert result.value.as_dict() == {
        "code": "missing_openai_api_key",
        "message": result.value.message,
    }


@pytest.mark.parametrize(
    "environment_variable",
    ["OPENAI_EMBEDDING_MODEL", "OPENAI_LLM_MODEL", "OPENAI_QUALITY_JUDGE_MODEL"],
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
    assert "must not be empty" in result.value.message
    assert result.value.as_dict()["code"] == "invalid_model_setting"
