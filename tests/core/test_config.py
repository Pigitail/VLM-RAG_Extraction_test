from __future__ import annotations

import os

import pytest

from vlmrag.config import Settings, load_dotenv


def test_defaults_from_empty_environment():
    settings = Settings.from_env({})
    assert settings.provider == "openai"
    assert settings.max_concurrency == 4
    assert settings.timeout_s == 600
    assert settings.max_retries == 3
    assert settings.reasoning_effort == "low"
    assert settings.structured_output is True
    assert settings.retrieval_summary_doc_types == ()
    assert settings.cache_dir is None


def test_reads_every_variable():
    env = {
        "VLMRAG_PROVIDER": "Azure",
        "VLMRAG_MODEL": "deployment-a",
        "OPENAI_API_KEY": "test-openai-placeholder",
        "OPENAI_BASE_URL": "http://localhost:8000/v1",
        "AZURE_OPENAI_API_KEY": "test-azure-placeholder",
        "AZURE_OPENAI_ENDPOINT": "https://example-resource.example.com",
        "AZURE_OPENAI_API_VERSION": "2025-04-01-preview",
        "VLMRAG_MAX_CONCURRENCY": "8",
        "VLMRAG_TIMEOUT_S": "120",
        "VLMRAG_MAX_RETRIES": "5",
        "VLMRAG_REASONING_EFFORT": "medium",
        "VLMRAG_STRUCTURED_OUTPUT": "0",
        "VLMRAG_CACHE_DIR": ".cache/vlm",
        "VLMRAG_SOFFICE": "/opt/libreoffice/program/soffice",
        "VLMRAG_RETRIEVAL_SUMMARY": "slides, docx",
    }
    settings = Settings.from_env(env)
    assert settings.provider == "azure"
    assert settings.model == "deployment-a"
    assert settings.azure_api_version == "2025-04-01-preview"
    assert settings.max_concurrency == 8
    assert settings.timeout_s == 120.0
    assert settings.max_retries == 5
    assert settings.reasoning_effort == "medium"
    assert settings.structured_output is False
    assert settings.cache_dir == ".cache/vlm"
    assert settings.retrieval_summary_doc_types == ("slides", "docx")
    assert settings.summary_enabled_for("slides") and not settings.summary_enabled_for("drawing")


def test_secrets_are_not_in_repr():
    settings = Settings.from_env({"OPENAI_API_KEY": "test-secret-value-placeholder", "AZURE_OPENAI_API_KEY": "test-other-placeholder"})
    text = repr(settings)
    assert "test-secret-value-placeholder" not in text and "test-other-placeholder" not in text


def test_empty_reasoning_effort_disables_parameter():
    assert Settings.from_env({"VLMRAG_REASONING_EFFORT": ""}).reasoning_effort is None


def test_invalid_values_are_rejected():
    with pytest.raises(ValueError):
        Settings.from_env({"VLMRAG_PROVIDER": "nope"})
    with pytest.raises(ValueError):
        Settings.from_env({"VLMRAG_STRUCTURED_OUTPUT": "maybe"})
    with pytest.raises(ValueError):
        Settings().replace(provider="nope")


def test_replace_ignores_none():
    settings = Settings(model="a").replace(model=None, provider="null")
    assert settings.model == "a" and settings.provider == "null"


def test_load_dotenv(tmp_path, monkeypatch):
    env_file = tmp_path / ".env.local"
    env_file.write_text(
        "# comment\nVLMRAG_TEST_A=one\nexport VLMRAG_TEST_B='two words'\nVLMRAG_TEST_C=kept # note\nVLMRAG_TEST_D=new\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("VLMRAG_TEST_A", raising=False)
    monkeypatch.delenv("VLMRAG_TEST_B", raising=False)
    monkeypatch.delenv("VLMRAG_TEST_C", raising=False)
    monkeypatch.setenv("VLMRAG_TEST_D", "existing")
    assert load_dotenv(env_file) is True
    assert os.environ["VLMRAG_TEST_A"] == "one"
    assert os.environ["VLMRAG_TEST_B"] == "two words"
    assert os.environ["VLMRAG_TEST_C"] == "kept"
    assert os.environ["VLMRAG_TEST_D"] == "existing"  # the environment wins
    assert load_dotenv(tmp_path / "missing.env") is False
    for name in ("VLMRAG_TEST_A", "VLMRAG_TEST_B", "VLMRAG_TEST_C"):
        os.environ.pop(name, None)  # set by load_dotenv itself, so monkeypatch cannot undo them
