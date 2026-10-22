"""Unit tests for GenerationProvider implementations."""

import pytest

from ragforge.core.config import Settings
from ragforge.services.generation_provider import (
    AzureOpenAIGenerationProvider,
    FakeGenerationProvider,
    GenerationConfigurationError,
    create_generation_provider,
)


@pytest.mark.asyncio
async def test_fake_generation_provider_with_sources() -> None:
    """Verify fake generation provider returns deterministic answer citing source_1."""
    provider = FakeGenerationProvider(model_name="fake-model")
    assert provider.provider_name == "fake"
    assert provider.model_name == "fake-model"

    output = await provider.generate(
        system_prompt="System instructions",
        user_prompt="Context contains [source_1] text.\nQuestion: What is it?",
    )
    assert "[source_1]" in output.text
    assert output.provider_name == "fake"
    assert output.model_name == "fake-model"
    assert output.latency_ms >= 0.0
    assert output.prompt_tokens is not None and output.prompt_tokens > 0
    assert output.completion_tokens is not None and output.completion_tokens > 0
    assert output.total_tokens == output.prompt_tokens + output.completion_tokens


@pytest.mark.asyncio
async def test_fake_generation_provider_without_sources() -> None:
    """Verify fake generation provider returns abstention when no sources exist."""
    provider = FakeGenerationProvider()
    output = await provider.generate(
        system_prompt="System",
        user_prompt="Context: None.\nQuestion: What is it?",
    )
    assert "does not contain sufficient information" in output.text.lower()


@pytest.mark.asyncio
async def test_fake_generation_provider_custom_override() -> None:
    """Verify custom response override works as expected."""
    provider = FakeGenerationProvider(custom_response_override="Custom response with [source_99].")
    output = await provider.generate(system_prompt="", user_prompt="")
    assert output.text == "Custom response with [source_99]."


def test_azure_generation_provider_configuration_validation() -> None:
    """Verify Azure generation provider fails fast if required settings are missing."""
    # Missing endpoint
    with pytest.raises(GenerationConfigurationError) as exc_endpoint:
        AzureOpenAIGenerationProvider(
            endpoint=None,
            api_key="key",
            deployment_name="gpt-4o-mini",
        )
    assert "RAGFORGE_AZURE_OPENAI_ENDPOINT is not configured" in exc_endpoint.value.message

    # Missing API key
    with pytest.raises(GenerationConfigurationError) as exc_key:
        AzureOpenAIGenerationProvider(
            endpoint="https://example.openai.azure.com",
            api_key="",
            deployment_name="gpt-4o-mini",
        )
    assert "RAGFORGE_AZURE_OPENAI_API_KEY is not configured" in exc_key.value.message

    # Missing deployment name
    with pytest.raises(GenerationConfigurationError) as exc_dep:
        AzureOpenAIGenerationProvider(
            endpoint="https://example.openai.azure.com",
            api_key="test-key",
            deployment_name="  ",
        )
    assert "RAGFORGE_AZURE_OPENAI_CHAT_DEPLOYMENT_NAME is not configured" in exc_dep.value.message


def test_create_generation_provider_factory() -> None:
    """Verify factory initializes providers according to settings."""
    settings_fake = Settings(
        ENVIRONMENT="testing",
        GENERATION_PROVIDER="fake",
        GENERATION_MODEL="test-slm",
    )
    prov_fake = create_generation_provider(settings_fake)
    assert isinstance(prov_fake, FakeGenerationProvider)
    assert prov_fake.model_name == "test-slm"

    settings_azure_missing = Settings(
        ENVIRONMENT="testing",
        GENERATION_PROVIDER="azure",
    )
    with pytest.raises(GenerationConfigurationError):
        create_generation_provider(settings_azure_missing)
