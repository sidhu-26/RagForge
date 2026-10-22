import asyncio
import json
from typing import Any

import httpx
import pytest

from ragforge.core.config import Settings
from ragforge.services.generation_provider import (
    AzureOpenAIGenerationProvider,
    FakeGenerationProvider,
    GeminiGenerationProvider,
    GenerationConfigurationError,
    GenerationError,
    GenerationRateLimitError,
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

    # Gemini missing config
    settings_gemini_missing = Settings(
        ENVIRONMENT="testing",
        GENERATION_PROVIDER="gemini",
        GEMINI_API_KEY=None,
    )
    with pytest.raises(GenerationConfigurationError):
        create_generation_provider(settings_gemini_missing)

    # Gemini valid config
    settings_gemini_valid = Settings(
        ENVIRONMENT="testing",
        GENERATION_PROVIDER="gemini",
        GEMINI_API_KEY="test-gemini-key",
        GEMINI_MODEL="gemini-2.5-flash",
    )
    prov_gemini = create_generation_provider(settings_gemini_valid)
    assert isinstance(prov_gemini, GeminiGenerationProvider)
    assert prov_gemini.provider_name == "gemini"
    assert prov_gemini.model_name == "gemini-2.5-flash"


def test_gemini_generation_provider_configuration_validation() -> None:
    """Verify Gemini generation provider fails fast if required settings are missing."""
    with pytest.raises(GenerationConfigurationError) as exc_key:
        GeminiGenerationProvider(
            api_key=None,
            model_name="gemini-2.5-flash",
        )
    assert "RAGFORGE_GEMINI_API_KEY is not configured" in exc_key.value.message

    with pytest.raises(GenerationConfigurationError) as exc_key_empty:
        GeminiGenerationProvider(
            api_key="   ",
            model_name="gemini-2.5-flash",
        )
    assert "RAGFORGE_GEMINI_API_KEY is not configured" in exc_key_empty.value.message

    with pytest.raises(GenerationConfigurationError) as exc_model:
        GeminiGenerationProvider(
            api_key="test-api-key",
            model_name="",
        )
    assert "RAGFORGE_GEMINI_MODEL is not configured" in exc_model.value.message


@pytest.mark.asyncio
async def test_gemini_generate_success_with_tokens() -> None:
    """Verify successful generation with valid text, token usage, and request structure."""
    captured_request: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_request["url"] = str(request.url)
        captured_request["headers"] = dict(request.headers)
        captured_request["json"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            status_code=200,
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": "According to the docs [source_1], PostgreSQL uses MVCC."}
                            ],
                            "role": "model",
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 42,
                    "candidatesTokenCount": 18,
                    "totalTokenCount": 60,
                },
            },
        )

    transport = httpx.MockTransport(handler)
    provider = GeminiGenerationProvider(
        api_key="test-key-123",
        model_name="gemini-2.5-flash",
        transport=transport,
    )

    output = await provider.generate(
        system_prompt="You are a helpful assistant.",
        user_prompt="Explain MVCC [source_1].",
        temperature=0.2,
        max_tokens=256,
    )

    assert output.text == "According to the docs [source_1], PostgreSQL uses MVCC."
    assert output.provider_name == "gemini"
    assert output.model_name == "gemini-2.5-flash"
    assert output.latency_ms >= 0.0
    assert output.prompt_tokens == 42
    assert output.completion_tokens == 18
    assert output.total_tokens == 60

    # Verify request details
    expected_url = (
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    )
    assert captured_request["url"] == expected_url
    assert captured_request["headers"]["x-goog-api-key"] == "test-key-123"
    assert captured_request["json"]["systemInstruction"] == {
        "parts": [{"text": "You are a helpful assistant."}]
    }
    assert captured_request["json"]["contents"] == [
        {"role": "user", "parts": [{"text": "Explain MVCC [source_1]."}]}
    ]
    assert captured_request["json"]["generationConfig"] == {
        "temperature": 0.2,
        "maxOutputTokens": 256,
    }


@pytest.mark.asyncio
async def test_gemini_generate_success_without_system_prompt_and_missing_usage() -> None:
    """Verify success when system prompt is empty and usage metadata is missing."""
    captured_request: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_request["json"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            status_code=200,
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": "Simple answer"}],
                            "role": "model",
                        },
                        "finishReason": "STOP",
                    }
                ],
            },
        )

    transport = httpx.MockTransport(handler)
    provider = GeminiGenerationProvider(
        api_key="test-key-123",
        model_name="gemini-2.5-flash",
        transport=transport,
    )

    output = await provider.generate(
        system_prompt="",
        user_prompt="Hello Gemini",
    )

    assert output.text == "Simple answer"
    assert output.prompt_tokens is None
    assert output.completion_tokens is None
    assert output.total_tokens is None
    assert "systemInstruction" not in captured_request["json"]


@pytest.mark.asyncio
async def test_gemini_generate_http_400_client_error() -> None:
    """Verify HTTP 400 client error raises GenerationError and is not retried."""
    call_count = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(status_code=400, json={"error": {"message": "Invalid argument"}})

    transport = httpx.MockTransport(handler)
    provider = GeminiGenerationProvider(
        api_key="test-key",
        model_name="gemini-2.5-flash",
        max_retries=3,
        transport=transport,
    )

    with pytest.raises(GenerationError) as exc_info:
        await provider.generate("System", "User")

    assert call_count == 1
    assert exc_info.value.error_code == "GEMINI_CLIENT_ERROR_400"
    assert "HTTP 400" in exc_info.value.message


@pytest.mark.asyncio
async def test_gemini_generate_http_401_403_auth_error() -> None:
    """Verify HTTP 401/403 errors raise GenerationError with auth error code."""
    call_count = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(status_code=403, json={"error": {"message": "API key expired"}})

    transport = httpx.MockTransport(handler)
    provider = GeminiGenerationProvider(
        api_key="test-key",
        model_name="gemini-2.5-flash",
        max_retries=3,
        transport=transport,
    )

    with pytest.raises(GenerationError) as exc_info:
        await provider.generate("System", "User")

    assert call_count == 1
    assert exc_info.value.error_code == "GEMINI_AUTH_ERROR_403"
    assert "HTTP 403" in exc_info.value.message


@pytest.mark.asyncio
async def test_gemini_generate_http_429_retry_and_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify HTTP 429 rate limit retries with Retry-After and raises GenerationRateLimitError."""
    call_count = 0
    sleeps: list[float] = []

    async def fake_sleep(duration: float) -> None:
        sleeps.append(duration)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(
            status_code=429,
            headers={"Retry-After": "0.1"},
            json={"error": {"message": "Resource exhausted"}},
        )

    transport = httpx.MockTransport(handler)
    provider = GeminiGenerationProvider(
        api_key="test-key",
        model_name="gemini-2.5-flash",
        max_retries=2,
        transport=transport,
    )

    with pytest.raises(GenerationRateLimitError) as exc_info:
        await provider.generate("System", "User")

    # 1 initial attempt + 2 retries = 3 attempts total
    assert call_count == 3
    assert len(sleeps) == 2
    assert sleeps[0] == 0.1
    assert sleeps[1] == 0.1
    assert exc_info.value.error_code == "GENERATION_RATE_LIMIT"


@pytest.mark.asyncio
async def test_gemini_generate_5xx_transient_retry_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify 5xx server errors retry and can succeed on subsequent attempt."""
    call_count = 0
    sleeps: list[float] = []

    async def fake_sleep(duration: float) -> None:
        sleeps.append(duration)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            return httpx.Response(status_code=503, json={"error": "Service unavailable"})
        return httpx.Response(
            status_code=200,
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": "Recovered response"}],
                            "role": "model",
                        },
                        "finishReason": "STOP",
                    }
                ],
            },
        )

    transport = httpx.MockTransport(handler)
    provider = GeminiGenerationProvider(
        api_key="test-key",
        model_name="gemini-2.5-flash",
        max_retries=2,
        transport=transport,
    )

    output = await provider.generate("System", "User")
    assert call_count == 2
    assert len(sleeps) == 1
    assert output.text == "Recovered response"


@pytest.mark.asyncio
async def test_gemini_generate_timeout_and_network_error_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify network timeouts retry and raise GenerationError upon exhaustion."""
    call_count = 0
    sleeps: list[float] = []

    async def fake_sleep(duration: float) -> None:
        sleeps.append(duration)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        raise httpx.ConnectTimeout("Connection timed out")

    transport = httpx.MockTransport(handler)
    provider = GeminiGenerationProvider(
        api_key="test-key",
        model_name="gemini-2.5-flash",
        max_retries=2,
        transport=transport,
    )

    with pytest.raises(GenerationError) as exc_info:
        await provider.generate("System", "User")

    assert call_count == 3
    assert len(sleeps) == 2
    assert exc_info.value.error_code == "GENERATION_ATTEMPTS_EXHAUSTED"


@pytest.mark.asyncio
async def test_gemini_generate_invalid_json() -> None:
    """Verify invalid JSON response raises GenerationError with GEMINI_INVALID_RESPONSE."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code=200, text="Not a JSON string {")

    transport = httpx.MockTransport(handler)
    provider = GeminiGenerationProvider(
        api_key="test-key",
        model_name="gemini-2.5-flash",
        transport=transport,
    )

    with pytest.raises(GenerationError) as exc_info:
        await provider.generate("System", "User")

    assert exc_info.value.error_code == "GEMINI_INVALID_RESPONSE"


@pytest.mark.asyncio
async def test_gemini_generate_empty_candidates_or_blocked() -> None:
    """Verify empty candidates or blocked prompt raises EMPTY_GENERATION_CHOICES."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json={
                "candidates": [],
                "promptFeedback": {"blockReason": "SAFETY"},
            },
        )

    transport = httpx.MockTransport(handler)
    provider = GeminiGenerationProvider(
        api_key="test-key",
        model_name="gemini-2.5-flash",
        transport=transport,
    )

    with pytest.raises(GenerationError) as exc_info:
        await provider.generate("System", "User")

    assert exc_info.value.error_code == "EMPTY_GENERATION_CHOICES"
    assert "SAFETY" in exc_info.value.message


@pytest.mark.asyncio
async def test_gemini_generate_empty_text_content() -> None:
    """Verify empty text content raises EMPTY_GENERATION_CONTENT with finishReason."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json={
                "candidates": [
                    {
                        "content": {"parts": [{"text": ""}], "role": "model"},
                        "finishReason": "RECITATION",
                    }
                ],
            },
        )

    transport = httpx.MockTransport(handler)
    provider = GeminiGenerationProvider(
        api_key="test-key",
        model_name="gemini-2.5-flash",
        transport=transport,
    )

    with pytest.raises(GenerationError) as exc_info:
        await provider.generate("System", "User")

    assert exc_info.value.error_code == "EMPTY_GENERATION_CONTENT"
    assert "RECITATION" in exc_info.value.message
