"""Language model generation provider abstractions and implementations.

Defines:
- GenerationProvider (ABC): Typed interface for calling language models.
- GenerationOutput: Structured output containing text, latency, token usage.
- FakeGenerationProvider: Deterministic responses for unit and integration testing.
- AzureOpenAIGenerationProvider: Azure-hosted chat/completions provider with retries.
"""

import abc
import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx

from ragforge.core.config import Settings

logger = logging.getLogger(__name__)


class GenerationError(Exception):
    """Base exception for text generation errors."""

    def __init__(self, message: str, error_code: str = "GENERATION_FAILED") -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code


class GenerationConfigurationError(GenerationError):
    """Raised when provider configuration or credentials are missing."""

    def __init__(self, message: str) -> None:
        super().__init__(message, error_code="GENERATION_CONFIG_ERROR")


class GenerationRateLimitError(GenerationError):
    """Raised when upstream LLM endpoint responds with rate limiting (HTTP 429)."""

    def __init__(self, message: str = "Upstream generation rate limit exceeded.") -> None:
        super().__init__(message, error_code="GENERATION_RATE_LIMIT")


@dataclass
class GenerationOutput:
    """Standardized response from a generation model provider."""

    text: str
    provider_name: str
    model_name: str
    latency_ms: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class GenerationProvider(abc.ABC):
    """Abstract interface for language model generation."""

    @property
    @abc.abstractmethod
    def provider_name(self) -> str:
        """Name of provider backend (e.g. 'azure', 'fake')."""
        pass

    @property
    @abc.abstractmethod
    def model_name(self) -> str:
        """Name of model deployment or engine."""
        pass

    @abc.abstractmethod
    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> GenerationOutput:
        """Generate a model completion.

        Args:
            system_prompt: Trusted system instructions and constraints.
            user_prompt: User question combined with retrieved evidence.
            temperature: Sampling temperature (0.0 = deterministic).
            max_tokens: Maximum tokens in generated completion.

        Returns:
            GenerationOutput: Text and telemetry.
        """
        pass


class FakeGenerationProvider(GenerationProvider):
    """Deterministic fake language model for unit and integration testing.

    Does not call external APIs. If context contains evidence passages, quotes them
    and appends '[source_1]' citations.
    """

    def __init__(
        self,
        model_name: str = "fake-slm-01",
        latency_simulation_ms: float = 5.0,
        custom_response_override: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._latency = latency_simulation_ms
        self._override = custom_response_override

    @property
    def provider_name(self) -> str:
        return "fake"

    @property
    def model_name(self) -> str:
        return self._model_name

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> GenerationOutput:
        _ = (temperature, max_tokens)
        start_time = time.perf_counter()
        if self._latency > 0:
            await asyncio.sleep(self._latency / 1000.0)

        if self._override is not None:
            text = self._override
        else:
            # Check if prompt contains source references
            if "[source_1]" in user_prompt:
                text = (
                    "Based on the provided documentation, the requested operation functions "
                    "as described in the retrieved passages [source_1]."
                )
            else:
                text = (
                    "The provided context does not contain sufficient information to "
                    "answer the question."
                )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        prompt_tokens_est = len(system_prompt.split()) + len(user_prompt.split())
        completion_tokens_est = len(text.split())

        return GenerationOutput(
            text=text,
            provider_name=self.provider_name,
            model_name=self.model_name,
            latency_ms=elapsed_ms,
            prompt_tokens=prompt_tokens_est,
            completion_tokens=completion_tokens_est,
            total_tokens=prompt_tokens_est + completion_tokens_est,
        )


class AzureOpenAIGenerationProvider(GenerationProvider):
    """Production provider for Azure OpenAI Chat/Completions API.

    Uses endpoint format:
    {endpoint}/openai/deployments/{deployment}/chat/completions?api-version={api_version}
    """

    def __init__(
        self,
        endpoint: str | None,
        api_key: str | None,
        deployment_name: str | None,
        api_version: str = "2024-02-01",
        timeout_seconds: float = 45.0,
        max_retries: int = 3,
    ) -> None:
        if not endpoint or not endpoint.strip():
            raise GenerationConfigurationError(
                "Azure OpenAI generation provider selected, but "
                "RAGFORGE_AZURE_OPENAI_ENDPOINT is not configured."
            )
        if not api_key or not api_key.strip():
            raise GenerationConfigurationError(
                "Azure OpenAI generation provider selected, but "
                "RAGFORGE_AZURE_OPENAI_API_KEY is not configured."
            )
        if not deployment_name or not deployment_name.strip():
            raise GenerationConfigurationError(
                "Azure OpenAI generation provider selected, but "
                "RAGFORGE_AZURE_OPENAI_CHAT_DEPLOYMENT_NAME is not configured."
            )

        self._endpoint = endpoint.rstrip("/")
        self._api_key = api_key
        self._deployment = deployment_name
        self._api_version = api_version
        self._timeout = timeout_seconds
        self._max_retries = max_retries

    @property
    def provider_name(self) -> str:
        return "azure"

    @property
    def model_name(self) -> str:
        return self._deployment

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> GenerationOutput:
        url = (
            f"{self._endpoint}/openai/deployments/{self._deployment}/chat/completions"
            f"?api-version={self._api_version}"
        )
        headers = {
            "api-key": self._api_key,
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        attempts = 0
        last_error: Exception | None = None
        start_time = time.perf_counter()

        while attempts <= self._max_retries:
            attempts += 1
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    resp = await client.post(url, headers=headers, json=payload)

                if resp.status_code == 200:
                    data = resp.json()
                    choices = data.get("choices", [])
                    if not choices:
                        raise GenerationError(
                            "Azure OpenAI returned empty choices array",
                            error_code="EMPTY_GENERATION_CHOICES",
                        )
                    message_obj = choices[0].get("message", {})
                    content = message_obj.get("content", "")
                    usage = data.get("usage", {})
                    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

                    return GenerationOutput(
                        text=content,
                        provider_name=self.provider_name,
                        model_name=self.model_name,
                        latency_ms=elapsed_ms,
                        prompt_tokens=usage.get("prompt_tokens"),
                        completion_tokens=usage.get("completion_tokens"),
                        total_tokens=usage.get("total_tokens"),
                    )

                if resp.status_code == 429:
                    retry_after = float(resp.headers.get("Retry-After", 1.5 * attempts))
                    logger.warning(
                        "Azure OpenAI generation rate limited (HTTP 429)",
                        extra={"attempt": attempts, "retry_after": retry_after},
                    )
                    if attempts <= self._max_retries:
                        await asyncio.sleep(retry_after)
                        continue
                    raise GenerationRateLimitError()

                if resp.status_code >= 500:
                    logger.warning(
                        "Azure OpenAI generation server error",
                        extra={"status_code": resp.status_code, "attempt": attempts},
                    )
                    if attempts <= self._max_retries:
                        await asyncio.sleep(1.0 * (2 ** (attempts - 1)))
                        continue
                    raise GenerationError(
                        f"Azure OpenAI service error HTTP {resp.status_code}",
                        error_code="AZURE_SERVER_ERROR",
                    )

                # Client error (4xx except 429) - do not retry
                raise GenerationError(
                    f"Azure OpenAI API error HTTP {resp.status_code}: {resp.text[:200]}",
                    error_code=f"AZURE_CLIENT_ERROR_{resp.status_code}",
                )

            except (httpx.TimeoutException, httpx.NetworkError) as net_err:
                last_error = net_err
                logger.warning(
                    "Network error during Azure OpenAI generation call",
                    extra={"attempt": attempts, "error": str(net_err)},
                )
                if attempts <= self._max_retries:
                    await asyncio.sleep(1.0 * (2 ** (attempts - 1)))
                    continue
                break

        raise GenerationError(
            f"Azure OpenAI generation failed after {self._max_retries + 1} attempts: {last_error}",
            error_code="GENERATION_ATTEMPTS_EXHAUSTED",
        )


class GeminiGenerationProvider(GenerationProvider):
    """Production provider for Google Gemini Developer API.

    Uses endpoint format:
    https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent
    """

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(
        self,
        api_key: str | None,
        model_name: str | None,
        timeout_seconds: float = 45.0,
        max_retries: int = 3,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not api_key or not api_key.strip():
            raise GenerationConfigurationError(
                "Gemini generation provider selected, but "
                "RAGFORGE_GEMINI_API_KEY is not configured."
            )
        if not model_name or not model_name.strip():
            raise GenerationConfigurationError(
                "Gemini generation provider selected, but RAGFORGE_GEMINI_MODEL is not configured."
            )

        self._api_key = api_key.strip()
        self._model = model_name.strip()
        self._timeout = timeout_seconds
        self._max_retries = max_retries
        self._transport = transport

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def model_name(self) -> str:
        return self._model

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> GenerationOutput:
        url = f"{self.BASE_URL}/models/{self._model}:generateContent"
        headers = {
            "x-goog-api-key": self._api_key,
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_prompt}],
                }
            ],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if system_prompt and system_prompt.strip():
            payload["systemInstruction"] = {
                "parts": [{"text": system_prompt}],
            }

        attempts = 0
        last_error: Exception | None = None
        start_time = time.perf_counter()

        while attempts <= self._max_retries:
            attempts += 1
            try:
                async with httpx.AsyncClient(
                    timeout=self._timeout, transport=self._transport
                ) as client:
                    resp = await client.post(url, headers=headers, json=payload)

                if resp.status_code == 200:
                    try:
                        data = resp.json()
                    except Exception as json_err:
                        raise GenerationError(
                            "Gemini returned invalid JSON response",
                            error_code="GEMINI_INVALID_RESPONSE",
                        ) from json_err

                    candidates = data.get("candidates", [])
                    if not candidates:
                        prompt_feedback = data.get("promptFeedback", {})
                        block_reason = prompt_feedback.get("blockReason")
                        msg = (
                            f"Gemini response contained no candidates (blockReason: {block_reason})"
                            if block_reason
                            else "Gemini returned empty candidates array"
                        )
                        raise GenerationError(
                            msg,
                            error_code="EMPTY_GENERATION_CHOICES",
                        )

                    first_candidate = candidates[0]
                    content_obj = first_candidate.get("content", {})
                    parts = content_obj.get("parts", [])

                    # Extract usable text from parts
                    text_parts = [
                        p["text"]
                        for p in parts
                        if isinstance(p, dict) and "text" in p and isinstance(p["text"], str)
                    ]
                    content = "".join(text_parts)

                    if not content:
                        finish_reason = first_candidate.get("finishReason", "UNKNOWN")
                        raise GenerationError(
                            f"Gemini returned empty text content (finishReason: {finish_reason})",
                            error_code="EMPTY_GENERATION_CONTENT",
                        )

                    # Extract usage metadata
                    usage = data.get("usageMetadata", {})
                    prompt_tokens = usage.get("promptTokenCount")
                    completion_tokens = usage.get("candidatesTokenCount")
                    total_tokens = usage.get("totalTokenCount")

                    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

                    return GenerationOutput(
                        text=content,
                        provider_name=self.provider_name,
                        model_name=self.model_name,
                        latency_ms=elapsed_ms,
                        prompt_tokens=prompt_tokens if isinstance(prompt_tokens, int) else None,
                        completion_tokens=(
                            completion_tokens if isinstance(completion_tokens, int) else None
                        ),
                        total_tokens=total_tokens if isinstance(total_tokens, int) else None,
                    )

                if resp.status_code == 429:
                    raw_retry_after = resp.headers.get("Retry-After")
                    retry_after = 1.5 * attempts
                    if raw_retry_after:
                        with contextlib.suppress(ValueError):
                            retry_after = float(raw_retry_after)

                    logger.warning(
                        "Gemini generation rate limited (HTTP 429)",
                        extra={"attempt": attempts, "retry_after": retry_after},
                    )
                    if attempts <= self._max_retries:
                        await asyncio.sleep(retry_after)
                        continue
                    raise GenerationRateLimitError()

                if resp.status_code >= 500:
                    logger.warning(
                        "Gemini generation server error",
                        extra={"status_code": resp.status_code, "attempt": attempts},
                    )
                    if attempts <= self._max_retries:
                        await asyncio.sleep(1.0 * (2 ** (attempts - 1)))
                        continue
                    raise GenerationError(
                        f"Gemini service error HTTP {resp.status_code}",
                        error_code="GEMINI_SERVER_ERROR",
                    )

                # Client errors (4xx except 429) - do not retry
                if resp.status_code in (401, 403):
                    raise GenerationError(
                        f"Gemini authentication or permission error HTTP {resp.status_code}",
                        error_code=f"GEMINI_AUTH_ERROR_{resp.status_code}",
                    )

                raise GenerationError(
                    f"Gemini API client error HTTP {resp.status_code}",
                    error_code=f"GEMINI_CLIENT_ERROR_{resp.status_code}",
                )

            except (httpx.TimeoutException, httpx.NetworkError) as net_err:
                last_error = net_err
                logger.warning(
                    "Network error during Gemini generation call",
                    extra={"attempt": attempts, "error": type(net_err).__name__},
                )
                if attempts <= self._max_retries:
                    await asyncio.sleep(1.0 * (2 ** (attempts - 1)))
                    continue
                break

        raise GenerationError(
            f"Gemini generation failed after {self._max_retries + 1} attempts: {last_error}",
            error_code="GENERATION_ATTEMPTS_EXHAUSTED",
        )


def create_generation_provider(settings: Settings) -> GenerationProvider:
    """Factory creating the configured GenerationProvider instance."""
    provider_type = settings.GENERATION_PROVIDER.lower()

    if provider_type == "fake":
        return FakeGenerationProvider(model_name=settings.GENERATION_MODEL)

    if provider_type == "azure":
        api_key_val = (
            settings.AZURE_OPENAI_API_KEY.get_secret_value()
            if settings.AZURE_OPENAI_API_KEY
            else None
        )
        return AzureOpenAIGenerationProvider(
            endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=api_key_val,
            deployment_name=settings.AZURE_OPENAI_CHAT_DEPLOYMENT_NAME,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            timeout_seconds=settings.GENERATION_REQUEST_TIMEOUT_SECONDS,
            max_retries=settings.GENERATION_MAX_RETRIES,
        )

    if provider_type == "gemini":
        gemini_api_key_val = (
            settings.GEMINI_API_KEY.get_secret_value() if settings.GEMINI_API_KEY else None
        )
        return GeminiGenerationProvider(
            api_key=gemini_api_key_val,
            model_name=settings.GEMINI_MODEL,
            timeout_seconds=settings.GENERATION_REQUEST_TIMEOUT_SECONDS,
            max_retries=settings.GENERATION_MAX_RETRIES,
        )

    raise GenerationConfigurationError(f"Unsupported GENERATION_PROVIDER: '{provider_type}'")
