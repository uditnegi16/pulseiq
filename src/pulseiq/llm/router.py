"""Multi-provider LLM routing: Groq primary, NVIDIA NIM fallback.

WHY A ROUTER RATHER THAN A CLIENT
---------------------------------
Free-tier LLM endpoints rate-limit, and a single-provider integration turns a
429 into a failed request. Routing to a second provider on failure costs a few
dozen lines and removes an entire class of outage.

The fallback is deliberately narrow. Retrying a 400 (malformed request) on a
different provider just produces the same 400 more slowly, so only transient
failures -- timeouts, rate limits, 5xx -- trigger a retry elsewhere.

DEGRADATION
-----------
With no provider configured, `generate()` raises `NoProviderAvailable` rather
than returning placeholder text. A recommendation endpoint that silently returns
canned advice when the LLM is down is worse than one that returns 503: the
caller cannot tell the difference between analysis and filler.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import requests

from config.settings import settings

logger = logging.getLogger(__name__)

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
NVIDIA_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

# Retrying these on another provider may succeed. Anything else is a request
# problem that will fail identically everywhere.
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


class NoProviderAvailable(RuntimeError):
    """No LLM provider is configured, or all configured providers failed."""


class TransientProviderError(RuntimeError):
    """A failure worth retrying on the next provider."""


@dataclass(frozen=True)
class Provider:
    name: str
    url: str
    api_key: str
    model: str


@dataclass(frozen=True)
class Completion:
    text: str
    provider: str
    model: str


def available_providers() -> list[Provider]:
    """Configured providers, in priority order. Groq first: faster free tier."""
    providers: list[Provider] = []

    if settings.groq_api_key is not None:
        providers.append(
            Provider(
                name="groq",
                url=GROQ_URL,
                api_key=settings.require("groq_api_key"),
                model=settings.groq_model,
            )
        )

    if settings.nvidia_nim_api_key is not None:
        providers.append(
            Provider(
                name="nvidia_nim",
                url=NVIDIA_URL,
                api_key=settings.require("nvidia_nim_api_key"),
                model=settings.nvidia_nim_model or "meta/llama-3.1-8b-instruct",
            )
        )

    return providers


def call_provider(
    provider: Provider,
    *,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 400,
    temperature: float = 0.3,
    timeout: float = 30.0,
    session: requests.Session | None = None,
) -> Completion:
    """Call one provider. Both expose an OpenAI-compatible schema.

    Raises TransientProviderError for failures worth retrying elsewhere, and
    RuntimeError for request problems that will fail identically everywhere.
    """
    caller = session or requests

    try:
        response = caller.post(
            provider.url,
            headers={
                "Authorization": f"Bearer {provider.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": provider.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=timeout,
        )
    except requests.Timeout as exc:
        raise TransientProviderError(f"{provider.name}: timed out after {timeout}s") from exc
    except requests.RequestException as exc:
        raise TransientProviderError(f"{provider.name}: {type(exc).__name__}") from exc

    if response.status_code in RETRYABLE_STATUS:
        raise TransientProviderError(f"{provider.name}: HTTP {response.status_code}")

    if response.status_code >= 400:
        # Never log or surface the response body verbatim -- provider errors can
        # echo the request, and the request contains the Authorization header.
        raise RuntimeError(f"{provider.name}: HTTP {response.status_code}")

    try:
        payload = response.json()
        text = payload["choices"][0]["message"]["content"].strip()
    except (ValueError, KeyError, IndexError) as exc:
        raise TransientProviderError(f"{provider.name}: unexpected response shape") from exc

    if not text:
        raise TransientProviderError(f"{provider.name}: empty completion")

    return Completion(text=text, provider=provider.name, model=provider.model)


def generate(
    *,
    system_prompt: str,
    user_prompt: str,
    providers: list[Provider] | None = None,
    **kwargs,
) -> Completion:
    """Try each provider in order. Raises NoProviderAvailable if all fail."""
    providers = providers if providers is not None else available_providers()

    if not providers:
        raise NoProviderAvailable(
            "No LLM provider configured. Set GROQ_API_KEY or NVIDIA_NIM_API_KEY "
            "in .env (see .env.example)."
        )

    failures: list[str] = []
    for provider in providers:
        try:
            completion = call_provider(
                provider, system_prompt=system_prompt, user_prompt=user_prompt, **kwargs
            )
            if failures:
                logger.info("%s succeeded after %d failure(s)", provider.name, len(failures))
            return completion
        except TransientProviderError as exc:
            logger.warning("provider failed, trying next -- %s", exc)
            failures.append(str(exc))
        except RuntimeError as exc:
            # A request-level error will fail identically on every provider.
            logger.error("non-retryable provider error -- %s", exc)
            failures.append(str(exc))
            break

    raise NoProviderAvailable("All providers failed:\n  " + "\n  ".join(failures))
