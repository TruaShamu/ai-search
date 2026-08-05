from __future__ import annotations

import os
import random
import threading
import time
from pathlib import Path
from typing import Optional

import httpx
from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
DEFAULT_API_VERSION = "2024-12-01-preview"
DEFAULT_DEPLOYMENT = "gpt-54-nano"
DEFAULT_TIMEOUT = 30
DEFAULT_RETRY_BACKOFF: tuple[float, ...] = (2, 5, 15, 30)
INPUT_TOKEN_PRICE_PER_MILLION = 0.20
OUTPUT_TOKEN_PRICE_PER_MILLION = 1.25


class AzureOpenAIClient:
    """Shared Azure OpenAI chat-completion wrapper with retry logic."""

    BACKOFF = DEFAULT_RETRY_BACKOFF

    def __init__(
        self,
        *,
        dry_run: bool = False,
        api_version: str = DEFAULT_API_VERSION,
        timeout: float = DEFAULT_TIMEOUT,
        retry_backoff: tuple[float, ...] = DEFAULT_RETRY_BACKOFF,
        rate_limit_max_attempts: int | None = None,
        rate_limit_base_wait: float = 0.6,
        rate_limit_max_wait: float = 8.0,
    ):
        self.dry_run = dry_run
        self.api_version = api_version
        self.retry_backoff = retry_backoff
        self.max_retries = len(retry_backoff) + 1
        self.rate_limit_max_attempts = rate_limit_max_attempts
        self.rate_limit_base_wait = rate_limit_base_wait
        self.rate_limit_max_wait = rate_limit_max_wait
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_calls = 0
        self._usage_lock = threading.Lock()

        if dry_run:
            self.endpoint = self.key = self.deployment = ""
            self.url = ""
            self.headers: dict[str, str] = {}
            self.client: Optional[httpx.Client] = None
            return

        load_dotenv(ENV_PATH)
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
        self.key = os.getenv("AZURE_OPENAI_KEY") or os.getenv("AZURE_OPENAI_API_KEY", "")
        self.deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", DEFAULT_DEPLOYMENT)

        if not self.endpoint or not self.key:
            raise ValueError(
                "AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_KEY must be set in .env"
            )

        self.url = (
            f"{self.endpoint}/openai/deployments/{self.deployment}"
            f"/chat/completions?api-version={self.api_version}"
        )
        self.headers = {"api-key": self.key, "Content-Type": "application/json"}
        self.client = httpx.Client(timeout=timeout)

    def close(self) -> None:
        if self.client is not None:
            self.client.close()

    def _record_usage(self, usage: dict) -> None:
        with self._usage_lock:
            self.total_input_tokens += usage.get("prompt_tokens", 0)
            self.total_output_tokens += usage.get("completion_tokens", 0)
            self.total_calls += 1

    @staticmethod
    def _is_content_filter_response(resp: httpx.Response) -> bool:
        try:
            err = resp.json().get("error") or {}
        except Exception:
            return False
        codes = " ".join(
            str(x)
            for x in (
                err.get("code", ""),
                (err.get("innererror") or {}).get("code", ""),
                err.get("message", ""),
            )
        ).lower()
        return "content_filter" in codes or "responsibleaipolicyviolation" in codes

    def call(
        self,
        system: str | None = None,
        user: str | None = None,
        *,
        messages: list[dict[str, str]] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 120,
        strip_quotes: bool = False,
        fail_on_4xx: bool = True,
        content_filter_as_none: bool = False,
        respect_retry_after: bool = True,
        jitter_rate_limit: bool = False,
    ) -> Optional[str]:
        """Send a chat completion and return assistant content, or None."""
        if self.dry_run:
            return None

        assert self.client is not None
        if messages is None:
            messages = []
            if system is not None:
                messages.append({"role": "system", "content": system})
            if user is not None:
                messages.append({"role": "user", "content": user})
        if not messages:
            raise ValueError("messages or system/user content must be provided")

        body = {
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
        }

        net_attempts = 0
        rate_attempts = 0

        while True:
            try:
                resp = self.client.post(self.url, json=body, headers=self.headers)

                if resp.status_code == 429:
                    rate_attempts += 1
                    if (
                        self.rate_limit_max_attempts is not None
                        and rate_attempts >= self.rate_limit_max_attempts
                    ):
                        return None
                    if respect_retry_after:
                        retry_after = float(resp.headers.get("Retry-After", 10))
                        time.sleep(retry_after)
                    else:
                        wait = min(
                            self.rate_limit_max_wait,
                            self.rate_limit_base_wait * (1.7 ** (rate_attempts - 1)),
                        )
                        if jitter_rate_limit:
                            wait *= random.uniform(0.7, 1.3)
                        time.sleep(wait)
                    continue

                if 400 <= resp.status_code < 500:
                    if content_filter_as_none and self._is_content_filter_response(resp):
                        return None
                    if fail_on_4xx:
                        raise RuntimeError(
                            f"Unrecoverable HTTP {resp.status_code}: {resp.text[:400]}"
                        )
                    return None

                resp.raise_for_status()
                data = resp.json()
                self._record_usage(data.get("usage", {}))
                content = data["choices"][0]["message"]["content"].strip()
                if strip_quotes:
                    content = content.strip('"').strip("'")
                return content

            except RuntimeError:
                raise
            except (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPStatusError):
                net_attempts += 1
                if net_attempts >= self.max_retries:
                    return None
                time.sleep(self.retry_backoff[min(net_attempts - 1, len(self.retry_backoff) - 1)])
            except Exception:
                net_attempts += 1
                if net_attempts >= self.max_retries:
                    return None
                time.sleep(self.retry_backoff[min(net_attempts - 1, len(self.retry_backoff) - 1)])

    def cost_estimate(self) -> dict:
        inp = (self.total_input_tokens / 1_000_000) * INPUT_TOKEN_PRICE_PER_MILLION
        out = (self.total_output_tokens / 1_000_000) * OUTPUT_TOKEN_PRICE_PER_MILLION
        return {
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_calls": self.total_calls,
            "cost_usd": round(inp + out, 4),
        }
