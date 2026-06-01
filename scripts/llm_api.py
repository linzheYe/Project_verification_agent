from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from typing import Any

import requests


class LLMConnectivityError(RuntimeError):
    """Network/proxy connectivity failure when calling LLM provider."""


@dataclass
class LLMClient:
    """统一 LLM 调用客户端（当前默认实现是 OpenRouter）。

    设计意图：
    - 文件名和类名都不绑定单一厂商，后续可扩展其他 provider。
    - 对 pipeline 只暴露一个 `call()` 接口，减少替换成本。
    """

    provider: str = "openrouter"
    model: str = os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash")
    timeout_seconds: int = 300
    max_retries: int = 6
    retry_base_delay_seconds: float = 1.5
    retry_max_delay_seconds: float = 30.0
    retry_jitter_seconds: float = 0.5

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str:
        """调用 LLM 并返回纯文本 content。"""
        if self.provider == "openrouter":
            return self._call_openrouter(
                system_prompt,
                user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        raise ValueError(f"unsupported provider: {self.provider}")

    def _call_openrouter(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float,
        max_tokens: int | None,
    ) -> str:
        """OpenRouter 实现细节。"""
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()},
            ],
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)
        last_connectivity_error: Exception | None = None
        attempts = self.max_retries + 1
        for attempt_idx in range(attempts):
            try:
                resp = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json=payload,
                    timeout=self.timeout_seconds,
                )
            except requests.exceptions.RequestException as e:
                last_connectivity_error = LLMConnectivityError(f"openrouter_request_error: {e}")
                if attempt_idx >= self.max_retries:
                    raise last_connectivity_error from e
                self._sleep_before_retry(attempt_idx)
                continue

            if self._is_retryable_status(resp.status_code):
                last_connectivity_error = LLMConnectivityError(
                    f"OpenRouter connectivity error {resp.status_code}: {resp.text[:500]}"
                )
                if attempt_idx >= self.max_retries:
                    raise last_connectivity_error
                self._sleep_before_retry(attempt_idx)
                continue

            if resp.status_code != 200:
                raise RuntimeError(f"OpenRouter API error {resp.status_code}: {resp.text[:500]}")

            text = str(resp.json()["choices"][0]["message"].get("content", "")).strip()
            if not text:
                raise RuntimeError("OpenRouter returned empty content")
            return text

        if last_connectivity_error is not None:
            raise last_connectivity_error
        raise RuntimeError("OpenRouter call failed without response")

    @staticmethod
    def _is_retryable_status(status_code: int) -> bool:
        retryable = {
            407, 408, 409, 423, 425, 429,
            500, 502, 503, 504,
            520, 521, 522, 523, 524, 529,
        }
        return status_code in retryable

    def _sleep_before_retry(self, attempt_idx: int) -> None:
        base = self.retry_base_delay_seconds * (2**attempt_idx)
        capped = min(base, self.retry_max_delay_seconds)
        jitter = random.uniform(0, self.retry_jitter_seconds) if self.retry_jitter_seconds > 0 else 0.0
        time.sleep(capped + jitter)
