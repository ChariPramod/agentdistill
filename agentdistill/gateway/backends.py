"""What the gateway talks to.

Both backends return the same shape -- OpenAI choice dicts with a `text` field -- so the cascade's feature code
has one input format regardless of which side produced a turn.
"""

from __future__ import annotations

from typing import Any

import httpx


class BackendError(RuntimeError):
    """An upstream failed. Carries the status so the gateway can decide whether to fall back."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class StudentBackend:
    """An OpenAI-compatible endpoint, normally `vllm serve`."""

    def __init__(self, base_url: str, client: httpx.AsyncClient | None = None, timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.http = client or httpx.AsyncClient(timeout=timeout)
        self._owns_client = client is None

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict],
        model: str = "student",
        n: int = 1,
        logprobs: bool = False,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict:
        body: dict[str, Any] = {"model": model, "messages": messages, "n": n, "temperature": temperature}
        if tools:
            body["tools"] = tools
        if max_tokens:
            body["max_tokens"] = max_tokens
        if logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = 5
        if extra:
            body.update(extra)

        try:
            response = await self.http.post(f"{self.base_url}/v1/chat/completions", json=body)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise BackendError(f"student returned {e.response.status_code}", e.response.status_code) from e
        except httpx.HTTPError as e:
            raise BackendError(f"student is unreachable: {e}") from e

        data = response.json()
        for choice in data.get("choices", []):
            # The feature code reads `text`; a server that returned no logprobs leaves it empty rather than absent.
            choice.setdefault(
                "text", "".join(t["token"] for t in (choice.get("logprobs") or {}).get("content", []))
            )
        return data

    async def models(self) -> list[str]:
        response = await self.http.get(f"{self.base_url}/v1/models")
        response.raise_for_status()
        return [m["id"] for m in response.json().get("data", [])]

    async def aclose(self) -> None:
        if self._owns_client:
            await self.http.aclose()


class TeacherBackend:
    """Any provider, through LiteLLM, returned in the same shape as the student."""

    def __init__(self, model: str, **litellm_kwargs: Any) -> None:
        self.model = model
        self.kwargs = litellm_kwargs

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict],
        n: int = 1,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        **_: Any,
    ) -> dict:
        import litellm

        try:
            response = await litellm.acompletion(
                model=self.model, messages=messages, tools=tools or None, n=n,
                temperature=temperature, max_tokens=max_tokens, **self.kwargs,
            )
        except Exception as e:
            raise BackendError(f"teacher call failed: {type(e).__name__}: {e}") from e
        data = response.model_dump() if hasattr(response, "model_dump") else dict(response)
        for choice in data.get("choices", []):
            choice.setdefault("text", choice.get("message", {}).get("content") or "")
        return data


class StubBackend:
    """Returns scripted choices. Used by the gateway's own tests and by `serve --dry-run`."""

    def __init__(self, choices: list[dict] | None = None, usage: dict | None = None) -> None:
        self.choices = choices or [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]
        self.usage = usage or {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        self.calls: list[dict] = []

    async def chat(self, messages: list[dict], tools: list[dict], n: int = 1, **kwargs: Any) -> dict:
        self.calls.append({"messages": messages, "tools": tools, "n": n, **kwargs})
        return {"choices": [dict(self.choices[min(i, len(self.choices) - 1)]) for i in range(n)],
                "usage": dict(self.usage)}
