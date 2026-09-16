"""Turn clients: anything that produces one assistant turn from a prefix.

Four implementations, all satisfying `TurnClient`:

- `RecordedTurnClient` — replays the recorded assistant turns. Proves the harness reproduces a trace exactly.
- `HfTurnClient` — `transformers.generate`. Works on CPU, so it runs in tests and inside training callbacks.
  Slow; keep turn counts small.
- `VllmOfflineTurnClient` — `vllm.LLM` with an optional LoRA. The fast path on a GPU, with a batched method.
- `HttpTurnClient` — any OpenAI-compatible endpoint: a vLLM server, a teacher API, or the gateway later.

The first two parse tool calls out of generated *text*, because that is what a served model emits. The last one
gets structured tool calls back from the API, so no parsing is needed -- and that asymmetry is exactly why the
round-trip check in `base-check` matters.
"""

from __future__ import annotations

import json
from typing import Any

from agentdistill.data.template_check import parse_fallback, parse_with_vllm


class ParseError(ValueError):
    """Generated text could not be turned into an assistant turn."""


def parse_assistant(text: str, tok: Any, parser_name: str | None, family: str | None) -> dict:
    """Turn generated text into a normalized assistant message.

    A model that emits something no parser recognizes has, for the harness's purposes, produced a text answer.
    That is the same thing the serving stack would do, so scoring it any other way would flatter the student.
    """
    calls = parse_with_vllm(text, tok, parser_name)
    if calls is None and family:
        calls = parse_fallback(text, family)
    if not calls:
        return {"role": "assistant", "content": text.strip(), "tool_calls": None}
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": f"call_{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
            for i, (name, args) in enumerate(calls)
        ],
    }


class RecordedTurnClient:
    """Emits the recorded assistant turns in order.

    The control for the whole harness: replaying a trace through it must reproduce that trace exactly, with zero
    divergences. If it does not, the harness is wrong and every number it produces is meaningless.
    """

    def __init__(self, trace: dict) -> None:
        self.turns = [m for m in trace["messages"] if m["role"] == "assistant"]
        self.i = 0

    def next_turn(self, messages: list[dict], tools: list[dict]) -> dict:
        if self.i >= len(self.turns):
            # The recording ended. Answering with empty text ends the trajectory cleanly rather than looping.
            return {"role": "assistant", "content": "", "tool_calls": None}
        turn = self.turns[self.i]
        self.i += 1
        return turn

    def reset(self) -> None:
        self.i = 0


class ScriptedTurnClient:
    """Emits a fixed list of turns. For tests that need a specific divergence."""

    def __init__(self, turns: list[dict]) -> None:
        self.turns = turns
        self.i = 0

    def next_turn(self, messages: list[dict], tools: list[dict]) -> dict:
        if self.i >= len(self.turns):
            return {"role": "assistant", "content": "", "tool_calls": None}
        turn = self.turns[self.i]
        self.i += 1
        return turn


class HfTurnClient:
    """`transformers.generate`. Greedy by default so eval is reproducible."""

    def __init__(
        self,
        model: Any,
        tok: Any,
        parser_name: str | None = None,
        family: str | None = None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ) -> None:
        self.model, self.tok = model, tok
        self.parser_name, self.family = parser_name, family
        self.max_new_tokens, self.temperature = max_new_tokens, temperature

    def next_turn(self, messages: list[dict], tools: list[dict]) -> dict:
        import torch

        prompt = self.tok.apply_chat_template(
            messages, tools=tools or None, tokenize=False, add_generation_prompt=True
        )
        enc = self.tok(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.tok.pad_token_id or self.tok.eos_token_id,
        }
        if self.temperature > 0:
            kwargs.update(do_sample=True, temperature=self.temperature)
        else:
            kwargs.update(do_sample=False)
        with torch.no_grad():
            out = self.model.generate(**enc, **kwargs)
        text = self.tok.decode(out[0, enc["input_ids"].shape[1] :], skip_special_tokens=False)
        return parse_assistant(text, self.tok, self.parser_name, self.family)


class VllmOfflineTurnClient:
    """`vllm.LLM` with an optional LoRA adapter. The fast path on a rented GPU.

    Prefix caching matters here: turns from one task share nearly all of their prompt, so the batched path is
    dramatically cheaper than it looks.
    """

    def __init__(
        self,
        base_model: str,
        tok: Any,
        parser_name: str | None = None,
        family: str | None = None,
        lora_path: str | None = None,
        max_new_tokens: int = 512,
        quantization: str | None = None,
        temperature: float = 0.0,
        max_model_len: int | None = None,
        seed: int = 0,
    ) -> None:
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest

        self.llm = LLM(
            model=base_model,
            enable_lora=lora_path is not None,
            max_lora_rank=64,
            quantization=quantization,
            enable_prefix_caching=True,
            max_model_len=max_model_len,
            seed=seed,
        )
        self.lora = LoRARequest("student", 1, lora_path) if lora_path else None
        self.sp = SamplingParams(temperature=temperature, max_tokens=max_new_tokens, seed=seed or None)
        self.tok, self.parser_name, self.family = tok, parser_name, family

    def _render(self, messages: list[dict], tools: list[dict]) -> str:
        return self.tok.apply_chat_template(
            messages, tools=tools or None, tokenize=False, add_generation_prompt=True
        )

    def next_turn(self, messages: list[dict], tools: list[dict]) -> dict:
        out = self.llm.generate([self._render(messages, tools)], self.sp, lora_request=self.lora)
        return parse_assistant(out[0].outputs[0].text, self.tok, self.parser_name, self.family)

    def next_turns_batch(self, prompts: list[tuple[list[dict], list[dict]]]) -> list[dict]:
        texts = [self._render(m, t) for m, t in prompts]
        outs = self.llm.generate(texts, self.sp, lora_request=self.lora)
        return [parse_assistant(o.outputs[0].text, self.tok, self.parser_name, self.family) for o in outs]


class HttpTurnClient:
    """Any OpenAI-compatible endpoint: a vLLM server, a teacher API, or the agentdistill gateway.

    Tool calls come back structured, so no text parsing happens here.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "none",
        logprobs: bool = False,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        timeout: float = 120.0,
    ) -> None:
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model, self.logprobs = model, logprobs
        self.temperature, self.max_tokens = temperature, max_tokens

    def next_turn(self, messages: list[dict], tools: list[dict]) -> dict:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if tools:
            kwargs["tools"] = tools
        if self.max_tokens:
            kwargs["max_tokens"] = self.max_tokens
        if self.logprobs:
            kwargs.update(logprobs=True, top_logprobs=5)
        response = self.client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        calls = [
            {
                "id": c.id,
                "type": "function",
                "function": {"name": c.function.name, "arguments": c.function.arguments},
            }
            for c in (message.tool_calls or [])
        ]
        return {
            "role": "assistant",
            "content": message.content,
            "tool_calls": calls or None,
            # Underscore-prefixed keys are stripped before a turn is appended to a trajectory; the cascade reads
            # this later for logprob features.
            "_raw": response.choices[0].model_dump(),
        }


def strip_private(message: dict) -> dict:
    """Remove transport-only keys before a turn joins a trajectory."""
    return {k: v for k, v in message.items() if not k.startswith("_")}
