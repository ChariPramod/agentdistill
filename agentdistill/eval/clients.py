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
    """`transformers.generate`. Greedy by default so eval is reproducible.

    `logprobs` and `n_samples` exist so the CPU rehearsal can run the calibration stage. They are correct but
    slow -- one forward pass per sample, no batching -- and on real hardware the vLLM client is the one to use.
    """

    def __init__(
        self,
        model: Any,
        tok: Any,
        parser_name: str | None = None,
        family: str | None = None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        logprobs: bool = False,
        n_samples: int = 0,
        top_logprobs: int = 5,
        seed: int = 0,
    ) -> None:
        self.model, self.tok = model, tok
        self.parser_name, self.family = parser_name, family
        self.max_new_tokens, self.temperature = max_new_tokens, temperature
        self.logprobs, self.n_samples, self.top_logprobs = logprobs, n_samples, top_logprobs
        self.seed = seed

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
        if self.logprobs:
            kwargs.update(output_scores=True, return_dict_in_generate=True)

        with torch.no_grad():
            out = self.model.generate(**enc, **kwargs)

        sequences = out.sequences if self.logprobs else out
        prompt_len = enc["input_ids"].shape[1]
        text = self.tok.decode(sequences[0, prompt_len:], skip_special_tokens=False)
        turn = parse_assistant(text, self.tok, self.parser_name, self.family)

        if self.logprobs:
            turn["logprobs"] = {"content": _hf_token_logprobs(
                out.scores, sequences[0, prompt_len:], self.tok, self.top_logprobs
            )}
        if self.n_samples:
            turn["samples"] = self._extra_samples(enc, messages, tools)
        return turn

    def _extra_samples(self, enc: Any, messages: list[dict], tools: list[dict]) -> list[dict]:
        """Additional sampled turns, for the agreement feature.

        Sampled rather than greedy: identical greedy turns would make agreement a constant 1.0 and the feature
        would carry no information at all.
        """
        import torch

        out: list[dict] = []
        for i in range(self.n_samples):
            # Seeded per sample so a rerun of the eval draws the same k turns; an unseeded agreement feature
            # would move between runs and the gate fitted on it would not be reproducible.
            torch.manual_seed(self.seed + i)
            with torch.no_grad():
                seq = self.model.generate(
                    **enc,
                    max_new_tokens=self.max_new_tokens,
                    pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id,
                    do_sample=True,
                    temperature=max(self.temperature, 0.7),
                    top_p=0.95,
                )
            text = self.tok.decode(seq[0, enc["input_ids"].shape[1]:], skip_special_tokens=False)
            out.append(parse_assistant(text, self.tok, self.parser_name, self.family))
        return out


def _hf_token_logprobs(scores: Any, tokens: Any, tok: Any, top_k: int) -> list[dict]:
    """Per-token logprobs in the OpenAI shape the feature extractor reads.

    One shape for both backends, so a gate fitted on vLLM output and one fitted on transformers output are
    fitted on the same columns.
    """
    import torch

    content = []
    for step, logits in enumerate(scores):
        if step >= len(tokens):
            break
        logprobs = torch.log_softmax(logits[0].float(), dim=-1)
        token_id = int(tokens[step])
        top = torch.topk(logprobs, k=min(top_k, logprobs.shape[-1]))
        content.append({
            "token": tok.decode([token_id]),
            "logprob": float(logprobs[token_id]),
            "top_logprobs": [
                {"token": tok.decode([int(i)]), "logprob": float(v)}
                for v, i in zip(top.values.tolist(), top.indices.tolist(), strict=True)
            ],
        })
    return content


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
        logprobs: bool = False,
        n_samples: int = 0,
        top_logprobs: int = 5,
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
        self.sp = SamplingParams(
            temperature=temperature, max_tokens=max_new_tokens, seed=seed or None,
            logprobs=top_logprobs if logprobs else None,
        )
        # Sampled rather than greedy: k identical greedy turns would make the agreement feature a constant.
        self.sample_sp = SamplingParams(
            n=n_samples, temperature=0.8, top_p=0.95, max_tokens=max_new_tokens, seed=seed or None
        ) if n_samples else None
        self.logprobs, self.n_samples = logprobs, n_samples
        self.tok, self.parser_name, self.family = tok, parser_name, family

    def _render(self, messages: list[dict], tools: list[dict]) -> str:
        return self.tok.apply_chat_template(
            messages, tools=tools or None, tokenize=False, add_generation_prompt=True
        )

    def next_turn(self, messages: list[dict], tools: list[dict]) -> dict:
        prompt = self._render(messages, tools)
        out = self.llm.generate([prompt], self.sp, lora_request=self.lora)
        turn = parse_assistant(out[0].outputs[0].text, self.tok, self.parser_name, self.family)
        if self.logprobs:
            turn["logprobs"] = {"content": _vllm_token_logprobs(out[0].outputs[0])}
        if self.sample_sp is not None:
            extra = self.llm.generate([prompt], self.sample_sp, lora_request=self.lora)
            turn["samples"] = [
                parse_assistant(o.text, self.tok, self.parser_name, self.family) for o in extra[0].outputs
            ]
        return turn

    def next_turns_batch(self, prompts: list[tuple[list[dict], list[dict]]]) -> list[dict]:
        texts = [self._render(m, t) for m, t in prompts]
        outs = self.llm.generate(texts, self.sp, lora_request=self.lora)
        return [parse_assistant(o.outputs[0].text, self.tok, self.parser_name, self.family) for o in outs]


def _vllm_token_logprobs(output: Any) -> list[dict]:
    """vLLM's per-token logprobs, reshaped to the OpenAI form the feature extractor reads."""
    content = []
    for token_id, entry in zip(output.token_ids, output.logprobs or [], strict=False):
        if not entry:
            continue
        chosen = entry.get(token_id)
        content.append({
            "token": getattr(chosen, "decoded_token", None) or str(token_id),
            "logprob": float(getattr(chosen, "logprob", float("nan"))) if chosen else float("nan"),
            "top_logprobs": [
                {"token": getattr(v, "decoded_token", None) or str(k), "logprob": float(v.logprob)}
                for k, v in entry.items()
            ],
        })
    return content


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


class LiteLLMTurnClient:
    """The configured teacher, through LiteLLM.

    One `teacher.model` string works for any provider, and it is the same path
    `examples/support_agent/record.py` uses to record the traces in the first place. Evaluating the teacher
    through a different client than recorded it would be measuring the client.

    Greedy by default, like every other eval client, so a teacher baseline is reproducible.
    """

    def __init__(
        self,
        model: str,
        logprobs: bool = False,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        completion: Any = None,
    ) -> None:
        self.model, self.logprobs = model, logprobs
        self.temperature, self.max_tokens = temperature, max_tokens
        self._completion = completion
        #: Running totals as the provider reports them; the harness diffs these per task for the cost block.
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "cached_prompt_tokens": 0}

    @property
    def completion(self) -> Any:
        if self._completion is None:
            import litellm

            self._completion = litellm.completion
        return self._completion

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
            # Not every provider supports logprobs; a teacher that cannot report them is still a valid
            # baseline, so this asks and tolerates their absence rather than failing the run.
            kwargs.update(logprobs=True, top_logprobs=5)

        response = self.completion(**kwargs)
        self._add_usage(getattr(response, "usage", None))
        choice = response.choices[0]
        message = choice.message
        calls = [
            {
                "id": c.id,
                "type": "function",
                "function": {"name": c.function.name, "arguments": c.function.arguments},
            }
            for c in (getattr(message, "tool_calls", None) or [])
        ]
        turn: dict[str, Any] = {
            "role": "assistant",
            "content": getattr(message, "content", None),
            "tool_calls": calls or None,
        }
        content = _litellm_logprobs(choice)
        if content:
            turn["logprobs"] = {"content": content}
        return turn


    def _add_usage(self, usage: Any) -> None:
        if usage is None:
            return
        get = usage.get if isinstance(usage, dict) else (lambda k, d=None: getattr(usage, k, d))
        self.usage["prompt_tokens"] += int(get("prompt_tokens", 0) or 0)
        self.usage["completion_tokens"] += int(get("completion_tokens", 0) or 0)
        details = get("prompt_tokens_details", None)
        cached = (details.get("cached_tokens") if isinstance(details, dict)
                  else getattr(details, "cached_tokens", None)) if details is not None else None
        self.usage["cached_prompt_tokens"] += int(cached or 0)


def _litellm_logprobs(choice: Any) -> list[dict]:
    """Per-token logprobs when the provider returned any, in the OpenAI shape the features read."""
    raw = getattr(choice, "logprobs", None)
    tokens = getattr(raw, "content", None) if raw is not None else None
    if not tokens:
        return []
    out = []
    for t in tokens:
        out.append({
            "token": getattr(t, "token", ""),
            "logprob": float(getattr(t, "logprob", float("nan"))),
            "top_logprobs": [
                {"token": getattr(x, "token", ""), "logprob": float(getattr(x, "logprob", float("nan")))}
                for x in (getattr(t, "top_logprobs", None) or [])
            ],
        })
    return out


def strip_private(message: dict) -> dict:
    """Remove transport-only keys before a turn joins a trajectory."""
    return {k: v for k, v in message.items() if not k.startswith("_")}
