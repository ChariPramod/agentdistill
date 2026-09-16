# agentdistill: Next Phase Plan

Milestone 2 completion, real traces, Milestone 3 harness, bridge to Milestone 4
Version 0.2, September 15, 2026
Builds on: agentdistill-implementation-plan.md v0.1 and docs/progress.md at Milestone 1

---

## 0. Where you are and what this phase must produce

Done: ingest, curate, dataset build with verified masks, SFT trainer that saves and reloads on a tiny model, 257 tests, CI on 3.11 to 3.13. Adapters can only reach `candidate`. Every command past Milestone 2 exits 2 naming its milestone.

This phase ends when one sentence is true and measured: **"On N real held-out tasks, the student's success rate is X% versus the teacher's Y%, delta with a 95% CI, from a run anyone can reproduce with one command."** Nothing else in the project matters until that sentence exists. The gateway, cascade, router, and cost report all consume that number.

Four workstreams, in dependency order:

1. **M2 close-out**: template round trip through a real tool parser, mask invariant tests, teacher-forced next-action accuracy, compat shim for TRL, one real 8B QLoRA run with numbers in the registry.
2. **M2.5 real traces**: an example agent with verifiable tools that produces real teacher trajectories and programmatic success labels. Synthetic traces are demoted to unit fixtures.
3. **M3 harness**: student clients, replay tool provider with canonical hashing, task runner, graders, metrics, paired statistics, `eval run` and `eval compare`.
4. **M4 bridge**: rollout collection reuses the harness; pairs and RFT sets are one function away.

Ten working days if nothing surprises you. Assume two surprises.

---

## 1. Patch the plan document first (30 minutes)

The code diverged from v0.1 in three places and the plan must say what the code does. Make these edits to `agentdistill-implementation-plan.md` and commit them with the code they describe.

**Section 3.2 near_dedupe**: replace the Jaccard rule with: "MinHash LSH proposes candidates; each candidate pair is verified with the MinHash Jaccard estimate against the threshold before a drop. LSH banding is approximate and returns pairs below threshold." Add: "`near_dedupe_normalize_literals` (mask numbers, ids, and timestamps before shingling) is off by default; it collapses short corpora to one sample per trajectory shape. Turn it on only for corpora with long assistant text."

**Section 4.3 loss masking**: replace the containment rule with: "A token is a target if its start offset lies inside an assistant span. Tokens straddling the end boundary (end-of-turn marker merged with a following newline) are targets; tokens straddling the start boundary (header merged with the first content token) are not." Add the invariant from 2.2 below as the test that guards it.

**Section 3.1 filters table**: the `near_dedupe` row gets "LSH candidates verified against threshold".

**Section 19**: append a "Divergences from plan" pointer to `docs/progress.md`.

Also add a top-level `docs/canonical-json.md` (section 8 of this document) because the harness, the pair builder, and later mcpgate and agentreplay all hash arguments and must agree.

---

## 2. Milestone 2 close-out

### 2.1 Template round trip through a real tool parser

This is the check that prevents a wasted GPU day. The student will be served by vLLM, and vLLM recovers tool calls from generated text with a template-specific parser. If the training data renders tool calls in a way the parser cannot read, the student is useless no matter how good its loss is.

```python
# agentdistill/data/template_check.py  (additions)
from __future__ import annotations

import json
import re
from typing import Any

FALLBACK_PARSERS: dict[str, re.Pattern[str]] = {
    # hermes / qwen style: <tool_call>{"name": ..., "arguments": {...}}</tool_call>
    "hermes": re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S),
    # llama 3.x json style: {"name": ..., "parameters": {...}}
    "llama3_json": re.compile(r"(\{\s*\"name\"\s*:\s*\".+?\"\s*,\s*\"parameters\"\s*:\s*\{.*\}\s*\})", re.S),
}


def example_from_schema(schema: dict[str, Any]) -> Any:
    t = schema.get("type")
    if "enum" in schema:
        return schema["enum"][0]
    if t == "object":
        props = schema.get("properties", {})
        req = schema.get("required", list(props))
        return {k: example_from_schema(props[k]) for k in req if k in props}
    if t == "array":
        return [example_from_schema(schema.get("items", {"type": "string"}))]
    if t == "integer":
        return 7
    if t == "number":
        return 7.5
    if t == "boolean":
        return True
    return "x"


def _render(tok, messages, tools, add_generation_prompt):
    return tok.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=add_generation_prompt)


def parse_with_vllm(model_output: str, tok, parser_name: str) -> list[tuple[str, dict]] | None:
    try:
        from vllm.entrypoints.openai.tool_parsers import ToolParserManager  # type: ignore
    except ImportError:
        return None
    parser = ToolParserManager.get_tool_parser(parser_name)(tok)
    info = parser.extract_tool_calls(model_output, request=None)  # verify signature against installed vLLM
    if not info.tools_called:
        return []
    return [(c.function.name, json.loads(c.function.arguments)) for c in info.tool_calls]


def parse_fallback(model_output: str, family: str) -> list[tuple[str, dict]]:
    pat = FALLBACK_PARSERS[family]
    out = []
    for m in pat.finditer(model_output):
        obj = json.loads(m.group(1))
        args = obj.get("arguments", obj.get("parameters", {}))
        out.append((obj["name"], args))
    return out


def roundtrip_tool_call(tok, tools: list[dict], parser_name: str, family: str) -> dict:
    """Render one assistant tool call with the template, strip the header, parse it back, compare."""
    fn = tools[0]["function"]
    args = example_from_schema(fn.get("parameters", {"type": "object"}))
    msgs = [{"role": "system", "content": "You are a test."}, {"role": "user", "content": "Do the thing."}]
    assistant = {"role": "assistant", "content": "", "tool_calls": [
        {"id": "call_0", "type": "function", "function": {"name": fn["name"], "arguments": json.dumps(args)}}]}
    full = _render(tok, msgs + [assistant], tools, False)
    header = _render(tok, msgs, tools, True)
    if not full.startswith(header):
        raise ValueError("template is not prefix-stable; cannot isolate model output")
    model_output = full[len(header):]
    parsed = parse_with_vllm(model_output, tok, parser_name)
    source = "vllm"
    if parsed is None:
        parsed = parse_fallback(model_output, family)
        source = "fallback"
    ok = parsed == [(fn["name"], args)]
    return {"ok": ok, "parser": source, "expected": [(fn["name"], args)], "parsed": parsed, "model_output": model_output}
```

Wire into `base-check`: print the round-trip result and refuse a non-`ok` template unless `--allow-unparsed` is passed. The `parser_name` (vLLM's name, e.g. `hermes`) and `family` (fallback regex key) live in `project.yaml` under `train.tool_parser` so the same values feed the serving command later.

Test: `tests/test_template_roundtrip.py` runs the fallback path on the three fixture templates and, when vLLM is importable, the vLLM path too (skip otherwise). Both must agree.

### 2.2 Mask invariant test

Your start-inside rule is right. Guard it.

```python
# tests/test_mask_invariants.py
from __future__ import annotations

import pytest

from agentdistill.data.build import build_trajectory_sample

TEMPLATES = ["fixtures/tok_hermes", "fixtures/tok_llama3", "fixtures/tok_chatml"]   # your three checked-in tokenizers


def _render(tok, messages, tools, gen):
    return tok.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=gen)


def _header_and_eot(tok, tools):
    """Derive the assistant header string and end-of-turn string from the template itself."""
    sys_user = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    with_gen = _render(tok, sys_user, tools, True)
    without = _render(tok, sys_user, tools, False)
    header = with_gen[len(without):]
    content = "REPLY_MARKER_9f3"
    full = _render(tok, sys_user + [{"role": "assistant", "content": content}], tools, False)
    tail = full[full.index(content) + len(content):]
    return header, tail.strip()


@pytest.mark.parametrize("path", TEMPLATES)
def test_label_invariants(load_tok, fixture_trace, path):
    tok = load_tok(path)
    tools, messages = fixture_trace["tools"], fixture_trace["messages"]
    header, eot = _header_and_eot(tok, tools)
    s = build_trajectory_sample(tok, messages, tools, 8192)
    assert s is not None
    target_ids = [i for i in s.labels if i != -100]
    text = tok.decode(target_ids, skip_special_tokens=False)
    n_assistant = sum(1 for m in messages if m["role"] == "assistant")
    assert text.count(eot) == n_assistant, f"expected {n_assistant} end-of-turn markers in targets, got {text.count(eot)}"
    assert header.strip() not in text, "assistant header leaked into targets"
    assert text.rstrip().endswith(eot), "last target does not end with end-of-turn"
    # every tool call's JSON must be fully inside the targets
    for m in messages:
        for c in m.get("tool_calls") or []:
            assert c["function"]["name"] in text
```

If a template puts the header on the same token as the first content character (rare, but it happens with some ChatML variants and no leading newline), the header assertion will fail and you will know to special-case that template rather than discover it as a student that emits `assistant\n` at the top of every reply.

### 2.3 TRL compatibility shim

Do not chase field renames by hand on the GPU box.

```python
# agentdistill/train/compat.py
from __future__ import annotations

import dataclasses
import importlib
from typing import Any


def sft_config_kwargs(cfg: dict[str, Any], out_dir: str) -> dict[str, Any]:
    from trl import SFTConfig
    fields = {f.name for f in dataclasses.fields(SFTConfig)}
    want: dict[str, Any] = {
        "output_dir": out_dir,
        "num_train_epochs": cfg["epochs"],
        "learning_rate": cfg["lr"],
        "lr_scheduler_type": cfg["scheduler"],
        "warmup_ratio": cfg["warmup_ratio"],
        "per_device_train_batch_size": cfg["per_device_batch"],
        "gradient_accumulation_steps": cfg["grad_accum"],
        "bf16": True,
        "gradient_checkpointing": True,
        "eval_strategy": "steps",
        "eval_steps": cfg["eval_every_steps"],
        "save_strategy": "steps",
        "save_steps": cfg["eval_every_steps"],
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "logging_steps": 10,
        "seed": cfg["seed"],
        "report_to": ["tensorboard"],
    }
    # renamed across TRL releases
    for cands, value in [(("max_length", "max_seq_length"), cfg["max_seq_len"])]:
        for c in cands:
            if c in fields:
                want[c] = value
                break
    packing = bool(cfg.get("packing")) and flash_attn_available()
    if "packing" in fields:
        want["packing"] = packing
    if "padding_free" in fields:
        want["padding_free"] = packing
    if "dataset_kwargs" in fields:
        want["dataset_kwargs"] = {"skip_prepare_dataset": True}
    unknown = [k for k in want if k not in fields]
    if unknown:
        raise RuntimeError(f"SFTConfig in installed TRL lacks fields {unknown}; update compat.py")
    return want


def flash_attn_available() -> bool:
    try:
        importlib.import_module("flash_attn")
        return True
    except Exception:
        return False


def attn_implementation(cfg: dict[str, Any]) -> str:
    return "flash_attention_2" if cfg.get("packing") and flash_attn_available() else "sdpa"
```

If flash-attn is absent, packing silently turns off and the run log says so. Do not spend the GPU day building flash-attn; the first run is for validating the pipeline.

### 2.4 Teacher-forced next-action accuracy

This is cheap, per-turn, runs without tool mocking, and it is the first real quality signal you will have. It answers "does the student choose the same action the teacher chose, given the teacher's own prefix."

```python
# agentdistill/eval/teacher_forced.py
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from agentdistill.canonical import args_hash


class TurnClient(Protocol):
    def next_turn(self, messages: list[dict], tools: list[dict]) -> dict:
        """Return an assistant message dict: {'content': str|None, 'tool_calls': [...] | None}."""


@dataclass
class TurnResult:
    task_id: str
    turn_idx: int
    teacher_kind: str      # 'tool' | 'text'
    student_kind: str
    name_match: bool       # same set of tool names (order-insensitive)
    args_match: bool       # same canonical arg hashes for every call
    kind_match: bool


def _sig(m: dict) -> tuple[str, set[str], set[str]]:
    calls = m.get("tool_calls") or []
    names = {c["function"]["name"] for c in calls}
    hashes = set()
    for c in calls:
        a = c["function"]["arguments"]
        args = json.loads(a) if isinstance(a, str) else a
        hashes.add(args_hash(c["function"]["name"], args))
    return ("tool" if calls else "text", names, hashes)


def teacher_forced(traces: list[dict], client: TurnClient, max_turns_per_trace: int | None = None) -> list[TurnResult]:
    out: list[TurnResult] = []
    for t in traces:
        msgs, tools = t["messages"], t["tools"]
        seen = 0
        for i, m in enumerate(msgs):
            if m["role"] != "assistant":
                continue
            if max_turns_per_trace and seen >= max_turns_per_trace:
                break
            seen += 1
            student = client.next_turn(msgs[:i], tools)
            tk, tn, th = _sig(m)
            sk, sn, sh = _sig(student)
            out.append(TurnResult(t.get("task_id", t["id"]), i, tk, sk, tn == sn, th == sh, tk == sk))
    return out


def summarize(results: list[TurnResult], iters: int = 2000, seed: int = 0) -> dict:
    by_task: dict[str, list[TurnResult]] = {}
    for r in results:
        by_task.setdefault(r.task_id, []).append(r)
    tasks = sorted(by_task)
    rng = np.random.default_rng(seed)

    def rate(fn) -> tuple[float, tuple[float, float]]:
        per_task = np.array([np.mean([fn(r) for r in by_task[t]]) for t in tasks])
        idx = rng.integers(0, len(tasks), size=(iters, len(tasks)))
        boots = per_task[idx].mean(axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        return float(per_task.mean()), (float(lo), float(hi))

    tool_turns = [r for r in results if r.teacher_kind == "tool"]
    return {
        "n_turns": len(results), "n_tasks": len(tasks),
        "kind_match": rate(lambda r: r.kind_match),
        "name_match_on_tool_turns": (float(np.mean([r.name_match for r in tool_turns])) if tool_turns else float("nan")),
        "args_match_on_tool_turns": (float(np.mean([r.args_match for r in tool_turns])) if tool_turns else float("nan")),
        "full_match": rate(lambda r: r.kind_match and r.name_match and r.args_match),
    }
```

Two clients implement `TurnClient`:

```python
# agentdistill/eval/clients.py
from __future__ import annotations

import json
from typing import Any

from agentdistill.data.template_check import parse_fallback, parse_with_vllm


def parse_assistant(text: str, tok, parser_name: str, family: str) -> dict:
    calls = parse_with_vllm(text, tok, parser_name)
    if calls is None:
        calls = parse_fallback(text, family)
    if not calls:
        return {"role": "assistant", "content": text.strip(), "tool_calls": None}
    return {"role": "assistant", "content": None, "tool_calls": [
        {"id": f"call_{i}", "type": "function", "function": {"name": n, "arguments": json.dumps(a)}} for i, (n, a) in enumerate(calls)]}


class HfTurnClient:
    """transformers generate; used inside training callbacks and in tests. Slow; keep turn counts small."""

    def __init__(self, model, tok, parser_name: str, family: str, max_new_tokens: int = 256):
        self.model, self.tok, self.parser_name, self.family, self.max_new_tokens = model, tok, parser_name, family, max_new_tokens

    def next_turn(self, messages: list[dict], tools: list[dict]) -> dict:
        import torch
        prompt = self.tok.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True)
        enc = self.tok(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False, pad_token_id=self.tok.eos_token_id)
        text = self.tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=False)
        return parse_assistant(text, self.tok, self.parser_name, self.family)


class VllmOfflineTurnClient:
    """vllm.LLM with an optional LoRA; the fast path for eval on a rented GPU."""

    def __init__(self, base_model: str, tok, parser_name: str, family: str, lora_path: str | None = None,
                 max_new_tokens: int = 512, quantization: str | None = None):
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
        self.llm = LLM(model=base_model, enable_lora=lora_path is not None, max_lora_rank=64,
                       quantization=quantization, enable_prefix_caching=True)
        self.lora = LoRARequest("student", 1, lora_path) if lora_path else None
        self.sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
        self.tok, self.parser_name, self.family = tok, parser_name, family

    def next_turn(self, messages: list[dict], tools: list[dict]) -> dict:
        prompt = self.tok.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True)
        out = self.llm.generate([prompt], self.sp, lora_request=self.lora)
        return parse_assistant(out[0].outputs[0].text, self.tok, self.parser_name, self.family)

    def next_turns_batch(self, prompts: list[tuple[list[dict], list[dict]]]) -> list[dict]:
        texts = [self.tok.apply_chat_template(m, tools=t, tokenize=False, add_generation_prompt=True) for m, t in prompts]
        outs = self.llm.generate(texts, self.sp, lora_request=self.lora)
        return [parse_assistant(o.outputs[0].text, self.tok, self.parser_name, self.family) for o in outs]


class HttpTurnClient:
    """OpenAI-compatible endpoint (vLLM serve, or the agentdistill gateway later)."""

    def __init__(self, base_url: str, model: str, api_key: str = "none", logprobs: bool = False):
        from openai import OpenAI
        self.c = OpenAI(base_url=base_url, api_key=api_key)
        self.model, self.logprobs = model, logprobs

    def next_turn(self, messages: list[dict], tools: list[dict]) -> dict:
        r = self.c.chat.completions.create(model=self.model, messages=messages, tools=tools or None, temperature=0,
                                           logprobs=self.logprobs, top_logprobs=5 if self.logprobs else None)
        m = r.choices[0].message
        return {"role": "assistant", "content": m.content,
                "tool_calls": [{"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}}
                               for c in (m.tool_calls or [])] or None,
                "_raw": r.choices[0].model_dump()}
```

`teacher_forced` with the batch method: add a `teacher_forced_batched(traces, client)` that collects every (prefix, tools) pair first and calls `next_turns_batch` once. On vLLM with prefix caching, 200 turns take under a minute.

### 2.5 Training callback

```python
# agentdistill/train/callbacks.py
from __future__ import annotations

from transformers import TrainerCallback

from agentdistill.eval.clients import HfTurnClient
from agentdistill.eval.teacher_forced import summarize, teacher_forced


class NextActionCallback(TrainerCallback):
    def __init__(self, traces: list[dict], tok, parser_name: str, family: str, n_turns: int = 50):
        self.traces, self.tok, self.parser_name, self.family, self.n_turns = traces, tok, parser_name, family, n_turns

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        model.eval()
        client = HfTurnClient(model, self.tok, self.parser_name, self.family)
        res = teacher_forced(self.traces, client, max_turns_per_trace=2)[: self.n_turns]
        s = summarize(res)
        if state.is_world_process_zero:
            print(f"[next_action] step={state.global_step} full_match={s['full_match'][0]:.3f} "
                  f"name={s['name_match_on_tool_turns']:.3f} args={s['args_match_on_tool_turns']:.3f}")
        # write to the trainer log so tensorboard picks it up
        if kwargs.get("logs") is not None:
            kwargs["logs"]["next_action_full_match"] = s["full_match"][0]
        model.train()
```

Register it in `train_sft` with 25 held-out traces. Generation under 4-bit with gradient checkpointing enabled is slow; 50 turns every 100 steps costs roughly a minute on an L4. If it is worse than that, drop to 25 turns and run the full 200 at the end only.

### 2.6 The real GPU run

Checklist for the rented box (an L4 or A10G, 24 GB):

1. `pip install -e ".[train]"`, then `python -c "from agentdistill.train.compat import sft_config_kwargs"` against a dummy config to catch field mismatches before loading a model.
2. `agentdistill base-check <model>` prints the round trip result. Stop if not `ok`.
3. `agentdistill train sft <dataset> --config project.yaml` on the example dataset. Expect 20 to 40 minutes for two epochs on a few hundred samples.
4. Confirm the registry row has `metrics.throughput_target_tok_per_s`, `metrics.eval_loss`, `metrics.next_action_full_match_final`, `metrics.packing` (true or false), `metrics.attn_implementation`.
5. `agentdistill adapter merge <id>` then reload the merged model and run `teacher_forced` again; the numbers must match the unmerged adapter within noise. This catches a wrong `target_modules` list or a merge into the wrong dtype.

Milestone 2 definition of done is unchanged from v0.1, with one addition: the round trip check is `ok` for the chosen base model and recorded in the registry row.

---

## 3. Milestone 2.5: real traces

### 3.1 Why this is not optional

Templated traces have a generator, and an 8B model will learn the generator. Every number computed on them will look wonderful and mean nothing, and the first person who runs the pipeline on real traffic will get a student that cannot handle a customer who writes in lowercase. You need trajectories produced by a real model reacting to real tool results, with success labels that do not depend on a judge.

### 3.2 The example agent

`examples/support_agent/` contains:

- **`crm.py`**: a fake CRM backed by SQLite, seeded deterministically from a scenario. Six tools with JSON schemas and annotations: `get_customer(email)`, `list_orders(customer_id, status?)`, `get_order(order_id)`, `create_ticket(customer_id, category, summary)`, `issue_refund(order_id, amount, reason)`, `update_address(customer_id, address)`. Each tool is a pure function of the database state; results are deterministic given the same calls.
- **`scenarios.py`**: 40 scenario templates with parameter spaces (names, order states, amounts, wrinkles like "customer gives the wrong order id first" or "two orders, only one eligible for refund"). Each scenario instantiates to a task: a user message, a seeded database, and an **expected end state predicate** (a Python function over the final database plus the final assistant message).
- **`agent.py`**: a plain tool-calling loop over any OpenAI-compatible or Anthropic endpoint via LiteLLM, max 12 turns, system prompt fixed and versioned.
- **`record.py`**: runs a task, records the normalized trace (messages, tools, tool results), runs the predicate, writes `success`, `grader="predicate"`, `teacher_model`, cost, and tokens. Emits JSONL that `agentdistill ingest jsonl` accepts.
- **`graders.py`**: the predicates. Examples: refund scenario succeeds iff exactly one refund row exists for the eligible order with the right amount and the final message mentions the amount; address scenario succeeds iff the address row matches and no ticket was created.

```python
# examples/support_agent/agent.py
from __future__ import annotations

import json
import time

import litellm

SYSTEM = open("examples/support_agent/system_prompt.md").read()


def run_agent(task: dict, crm, tools: list[dict], model: str, max_turns: int = 12) -> dict:
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task["user_message"]}]
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    started = time.time()
    for _ in range(max_turns):
        r = litellm.completion(model=model, messages=messages, tools=tools, temperature=0.2)
        m = r.choices[0].message
        usage["prompt_tokens"] += r.usage.prompt_tokens
        usage["completion_tokens"] += r.usage.completion_tokens
        assistant = {"role": "assistant", "content": m.content}
        if m.tool_calls:
            assistant["tool_calls"] = [{"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}} for c in m.tool_calls]
        messages.append(assistant)
        if not m.tool_calls:
            break
        for c in m.tool_calls:
            try:
                result = crm.call(c.function.name, json.loads(c.function.arguments))
                content = json.dumps(result)
            except Exception as e:  # tool errors are part of the trajectory
                content = json.dumps({"error": str(e)})
            messages.append({"role": "tool", "tool_call_id": c.id, "content": content})
    return {"messages": messages, "tools": tools, "usage": usage, "latency_ms": int((time.time() - started) * 1000)}
```

```python
# examples/support_agent/record.py
from __future__ import annotations

import hashlib
import json
import sys

from examples.support_agent import scenarios
from examples.support_agent.crm import CRM, TOOLS
from examples.support_agent.agent import run_agent


def main(model: str, n_tasks: int, out_path: str, seed: int = 0) -> None:
    tasks = scenarios.sample(n_tasks, seed=seed)
    with open(out_path, "w") as f:
        for task in tasks:
            crm = CRM.from_seed(task["db_seed"])
            run = run_agent(task, crm, TOOLS, model)
            final_text = next((m.get("content") or "" for m in reversed(run["messages"]) if m["role"] == "assistant"), "")
            success = task["predicate"](crm, final_text)
            trace = {
                "id": hashlib.sha256(json.dumps(run["messages"], sort_keys=True).encode()).hexdigest()[:24],
                "source": "jsonl", "task_id": task["task_id"], "task_input": {"text": task["user_message"], "scenario": task["scenario"]},
                "messages": run["messages"], "tools": TOOLS, "teacher_model": model,
                "success": bool(success), "grader": "predicate", "score": 1.0 if success else 0.0,
                "prompt_tokens": run["usage"]["prompt_tokens"], "completion_tokens": run["usage"]["completion_tokens"],
                "metadata": {"scenario": task["scenario"], "db_seed": task["db_seed"], "latency_ms": run["latency_ms"]},
            }
            f.write(json.dumps(trace) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3])
```

### 3.3 Volume, splits, and cost

- 400 tasks from 40 scenarios, 10 instances each, teacher run once per task. At roughly 6 turns and 3k prompt tokens per turn, a frontier-model teacher costs on the order of tens of dollars for the whole set; an open-weight teacher through a hosted provider costs less. Record the exact spend in `metadata`.
- Split by scenario **instance**, not by scenario: 320 train, 80 held out as `eval_sets.support-holdout-v1`. Every scenario appears in both, so the student is tested on new instances of things it has seen. A second eval set, `support-unseen-v1`, holds 4 scenarios entirely out of training for the generalization number. Report both; the second will be worse and that is the honest number.
- Freeze both eval sets now. They do not change until the project has a v1.0 tag.
- Keep the 679 synthetic traces as `tests/fixtures/synthetic/` for unit tests. Delete them from the example dataset.

### 3.4 Teacher terms

The README warning from v0.1 applies. For the example project, default to an open-weight teacher so the shipped example is clean, and document how to point `record.py` at a frontier model for a user's own internal use. Do not ship traces recorded from a provider whose terms you have not read for this purpose.

### 3.5 Definition of done

- `examples/support_agent/record.py` produces 400 real traces with predicate labels; teacher success rate is between 60% and 90% (if it is above 95%, the scenarios are too easy to separate student from teacher; add wrinkles).
- `agentdistill ingest jsonl` accepts them; `curate` keeps at least 250 after filters (report says why the rest dropped).
- Two frozen eval sets exist in the registry.
- Predicates are unit-tested against hand-built final states.

---

## 4. Milestone 3: the harness

### 4.1 Shape

```
eval run <subject> --eval-set support-holdout-v1 --n 5 --policy strict
   subject: adapter id | 'teacher' | 'base' | 'http:<model>@<url>'
   -> for each task in the eval set, N times:
        run_task(trace, client, ReplayToolProvider(trace), policy)  -> outcome, steps, tokens, divergence?
   -> grade with the eval set's grader (predicate for the example project)
   -> eval_runs row with metrics, per_cluster, and raw per-repeat results in eval_results
eval compare <run_a> <run_b>  -> paired report
```

The student never touches the fake CRM during eval. Tool results come from the recorded teacher trajectory by canonical argument hash. This is the same design as agentreplay's replay engine; when agentreplay exists, `agentreplay_bridge.py` swaps in. Until then, this is the harness.

### 4.2 Canonical hashing

Port `canonical.py` from the agentreplay plan unchanged into `agentdistill/canonical.py` and write `docs/canonical-json.md` (section 8). The same rules must produce the same hash in mcpgate's audit `args_hash` later, or the agentreplay export will never line up.

### 4.3 Replay tool provider

```python
# agentdistill/eval/replay.py
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from agentdistill.canonical import args_hash, canonical_json, normalize, rules_for


class Divergence(Exception):
    def __init__(self, tool: str, args: dict, h: str, nearest: "Recorded | None", score: float):
        super().__init__(f"divergence on {tool} (hash {h[:12]})")
        self.tool, self.args, self.hash, self.nearest, self.score = tool, args, h, nearest, score


@dataclass
class Recorded:
    tool: str
    args: dict
    args_hash: str
    result: str          # the tool message content as recorded (string)


class ReplayToolProvider:
    def __init__(self, trace: dict, policy: str = "strict", fuzzy_threshold: float = 0.92, embedder: Any = None):
        self.policy, self.threshold, self.embedder = policy, fuzzy_threshold, embedder
        self.index: dict[tuple[str, str], Recorded] = {}
        self.by_tool: dict[str, list[Recorded]] = {}
        calls_by_id: dict[str, tuple[str, dict]] = {}
        for m in trace["messages"]:
            for c in m.get("tool_calls") or []:
                a = c["function"]["arguments"]
                calls_by_id[c["id"]] = (c["function"]["name"], json.loads(a) if isinstance(a, str) else a)
        for m in trace["messages"]:
            if m["role"] != "tool":
                continue
            name, args = calls_by_id[m["tool_call_id"]]
            rec = Recorded(name, args, args_hash(name, args), m["content"])
            self.index.setdefault((name, rec.args_hash), rec)     # first occurrence wins
            self.by_tool.setdefault(name, []).append(rec)
        self.stats = {"replayed": 0, "fuzzy": 0}

    def lookup(self, tool: str, args: dict) -> str:
        h = args_hash(tool, args)
        hit = self.index.get((tool, h))
        if hit is not None:
            self.stats["replayed"] += 1
            return hit.result
        nearest, score = self._nearest(tool, args)
        if self.policy == "fuzzy" and nearest is not None and score >= self.threshold:
            self.stats["fuzzy"] += 1
            return nearest.result
        raise Divergence(tool, args, h, nearest, score)

    def _nearest(self, tool: str, args: dict) -> tuple[Recorded | None, float]:
        recs = self.by_tool.get(tool, [])
        if not recs:
            return None, 0.0
        if self.embedder is None:
            # cheap structural similarity: shared canonical key/value pairs
            q = set(canonical_json(normalize(args, rules_for(tool))).split(","))
            best, best_s = None, 0.0
            for r in recs:
                d = set(canonical_json(normalize(r.args, rules_for(tool))).split(","))
                s = len(q & d) / max(len(q | d), 1)
                if s > best_s:
                    best, best_s = r, s
            return best, best_s
        import numpy as np
        vecs = self.embedder.embed([canonical_json(normalize(r.args, rules_for(tool))) for r in recs])
        q = self.embedder.embed([canonical_json(normalize(args, rules_for(tool)))])[0]
        sims = vecs @ q / (np.linalg.norm(vecs, axis=1) * np.linalg.norm(q) + 1e-9)
        i = int(sims.argmax())
        return recs[i], float(sims[i])
```

### 4.4 Task runner

```python
# agentdistill/eval/harness.py
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from agentdistill.curate.schema import tool_calls_valid
from agentdistill.eval.replay import Divergence, ReplayToolProvider
from agentdistill.eval.teacher_forced import TurnClient


@dataclass
class TaskOutcome:
    task_id: str
    repeat_idx: int
    messages: list[dict]
    final_text: str
    n_turns: int
    n_tool_calls: int
    schema_valid: bool
    diverged: bool
    divergence: dict | None
    replay_stats: dict
    latency_ms: int
    completion_tokens_est: int
    success: bool | None = None
    grader_out: dict = field(default_factory=dict)


def run_task(trace: dict, client: TurnClient, provider: ReplayToolProvider, repeat_idx: int = 0, max_turns: int = 12,
             token_estimate=lambda s: len(s) // 4) -> TaskOutcome:
    tools = trace["tools"]
    # prefix = system + first user message from the recorded trace
    msgs = [m for m in trace["messages"][:2] if m["role"] in ("system", "user")]
    started = time.time()
    n_calls, diverged, div = 0, False, None
    est = 0
    for _ in range(max_turns):
        a = client.next_turn(msgs, tools)
        a = {k: v for k, v in a.items() if not k.startswith("_")}
        msgs.append(a)
        est += token_estimate((a.get("content") or "") + json.dumps(a.get("tool_calls") or []))
        if not a.get("tool_calls"):
            break
        for c in a["tool_calls"]:
            n_calls += 1
            args_raw = c["function"]["arguments"]
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            except json.JSONDecodeError:
                msgs.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps({"error": "arguments were not valid JSON"})})
                continue
            try:
                content = provider.lookup(c["function"]["name"], args)
            except Divergence as d:
                diverged, div = True, {"tool": d.tool, "args": d.args, "nearest_score": d.score}
                msgs.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps({"error": "replay divergence"})})
                break
            msgs.append({"role": "tool", "tool_call_id": c["id"], "content": content})
        if diverged:
            break
    final = next((m.get("content") or "" for m in reversed(msgs) if m["role"] == "assistant"), "")
    valid, _ = tool_calls_valid({"messages": msgs, "tools": tools})
    return TaskOutcome(
        task_id=trace["task_id"], repeat_idx=repeat_idx, messages=msgs, final_text=final,
        n_turns=sum(1 for m in msgs if m["role"] == "assistant"), n_tool_calls=n_calls, schema_valid=valid,
        diverged=diverged, divergence=div, replay_stats=dict(provider.stats),
        latency_ms=int((time.time() - started) * 1000), completion_tokens_est=est,
    )


class RecordedTurnClient:
    """Emits the recorded assistant turns in order. Used to prove the harness reproduces a trace exactly."""

    def __init__(self, trace: dict):
        self.turns = [m for m in trace["messages"] if m["role"] == "assistant"]
        self.i = 0

    def next_turn(self, messages, tools):
        m = self.turns[self.i]
        self.i += 1
        return m
```

Strict-mode divergence is recorded and the task is graded as-is (the predicate will usually fail, since the trajectory stopped). The report shows divergence rate separately from success so the two failure modes are not confused.

### 4.5 Grading in the replay setting

The example project's predicates inspect a CRM database, but during replay there is no database. Two graders cover this:

- **`replay_predicate`**: reconstruct the final state by applying the student's *replayed* tool calls to a fresh seeded CRM in order. Because results were served from the recording, a student that made the same calls reaches the same state. A student that made different calls diverged, and the predicate sees whatever state its calls produced. This is exact for the example project and it is what `graders.py` should implement as `predicate_from_calls(db_seed, tool_calls, final_text)`.
- **`llm_judge`**: rubric over final text plus trajectory summary, for users without state predicates. Calibration rules from v0.1 section 7 apply; vendor `calibration.py` and the reporting rule (never a bare judge number).

### 4.6 Runner, metrics, statistics

```python
# agentdistill/eval/runner.py
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from agentdistill.eval.harness import run_task
from agentdistill.eval.replay import ReplayToolProvider
from agentdistill.eval.stats import cluster_bootstrap_diff, mcnemar_paired, minimum_n_guard, wilcoxon_metric, holm


def run_eval(registry, eval_set: dict, traces_by_task: dict[str, dict], client, grader, subject: str,
             n_per_task: int, policy: str) -> str:
    run_id = uuid.uuid4().hex
    registry.start_eval_run(run_id, eval_set["id"], subject, n_per_task)
    for task_id in eval_set["task_ids"]:
        trace = traces_by_task[task_id]
        for k in range(n_per_task):
            provider = ReplayToolProvider(trace, policy=policy)
            out = run_task(trace, client, provider, repeat_idx=k)
            out.success, out.grader_out = grader(trace, out)
            registry.write_eval_result(run_id, out)
    registry.finish_eval_run(run_id, metrics=aggregate(registry, run_id, traces_by_task))
    return run_id


def aggregate(registry, run_id: str, traces_by_task: dict[str, dict]) -> dict:
    rows = registry.eval_results(run_id)
    by_task: dict[str, list] = {}
    for r in rows:
        by_task.setdefault(r.task_id, []).append(r)
    succ = {t: [bool(r.success) for r in rs] for t, rs in by_task.items()}
    per_cluster: dict[str, dict] = {}
    for t, rs in by_task.items():
        c = str(traces_by_task[t].get("cluster", "none"))
        d = per_cluster.setdefault(c, {"n_tasks": 0, "success": 0.0})
        d["n_tasks"] += 1
        d["success"] += sum(bool(r.success) for r in rs) / len(rs)
    for d in per_cluster.values():
        d["success"] = d["success"] / d["n_tasks"]
    n = len(rows)
    return {
        "success": sum(bool(r.success) for r in rows) / n,
        "schema_valid": sum(bool(r.schema_valid) for r in rows) / n,
        "divergence_rate": sum(bool(r.diverged) for r in rows) / n,
        "turns_median": sorted(r.n_turns for r in rows)[n // 2],
        "tokens_est_median": sorted(r.completion_tokens_est for r in rows)[n // 2],
        "latency_ms_median": sorted(r.latency_ms for r in rows)[n // 2],
        "per_cluster": per_cluster,
        "n_tasks": len(by_task), "n_per_task": max(len(v) for v in by_task.values()),
    }


def compare(registry, run_a: str, run_b: str, alpha: float = 0.05) -> dict:
    a_rows, b_rows = registry.eval_results(run_a), registry.eval_results(run_b)

    def outcomes(rows):
        d: dict[str, list[bool]] = {}
        for r in rows:
            d.setdefault(r.task_id, []).append(bool(r.success))
        return d

    def metric(rows, attr):
        d: dict[str, list[float]] = {}
        for r in rows:
            d.setdefault(r.task_id, []).append(float(getattr(r, attr)))
        return {t: sum(v) / len(v) for t, v in d.items()}

    oa, ob = outcomes(a_rows), outcomes(b_rows)
    minimum_n_guard(len(set(oa) & set(ob)), min(len(v) for v in list(oa.values()) + list(ob.values())))
    success = cluster_bootstrap_diff(oa, ob)
    mc = mcnemar_paired(oa, ob)
    tokens = wilcoxon_metric(metric(a_rows, "completion_tokens_est"), metric(b_rows, "completion_tokens_est"))
    turns = wilcoxon_metric(metric(a_rows, "n_turns"), metric(b_rows, "n_turns"))
    sig = holm({"success": mc["p"], "tokens": tokens["p"], "turns": turns["p"]}, alpha)
    return {"success": success, "mcnemar": mc, "tokens": tokens, "turns": turns, "holm": sig,
            "generated_at": datetime.now(timezone.utc).isoformat()}
```

`stats.py` is the module from the agentreplay plan, vendored verbatim, with its simulation tests.

### 4.7 CLI

- `eval run <subject> --eval-set <id> --n 5 --policy strict --backend vllm|hf|http` replaces the exit-2 stub. `subject` resolves to a client: an adapter id loads the base plus LoRA in vLLM offline; `teacher` uses `HttpTurnClient` against the teacher endpoint; `base` is the base model with no adapter (the zero-shot baseline every report must include).
- `eval compare <a> <b>` prints the v0.1 section 7.3 report plus `divergence_rate` and `schema_valid` for each side and the weakest clusters.
- `eval show <run>` dumps per-task outcomes for hand review.

### 4.8 Tests

- `test_harness_reproduces_trace`: `RecordedTurnClient` on every fixture trace reproduces the recorded messages exactly, zero divergences, predicate outcome equals the recorded `success`.
- `test_harness_divergence`: a client that calls a tool with unrecorded args triggers `Divergence` in strict mode and a fuzzy hit above threshold in fuzzy mode.
- `test_replay_predicate_matches_live`: for 20 recorded traces, `predicate_from_calls` on the recorded calls equals the live predicate outcome.
- `test_stats_simulation`: from the agentreplay plan (CI coverage, McNemar power, Holm false positives).
- `test_compare_guard`: `compare` refuses N=1.
- `test_teacher_forced_stub`: a stub client that echoes the teacher turn scores 1.0; one that swaps a tool name scores 0 on `name_match`.

### 4.9 Definition of done

- `eval run base`, `eval run <sft-adapter>`, and `eval run teacher` complete on `support-holdout-v1` with N=5 on one GPU in under an hour.
- `eval compare` prints success delta with CI, McNemar p, tokens and turns with Wilcoxon, Holm flags, divergence rate, schema validity, weakest clusters.
- `base` is worse than the SFT adapter with a CI that excludes zero. If it is not, stop and look at the data before building anything else.
- The report is committed under `examples/support_agent/reports/` with the exact commands that produced it.

---

## 5. Bridge to Milestone 4

With the harness in place, on-policy training is three functions:

- `collect_rollouts(adapter, train_task_ids, k=8)`: `run_task` k times per training task with `VllmOfflineTurnClient(temperature=0.8)` (add a `SamplingParams` override to the client), grade with `replay_predicate`, write traces with `source="rollout"` and `parent_adapter_id`.
- `build_rft(rollouts, cap_per_task=2)`: successful rollouts to SFT samples through the existing dataset builder.
- `build_pairs(rollouts, teacher_traces)`: `first_divergent_pair` from v0.1 section 6.1 over success-versus-failure rollouts of the same task, and student-failure-versus-teacher-success.

Rollout sampling must use the **fuzzy** replay policy; on-policy trajectories drift from the teacher's argument phrasing and strict mode would stop most of them. Report the fuzzy-hit share so nobody mistakes a fuzzily replayed success for a real one; when in doubt, lower the threshold and re-check with the predicate.

---

## 6. Day-by-day

**Day 1**: plan patches; `canonical.py` port with tests; `docs/canonical-json.md`; `template_check.roundtrip_tool_call` with the fallback parser; `test_template_roundtrip.py`.
**Day 2**: `test_mask_invariants.py` on three templates; `train/compat.py` with a test that fakes an `SFTConfig` missing a field; `eval/teacher_forced.py` and `eval/clients.py` (`HfTurnClient` only) with stub tests.
**Day 3**: `train/callbacks.py`; wire into `train_sft`; rent the GPU; run the checklist in 2.6; record numbers; merge and re-check. Milestone 2 review.
**Day 4**: `examples/support_agent/crm.py`, `scenarios.py` (first 15 scenarios), `graders.py` with predicate tests.
**Day 5**: `agent.py`, `record.py`, remaining scenarios; record 40 tasks against the chosen teacher and read every one of them by hand. Fix scenarios that are trivial or ambiguous. Record the full 400. Ingest, curate, freeze the two eval sets. Milestone 2.5 review.
**Day 6**: `eval/replay.py`, `eval/harness.py`, `RecordedTurnClient`, `test_harness_reproduces_trace`, `test_harness_divergence`.
**Day 7**: `replay_predicate`, `test_replay_predicate_matches_live`; vendor `stats.py` and its simulation tests; `eval/runner.py`.
**Day 8**: `VllmOfflineTurnClient` batched path, `HttpTurnClient`; `eval run` and `eval compare` CLI; `eval show`.
**Day 9**: retrain SFT on the real dataset; `eval run` for base, adapter, teacher; `eval compare`; per-cluster review; commit the report.
**Day 10**: Milestone 3 review against 4.9; write `docs/evaluation.md` (the first doc for a built feature since quickstart); implement `collect_rollouts` and run one round of 8 rollouts on 20 tasks to confirm the fuzzy replay share is sane. Decide whether Milestone 4 starts or the data needs another pass.

---

## 7. Things that will go wrong

| Symptom | Likely cause | Fix |
|---|---|---|
| Round trip fails on the vLLM path but passes the fallback | Parser name does not match the template family, or vLLM version changed the parser API | Check `vllm serve --help` for parser names; pin vLLM; adjust `parse_with_vllm` signature |
| Next-action accuracy is high but end-to-end success is low | Exposure bias; the student follows the teacher's prefix but not its own | Expected. That gap is what Milestone 4 closes; report both numbers |
| Divergence rate above 30% in strict mode for the SFT student | Argument phrasing drift (whitespace, casing, key order not covered by canonicalization) | Inspect `Divergence.nearest` and `score`; add normalization rules per tool; re-run |
| Teacher success above 95% on the example | Scenarios too easy | Add wrinkles: wrong ids, ineligible refunds, two customers with the same name |
| Merged model scores differ from the adapter | Wrong `target_modules` or merge dtype | Compare state-dict deltas; merge in bf16, not 4-bit |
| Eval takes hours | Not using the batched vLLM path, or no prefix caching | `next_turns_batch` per turn index across tasks; `enable_prefix_caching=True` |
| Predicate passes but the final message is wrong | Predicate checks state only | Predicates check state **and** the final message where the task demands a statement to the customer |

---

## 8. docs/canonical-json.md (shared across agentdistill, mcpgate, agentreplay)

```
Canonical JSON for argument hashing, v1

1. Apply per-tool custom normalizers first, if registered.
2. Objects: sort keys byte-wise; drop keys in the drop list
   (default: request_id, trace_id, timestamp, ts, cursor, page_token, nonce).
3. Arrays: preserve order; normalize elements.
4. Strings: trim; ISO-8601 timestamps -> "<ts>"; RFC 4122 UUIDs -> "<uuid>".
5. Floats: round to 6 decimals. Booleans and integers unchanged. null unchanged.
6. Serialize with sorted keys, separators "," and ":", UTF-8, no ASCII escaping.
7. args_hash = sha256(canonical({"tool": <name>, "args": <normalized args>})) as lowercase hex.

Implementations: agentdistill/canonical.py (Python), mcpgate packages/gateway/src/audit/canonical.ts (TypeScript).
A shared test vector file lives at schemas/canonical-vectors.json; both implementations must pass it.
```

Write the vector file with ten cases (nested objects, unicode, floats, timestamps in three formats, a UUID inside an array) and make both repos run it.
