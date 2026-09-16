"""Chat-template compatibility checks.

Do not hand-write a tool format; that is how you get a student that emits a syntax the serving stack cannot parse.
Use the tokenizer's own `apply_chat_template(messages, tools=...)`, and refuse to build a dataset if the template
cannot support it. Supporting a template without tool calling is a v2 feature, not a silent fallback.

Checks, in the order they run:

1. **has_template** — the tokenizer defines `chat_template`.
2. **accepts_tools** — `apply_chat_template(..., tools=[...])` runs, and the rendered text actually mentions the
   tool. A template that silently ignores `tools` is worse than one that errors: the student would be trained to
   call tools it was never shown.
3. **renders_tool_calls** — an assistant turn carrying `tool_calls` renders the name and the argument values.
4. **eos_defined** — the end-of-turn token exists, so generation can stop.
5. **prefix_stable** — rendering the first `i` messages is a string prefix of rendering all of them, at every
   assistant boundary. Offsets-based loss masking depends on this; a template that re-writes earlier turns when a
   later one arrives (some add a trailing summary, or move the tool block) cannot be masked this way.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

SAMPLE_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_orders",
            "description": "Look up orders for a customer.",
            "parameters": {
                "type": "object",
                "properties": {"customer_id": {"type": "string"}, "limit": {"type": "integer"}},
                "required": ["customer_id"],
            },
        },
    }
]

SAMPLE_MESSAGES: list[dict] = [
    {"role": "system", "content": "You are a support agent."},
    {"role": "user", "content": "Where is the order for customer c_9?"},
    {
        "role": "assistant",
        "content": "I will look that up.",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "search_orders", "arguments": '{"customer_id": "c_9", "limit": 5}'},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call_1", "content": '[{"order_id": "o_1", "status": "shipped"}]'},
    {"role": "assistant", "content": "Order o_1 shipped yesterday."},
]


class TemplateError(Exception):
    """Raised when a template cannot be used. The message names the template and the failing check."""


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class TemplateReport:
    model: str
    checks: list[CheckResult] = field(default_factory=list)
    is_fast: bool = True

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)

    @property
    def failures(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.ok]

    def raise_if_failed(self) -> None:
        if self.ok:
            return
        lines = [f"chat template for {self.model!r} cannot be used to build a dataset:"]
        lines += [f"  - {c.name}: {c.detail}" for c in self.failures]
        lines.append(
            "  Pick a base model whose template renders tools, or wait for v2 template shims. "
            "`agentdistill base-check <model>` prints this report for any candidate."
        )
        raise TemplateError("\n".join(lines))

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "ok": self.ok,
            "is_fast": self.is_fast,
            "checks": [{"name": c.name, "ok": c.ok, "detail": c.detail} for c in self.checks],
        }


def render(tok: Any, messages: list[dict], tools: list[dict] | None, add_generation_prompt: bool = False) -> str:
    """One place that calls `apply_chat_template`, so every stage renders identically."""
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    if tools:
        kwargs["tools"] = tools
    return tok.apply_chat_template(messages, **kwargs)


def check_prefix_stability(tok: Any, messages: list[dict], tools: list[dict] | None) -> tuple[bool, str]:
    """At every assistant turn, the render of the preceding messages (with a generation prompt) must be a prefix
    of the full render."""
    full = render(tok, messages, tools, add_generation_prompt=False)
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        prefix = render(tok, messages[:i], tools, add_generation_prompt=True)
        if not full.startswith(prefix):
            return False, (
                f"render of messages[:{i}] + generation prompt is not a prefix of the full render; "
                f"the template rewrites earlier turns, so offsets-based loss masking would mask the wrong tokens"
            )
    return True, ""


def check_template(tok: Any, model: str = "<tokenizer>") -> TemplateReport:
    report = TemplateReport(model=model, is_fast=bool(getattr(tok, "is_fast", False)))

    if not getattr(tok, "chat_template", None):
        report.checks.append(CheckResult("has_template", False, "tokenizer defines no chat_template"))
        return report
    report.checks.append(CheckResult("has_template", True))

    if not report.is_fast:
        report.checks.append(
            CheckResult(
                "fast_tokenizer",
                False,
                "offsets-based loss masking needs a fast tokenizer (`return_offsets_mapping`); this is a slow one",
            )
        )
    else:
        report.checks.append(CheckResult("fast_tokenizer", True))

    try:
        with_tools = render(tok, SAMPLE_MESSAGES[:2], SAMPLE_TOOLS)
    except Exception as e:  # templates raise anything from TemplateError to KeyError
        report.checks.append(CheckResult("accepts_tools", False, f"{type(e).__name__}: {e}"))
        return report
    if "search_orders" not in with_tools:
        report.checks.append(
            CheckResult(
                "accepts_tools",
                False,
                "the template accepted `tools` but did not render the tool name; it is ignoring the argument",
            )
        )
        return report
    report.checks.append(CheckResult("accepts_tools", True))

    try:
        full = render(tok, SAMPLE_MESSAGES, SAMPLE_TOOLS)
    except Exception as e:
        report.checks.append(CheckResult("renders_tool_calls", False, f"{type(e).__name__}: {e}"))
        return report
    missing = [needle for needle in ("search_orders", "c_9") if needle not in full]
    if missing:
        report.checks.append(
            CheckResult("renders_tool_calls", False, f"rendered conversation is missing {missing} from the tool call")
        )
    else:
        report.checks.append(CheckResult("renders_tool_calls", True))

    eos = getattr(tok, "eos_token", None)
    if not eos:
        report.checks.append(CheckResult("eos_defined", False, "tokenizer has no eos_token, so generation cannot stop"))
    else:
        report.checks.append(CheckResult("eos_defined", True, f"eos_token={eos!r}"))

    ok, detail = check_prefix_stability(tok, SAMPLE_MESSAGES, SAMPLE_TOOLS)
    report.checks.append(CheckResult("prefix_stable", ok, detail))

    return report


def roundtrip_tool_call(tok: Any, parser: Any = None) -> tuple[bool, str]:
    """Render a tool call and parse it back.

    With no parser supplied this is a containment check: the name and every argument value survive rendering.
    Pass a callable `parser(text) -> [{"name":..., "arguments": {...}}]` (for instance a vLLM tool parser) to make
    it a true round trip; the serving stack is the only authority on whether its parser can read the template.
    """
    text = render(tok, SAMPLE_MESSAGES[:3], SAMPLE_TOOLS)
    expected_args = json.loads(SAMPLE_MESSAGES[2]["tool_calls"][0]["function"]["arguments"])
    if parser is None:
        missing = [str(v) for v in expected_args.values() if str(v) not in text]
        if "search_orders" not in text or missing:
            return False, f"rendered tool call is missing name or values {missing}"
        return True, "containment check only; supply a parser for a true round trip"
    try:
        parsed = parser(text)
    except Exception as e:
        return False, f"parser raised {type(e).__name__}: {e}"
    if not parsed:
        return False, "parser found no tool call in the rendered text"
    got = parsed[0]
    if got.get("name") != "search_orders":
        return False, f"parser recovered tool name {got.get('name')!r}, expected 'search_orders'"
    if got.get("arguments") != expected_args:
        return False, f"parser recovered arguments {got.get('arguments')!r}, expected {expected_args!r}"
    return True, "round trip through the parser recovered name and arguments"
