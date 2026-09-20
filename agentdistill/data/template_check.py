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
import re
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


def object_arguments(messages: list[dict]) -> list[dict]:
    """Tool-call arguments as objects, which is the shape a chat template expects.

    Traces store arguments the way the OpenAI wire format does, as a JSON *string*. Chat templates serialize what
    they are given (`{{ tool_call.arguments | tojson }}`), so handing them the string emits a quoted, escaped
    string -- `"arguments": "{\"a\": 1}"` -- which the serving stack's parser then recovers as a string rather
    than a call's arguments. Qwen2.5's template failed `base-check`'s round trip for exactly this reason, and a
    student trained on that text would emit tool calls the parser drops.

    Copies only the messages it changes, so callers keep their own objects.
    """
    out = []
    for m in messages:
        calls = m.get("tool_calls")
        if not calls:
            out.append(m)
            continue
        fixed = []
        for c in calls:
            fn = (c or {}).get("function") or {}
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, TypeError):
                    # Unparseable arguments stay as they are: a template that renders them is how a malformed
                    # call stays visible instead of being silently repaired here.
                    fixed.append(c)
                    continue
                fixed.append({**c, "function": {**fn, "arguments": args}})
            else:
                fixed.append(c)
        out.append({**m, "tool_calls": fixed})
    return out


def render(tok: Any, messages: list[dict], tools: list[dict] | None, add_generation_prompt: bool = False) -> str:
    """One place that calls `apply_chat_template`, so every stage renders identically."""
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    if tools:
        kwargs["tools"] = tools
    return tok.apply_chat_template(object_arguments(messages), **kwargs)


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


# --------------------------------------------------------------------------------------------------------------
# Tool-call round trip
# --------------------------------------------------------------------------------------------------------------
#
# The student will be served by vLLM, which recovers tool calls from generated *text* with a template-specific
# parser. If the training data renders tool calls in a shape that parser cannot read, the student is useless no
# matter how good its loss is -- and you find out after the GPU bill, not before. This check runs before any
# training and costs nothing.

FALLBACK_PARSERS: dict[str, re.Pattern[str]] = {
    # hermes / qwen style: <tool_call>{"name": ..., "arguments": {...}}</tool_call>
    "hermes": re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S),
    # llama 3.x json style: {"name": ..., "parameters": {...}}
    "llama3_json": re.compile(r'(\{\s*"name"\s*:\s*".+?"\s*,\s*"parameters"\s*:\s*\{.*\}\s*\})', re.S),
    # the repo's fixture templates: <|call|>name{json}<|/call|>
    "agentdistill_fixture": re.compile(r"<\|call\|>([A-Za-z0-9_]+)(\{.*?\})<\|/call\|>", re.S),
}


def example_from_schema(schema: dict[str, Any]) -> Any:
    """A minimal instance satisfying a JSON schema, used as the payload for the round trip.

    Only required properties are filled: the point is to exercise the template and the parser, not to produce a
    realistic call.
    """
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


def parse_with_vllm(model_output: str, tok: Any, parser_name: str | None) -> list[tuple[str, dict]] | None:
    """Parse with vLLM's own tool parser. Returns None when vLLM is unavailable, so callers fall back.

    None means "could not check", which is different from [] meaning "no tool call found". Conflating them would
    turn a missing dependency into a passing test.
    """
    if not parser_name:
        return None
    try:
        from vllm.entrypoints.openai.tool_parsers import ToolParserManager
    except Exception:
        return None
    try:
        parser = ToolParserManager.get_tool_parser(parser_name)(tok)
        info = parser.extract_tool_calls(model_output, request=None)
    except Exception as e:  # pragma: no cover - depends on the installed vLLM
        raise TemplateError(
            f"vLLM tool parser {parser_name!r} failed on rendered output: {type(e).__name__}: {e}. "
            f"Check the parser name against `vllm serve --help` for this template family."
        ) from e
    if not getattr(info, "tools_called", False):
        return []
    return [(c.function.name, json.loads(c.function.arguments)) for c in info.tool_calls]


def parse_fallback(model_output: str, family: str) -> list[tuple[str, dict]]:
    """Regex parser standing in for vLLM when it is not installed (it is Linux/CUDA only).

    This is a check that the rendered shape is *recoverable*, not a claim that vLLM will recover it. The
    definitive check is the vLLM path; `base-check` reports which one ran.
    """
    pattern = FALLBACK_PARSERS.get(family)
    if pattern is None:
        raise TemplateError(
            f"unknown tool-call family {family!r}; known families are {sorted(FALLBACK_PARSERS)}. "
            f"Set train.tool_parser.family in project.yaml."
        )
    out: list[tuple[str, dict]] = []
    for m in pattern.finditer(model_output):
        if family == "agentdistill_fixture":
            out.append((m.group(1), json.loads(m.group(2))))
            continue
        obj = json.loads(m.group(1))
        args = obj.get("arguments", obj.get("parameters", {}))
        out.append((obj["name"], args))
    return out


def detect_family(tok: Any, tools: list[dict] | None = None) -> str | None:
    """Guess the tool-call family from what the template actually renders.

    A guess is offered so `base-check` is useful on a new model without configuration, but it is reported as a
    guess: `train.tool_parser` is what feeds the serving command, and it should be set explicitly.
    """
    tools = tools or SAMPLE_TOOLS
    try:
        rendered = render(tok, SAMPLE_MESSAGES[:3], tools)
    except Exception:
        return None
    for family, pattern in FALLBACK_PARSERS.items():
        if pattern.search(rendered):
            return family
    return None


def roundtrip_tool_call(
    tok: Any,
    tools: list[dict] | None = None,
    parser_name: str | None = None,
    family: str | None = None,
) -> dict:
    """Render one assistant tool call, strip the prompt header, parse the remainder back, and compare.

    Returns a dict rather than raising: `base-check` prints the mismatch so a user can see what the template
    produced versus what the parser recovered.
    """
    tools = tools or SAMPLE_TOOLS
    fn = tools[0]["function"]
    args = example_from_schema(fn.get("parameters", {"type": "object"}))
    msgs = [
        {"role": "system", "content": "You are a test."},
        {"role": "user", "content": "Do the thing."},
    ]
    assistant = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": "call_0", "type": "function", "function": {"name": fn["name"], "arguments": json.dumps(args)}}
        ],
    }
    full = render(tok, [*msgs, assistant], tools, add_generation_prompt=False)
    header = render(tok, msgs, tools, add_generation_prompt=True)
    if not full.startswith(header):
        return {
            "ok": False,
            "parser": "none",
            "detail": "template is not prefix-stable; the model's own output cannot be isolated from the prompt",
            "expected": [(fn["name"], args)],
            "parsed": None,
            "model_output": "",
        }

    model_output = full[len(header) :]
    resolved_family = family or detect_family(tok, tools)

    parsed = parse_with_vllm(model_output, tok, parser_name)
    source = "vllm"
    if parsed is None:
        if resolved_family is None:
            return {
                "ok": False,
                "parser": "none",
                "detail": (
                    "vLLM is not installed and the rendered tool call matches no known family, so nothing could "
                    "verify it. Set train.tool_parser.family, or install vLLM to check against the real parser."
                ),
                "expected": [(fn["name"], args)],
                "parsed": None,
                "model_output": model_output,
            }
        parsed = parse_fallback(model_output, resolved_family)
        source = f"fallback:{resolved_family}"

    expected = [(fn["name"], args)]
    ok = parsed == expected
    detail = "" if ok else "the parser did not recover the rendered call; the serving stack would drop it"
    if ok and source.startswith("fallback"):
        detail = "verified with the fallback regex; install vLLM to check against the parser that will serve it"
    return {
        "ok": ok,
        "parser": source,
        "detail": detail,
        "expected": expected,
        "parsed": parsed,
        "model_output": model_output,
    }
