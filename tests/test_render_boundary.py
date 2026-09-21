"""Every chat-template render, on real traces, through the pinned tokenizer.

Traces store tool-call arguments the way the OpenAI wire format does, as a JSON *string*. Real chat templates
serialize what they are handed (`{{ tool_call.arguments | tojson }}`), so a render that is given the string emits

    "arguments": "{\\"customer_id\\": \\"c_9\\"}"

and the serving stack's parser recovers a string where a call's arguments should be. A student trained on that
text emits tool calls vLLM drops, silently, and the only symptom is an agent that never seems to do anything.

`data/template_check.render` is the one place `apply_chat_template` is called, and it converts arguments to
objects first. This file is the check that *every* consumer goes through it. It does not read the source for
`apply_chat_template`; it renders three real traces down each path that exists and looks at the text that comes
out, because a path that renders correctly is the property, not a path that imports the right function.

Two properties per path:

- nothing rendered contains `"arguments": "{`, in any spelling;
- every rendered assistant turn parses back, through the configured hermes parser, to the tool name and
  arguments it was rendered from.

Against the *pinned* tokenizer, not a fixture: the fixture templates hid this defect once already by
interpolating arguments raw, which only works when the value is already a serialized string. Skipped when the
pinned tokenizer is not in the local HF cache, so the suite stays offline by default. To run it:

    HF_HUB_OFFLINE=0 python -m pytest tests/test_render_boundary.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from agentdistill.cascade.arg_mask import arg_char_spans, arg_token_mask, mask_coverage
from agentdistill.config import ProjectConfig
from agentdistill.data.build import assistant_spans
from agentdistill.data.template_check import parse_fallback, parse_with_vllm, render
from agentdistill.eval.clients import HfTurnClient, VllmOfflineTurnClient
from agentdistill.eval.teacher_forced import teacher_forced, teacher_forced_batched
from agentdistill.train.dpo_data import render_pair

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "examples" / "support_agent" / "project.yaml"
TRACES = ROOT / "examples" / "support_agent" / "traces-train.jsonl"

#: The defect, in both spellings a serializer might produce.
STRINGIFIED = ('"arguments": "{', '"arguments":"{')

N_TRACES = 3


# --------------------------------------------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def project():
    return ProjectConfig.load(str(CONFIG))


@pytest.fixture(scope="module")
def pinned_tokenizer(project):
    """The configured base model's tokenizer at its pinned revision, from the local cache; skip rather than
    download. Same contract as `tests/test_real_template.py`."""
    model = project.train.base_model
    if Path(model).exists():
        pytest.skip(f"train.base_model is a local path ({model}); this test is for a Hub model")
    try:
        from transformers import AutoTokenizer

        kwargs = {"revision": project.train.base_model_revision} if project.train.base_model_revision else {}
        return AutoTokenizer.from_pretrained(model, local_files_only=True, **kwargs)
    except Exception as e:  # any failure here means "not available offline"
        pytest.skip(f"{model} is not in the local HF cache: {type(e).__name__}")


@pytest.fixture(scope="module")
def traces():
    """Three recorded traces that actually call tools.

    Recorded, not synthesized: the point of this file is that the shapes the corpus really contains survive the
    render, and a hand-written trace would only prove that the shape someone thought of survives.
    """
    if not TRACES.exists():
        pytest.skip(f"{TRACES} is not present; run examples/support_agent/record.py to rebuild the corpus")
    out: list[dict] = []
    with TRACES.open() as f:
        for line in f:
            trace = json.loads(line)
            if any(m.get("tool_calls") for m in trace["messages"]):
                out.append(trace)
            if len(out) == N_TRACES:
                break
    assert len(out) == N_TRACES, f"only {len(out)} traces with tool calls in {TRACES}"
    return out


# --------------------------------------------------------------------------------------------------------------
# what a path produces
# --------------------------------------------------------------------------------------------------------------


@dataclass
class Rendered:
    """Everything one render path produced for one trace.

    `texts` is every string the path handed onward -- prompts included, since a prompt carries the whole history
    and a stringified argument anywhere in it is a stringified argument the student reads. `turns` is the subset
    that is exactly one assistant turn, which is the only thing a tool parser can be pointed at.
    """

    texts: list[str] = field(default_factory=list)
    turns: list[tuple[str, list[tuple[str, dict]]]] = field(default_factory=list)


def expected_calls(message: dict) -> list[tuple[str, dict]]:
    """(name, arguments) for a recorded assistant turn, arguments parsed out of the wire-format string."""
    return [
        (c["function"]["name"], json.loads(c["function"]["arguments"]))
        for c in message.get("tool_calls") or []
    ]


def assistant_indices(messages: list[dict]) -> list[int]:
    return [i for i, m in enumerate(messages) if m["role"] == "assistant"]


class HfProbe(HfTurnClient):
    """`HfTurnClient`'s real render call site, without a model on the machine.

    `__init__` is replaced rather than mocked: the only thing `_render` touches is `self.tok`, and building a
    7B model to observe a string would make this test unrunnable on the machine it has to run on.
    """

    def __init__(self, tok: Any) -> None:
        self.tok = tok
        self.prompts: list[str] = []

    def next_turn(self, messages: list[dict], tools: list[dict]) -> dict:
        self.prompts.append(self._render(messages, tools))
        return {"role": "assistant", "content": "ok", "tool_calls": None}

    def next_turns_batch(self, prompts: list[tuple[list[dict], list[dict]]]) -> list[dict]:
        return [self.next_turn(m, t) for m, t in prompts]


class VllmProbe(VllmOfflineTurnClient):
    """`VllmOfflineTurnClient._render`, without vLLM (which does not install on macOS ARM)."""

    def __init__(self, tok: Any) -> None:
        self.tok = tok


def _prompt_path_turns(tok: Any, render_prompt: Any, trace: dict) -> Rendered:
    """A client render call site, which produces prompts rather than assistant turns.

    The turn is recovered the way the serving stack's own boundary falls: the prompt for turn `i` must be a
    prefix of the full render through turn `i`, so the difference is exactly what the model is asked to produce.
    Asserting that prefix relation here is not incidental -- if a client's prompt were not a prefix of what the
    dataset rendered, the student would be trained on one string and served another.
    """
    messages, tools = trace["messages"], trace.get("tools") or []
    out = Rendered()
    for i in assistant_indices(messages):
        header = render_prompt(messages[:i], tools)
        full = render(tok, messages[: i + 1], tools, add_generation_prompt=False)
        assert full.startswith(header), (
            "the client's prompt is not a prefix of the dataset's render of the same conversation; "
            "the student would be trained on one string and prompted with another"
        )
        out.texts.append(header)
        out.turns.append((full[len(header) :], expected_calls(messages[i])))
    return out


def path_dataset_build(tok: Any, trace: dict) -> Rendered:
    """`data/build.assistant_spans` -- what `dataset build` writes into the training artifact."""
    messages, tools = trace["messages"], trace.get("tools") or []
    full, spans = assistant_spans(tok, messages, tools)
    out = Rendered(texts=[full])
    for (a, b), i in zip(spans, assistant_indices(messages), strict=True):
        out.turns.append((full[a:b], expected_calls(messages[i])))
    return out


def path_dpo_pairs(tok: Any, trace: dict) -> Rendered:
    """`train/dpo_data.render_pair` -- the pre-rendered strings TRL trains a preference loss on."""
    messages, tools = trace["messages"], trace.get("tools") or []
    out = Rendered()
    for i in assistant_indices(messages):
        if not messages[i].get("tool_calls"):
            continue
        pair = {
            "prompt": messages[:i],
            "chosen": [messages[i]],
            "rejected": [{"role": "assistant", "content": "I am not able to help with that.", "tool_calls": None}],
            "tools": tools,
            "task_id": trace.get("task_id"),
        }
        rendered = render_pair(tok, pair)
        out.texts += [rendered["prompt"], rendered["chosen"], rendered["rejected"]]
        out.turns.append((rendered["chosen"], expected_calls(messages[i])))
    return out


def path_teacher_forced(tok: Any, trace: dict) -> Rendered:
    """`eval/teacher_forced.teacher_forced` -- the per-turn next-action eval, through a client's render."""
    probe = HfProbe(tok)
    teacher_forced([trace], probe)
    return _replay_prompts(tok, probe, trace)


def path_teacher_forced_batched(tok: Any, trace: dict) -> Rendered:
    """The batched collector, which builds every prefix up front before generating."""
    probe = HfProbe(tok)
    teacher_forced_batched([trace], probe)
    return _replay_prompts(tok, probe, trace)


def _replay_prompts(tok: Any, probe: HfProbe, trace: dict) -> Rendered:
    """The prompts teacher-forcing actually produced, paired with the turns they ask for."""
    messages = trace["messages"]
    idx = assistant_indices(messages)
    assert len(probe.prompts) == len(idx), (
        f"teacher forcing asked for {len(probe.prompts)} turns but the trace has {len(idx)} assistant turns"
    )
    out = Rendered(texts=list(probe.prompts))
    tools = trace.get("tools") or []
    for prompt, i in zip(probe.prompts, idx, strict=True):
        full = render(tok, messages[: i + 1], tools, add_generation_prompt=False)
        assert full.startswith(prompt)
        out.turns.append((full[len(prompt) :], expected_calls(messages[i])))
    return out


def path_hf_turn_client(tok: Any, trace: dict) -> Rendered:
    return _prompt_path_turns(tok, HfProbe(tok)._render, trace)


def path_vllm_turn_client(tok: Any, trace: dict) -> Rendered:
    return _prompt_path_turns(tok, VllmProbe(tok)._render, trace)


#: Every place this repo renders messages through a chat template. A new one belongs here, or it is untested.
PATHS = {
    "dataset_build": path_dataset_build,
    "dpo_pairs": path_dpo_pairs,
    "teacher_forced": path_teacher_forced,
    "teacher_forced_batched": path_teacher_forced_batched,
    "hf_turn_client": path_hf_turn_client,
    "vllm_turn_client": path_vllm_turn_client,
}


@pytest.fixture(scope="module")
def rendered(pinned_tokenizer, traces) -> dict[str, list[Rendered]]:
    """Every path run over every trace, once, because rendering a 7B template is not free."""
    return {name: [fn(pinned_tokenizer, t) for t in traces] for name, fn in PATHS.items()}


# --------------------------------------------------------------------------------------------------------------
# the two properties
# --------------------------------------------------------------------------------------------------------------


def test_rendered_prompts_never_contain_stringified_arguments(rendered):
    """The defect itself, on every path, on three real traces, through the pinned template."""
    for name, per_trace in rendered.items():
        for i, result in enumerate(per_trace):
            assert result.texts, f"{name} rendered nothing for trace {i}"
            for text in result.texts:
                for needle in STRINGIFIED:
                    assert needle not in text, (
                        f"{name} rendered tool-call arguments as a quoted string ({needle!r}) for trace {i}; "
                        f"the serving stack's parser would recover a string instead of a call"
                    )


@pytest.mark.parametrize("path", sorted(PATHS))
def test_every_rendered_assistant_turn_parses_back_to_its_tool_call(path, rendered, pinned_tokenizer, project):
    """Round trip per path: what the template emitted, the parser that will serve it recovers exactly.

    vLLM's own hermes parser when vLLM is installed, the regex fallback otherwise -- the same choice
    `base-check` makes, and it is reported rather than assumed, because a missing dependency must not read as a
    pass.
    """
    parser = project.train.tool_parser
    checked = 0
    for i, result in enumerate(rendered[path]):
        for text, expected in result.turns:
            parsed = parse_with_vllm(text, pinned_tokenizer, parser.name if parser else None)
            if parsed is None:
                parsed = parse_fallback(text, parser.family if parser else "hermes")
            assert parsed == expected, (
                f"{path}, trace {i}: the parser recovered {parsed!r} from a turn rendered for {expected!r}"
            )
            checked += 1
    assert checked, f"{path} produced no assistant turn to round trip"


def test_the_round_trip_actually_saw_tool_calls(rendered):
    """A round trip over turns that contain no tool call passes trivially. This is the check that it did not."""
    for path, per_trace in rendered.items():
        calls = sum(len(expected) for r in per_trace for _, expected in r.turns)
        assert calls >= N_TRACES, f"{path} round-tripped {calls} tool calls; the traces carry more than that"


def test_every_path_agrees_on_the_text_of_a_turn(rendered):
    """Six paths, one string per assistant turn.

    Not a restatement of the two properties above: a path could render a *correct* tool call in a shape another
    path does not produce, and then the student is trained on one text and evaluated on another. The paths that
    isolate turns must agree character for character.
    """
    dataset = [t for r in rendered["dataset_build"] for t, _ in r.turns]
    for path in ("teacher_forced", "teacher_forced_batched", "hf_turn_client", "vllm_turn_client"):
        assert [t for r in rendered[path] for t, _ in r.turns] == dataset, (
            f"{path} renders an assistant turn differently from the dataset builder"
        )


# --------------------------------------------------------------------------------------------------------------
# the argument mask, on text the pinned template produced
# --------------------------------------------------------------------------------------------------------------


def test_the_argument_mask_covers_a_qwen_rendered_tool_call(rendered, pinned_tokenizer, traces):
    """`arg_char_spans` on real rendered output rather than on a hand-typed approximation of it.

    The mask used to search for the trace's own argument *string*. Now that arguments reach the template as
    objects, the text carries the template's serialization of the object, and matching it means matching an
    object -- which is what the structural search does, with nothing in it that knows Qwen's spacing.
    """
    checked = 0
    for trace, result in zip(traces, rendered["dataset_build"], strict=True):
        calls_by_turn = [m.get("tool_calls") or [] for m in trace["messages"] if m["role"] == "assistant"]
        for (text, expected), calls in zip(result.turns, calls_by_turn, strict=True):
            if not calls:
                continue

            # The character spans are exact: each one is the template's own serialization of that call's
            # arguments, and nothing less or more.
            spans = arg_char_spans(text, calls)
            assert len(spans) == len(calls), (
                f"{len(spans)} argument span(s) for {len(calls)} call(s) in a Qwen-rendered turn"
            )
            for (a, b), (_, args) in zip(spans, expected, strict=True):
                assert json.loads(text[a:b]) == args, (
                    f"the span {text[a:b]!r} is not the arguments {args!r} the turn was rendered from"
                )

            # The token mask rounds those spans out to whole tokens, which is the unit the gate's features are
            # computed over. It may over-cover by the token that straddles a span's edge -- Qwen tokenizes the
            # call's `}}` as one token -- and must not reach the template's markers.
            tokens = [
                pinned_tokenizer.decode([t])
                for t in pinned_tokenizer(text, add_special_tokens=False)["input_ids"]
            ]
            mask = arg_token_mask(tokens, text, calls)
            masked = "".join(t for t, m in zip(tokens, mask, strict=True) if m)
            assert 0 < mask_coverage(mask) < 1, "the mask covers no tokens, or the whole turn"
            assert "<tool_call>" not in masked and '"name"' not in masked, (
                f"the mask reached past the arguments into the template's own text: {masked!r}"
            )
            for _, args in expected:
                for value in args.values():
                    assert str(value) in masked, f"the mask misses the argument value {value!r}"
            checked += 1
    assert checked, "no tool-calling turn was masked"


def test_the_argument_mask_does_not_depend_on_the_templates_spacing(pinned_tokenizer):
    """The corpus is ASCII, so its rendered arguments happen to match `json.dumps`'s default spelling exactly.
    That coincidence is what the old mask relied on, and it is not a property of anything.

    A non-ASCII value breaks it: the wire-format string escapes it (`\\u00fc`), `tojson` does not, so no
    re-serialization of the trace's arguments appears in the text at all. A float and a nested object are here
    for the same reason -- their spelling is the template's choice, not ours.
    """
    args = {"customer_id": "München", "limit": 5, "ratio": 0.5, "flags": {"expedite": True}}
    calls = [{
        "id": "c1", "type": "function",
        "function": {"name": "search_orders", "arguments": json.dumps(args)},
    }]
    messages = [
        {"role": "user", "content": "Where is the order?"},
        {"role": "assistant", "content": "", "tool_calls": calls},
    ]
    tools = [{"type": "function", "function": {"name": "search_orders", "description": "Find orders.",
                                               "parameters": {"type": "object", "properties": {}}}}]

    header = render(pinned_tokenizer, messages[:1], tools, add_generation_prompt=True)
    turn = render(pinned_tokenizer, messages, tools, add_generation_prompt=False)[len(header) :]

    assert json.dumps(args) not in turn and json.dumps(args, separators=(",", ":")) not in turn, (
        "this trace was meant to render in a spelling no re-serialization reproduces; it did not, so the test "
        "no longer exercises what it claims to"
    )
    spans = arg_char_spans(turn, calls)
    assert len(spans) == 1
    a, b = spans[0]
    assert json.loads(turn[a:b]) == args
    assert turn[a] == "{" and turn[b - 1] == "}", f"the span is not the whole object: {turn[a:b]!r}"
