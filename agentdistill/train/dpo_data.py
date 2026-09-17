"""Rendering preference pairs for DPO.

TRL's conversational DPO format may or may not thread `tools` through the chat template, depending on the
release. Relying on it means a pair set that silently trains without the tool schemas the student needs. So
pairs are pre-rendered to the plain string format, which every TRL version accepts, and the tools go in through
the template here where we can assert they arrived.
"""

from __future__ import annotations

import json
from typing import Any


def render(tok: Any, messages: list[dict], tools: list[dict] | None, gen: bool) -> str:
    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": gen}
    if tools:
        kwargs["tools"] = tools
    return tok.apply_chat_template(messages, **kwargs)


def render_pair(tok: Any, pair: dict, strip_bos: str | None = None) -> dict:
    """Render one pair to `{prompt, chosen, rejected}` strings.

    `chosen` and `rejected` exclude the prompt, which is what DPO expects: the loss compares two continuations of
    one shared context.
    """
    tools = pair.get("tools") or []
    prompt = render(tok, pair["prompt"], tools, gen=True)
    full_chosen = render(tok, [*pair["prompt"], *pair["chosen"]], tools, gen=False)
    full_rejected = render(tok, [*pair["prompt"], *pair["rejected"]], tools, gen=False)
    if not (full_chosen.startswith(prompt) and full_rejected.startswith(prompt)):
        raise ValueError(
            "chat template is not prefix-stable, so a DPO pair cannot be split into prompt and continuation. "
            "Run `agentdistill base-check <model>`."
        )
    if strip_bos and prompt.startswith(strip_bos):
        # TRL adds BOS when it tokenizes the pre-rendered prompt. If the template already emitted one, the
        # sequence gets two, which shifts every position and quietly degrades the run.
        prompt = prompt[len(strip_bos) :]
    return {
        "prompt": prompt,
        "chosen": full_chosen[len(render(tok, pair["prompt"], tools, gen=True)) :],
        "rejected": full_rejected[len(render(tok, pair["prompt"], tools, gen=True)) :],
        "task_id": pair.get("task_id"),
        "pair_kind": pair.get("pair_kind", "rollout"),
        "diff_kind": pair.get("diff_kind") or diff_kind(pair),
    }


#: What a pair actually teaches. Recorded per pair so the report can show what the DPO set was made of; a set
#: that is 90% `text` is a warning sign, because it is mostly teaching phrasing under a preference loss.
DIFF_KINDS = ("tool_choice", "tool_args", "tool_vs_text", "text")

#: Characters ignored when deciding whether two text answers genuinely differ.
_TRIVIAL = str.maketrans("", "", " \t\n\r.,;:!?'\"-()")


def text_signature(message: dict) -> str:
    """Prose reduced to what it says, so casing, spacing and punctuation are not mistaken for a decision."""
    return (message.get("content") or "").translate(_TRIVIAL).lower()


def diff_kind(pair: dict) -> str:
    """Classify what separates the two sides."""
    chosen, rejected = pair["chosen"][0], pair["rejected"][0]
    chosen_calls, rejected_calls = _calls(chosen), _calls(rejected)
    if bool(chosen_calls) != bool(rejected_calls):
        return "tool_vs_text"
    if not chosen_calls:
        return "text"
    if _names(chosen) != _names(rejected):
        return "tool_choice"
    return "tool_args"


def _names(message: dict) -> frozenset[str]:
    return frozenset(c["function"]["name"] for c in message.get("tool_calls") or [])


def pair_is_valid(pair: dict) -> tuple[bool, str]:
    """Reject pairs that cannot teach anything, before they reach the trainer.

    A DPO set full of these is the usual reason `rewards/accuracies` sits at 0.5: the model is being asked to
    prefer one of two things that are the same, or to compare continuations of different contexts.
    """
    if not pair.get("prompt"):
        return False, "prompt is empty"
    if pair["prompt"][-1]["role"] == "assistant":
        return False, "prompt ends on an assistant turn, so the two sides continue different contexts"
    for side in ("chosen", "rejected"):
        turns = pair.get(side) or []
        if not turns:
            return False, f"{side} is empty"
        if turns[0]["role"] != "assistant":
            return False, f"{side} is not an assistant turn"
        for call in turns[0].get("tool_calls") or []:
            try:
                json.loads(call["function"]["arguments"])
            except (json.JSONDecodeError, KeyError, TypeError):
                return False, f"{side} has malformed tool call arguments"

    chosen, rejected = pair["chosen"][0], pair["rejected"][0]
    chosen_calls, rejected_calls = _calls(chosen), _calls(rejected)

    if chosen_calls or rejected_calls:
        # At least one side acts. The preference must be about *what was done* -- a different tool, or different
        # arguments, or acting versus answering. Two calls to the same tool with the same arguments but different
        # prose is a phrasing preference dressed up as a decision, and training on it teaches house style.
        if chosen_calls == rejected_calls:
            return False, "chosen and rejected make the same tool calls; the pair only differs in wording"
        return True, ""

    # Neither side acts, so the text is the decision -- but only if the two answers genuinely differ. Casing,
    # spacing and punctuation are style, and a pair built from them teaches house style under a preference loss.
    if text_signature(chosen) == text_signature(rejected):
        return False, "chosen and rejected say the same thing apart from formatting"
    return True, ""


def _normalized_text(message: dict) -> str:
    return " ".join((message.get("content") or "").split())


def _calls(message: dict) -> tuple[str, ...]:
    """Canonical hashes of a turn's tool calls, order-insensitive.

    Canonical rather than raw so that key order and whitespace inside the arguments are not mistaken for a
    behavioural difference.
    """
    from agentdistill.canonical import args_hash

    out = []
    for c in message.get("tool_calls") or []:
        raw = c["function"]["arguments"]
        try:
            args = json.loads(raw) if isinstance(raw, str) else raw
        except (json.JSONDecodeError, TypeError):
            args = {"__unparseable__": str(raw)}
        out.append(args_hash(c["function"]["name"], args))
    return tuple(sorted(out))


def filter_pairs(pairs: list[dict]) -> tuple[list[dict], dict[str, int]]:
    """Keep the usable pairs, tag each with what it teaches, and count why the rest were dropped."""
    kept: list[dict] = []
    reasons: dict[str, int] = {}
    for pair in pairs:
        ok, why = pair_is_valid(pair)
        if ok:
            kept.append({**pair, "diff_kind": diff_kind(pair)})
        else:
            reasons[why] = reasons.get(why, 0) + 1
    return kept, reasons


def diff_kind_mix(pairs: list[dict]) -> dict[str, int]:
    """How many pairs of each kind. Reported so a set dominated by `text` pairs is visible."""
    out = dict.fromkeys(DIFF_KINDS, 0)
    for pair in pairs:
        kind = pair.get("diff_kind") or diff_kind(pair)
        out[kind] = out.get(kind, 0) + 1
    return out


def diff_kind_warnings(mix: dict[str, int]) -> list[str]:
    total = sum(mix.values())
    if not total:
        return []
    out = []
    if mix.get("text", 0) / total > 0.6:
        out.append(
            f"{mix['text'] / total:.0%} of pairs differ only in prose. DPO on those teaches phrasing, not "
            f"decisions; check the rollouts are actually diverging on actions."
        )
    if mix.get("tool_args", 0) + mix.get("tool_choice", 0) == 0:
        out.append("no pair differs on a tool call, so this set cannot teach tool selection at all")
    return out


def balance_kinds(pairs: list[dict], max_teacher_ratio: float = 1.0) -> tuple[list[dict], dict[str, int]]:
    """Cap teacher pairs against rollout pairs.

    A set dominated by teacher-versus-student pairs is supervised fine-tuning wearing a DPO loss: every chosen
    side is the teacher's, so the gradient mostly re-teaches imitation rather than the student's own failure
    modes -- which is the whole point of going on-policy.
    """
    rollout = [p for p in pairs if p.get("pair_kind", "rollout") == "rollout"]
    teacher = [p for p in pairs if p.get("pair_kind") == "teacher"]
    cap = int(len(rollout) * max_teacher_ratio) if rollout else len(teacher)
    kept_teacher = teacher[:cap]
    return rollout + kept_teacher, {
        "rollout": len(rollout),
        "teacher": len(kept_teacher),
        "teacher_dropped": len(teacher) - len(kept_teacher),
    }


def bos_token_text(tok: Any) -> str | None:
    """The BOS string a template emits, if it emits one.

    Used to avoid a double BOS: TRL tokenizes the pre-rendered prompt and may add its own.
    """
    bos = getattr(tok, "bos_token", None)
    if not bos:
        return None
    rendered = render(tok, [{"role": "user", "content": "x"}], None, gen=True)
    return bos if rendered.startswith(bos) else None
