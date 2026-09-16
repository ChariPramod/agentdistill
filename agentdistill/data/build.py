"""Sample construction and loss masking.

Loss falls on assistant tokens only -- text and tool calls -- and on nothing else. Masking is done from character
offsets rather than by re-tokenizing pieces, because templates routinely change tokenization at message
boundaries: a token can straddle the join between a tool result and the assistant turn that follows it, and
piecewise tokenization would put the boundary in the wrong place.

The approach: render the full conversation once, compute the character span of each assistant turn by rendering
prefixes, tokenize the full string with offsets, and unmask exactly the tokens that fall inside a span.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

from agentdistill.data.template_check import render

IGNORE_INDEX = -100

TargetMode = Literal["all_assistant", "last_turn"]


class MaskingError(Exception):
    """The template cannot be masked from offsets. Named explicitly so `base-check` and the dataset builder can
    report the same actionable failure."""


@dataclass
class Sample:
    input_ids: list[int]
    labels: list[int]
    n_target_tokens: int
    kind: str = "trajectory"
    trace_id: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_tokens(self) -> int:
        return len(self.input_ids)

    def hash(self) -> str:
        """Identifies the sample's content for the dataset hash. Token ids already encode the text and the
        template, so hashing them makes the dataset hash sensitive to a tokenizer change, which is correct."""
        payload = json.dumps(
            [self.input_ids, [i for i, x in enumerate(self.labels) if x != IGNORE_INDEX], self.kind],
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()


def assistant_spans(tok: Any, messages: list[dict], tools: list[dict] | None) -> tuple[str, list[tuple[int, int]]]:
    """Return the full rendered text and the character span of every assistant turn.

    Raises MaskingError if the template is not prefix-stable, since the resulting spans would be meaningless.
    """
    full = render(tok, messages, tools, add_generation_prompt=False)
    spans: list[tuple[int, int]] = []
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        prefix = render(tok, messages[:i], tools, add_generation_prompt=True)
        if not full.startswith(prefix):
            raise MaskingError(
                f"chat template is not prefix-stable at assistant turn {i}; "
                "this template cannot be used for offsets-based masking. "
                "Run `agentdistill base-check <model>` for the full template report."
            )
        end = len(render(tok, messages[: i + 1], tools, add_generation_prompt=False))
        if end <= len(prefix):
            raise MaskingError(
                f"chat template produced an empty span for assistant turn {i}; "
                "the turn rendered to nothing, so it would contribute no loss"
            )
        spans.append((len(prefix), end))
    return full, spans


def _encode_with_offsets(tok: Any, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    if not getattr(tok, "is_fast", False):
        raise MaskingError(
            "offsets-based loss masking requires a fast tokenizer; "
            "this tokenizer does not support return_offsets_mapping"
        )
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    return list(enc["input_ids"]), [tuple(o) for o in enc["offset_mapping"]]


def _labels_from_spans(
    ids: list[int], offsets: list[tuple[int, int]], spans: list[tuple[int, int]]
) -> tuple[list[int], int]:
    """Unmask every token that *starts* inside an assistant span.

    "Starts inside" rather than "is contained in": a BPE token can straddle the end of a turn, merging the last
    character of the end-of-turn marker with the first character of the next role header. Requiring full
    containment would drop that token, and the one token most likely to straddle is the end-of-turn marker itself
    -- so the student would never be trained to stop. A token that starts before the span belongs to the prompt
    and stays masked; partial tokens cannot be trained either way.

    With a real instruct template this distinction is moot, because end-of-turn markers are added special tokens
    and tokenize atomically. It matters for templates that spell their markers in ordinary text.
    """
    labels = [IGNORE_INDEX] * len(ids)
    for t, (s, e) in enumerate(offsets):
        if e <= s:  # special tokens carry an empty offset and belong to no turn
            continue
        if any(a <= s < b for a, b in spans):
            labels[t] = ids[t]
    return labels, sum(1 for x in labels if x != IGNORE_INDEX)


def build_trajectory_sample(
    tok: Any,
    messages: list[dict],
    tools: list[dict] | None,
    max_seq_len: int,
    target: TargetMode = "all_assistant",
    trace_id: str | None = None,
) -> Sample | None:
    """The full conversation as one sequence, loss on assistant turns only.

    Returns None when the sample does not fit in `max_seq_len` or would carry no loss; the caller decides whether
    to drop it or fall back to turn windows.
    """
    full, spans = assistant_spans(tok, messages, tools)
    if not spans:
        return None
    if target == "last_turn":
        spans = spans[-1:]
    ids, offsets = _encode_with_offsets(tok, full)
    if len(ids) > max_seq_len:
        return None
    labels, n_target = _labels_from_spans(ids, offsets, spans)
    if n_target == 0:
        return None
    return Sample(
        input_ids=ids,
        labels=labels,
        n_target_tokens=n_target,
        kind="trajectory",
        trace_id=trace_id,
        meta={"n_assistant_turns": len(spans), "target": target},
    )


def build_turn_windows(
    tok: Any,
    messages: list[dict],
    tools: list[dict] | None,
    max_seq_len: int,
    window_turns: int = 12,
    target: TargetMode = "all_assistant",
    trace_id: str | None = None,
) -> Iterator[Sample]:
    """One sample per assistant turn, keeping the system prompt and the last `window_turns` messages before it.

    Used for trajectories that do not fit in `max_seq_len`. These samples are flagged `turn_window` so the eval
    can measure long-context degradation separately.

    With `target="all_assistant"` the earlier assistant turns inside a window also receive loss, which repeats
    them across windows; `target="last_turn"` trains only the final turn of each window. Both are worth an
    ablation, so both are supported and the choice is recorded in the sample metadata.
    """
    system = [m for m in messages[:1] if m["role"] == "system"]
    body = messages[len(system) :]
    for i, m in enumerate(body):
        if m["role"] != "assistant":
            continue
        start = max(0, i - window_turns)
        # Never start a window on a tool result: it would be an answer to a call the model cannot see.
        while start > 0 and body[start]["role"] == "tool":
            start -= 1
        window = system + body[start : i + 1]
        try:
            s = build_trajectory_sample(tok, window, tools, max_seq_len, target=target, trace_id=trace_id)
        except MaskingError:
            raise
        if s is None:
            continue
        s.kind = "turn_window"
        s.meta.update({"turn_index": i, "window_start": start, "window_turns": window_turns})
        yield s


def build_samples_for_trace(
    tok: Any,
    trace: dict,
    max_seq_len: int,
    window_turns: int = 12,
    windows_for_long: bool = True,
    target: TargetMode = "all_assistant",
) -> tuple[list[Sample], str]:
    """Build every sample for one trace. Returns (samples, note).

    A trajectory that fits becomes one sample. One that does not becomes turn windows, if enabled; otherwise it is
    dropped and the note says why, so the dataset report can account for every trace that went in.
    """
    messages, tools, trace_id = trace["messages"], trace.get("tools") or [], trace.get("id")
    sample = build_trajectory_sample(tok, messages, tools, max_seq_len, target=target, trace_id=trace_id)
    if sample is not None:
        return [sample], ""
    if not windows_for_long:
        return [], f"trajectory exceeds max_seq_len={max_seq_len} and windows are disabled"
    windows = list(
        build_turn_windows(tok, messages, tools, max_seq_len, window_turns, target=target, trace_id=trace_id)
    )
    if not windows:
        return [], f"trajectory exceeds max_seq_len={max_seq_len} and no single turn window fits either"
    return windows, f"trajectory exceeds max_seq_len={max_seq_len}; emitted {len(windows)} turn windows"


def decode_targets(tok: Any, sample: Sample) -> list[str]:
    """The text the model is actually trained to produce. Used in tests and by `dataset inspect`: if this does not
    read like assistant turns, the mask is wrong."""
    out: list[str] = []
    buf: list[int] = []
    for tid, lab in zip(sample.input_ids, sample.labels, strict=True):
        if lab == IGNORE_INDEX:
            if buf:
                out.append(tok.decode(buf))
                buf = []
        else:
            buf.append(tid)
    if buf:
        out.append(tok.decode(buf))
    return out
