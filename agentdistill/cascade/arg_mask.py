"""Which generated tokens belong to tool-call argument JSON.

The confidence gate's most useful features are about the *arguments*: a model is often sure which tool to call
and much less sure what to put in it, and averaging that uncertainty across the whole turn washes it out. So the
features need a per-token mask marking the argument spans.

Nothing about this is exact. Servers return token strings that do not always concatenate to the text they
report (byte fallbacks, stripped spaces), and a template may re-serialize arguments on the way out. Every step
here degrades to something usable rather than raising, because a turn with no mask should still be scorable --
with the argument features absent rather than wrong.
"""

from __future__ import annotations

import json
from typing import Any


def token_spans(tokens: list[str]) -> list[tuple[int, int]]:
    """Character span of each token within the concatenation of all tokens."""
    spans: list[tuple[int, int]] = []
    pos = 0
    for t in tokens:
        spans.append((pos, pos + len(t)))
        pos += len(t)
    return spans


def arg_char_spans(text: str, tool_calls: list[dict]) -> list[tuple[int, int]]:
    """Locate each tool call's argument JSON inside the generated text.

    Three attempts, in order of fidelity: the exact serialized string, a re-serialization (the server may have
    compacted or expanded it), and finally the individual argument values. The last one is a partial mask -- it
    covers the values but not the keys or braces -- which is still better than no argument features at all.
    """
    out: list[tuple[int, int]] = []
    cursor = 0
    for call in tool_calls or []:
        raw = call.get("function", {}).get("arguments")
        candidates: list[str] = [raw] if isinstance(raw, str) else []
        obj: Any = None
        try:
            obj = json.loads(raw) if isinstance(raw, str) else raw
            candidates += [json.dumps(obj, separators=(",", ":")), json.dumps(obj)]
        except (json.JSONDecodeError, TypeError):
            obj = None

        found = None
        for candidate in candidates:
            if not candidate:
                continue
            i = text.find(candidate, cursor)
            if i >= 0:
                found = (i, i + len(candidate))
                break
        if found is not None:
            out.append(found)
            cursor = found[1]
            continue

        if isinstance(obj, dict):
            for value in obj.values():
                needle = value if isinstance(value, str) else json.dumps(value)
                i = text.find(needle, cursor)
                if i >= 0:
                    out.append((i, i + len(needle)))
                    cursor = i + len(needle)
    return out


def arg_token_mask(tokens: list[str], text: str, tool_calls: list[dict]) -> list[bool]:
    """True for each token that overlaps an argument span.

    When the tokens do not concatenate to the reported text, the joined string wins: the token spans index into
    it, so aligning on anything else would mark the wrong tokens.
    """
    joined = "".join(tokens)
    if joined != text:
        text = joined
    spans = token_spans(tokens)
    arg_spans = arg_char_spans(text, tool_calls)
    return [any(start < b and end > a for a, b in arg_spans) for start, end in spans]


def mask_coverage(mask: list[bool]) -> float:
    """Share of tokens inside argument spans. A coverage of 0 on a turn with tool calls means the mask failed."""
    return (sum(mask) / len(mask)) if mask else 0.0
