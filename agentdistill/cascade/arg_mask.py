"""Which generated tokens belong to tool-call argument JSON.

The confidence gate's most useful features are about the *arguments*: a model is often sure which tool to call
and much less sure what to put in it, and averaging that uncertainty across the whole turn washes it out. So the
features need a per-token mask marking the argument spans.

Nothing about this is exact. Servers return token strings that do not always concatenate to the text they
report (byte fallbacks, stripped spaces), and a template may re-serialize arguments on the way out. Every step
here degrades to something usable rather than raising, because a turn with no mask should still be scorable --
with the argument features absent rather than wrong.

Since the render-boundary fix, arguments reach a chat template as an *object* rather than as the wire format's
JSON string, so the generated text carries the template's own serialization of that object -- Qwen's hermes
block is `{"name": ..., "arguments": {...}}`. Which is to say the exact bytes of the trace's argument string are
no longer what appears in the text, and matching on them is matching on a coincidence of spacing. So the search
for the object is structural: parse every JSON object in the text and compare values. Nothing here knows what
separators, key order or escaping the template chose.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

#: One decoder, reused: `raw_decode` finds where a JSON value ends, which is what makes a structural scan
#: possible without a brace counter that string literals would fool.
_DECODER = json.JSONDecoder()


def _json_values_in(text: str, start: int) -> Iterator[tuple[tuple[int, int], Any]]:
    """Every JSON object in `text` at or after `start`, as ((begin, end), value), left to right.

    A generator, so the common case stops at the first match instead of decoding the whole turn. Objects nest --
    the hermes block's `arguments` sits inside the call object -- so every `{` is a candidate, not just the
    outermost.
    """
    i = text.find("{", start)
    while i >= 0:
        try:
            value, end = _DECODER.raw_decode(text, i)
        except ValueError:
            pass
        else:
            yield (i, end), value
        i = text.find("{", i + 1)


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

    Four attempts, in order of fidelity:

    1. the exact serialized string, for a server that echoed the trace's bytes;
    2. two re-serializations, compact and spaced, which cover the two shapes most servers emit;
    3. **the parsed object**, found by decoding every JSON object in the text and comparing values. This is the
       one that matches what a real chat template emits, because it is indifferent to the separators, key order
       and escaping the template chose -- none of which are knowable from here;
    4. the individual argument values. A partial mask -- it covers the values but not the keys or braces --
       which is still better than no argument features at all.

    Unparseable arguments stop at (1): there is no object to compare, and that is the point, since a malformed
    call should not be quietly matched to a well-formed one in the text.
    """
    out: list[tuple[int, int]] = []
    cursor = 0
    for call in tool_calls or []:
        raw = call.get("function", {}).get("arguments")
        candidates: list[str] = [raw] if isinstance(raw, str) else []
        obj: Any = None
        parsed_ok = False
        try:
            obj = json.loads(raw) if isinstance(raw, str) else raw
            candidates += [json.dumps(obj, separators=(",", ":")), json.dumps(obj)]
            parsed_ok = True
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

        if found is None and parsed_ok and isinstance(obj, dict):
            for span, value in _json_values_in(text, cursor):
                if value == obj:
                    found = span
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
