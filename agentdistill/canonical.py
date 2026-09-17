"""Canonical JSON for argument hashing, v1.

The replay harness looks up a recorded tool result by hashing the call's arguments. Two calls that mean the same
thing must hash the same, or a student that phrased an argument differently is scored as a divergence when it
actually behaved identically. Two calls that mean different things must hash differently, or the harness serves
the wrong recorded result and silently inflates the score.

The same rules must hold in mcpgate's audit `args_hash` and in agentreplay's exporter, or the three stores will
never line up. The spec lives in `docs/canonical-json.md` and the shared vectors in
`schemas/canonical-vectors.json`; every implementation runs those vectors.

Normalization, in order:

1. Per-tool custom normalizers, if registered.
2. Objects: sort keys byte-wise; drop volatile keys (request ids, timestamps, cursors, nonces).
3. Arrays: preserve order; normalize elements.
4. Strings: trim; ISO-8601 timestamps -> "<ts>"; RFC 4122 UUIDs -> "<uuid>".
5. Floats: round to 6 decimals. Booleans, integers, null unchanged.
6. Serialize with sorted keys, separators "," and ":", UTF-8, no ASCII escaping.
7. args_hash = sha256(canonical({"tool": name, "args": normalized})) as lowercase hex.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal, localcontext
from typing import Any, Literal

CANONICAL_VERSION = 1

#: Keys whose values change between otherwise identical calls. Dropping them is what makes a replay lookup work
#: across runs; keeping them would make every call unique.
DEFAULT_DROP_KEYS: frozenset[str] = frozenset(
    {"request_id", "trace_id", "timestamp", "ts", "cursor", "page_token", "nonce"}
)

FLOAT_PRECISION = 6

TS_PLACEHOLDER = "<ts>"
UUID_PLACEHOLDER = "<uuid>"

# RFC 4122: 8-4-4-4-12 hex, any version. Anchored so an id embedded in a sentence is not rewritten.
_UUID = re.compile(r"\A[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\Z")

# ISO-8601: date, optional time, optional fractional seconds, optional zone (Z or +/-HH:MM or +/-HHMM).
_ISO8601 = re.compile(
    r"\A\d{4}-\d{2}-\d{2}"
    r"(?:[T ]\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?"
    r"(?:Z|z|[+-]\d{2}:?\d{2})?)?\Z"
)


#: How numbers are rendered. The only axis on which the two hash versions in this repo differ.
#:
#: `json`  - Python/JSON semantics: `1.0` renders as `1.0` and stays distinct from `1`. A tool whose schema says
#:           `{"type": "integer"}` would reject one and accept the other, so they are different arguments.
#: `js`    - JavaScript semantics, which MCPGate's TypeScript implementation produces: an integral float renders
#:           as an integer, so `1.0` and `1` collide, and signed zero is flattened.
NumberFormat = Literal["json", "js"]


@dataclass(frozen=True)
class Rules:
    """How to normalize one tool's arguments."""

    drop_keys: frozenset[str] = DEFAULT_DROP_KEYS
    float_precision: int = FLOAT_PRECISION
    normalize_timestamps: bool = True
    normalize_uuids: bool = True
    number_format: NumberFormat = "json"
    #: Applied to the whole argument object before the generic rules. Use for tool-specific quirks, such as a
    #: tool that accepts either `email` or `customer_email` for the same thing.
    custom: Callable[[dict], dict] | None = field(default=None, compare=False)


_DEFAULT_RULES = Rules()
_TOOL_RULES: dict[str, Rules] = {}


def register_rules(tool: str, rules: Rules) -> None:
    """Register per-tool normalization. Call at import time, before any hashing."""
    _TOOL_RULES[tool] = rules


def clear_rules() -> None:
    """Drop every per-tool rule. For tests; resetting global state between cases."""
    _TOOL_RULES.clear()


def rules_for(tool: str) -> Rules:
    return _TOOL_RULES.get(tool, _DEFAULT_RULES)


def _normalize_string(s: str, rules: Rules) -> str:
    s = s.strip()
    if rules.normalize_uuids and _UUID.match(s):
        return UUID_PLACEHOLDER
    if rules.normalize_timestamps and _ISO8601.match(s):
        return TS_PLACEHOLDER
    return s


def _normalize_number(value: float | int, rules: Rules) -> Any:
    if isinstance(value, bool):  # bool is a subclass of int; must be checked first
        return value
    if isinstance(value, int):
        return value
    if not math.isfinite(value):
        raise ValueError(f"cannot canonicalize a non-finite number: {value!r}")
    if rules.number_format == "js":
        # Half-up at the sixth decimal, matching the TypeScript implementation. Python's round() is
        # banker's rounding, which would disagree on exact halves.
        if float(value).is_integer():
            return float(value) + 0.0
        with localcontext() as ctx:
            ctx.prec = 100
            quantum = Decimal(1).scaleb(-rules.float_precision)
            return float(Decimal.from_float(value).quantize(quantum, rounding=ROUND_HALF_UP))
    rounded = round(float(value), rules.float_precision)
    # 7.0 and 7 must not collide under `json` rules: a float stays a float. But -0.0 normalizes to 0.0, since
    # the sign of zero is not something any tool means anything by.
    return rounded + 0.0


def normalize(value: Any, rules: Rules | None = None) -> Any:
    """Recursively normalize a JSON value. Pure; the input is not mutated."""
    rules = rules or _DEFAULT_RULES
    if isinstance(value, dict):
        if rules.custom is not None:
            value = rules.custom(dict(value))
        out = {}
        for key in sorted(value, key=lambda k: str(k).encode("utf-8")):
            if key in rules.drop_keys:
                continue
            out[key] = normalize(value[key], rules)
        return out
    if isinstance(value, (list, tuple)):
        return [normalize(v, rules) for v in value]
    if isinstance(value, str):
        return _normalize_string(value, rules)
    if isinstance(value, (int, float)):
        return _normalize_number(value, rules)
    return value


def canonical_json(value: Any, number_format: NumberFormat = "json") -> str:
    """Serialize an already-normalized value to its canonical text form."""
    if number_format == "json":
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return _canonical_js(value)


def _canonical_js(value: Any) -> str:
    """Serialize the way `JSON.stringify` would render numbers.

    Keys sort byte-wise, and an integral float prints without its fractional part -- which is what a TypeScript
    implementation produces and what MCPGate's stored hashes assume.
    """
    if isinstance(value, dict):
        keys = sorted(value, key=lambda k: str(k).encode("utf-8"))
        body = ",".join(json.dumps(k, ensure_ascii=False) + ":" + _canonical_js(value[k]) for k in keys)
        return "{" + body + "}"
    if isinstance(value, list):
        return "[" + ",".join(_canonical_js(v) for v in value) + "]"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if value == 0:
        return "0"
    if abs(value) >= 1e21:
        return json.dumps(value, separators=(",", ":"))
    if int(value) == value:
        return str(int(value))
    return format(value, ".6f").rstrip("0").rstrip(".")


def canonical_args(tool: str, args: Any, rules: Rules | None = None) -> str:
    """The canonical text whose sha256 is `args_hash`."""
    r = rules or rules_for(tool)
    return canonical_json({"args": normalize(args, r), "tool": tool}, r.number_format)


def args_hash(tool: str, args: Any, rules: Rules | None = None) -> str:
    """Lowercase hex sha256 of the canonical form. This is the replay lookup key."""
    return hashlib.sha256(canonical_args(tool, args, rules).encode("utf-8")).hexdigest()


def hash_json(value: Any) -> str:
    """sha256 of an arbitrary already-normalized JSON value. For content addressing outside tool calls."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


#: The MCPGate-compatible ruleset (`args_hash_v=1` on their side). Same code path as the default; only the
#: number rendering differs. Exposed here so there is exactly one implementation in the repo.
SHARED_RULES = Rules(number_format="js")
