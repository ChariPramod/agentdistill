"""Canonical JSON. The shared vectors are the contract; the rest of these tests explain why each rule exists."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentdistill.canonical import (
    DEFAULT_DROP_KEYS,
    SHARED_RULES,
    Rules,
    args_hash,
    canonical_args,
    canonical_json,
    clear_rules,
    hash_json,
    normalize,
    register_rules,
    rules_for,
)

VECTORS = json.loads((Path(__file__).resolve().parents[1] / "schemas" / "canonical-vectors.json").read_text())


@pytest.fixture(autouse=True)
def _clean_rules():
    """Per-tool rules are global; a leaked registration would silently change another test's hashes."""
    clear_rules()
    yield
    clear_rules()


# --------------------------------------------------------------------------------------------------------------
# the shared contract
# --------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("vector", VECTORS["vectors"], ids=lambda v: v["name"])
def test_shared_vectors(vector):
    """Every implementation of this spec must reproduce these byte for byte."""
    assert normalize(vector["args"], rules_for(vector["tool"])) == vector["normalized"]
    assert canonical_args(vector["tool"], vector["args"]) == vector["canonical"]
    assert args_hash(vector["tool"], vector["args"]) == vector["args_hash"]


def test_vectors_cover_the_documented_rules():
    """A rule with no vector is a rule another implementation will get wrong."""
    names = {v["name"] for v in VECTORS["vectors"]}
    required = {
        "key_order", "nested_objects", "unicode_preserved", "string_trimmed", "float_rounding",
        "int_not_float", "negative_zero", "booleans_and_null", "drop_volatile_keys", "timestamp_formats",
        "uuid_in_array", "array_order_significant", "empty_args",
    }
    assert required <= names, f"vectors missing: {sorted(required - names)}"


def test_vector_hashes_are_distinct():
    """Two different vectors colliding would mean the spec cannot distinguish two different calls."""
    hashes = [v["args_hash"] for v in VECTORS["vectors"]]
    assert len(set(hashes)) == len(hashes)


# --------------------------------------------------------------------------------------------------------------
# equivalence: these must collide
# --------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("a", "b", "why"),
    [
        ({"b": 2, "a": 1}, {"a": 1, "b": 2}, "key order is not meaning"),
        ({"a": " x "}, {"a": "x"}, "surrounding whitespace is not meaning"),
        ({"a": 1.00000001}, {"a": 1.0}, "float noise below 6dp is not meaning"),
        ({"a": -0.0}, {"a": 0.0}, "sign of zero is not meaning"),
        ({"a": 1, "request_id": "r1"}, {"a": 1, "request_id": "r2"}, "volatile keys are dropped"),
        ({"t": "2026-09-15T12:00:00Z"}, {"t": "2025-01-01"}, "timestamps collapse"),
        (
            {"u": "7c9e6679-7425-40de-944b-e07fc1f90ae7"},
            {"u": "00000000-0000-4000-8000-000000000000"},
            "uuids collapse",
        ),
        ({"a": {"y": 1, "x": 2}}, {"a": {"x": 2, "y": 1}}, "sorting is recursive"),
    ],
)
def test_equivalent_calls_share_a_hash(a, b, why):
    assert args_hash("tool", a) == args_hash("tool", b), why


# --------------------------------------------------------------------------------------------------------------
# distinction: these must not collide
# --------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("a", "b", "why"),
    [
        ({"a": 1}, {"a": 2}, "different values are different calls"),
        ({"a": 1}, {"a": 1.0}, "an integer argument is not a float argument"),
        ({"a": [1, 2]}, {"a": [2, 1]}, "array order is meaning"),
        ({"a": "SHIPPED"}, {"a": "shipped"}, "case is preserved by default"),
        ({"a": 1}, {"a": 1, "b": 2}, "an extra argument is a different call"),
        ({"a": True}, {"a": 1}, "a bool is not an int, despite Python's type hierarchy"),
        ({"a": None}, {}, "an explicit null is not an absent key"),
        ({"a": "x"}, {"a": ["x"]}, "a scalar is not a one-element array"),
    ],
)
def test_distinct_calls_have_distinct_hashes(a, b, why):
    assert args_hash("tool", a) != args_hash("tool", b), why


def test_tool_name_is_part_of_the_hash():
    assert args_hash("refund", {"id": 1}) != args_hash("cancel", {"id": 1})


# --------------------------------------------------------------------------------------------------------------
# anchoring
# --------------------------------------------------------------------------------------------------------------


def test_identifiers_inside_prose_are_not_rewritten():
    """A summary mentioning a date is content. Rewriting it would merge genuinely different tickets."""
    a = {"summary": "customer called on 2026-09-15 about a refund"}
    b = {"summary": "customer called on 2024-01-02 about a refund"}
    assert args_hash("create_ticket", a) != args_hash("create_ticket", b)
    assert "<ts>" not in canonical_args("create_ticket", a)


def test_uuid_inside_prose_is_not_rewritten():
    a = {"summary": "see ticket 7c9e6679-7425-40de-944b-e07fc1f90ae7 for details"}
    assert "<uuid>" not in canonical_args("create_ticket", a)


@pytest.mark.parametrize(
    "value",
    ["2026-09-15", "2026-09-15T12:00:00Z", "2026-09-15T12:00:00.123+05:30", "2026-09-15 12:00", "2026-09-15T12:00:00+0530"],
)
def test_iso8601_shapes_all_collapse(value):
    assert normalize({"t": value})["t"] == "<ts>"


@pytest.mark.parametrize("value", ["2026-13-45x", "not a date", "20260915", "c_2026-09-15"])
def test_non_timestamps_are_left_alone(value):
    assert normalize({"t": value})["t"] == value


# --------------------------------------------------------------------------------------------------------------
# per-tool rules
# --------------------------------------------------------------------------------------------------------------


def test_custom_normalizer_runs_first():
    """A tool that accepts two names for one field: the alias is folded before the generic rules."""

    def fold_alias(args: dict) -> dict:
        if "customer_email" in args:
            args["email"] = args.pop("customer_email")
        return args

    register_rules("get_customer", Rules(custom=fold_alias))
    assert args_hash("get_customer", {"customer_email": "a@b.c"}) == args_hash("get_customer", {"email": "a@b.c"})
    # Another tool is unaffected.
    assert args_hash("other", {"customer_email": "a@b.c"}) != args_hash("other", {"email": "a@b.c"})


def test_per_tool_drop_keys():
    register_rules("noisy", Rules(drop_keys=frozenset({"session"})))
    assert args_hash("noisy", {"a": 1, "session": "s1"}) == args_hash("noisy", {"a": 1, "session": "s2"})
    assert args_hash("quiet", {"a": 1, "session": "s1"}) != args_hash("quiet", {"a": 1, "session": "s2"})


def test_per_tool_rules_can_disable_timestamp_collapsing():
    register_rules("audit", Rules(normalize_timestamps=False))
    assert args_hash("audit", {"t": "2026-09-15"}) != args_hash("audit", {"t": "2024-01-01"})


def test_default_drop_keys_are_the_documented_set():
    documented = {"request_id", "trace_id", "timestamp", "ts", "cursor", "page_token", "nonce"}
    assert set(DEFAULT_DROP_KEYS) == documented


# --------------------------------------------------------------------------------------------------------------
# serialization
# --------------------------------------------------------------------------------------------------------------


def test_serialization_is_compact_and_utf8():
    out = canonical_json({"b": 1, "a": "café"})
    assert out == '{"a":"café","b":1}'
    assert "\\u" not in out


def test_normalize_does_not_mutate_input():
    original = {"b": 2, "a": {"nested": " x "}, "request_id": "r"}
    snapshot = json.loads(json.dumps(original))
    normalize(original)
    assert original == snapshot


def test_hash_json_is_stable():
    assert hash_json({"a": 1, "b": 2}) == hash_json({"b": 2, "a": 1})


def test_deeply_nested_structures():
    deep: dict = {"v": 1}
    for _ in range(20):
        deep = {"n": deep, "request_id": "drop me"}
    out = normalize(deep)
    for _ in range(20):
        assert "request_id" not in out
        out = out["n"]
    assert out == {"v": 1}


# --------------------------------------------------------------------------------------------------------------
# one implementation, two rulesets
#
# The repo has exactly one canonicalizer. The MCPGate-compatible hash is a `Rules` selection on the same code
# path, not a second implementation, because two that agree today are two that disagree after the next edit.
# --------------------------------------------------------------------------------------------------------------


def test_shared_module_is_a_thin_wrapper_not_a_second_implementation():
    import agentdistill.canonical_shared as shared

    source = Path(shared.__file__).read_text()
    assert "hashlib" not in source, "the shared module must not compute its own digest"
    assert "def _canonical" not in source, "the shared module must not have its own serializer"
    assert shared.RULES is SHARED_RULES


@pytest.mark.parametrize(
    ("args", "same"),
    [
        ({"a": 1}, True),
        ({"a": 1.5}, True),
        ({"s": "x"}, True),
        ({"a": True}, True),
        ({"a": None}, True),
        ({"a": 1.0}, False),        # integral float: 1.0 vs 1
        ({"a": -0.0}, False),       # signed zero
        ({"a": [1, 2.0]}, False),   # recursively
    ],
)
def test_the_two_rulesets_differ_only_on_number_rendering(args, same):
    from agentdistill.canonical_shared import args_hash as shared_hash

    assert (args_hash("t", args) == shared_hash("t", args)) is same


def test_json_rules_keep_int_and_float_distinct():
    """A tool whose schema says {"type": "integer"} accepts one and rejects the other."""
    assert args_hash("t", {"a": 1}) != args_hash("t", {"a": 1.0})


def test_js_rules_collide_int_and_integral_float():
    """JavaScript renders 1.0 as 1, and MCPGate's stored rows assume it."""
    from agentdistill.canonical_shared import args_hash as shared_hash

    assert shared_hash("t", {"a": 1}) == shared_hash("t", {"a": 1.0})


def test_js_rules_round_to_six_decimals():
    """Both rulesets round at six decimals; float noise below that must not split a hash.

    The shared ruleset uses Decimal half-up to match the TypeScript implementation. On IEEE doubles at this
    precision the two rounding modes never actually diverge -- no binary double lands on an exact half at the
    sixth decimal -- so this asserts the rounding, not the tie-breaking rule.
    """
    from agentdistill.canonical_shared import args_hash as shared_hash
    from agentdistill.canonical_shared import normalize as shared_normalize

    assert shared_normalize({"a": 1.23456789})["a"] == pytest.approx(1.234568)
    assert shared_hash("t", {"a": 1.2345678}) == shared_hash("t", {"a": 1.2345681})


def test_non_finite_numbers_are_refused():
    with pytest.raises(ValueError, match="non-finite"):
        args_hash("t", {"a": float("inf")})


@pytest.mark.parametrize("vector_file", ["canonical-vectors.json", "canonical-shared-vectors.json"])
def test_both_vector_files_pass_off_the_same_code_path(vector_file):
    """The definition of done: 19 shared vectors and the v1 vectors, one implementation."""
    from agentdistill.canonical_shared import args_hash as shared_hash

    path = Path(__file__).resolve().parents[1] / "schemas" / vector_file
    doc = json.loads(path.read_text())
    vectors = doc if isinstance(doc, list) else doc.get("vectors", [])
    assert vectors, f"{vector_file} has no vectors"
    hasher = shared_hash if "shared" in vector_file else args_hash
    for v in vectors:
        expected = v.get("args_hash") or v.get("hash") or v.get("expected")
        if not expected:
            continue
        assert hasher(v.get("tool", "t"), v.get("args", {})) == expected, f"{vector_file}: {v.get('name')}"
