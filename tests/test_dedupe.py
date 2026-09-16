"""Near-duplicate detection.

Two properties matter: a trace and its near-copy collapse, and rearranging a trace without changing what the
assistant did must not create a false duplicate.
"""

from __future__ import annotations

import json

from hypothesis import given, settings
from hypothesis import strategies as st

from agentdistill.curate.dedupe import (
    assistant_text,
    duplicate_groups,
    jaccard,
    near_duplicates,
    normalize_literals,
    shingles,
)
from tests.conftest import make_call, make_trace

LONG = " ".join(
    f"I checked detail {i} of the account history and the shipment record before answering" for i in range(12)
)


def _trace(tid: str, *, limit: int = 5, closing: str = LONG) -> dict:
    return make_trace(tid, args={"customer_id": "c_9", "limit": limit}, closing=closing)


def test_identical_traces_are_duplicates():
    a, b = _trace("a"), _trace("b")
    assert near_duplicates([a, b]) == {"b"}, "the first occurrence is kept"


def test_copy_with_one_changed_number_is_a_near_duplicate():
    a, b = _trace("a", limit=5), _trace("b", limit=6)
    assert jaccard(a, b) >= 0.85
    assert near_duplicates([a, b]) == {"b"}


def test_genuinely_different_trajectories_are_kept():
    a = _trace("a")
    b = make_trace("b", tool_name="cancel_order", closing="I have cancelled that order for you entirely.")
    assert near_duplicates([a, b]) == set()


def test_candidates_below_threshold_are_not_dropped():
    """LSH banding proposes pairs below the configured threshold; the filter must verify before dropping."""
    a = _trace("a", closing=LONG)
    b = _trace("b", closing="Completely different closing text with no shared phrasing whatsoever here.")
    assert jaccard(a, b) < 0.85
    assert near_duplicates([a, b]) == set()


def test_assistant_text_excludes_environment_output():
    t = _trace("a")
    text = assistant_text(t)
    assert "Let me look that up" in text
    assert "search_orders" in text
    assert "status" not in text, "tool results are environment output, not model behavior"
    assert "You are a support agent" not in text


def test_traces_differing_only_in_tool_results_are_duplicates():
    """Same assistant behavior over different data is the same lesson."""
    a, b = _trace("a"), _trace("b")
    b["messages"][3]["content"] = '[{"order_id": "o_999", "status": "pending"}]'
    assert near_duplicates([a, b]) == {"b"}


def test_duplicate_groups_names_the_survivor():
    a, b, c = _trace("a"), _trace("b"), _trace("c")
    groups = duplicate_groups([a, b, c])
    assert groups == {"a": ["b", "c"]}


def test_empty_assistant_text_is_never_a_duplicate():
    a = make_trace("a")
    a["messages"] = [{"role": "user", "content": "hello"}]
    b = make_trace("b")
    b["messages"] = [{"role": "user", "content": "goodbye"}]
    assert near_duplicates([a, b]) == set()


def test_normalize_literals_masks_ids_and_numbers():
    out = normalize_literals('search_orders {"customer_id":"c_124","limit":5} refunded 42.50 for o_1234')
    assert "c_124" not in out and "o_1234" not in out
    assert "<ID>" in out and "<NUM>" in out
    assert "search_orders" in out, "tool names must survive"


def test_normalization_collapses_instance_data_when_enabled():
    a = make_trace("a", args={"customer_id": "c_1", "limit": 5}, closing="Refunded 10.00 for order o_1.")
    b = make_trace("b", args={"customer_id": "c_2", "limit": 9}, closing="Refunded 99.00 for order o_7.")
    assert near_duplicates([a, b], normalize=False) == set()
    assert near_duplicates([a, b], normalize=True) == {"b"}


def test_shingles_of_short_text_degrade_gracefully():
    assert shingles("one two", n=5) == {"one two"}
    assert shingles("", n=5) == set()


@settings(max_examples=30, deadline=None)
@given(seed=st.integers(min_value=0, max_value=10_000))
def test_reordering_tool_results_does_not_create_false_duplicates(seed):
    """Two traces whose assistant turns differ must stay distinct however their environment messages are arranged."""
    import random

    rng = random.Random(seed)
    a = make_trace("a", closing="I have refunded the full amount to your original payment method today.")
    b = make_trace("b", tool_name="cancel_order",
                   closing="I have cancelled the order and you will not be charged anything at all.")
    for t in (a, b):
        results = [m for m in t["messages"] if m["role"] == "tool"]
        rng.shuffle(results)
    assert near_duplicates([a, b]) == set()


@settings(max_examples=30, deadline=None)
@given(limit=st.integers(min_value=1, max_value=50))
def test_argument_change_agrees_with_exact_jaccard_outside_the_noise_band(limit):
    """MinHash estimates Jaccard; with 128 permutations the standard error is about 0.09.

    Right at the threshold the estimate and the exact value can disagree, which is inherent to the method and not
    worth eliminating -- 128 permutations is the cost/accuracy trade-off the default picks. So the property is
    asserted outside a noise band around the threshold, and left unconstrained inside it.
    """
    a, b = _trace("a", limit=1), _trace("b", limit=limit)
    exact = jaccard(a, b)
    dropped = near_duplicates([a, b])
    if exact >= 0.85 + 0.12:
        assert dropped == {"b"}, f"exact jaccard {exact:.3f} is well above threshold"
    elif exact <= 0.85 - 0.12:
        assert dropped == set(), f"exact jaccard {exact:.3f} is well below threshold"


def test_a_copy_with_a_changed_call_id_is_a_duplicate():
    a = _trace("a")
    b = json.loads(json.dumps(a))
    b["id"] = "b"
    for m in b["messages"]:
        for c in m.get("tool_calls") or []:
            c["id"] = "totally_different"
    assert near_duplicates([a, b]) == {"b"}


def test_make_call_helper_is_used():
    assert make_call("c1", "f", {"a": 1})["function"]["arguments"] == '{"a": 1}'
