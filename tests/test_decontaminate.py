"""Decontamination. The planted-task test is the one that must never regress."""

from __future__ import annotations

from agentdistill.curate.decontaminate import Decontaminator, contaminated_ids, ngrams, normalize_text, task_text
from tests.conftest import make_trace

EVAL_TASK = (
    "Please refund order 12345 for the customer who complained about the late shipment that arrived "
    "three weeks after the promised delivery date"
)


def test_planted_eval_task_is_always_removed():
    """An eval task planted verbatim in the training corpus must be dropped."""
    planted = make_trace("planted", task=EVAL_TASK)
    innocent = make_trace("innocent", task="What is the weather in Paris tomorrow afternoon?")
    hits = contaminated_ids([planted, innocent], [{"system": "s", "user": EVAL_TASK}])
    assert set(hits) == {"planted"}
    assert "exact" in hits["planted"]


def test_planted_task_removed_despite_case_and_punctuation():
    planted = make_trace("planted", task=EVAL_TASK.upper() + " !!!")
    hits = contaminated_ids([planted], [{"system": "s", "user": EVAL_TASK}])
    assert "planted" in hits


def test_partial_overlap_above_threshold_is_caught():
    train = EVAL_TASK + " and also tell me about the return policy for opened items"
    hits = contaminated_ids([make_trace("t", task=train)], [{"system": "s", "user": EVAL_TASK}], overlap=0.5)
    assert "t" in hits


def test_unrelated_task_is_kept():
    hits = contaminated_ids(
        [make_trace("t", task="How do I change the language setting in my account preferences?")],
        [{"system": "s", "user": EVAL_TASK}],
    )
    assert hits == {}


def test_system_prompt_is_excluded_from_matching():
    """Every task in a project shares a system prompt; matching on it would flag the whole corpus."""
    shared = "You are a support agent for an online store and must use the tools provided to you."
    dec = Decontaminator([{"system": shared, "user": EVAL_TASK}])
    hit, _ = dec.check({"system": shared, "user": "A completely unrelated question about gift cards."})
    assert not hit


def test_no_eval_set_means_no_drops():
    assert contaminated_ids([make_trace("t")], []) == {}
    assert not Decontaminator([])


def test_overlap_is_measured_against_the_training_trace():
    """A short training task contained in a long eval task is contamination.

    Normalizing by the eval task's length instead would bury a fully-contained short task: its overlap would look
    negligible against a long eval prompt, and the contaminated trace would survive.
    """
    short = "refund order 12345 for the customer who complained about the late shipment"
    assert short in EVAL_TASK, "the fixture must actually be contained for this property to be under test"
    long_eval = EVAL_TASK + " " + " ".join(f"extra context sentence number {i}" for i in range(40))
    hits = contaminated_ids([make_trace("t", task=short)], [{"system": "", "user": long_eval}])
    assert "t" in hits


def test_task_text_handles_every_shape():
    assert task_text(None) == ""
    assert task_text("plain") == "plain"
    assert task_text({"system": "s", "user": "u"}) == "u"
    assert "v" in task_text({"other": "v"})


def test_ngrams_of_short_text():
    assert ngrams(normalize_text("one two"), 8) == {"one two"}
    assert ngrams([], 8) == set()


def test_blank_task_input_is_never_contaminated():
    dec = Decontaminator([{"system": "s", "user": EVAL_TASK}])
    assert dec.check({"system": "s", "user": "   "}) == (False, "")
