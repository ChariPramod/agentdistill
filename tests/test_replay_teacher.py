"""The replay teacher, and the run metrics that make the cost block executable.

The replay teacher is a tiny-mode stand-in. What matters is that it produces a teacher-shaped row -- usage, a
backend tag the report discloses -- and that it can answer a turn mid-cascade on a prefix it did not build.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agentdistill.eval.clients import LiteLLMTurnClient
from agentdistill.eval.fake_teacher import COMPLETION_TOKENS, PROMPT_TOKENS, ReplayTeacherClient, UnknownTask
from agentdistill.eval.runner import RunSpec, label_grader, run_eval, throughput_metrics, usage_metrics
from tests.test_eval_runner import make_eval_trace


@pytest.fixture
def traces(registry):
    ts = [make_eval_trace(i) for i in range(4)]
    registry.insert_traces(ts)
    registry.insert_eval_set({"id": "es_t", "name": "t", "trace_ids": [t["id"] for t in ts],
                              "grader": {"type": "label"}})
    return {t["id"]: t for t in ts}


def test_the_replay_teacher_reproduces_the_recording(registry, traces):
    client = ReplayTeacherClient(list(traces.values()))
    run_id = run_eval(registry, registry.get_eval_set("t"), traces, client, label_grader,
                      RunSpec(subject="teacher", eval_set="t", n_per_task=2))
    m = registry.get_eval_run(run_id)["metrics"]
    assert m["success"] == 1.0
    assert m["teacher_backend"] == "replay", "the report keys its disclosure on this"
    # Two assistant turns per trace, each billed at the fixed counts.
    assert m["prompt_tokens_median"] == 2 * PROMPT_TOKENS
    assert m["completion_tokens_median"] == 2 * COMPLETION_TOKENS


def test_it_answers_mid_conversation_on_a_prefix_it_did_not_build(traces):
    """In a cascade the teacher is asked for turn i on a prefix the student wrote."""
    trace = traces["task2"]
    client = ReplayTeacherClient(list(traces.values()))
    prefix = trace["messages"][:4]  # system, user, the student's first assistant turn, a tool result
    turn = client.next_turn(prefix, [])
    assert turn["content"] == "All done for case 2."


def test_an_unknown_task_is_an_error_not_an_invented_answer(traces):
    client = ReplayTeacherClient(list(traces.values()))
    with pytest.raises(UnknownTask):
        client.next_turn([{"role": "user", "content": "a task nobody recorded"}], [])


def test_rollouts_are_never_replayed_as_the_teacher(registry, traces):
    rollout = make_eval_trace(9)
    rollout.update(id="ro_1", source="rollout")
    registry.insert_traces([rollout])
    client = ReplayTeacherClient.from_registry(registry)
    with pytest.raises(UnknownTask):
        client.next_turn(rollout["messages"][:2], [])


def test_a_client_without_usage_records_no_token_medians(registry, traces):
    from tests.test_eval_runner import PerTaskRecorded

    run_id = run_eval(registry, registry.get_eval_set("t"), traces, PerTaskRecorded(traces), label_grader,
                      RunSpec(subject="x", eval_set="t", n_per_task=1))
    m = registry.get_eval_run(run_id)["metrics"]
    assert "prompt_tokens_median" not in m, "an estimate would understate the prompt; absent is honest"
    assert "teacher_backend" not in m


def test_every_run_records_throughput_and_its_conditions(registry, traces):
    from tests.test_eval_runner import PerTaskRecorded

    run_id = run_eval(registry, registry.get_eval_set("t"), traces, PerTaskRecorded(traces), label_grader,
                      RunSpec(subject="x", eval_set="t", n_per_task=1))
    m = registry.get_eval_run(run_id)["metrics"]
    if m.get("throughput_tok_per_s") is not None:
        assert "unbatched" in m["throughput_conditions"]


def test_throughput_from_rows():
    rows = [{"completion_tokens_est": 100, "latency_ms": 500}, {"completion_tokens_est": 300, "latency_ms": 1500}]
    t = throughput_metrics(rows)
    assert t["throughput_tok_per_s"] == pytest.approx(200.0)
    assert throughput_metrics([{"completion_tokens_est": 0, "latency_ms": 0}]) == {}


def test_usage_medians_and_cache_share():
    m = usage_metrics([{"prompt_tokens": 1000, "completion_tokens": 50, "cached_prompt_tokens": 500},
                       {"prompt_tokens": 3000, "completion_tokens": 70, "cached_prompt_tokens": 500}])
    assert m["prompt_tokens_median"] == 2000 and m["completion_tokens_median"] == 60
    assert m["cache_hit_frac"] == pytest.approx(0.25)
    assert usage_metrics([]) == {}


def test_the_litellm_teacher_accumulates_provider_usage():
    def completion(**kw):
        message = SimpleNamespace(content="ok", tool_calls=None)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, logprobs=None)],
            usage=SimpleNamespace(prompt_tokens=1500, completion_tokens=40,
                                  prompt_tokens_details=SimpleNamespace(cached_tokens=1000)),
        )

    client = LiteLLMTurnClient("some-model", completion=completion)
    client.next_turn([{"role": "user", "content": "hi"}], [])
    client.next_turn([{"role": "user", "content": "hi"}], [])
    assert client.usage == {"prompt_tokens": 3000, "completion_tokens": 80, "cached_prompt_tokens": 2000}
