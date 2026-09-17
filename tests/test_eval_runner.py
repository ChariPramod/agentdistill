"""Eval runs, comparisons, and the replay grader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentdistill.eval.clients import RecordedTurnClient, ScriptedTurnClient
from agentdistill.eval.harness import run_task
from agentdistill.eval.replay import ReplayToolProvider
from agentdistill.eval.report import per_task_table, render_comparison, render_run, summarize_divergences
from agentdistill.eval.runner import RunSpec, aggregate, compare, label_grader, run_eval, weakest_clusters
from agentdistill.eval.stats import TooFewTasks
from agentdistill.ingest.normalize import normalize_trace
from tests.conftest import make_call

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "support_agent"


def make_eval_trace(i: int, tool_result: str = '{"customer_id": "c_1"}') -> dict:
    raw = {
        "id": f"task{i}",
        "task_id": f"task{i}",
        "success": True,
        "tools": [{"type": "function", "function": {"name": "get_customer",
                                                    "parameters": {"type": "object",
                                                                   "properties": {"email": {"type": "string"}},
                                                                   "required": ["email"]}}}],
        "messages": [
            {"role": "system", "content": "You are support."},
            {"role": "user", "content": f"help me with case {i}, a{i}@b.com"},
            {"role": "assistant", "content": "Looking.",
             "tool_calls": [make_call("c1", "get_customer", {"email": f"a{i}@b.com"})]},
            {"role": "tool", "tool_call_id": "c1", "content": tool_result},
            {"role": "assistant", "content": f"All done for case {i}.", "tool_calls": None},
        ],
    }
    trace = normalize_trace(raw, source="jsonl")
    trace["id"] = f"task{i}"
    trace["cluster"] = i % 3
    return trace


@pytest.fixture
def populated(registry):
    traces = [make_eval_trace(i) for i in range(12)]
    registry.insert_traces(traces)
    registry.insert_eval_set({
        "id": "es_test", "name": "test-set", "trace_ids": [t["id"] for t in traces],
        "grader": {"type": "label"},
    })
    return registry, {t["id"]: t for t in traces}


class PerTaskRecorded:
    """A client that replays whichever task it is currently being asked about."""

    def __init__(self, traces_by_task: dict[str, dict]) -> None:
        self.traces = traces_by_task
        self.current: RecordedTurnClient | None = None

    def next_turn(self, messages, tools):
        if self.current is None:
            user = next((m["content"] for m in messages if m["role"] == "user"), "")
            trace = next(t for t in self.traces.values()
                         if t["messages"][1]["content"] == user)
            self.current = RecordedTurnClient(trace)
        return self.current.next_turn(messages, tools)

    def reset(self):
        self.current = None


# --------------------------------------------------------------------------------------------------------------
# running
# --------------------------------------------------------------------------------------------------------------


def test_run_eval_stores_a_row_per_repeat(populated):
    registry, traces = populated
    run_id = run_eval(registry, registry.get_eval_set("test-set"), traces, PerTaskRecorded(traces),
                      label_grader, RunSpec(subject="recorded", eval_set="test-set", n_per_task=3))
    rows = registry.eval_results(run_id)
    assert len(rows) == 12 * 3
    assert {r["repeat_idx"] for r in rows} == {0, 1, 2}


def test_run_eval_records_metrics(populated):
    registry, traces = populated
    run_id = run_eval(registry, registry.get_eval_set("test-set"), traces, PerTaskRecorded(traces),
                      label_grader, RunSpec(subject="recorded", eval_set="test-set", n_per_task=2))
    run = registry.get_eval_run(run_id)
    assert run["metrics"]["success"] == 1.0, "replaying the recording must reproduce it"
    assert run["metrics"]["divergence_rate"] == 0.0
    assert run["metrics"]["n_tasks"] == 12
    assert run["per_cluster"], "per-cluster breakdown drives the next curation round"


def test_replay_stats_are_per_repeat_not_cumulative(populated):
    """A shared provider across repeats would report a divergence rate several times too high."""
    registry, traces = populated
    run_id = run_eval(registry, registry.get_eval_set("test-set"), traces, PerTaskRecorded(traces),
                      label_grader, RunSpec(subject="recorded", eval_set="test-set", n_per_task=3))
    for row in registry.eval_results(run_id):
        assert row["replay_stats"]["replayed"] == 1, "one call per repeat, not an accumulating count"


def test_run_eval_can_skip_storing_messages(populated):
    registry, traces = populated
    run_id = run_eval(registry, registry.get_eval_set("test-set"), traces, PerTaskRecorded(traces),
                      label_grader, RunSpec(subject="recorded", eval_set="test-set", n_per_task=1),
                      store_messages=False)
    assert all(r["messages"] is None for r in registry.eval_results(run_id))


def test_aggregate_weights_tasks_equally():
    """Row-level averaging would let a task with more repeats count more."""
    rows = [
        {"task_id": "a", "success": True, "schema_valid": True, "diverged": False, "n_turns": 2,
         "n_tool_calls": 1, "completion_tokens_est": 10, "latency_ms": 1, "stop_reason": "answered",
         "replay_stats": {}},
        {"task_id": "a", "success": True, "schema_valid": True, "diverged": False, "n_turns": 2,
         "n_tool_calls": 1, "completion_tokens_est": 10, "latency_ms": 1, "stop_reason": "answered",
         "replay_stats": {}},
        {"task_id": "b", "success": False, "schema_valid": True, "diverged": False, "n_turns": 2,
         "n_tool_calls": 1, "completion_tokens_est": 10, "latency_ms": 1, "stop_reason": "answered",
         "replay_stats": {}},
    ]
    metrics, _ = aggregate(rows, {"a": {}, "b": {}})
    assert metrics["success"] == 0.5, "two tasks, one succeeded"


def test_aggregate_on_no_rows():
    metrics, per_cluster = aggregate([], {})
    assert metrics["n_rows"] == 0 and per_cluster == {}


# --------------------------------------------------------------------------------------------------------------
# comparing
# --------------------------------------------------------------------------------------------------------------


def _run_two(registry, traces, degrade: bool):
    """A second run whose client fails a third of the tasks, for a comparison with a real effect."""

    class Degraded(PerTaskRecorded):
        def next_turn(self, messages, tools):
            user = next((m["content"] for m in messages if m["role"] == "user"), "")
            if degrade and any(f"case {i}," in user for i in (0, 1, 2, 3, 4, 5, 6)):
                return {"role": "assistant", "content": "I cannot help with that.", "tool_calls": None}
            return super().next_turn(messages, tools)

    return run_eval(registry, registry.get_eval_set("test-set"), traces, Degraded(traces), label_grader,
                    RunSpec(subject="degraded" if degrade else "control", eval_set="test-set", n_per_task=3))


def test_compare_detects_a_real_difference(populated):
    registry, traces = populated
    good = _run_two(registry, traces, degrade=False)
    bad = _run_two(registry, traces, degrade=True)
    result = compare(registry, good, bad)
    assert result["success"]["delta"] > 0, "the control should beat the degraded subject"
    assert result["success"]["excludes_zero"]
    assert result["n_shared_tasks"] == 12
    # Exact McNemar on d discordant tasks cannot go below 2 / 2**d. With 7 degraded tasks the floor is 0.016,
    # so a p under 0.05 is attainable; with 4 it would be 0.125 and no amount of effect size would help.
    assert result["mcnemar"]["n_discordant"] == 7
    assert result["mcnemar"]["p"] < 0.05


def test_compare_reports_no_difference_between_identical_runs(populated):
    registry, traces = populated
    a = _run_two(registry, traces, degrade=False)
    b = _run_two(registry, traces, degrade=False)
    result = compare(registry, a, b)
    assert result["success"]["delta"] == 0.0
    assert not result["success"]["excludes_zero"]


def test_compare_refuses_runs_from_different_eval_sets(populated, registry):
    registry, traces = populated
    registry.insert_eval_set({"id": "es_other", "name": "other", "trace_ids": list(traces)[:9],
                              "grader": {"type": "label"}})
    a = _run_two(registry, traces, degrade=False)
    b = run_eval(registry, registry.get_eval_set("other"), traces, PerTaskRecorded(traces), label_grader,
                 RunSpec(subject="x", eval_set="other", n_per_task=1))
    with pytest.raises(ValueError, match="different eval sets"):
        compare(registry, a, b)


def test_compare_refuses_an_eval_set_too_small(registry):
    traces = [make_eval_trace(i) for i in range(3)]
    registry.insert_traces(traces)
    registry.insert_eval_set({"id": "es_tiny", "name": "tiny", "trace_ids": [t["id"] for t in traces],
                              "grader": {"type": "label"}})
    by_task = {t["id"]: t for t in traces}
    spec = RunSpec(subject="s", eval_set="tiny", n_per_task=1)
    a = run_eval(registry, registry.get_eval_set("tiny"), by_task, PerTaskRecorded(by_task), label_grader, spec)
    b = run_eval(registry, registry.get_eval_set("tiny"), by_task, PerTaskRecorded(by_task), label_grader, spec)
    with pytest.raises(TooFewTasks, match="at least 8"):
        compare(registry, a, b)


def test_compare_unknown_run(populated):
    registry, _ = populated
    with pytest.raises(ValueError, match="unknown eval run"):
        compare(registry, "nope", "also-nope")


def test_weakest_clusters_sorts_by_deficit():
    a = {"0": {"n_tasks": 5, "success": 0.4}, "1": {"n_tasks": 5, "success": 0.9}}
    b = {"0": {"n_tasks": 5, "success": 0.8}, "1": {"n_tasks": 5, "success": 0.85}}
    weak = weakest_clusters(a, b)
    assert weak[0]["cluster"] == "0"
    assert weak[0]["delta"] == pytest.approx(-0.4)


# --------------------------------------------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------------------------------------------


def test_comparison_report_contains_every_required_number(populated):
    registry, traces = populated
    result = compare(registry, _run_two(registry, traces, False), _run_two(registry, traces, True))
    text = render_comparison(result)
    for needle in ["Task success", "95% CI", "McNemar", "Schema validity", "Divergence rate",
                   "Tokens / task", "Turns / task", "Holm-corrected", "VERDICT"]:
        assert needle in text, f"missing {needle!r} from the report"


def test_report_verdict_is_explicit_about_an_inconclusive_result(populated):
    registry, traces = populated
    result = compare(registry, _run_two(registry, traces, False), _run_two(registry, traces, False))
    text = render_comparison(result)
    assert "includes zero" in text
    assert "not the same as showing they are equivalent" in text


def test_run_report_renders(populated):
    registry, traces = populated
    run_id = _run_two(registry, traces, degrade=False)
    text = render_run(registry.get_eval_run(run_id))
    assert "success" in text and "divergence rate" in text


def test_per_task_table_flags_failures(populated):
    registry, traces = populated
    rows = registry.eval_results(_run_two(registry, traces, degrade=True))
    table = per_task_table(rows)
    assert len(table) == 12
    failures = [row for row in table if row[1].startswith("0/")]
    assert failures, "the degraded run should have failing tasks"


def test_divergence_summary_groups_by_tool():
    rows = [
        {"divergence": {"tool": "get_customer", "nearest_score": 0.5, "args": {"email": "x"}}},
        {"divergence": {"tool": "get_customer", "nearest_score": 0.9, "args": {"email": "y"}}},
        {"divergence": None},
    ]
    summary = summarize_divergences(rows)
    assert len(summary) == 1
    assert summary[0]["tool"] == "get_customer" and summary[0]["n"] == 2
    assert summary[0]["max_score"] == 0.9


# --------------------------------------------------------------------------------------------------------------
# the replay predicate, against the example corpus
# --------------------------------------------------------------------------------------------------------------


def _corpus(name: str) -> list[dict]:
    path = EXAMPLE / name
    if not path.exists():
        pytest.skip(f"{name} not generated; run examples/support_agent/record.py")
    return [json.loads(line) for line in path.open()]


@pytest.mark.parametrize("corpus", ["eval-holdout.jsonl", "eval-unseen.jsonl"])
def test_replay_predicate_matches_live(corpus):
    """Reconstructing the end state from the recorded calls must reproduce the label recorded live.

    This is what catches nondeterminism in the example's own state: it previously failed because tracking numbers
    were built from Python's per-process-randomized `hash()`.
    """
    from examples.support_agent.replay_grader import calls_from_trace, predicate_from_calls

    traces = _corpus(corpus)
    mismatches = []
    for trace in traces:
        final = next((m.get("content") or "" for m in reversed(trace["messages"]) if m["role"] == "assistant"), "")
        ok, detail, _ = predicate_from_calls(trace, calls_from_trace(trace), final)
        if ok != bool(trace["success"]):
            mismatches.append((trace["metadata"]["scenario"], trace["success"], ok, detail))
    assert not mismatches, f"{len(mismatches)}/{len(traces)} disagree: {mismatches[:3]}"


def test_replay_grader_grades_a_harness_outcome():
    from examples.support_agent.replay_grader import grade_outcome

    traces = _corpus("eval-holdout.jsonl")
    trace = next(t for t in traces if t["success"])
    outcome = run_task(trace, RecordedTurnClient(trace), ReplayToolProvider(trace))
    success, detail = grade_outcome(trace, outcome)
    assert success is True
    assert detail["scenario"] == trace["metadata"]["scenario"]
    assert detail["n_calls_applied"] >= 1


def test_replay_grader_fails_a_student_that_did_nothing():
    from examples.support_agent.replay_grader import grade_outcome

    traces = _corpus("eval-holdout.jsonl")
    trace = next(t for t in traces if t["success"] and t["metadata"]["scenario"].startswith("refund"))
    outcome = run_task(trace, ScriptedTurnClient([{"role": "assistant", "content": "All sorted!"}]),
                       ReplayToolProvider(trace))
    success, _ = grade_outcome(trace, outcome)
    assert success is False, "claiming success without acting must not grade as success"


def test_replay_grader_rejects_a_trace_without_scenario_metadata():
    from examples.support_agent.replay_grader import UngradeableTrace, task_for_trace

    with pytest.raises(UngradeableTrace, match="db_seed"):
        task_for_trace({"id": "x", "metadata": {}})


# --------------------------------------------------------------------------------------------------------------
# what `teacher` resolves to
#
# It used to fall through to the local-model branch and evaluate `train.base_model` under the label "teacher",
# which turns the single most important comparison in the report -- student against teacher -- into a comparison
# of the student against the base model. A wrong baseline is worse than a missing one.
# --------------------------------------------------------------------------------------------------------------


def _project(tmp_path, teacher: dict | None) -> str:
    import yaml

    body = {
        "name": "t",
        "registry": f"sqlite:///{tmp_path}/registry.db",
        "artifacts": str(tmp_path / "artifacts"),
        "reports": str(tmp_path / "reports"),
        "train": {"base_model": "some/base-model", "max_seq_len": 512},
        "dataset": {"max_seq_len": 512},
    }
    if teacher:
        body["teacher"] = teacher
    path = tmp_path / "project.yaml"
    path.write_text(yaml.safe_dump(body))
    return str(path)


def test_the_teacher_subject_calls_the_teacher_not_the_base_model(tmp_path):
    from agentdistill.cli import _resolve_client
    from agentdistill.config import ProjectConfig
    from agentdistill.eval.clients import LiteLLMTurnClient

    cfg = ProjectConfig.load(_project(tmp_path, {"model": "anthropic/claude-sonnet-5"}))
    client = _resolve_client("teacher", cfg, backend="hf")

    assert isinstance(client, LiteLLMTurnClient)
    assert client.model == "anthropic/claude-sonnet-5"
    assert client.model != cfg.base_model


def test_no_teacher_configured_refuses_rather_than_substituting_the_base_model(tmp_path):
    import typer

    from agentdistill.cli import _resolve_client
    from agentdistill.config import ProjectConfig

    cfg = ProjectConfig.load(_project(tmp_path, None))
    with pytest.raises(typer.Exit):
        _resolve_client("teacher", cfg, backend="hf")


def test_the_teacher_client_sends_tools_and_reads_a_tool_call_back():
    from agentdistill.eval.clients import LiteLLMTurnClient

    seen = {}

    class Fn:
        name, arguments = "search_orders", '{"q": "A1"}'

    class Call:
        id, function = "c1", Fn()

    class Message:
        content, tool_calls = None, [Call()]

    class Choice:
        message, logprobs = Message(), None

    class Response:
        choices = [Choice()]

    def fake_completion(**kwargs):
        seen.update(kwargs)
        return Response()

    client = LiteLLMTurnClient("anthropic/claude-sonnet-5", completion=fake_completion)
    tools = [{"type": "function", "function": {"name": "search_orders", "parameters": {}}}]
    turn = client.next_turn([{"role": "user", "content": "where is A1"}], tools)

    assert seen["model"] == "anthropic/claude-sonnet-5"
    assert seen["tools"] == tools
    # Greedy, like every other eval client, so a teacher baseline is reproducible.
    assert seen["temperature"] == 0.0
    assert turn["tool_calls"][0]["function"]["name"] == "search_orders"


def test_a_teacher_that_reports_no_logprobs_is_still_a_valid_baseline():
    """Not every provider supports them; asking and tolerating their absence beats failing the run."""
    from agentdistill.eval.clients import LiteLLMTurnClient

    class Message:
        content, tool_calls = "done", None

    class Choice:
        message, logprobs = Message(), None

    class Response:
        choices = [Choice()]

    client = LiteLLMTurnClient("x/y", logprobs=True, completion=lambda **kw: Response())
    turn = client.next_turn([{"role": "user", "content": "hi"}], [])
    assert turn["content"] == "done"
    assert "logprobs" not in turn
