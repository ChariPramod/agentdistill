"""The batched lockstep runner against the sequential harness, and `run_eval` over both.

Both production paths drive one `TaskStepper`, so agreeing with each other is a consistency check, not a proof:
the proof is `tests/test_lockstep_equivalence.py`, which holds the runner to an independent oracle. What this
file is still for is everything the oracle cannot see -- the grader label, `replay_stats`, the rows and metrics
`run_eval` writes, and the report's refusal to price an unbatched cost.

The synthetic corpus and the student live in `tests/lockstep_corpus.py`, shared with the oracle tests so both
files exercise the same trajectories.
"""

from __future__ import annotations

import json
import random
from types import SimpleNamespace

import pytest
from hypothesis import given
from hypothesis import strategies as st

from agentdistill.eval.clients import VllmOfflineTurnClient
from agentdistill.eval.harness import run_task
from agentdistill.eval.lockstep import REPLY_INDEX, run_lockstep
from agentdistill.eval.replay import ReplayToolProvider
from agentdistill.eval.runner import RunSpec, label_grader, lockstep_eligible, run_eval
from tests.lockstep_corpus import KINDS, TOOL, FakeBatchClient, Sequential, items, make_task, task_index

FIELDS = ("messages", "n_turns", "n_tool_calls", "diverged", "divergence", "completion_tokens_est", "stop_reason",
          "schema_valid", "final_text", "replay_stats")


def test_forty_tasks_give_identical_outcomes_under_both_runners():
    traces = [make_task(i) for i in range(40)]
    sequential = [run_task(t, Sequential(**KINDS), ReplayToolProvider(t), max_turns=6) for t in traces]
    client = FakeBatchClient(**KINDS)
    batched, stats = run_lockstep(items(traces), client, batch_size=40, max_turns=6)

    assert len(batched) == 40
    for s, b in zip(sequential, batched, strict=True):
        assert (s.task_id, s.repeat_idx) == (b.task_id, b.repeat_idx)
        for name in FIELDS:
            assert getattr(s, name) == getattr(b, name), f"{s.task_id}: {name} differs"
        assert label_grader(traces[int(s.task_id[1:])], s) == label_grader(traces[int(b.task_id[1:])], b)

    # Every stop reason is exercised, so equality is not vacuous.
    assert {o.stop_reason for o in batched} == {"answered", "diverged", "max_turns"}
    assert any("not valid JSON" in json.dumps(o.messages) for o in batched)
    assert stats.batched
    # One call per step, not per turn: the deepest task sets the number of steps.
    assert stats.calls == len(client.batch_sizes) == max(o.n_turns for o in batched)
    assert stats.calls * 5 <= stats.item_turns


def test_a_client_without_a_batched_method_is_driven_per_item_and_not_called_batched():
    traces = [make_task(i) for i in range(6)]
    outcomes, stats = run_lockstep(items(traces), Sequential(), batch_size=3)
    assert len(outcomes) == 6
    assert not stats.batched
    assert stats.calls == stats.item_turns


def test_a_short_or_malformed_batch_is_refused():
    class Short(FakeBatchClient):
        def next_turns_batch(self, requests):
            return super().next_turns_batch(requests)[:-1]

    class NotDicts(FakeBatchClient):
        def next_turns_batch(self, requests):
            return [json.dumps(r) for r in super().next_turns_batch(requests)]

    traces = [make_task(i) for i in range(3)]
    with pytest.raises(AssertionError, match="2 replies for 3 requests"):
        run_lockstep(items(traces), Short(), batch_size=3)
    with pytest.raises(AssertionError, match="not an assistant dict"):
        run_lockstep(items(traces), NotDicts(), batch_size=3)


def test_batch_size_must_be_positive():
    with pytest.raises(ValueError, match="at least 1"):
        run_lockstep(items([make_task(0)]), FakeBatchClient(), batch_size=0)


# --------------------------------------------------------------------------------------------------------------
# the vLLM client's order check
# --------------------------------------------------------------------------------------------------------------


class _Tok:
    def apply_chat_template(self, messages, tools=None, tokenize=False, add_generation_prompt=True):
        return json.dumps({"m": messages, "t": tools})


class _FakeLLM:
    """Echoes a per-prompt answer. `shuffle` returns the outputs rotated, as an engine that lost order would."""

    def __init__(self, shuffle: bool = False) -> None:
        self.shuffle, self.calls = shuffle, 0

    def generate(self, prompts, sp, lora_request=None):
        self.calls += 1
        outs = [SimpleNamespace(prompt=p, outputs=[SimpleNamespace(text=f"answer for {task_index(json.loads(p)['m'])}",
                                                                   token_ids=[], logprobs=None)])
                for p in prompts]
        return outs[1:] + outs[:1] if self.shuffle else outs


def _vllm(shuffle: bool = False) -> VllmOfflineTurnClient:
    client = object.__new__(VllmOfflineTurnClient)
    client.llm, client.lora, client.sp, client.sample_sp = _FakeLLM(shuffle), None, None, None
    client.logprobs, client.n_samples = False, 0
    client.tok, client.parser_name, client.family = _Tok(), None, None
    return client


def test_vllm_batch_returns_one_turn_per_prompt_in_order_from_one_generate_call():
    client = _vllm()
    requests = [([{"role": "user", "content": f"task {i}: x"}], [TOOL]) for i in range(5)]
    turns = client.next_turns_batch(requests)
    assert [t["content"] for t in turns] == [f"answer for {i}" for i in range(5)]
    assert client.llm.calls == 1
    # Each reply names the prompt it answers, which is what the runner checks before it zips.
    assert [t[REPLY_INDEX] for t in turns] == list(range(5))
    one = client.next_turn(*requests[2])
    assert one[REPLY_INDEX] == 0, "a batch of one answers prompt 0 of that batch"
    assert {k: v for k, v in one.items() if k != REPLY_INDEX} == \
           {k: v for k, v in turns[2].items() if k != REPLY_INDEX}, \
           "next_turn is a batch of one, so it parses identically"


def test_vllm_batch_detects_outputs_returned_out_of_order():
    client = _vllm(shuffle=True)
    requests = [([{"role": "user", "content": f"task {i}: x"}], []) for i in range(3)]
    with pytest.raises(AssertionError, match="batch order was not preserved"):
        client.next_turns_batch(requests)


def test_the_vllm_client_labels_every_reply_with_its_prompt_index():
    """The two order checks are independent. `_in_order` compares each output with the prompt it was generated
    from; `REPLY_INDEX` is what survives into the runner, which never sees the prompts."""
    client = _vllm()
    requests = [([{"role": "user", "content": f"task {i}: x"}], []) for i in range(3)]
    assert [t[REPLY_INDEX] for t in client.next_turns_batch(requests)] == [0, 1, 2]


# --------------------------------------------------------------------------------------------------------------
# run_eval
# --------------------------------------------------------------------------------------------------------------


@pytest.fixture
def forty(registry):
    traces = [make_task(i) for i in range(40)]
    registry.insert_traces(traces)
    registry.insert_eval_set({"id": "es_b", "name": "batched", "trace_ids": [t["id"] for t in traces],
                              "grader": {"type": "label"}})
    return registry, {t["id"]: t for t in traces}


VOLATILE = ("id", "eval_run_id", "latency_ms", "created_at")


def _content(rows: list[dict]) -> list[dict]:
    return [{k: v for k, v in r.items() if k not in VOLATILE} for r in rows]


def test_run_eval_rows_are_identical_under_both_paths(forty):
    registry, traces = forty
    spec = RunSpec(subject="s", eval_set="batched", n_per_task=2, max_turns=6)
    es = registry.get_eval_set("batched")
    seq = run_eval(registry, es, traces, Sequential(**KINDS), label_grader, spec)
    client = FakeBatchClient(**KINDS)
    bat = run_eval(registry, es, traces, client, label_grader, spec, batch_size=16)

    assert _content(registry.eval_results(seq)) == _content(registry.eval_results(bat))
    m_seq, m_bat = registry.get_eval_run(seq)["metrics"], registry.get_eval_run(bat)["metrics"]
    for key in ("success", "divergence_rate", "schema_valid", "turns_median", "tokens_est_median", "stop_reasons"):
        assert m_seq[key] == m_bat[key]
    assert max(client.batch_sizes) == 16

    assert m_seq["throughput_mode"] == "unbatched"
    # Sequential throughput divides by per-row wall clock, which rounds to zero on a fast fake, so it may be absent.
    assert "unbatched" in m_seq.get("throughput_conditions", "unbatched")
    assert m_bat["throughput_mode"] == "batched"
    assert m_bat["throughput_conditions"] == "batched lockstep eval, batch=16"
    assert m_bat["throughput_tok_per_s"] > 0
    assert m_bat["lockstep"]["calls"] < m_bat["lockstep"]["item_turns"]


def test_run_eval_falls_back_to_sequential_for_clients_that_cannot_batch_safely(forty):
    registry, traces = forty
    spec = RunSpec(subject="s", eval_set="batched", n_per_task=1, max_turns=6)

    class Stateful(FakeBatchClient):
        """Per-task state (`reset`) would be scrambled by interleaving tasks."""

        def reset(self):
            pass

    assert not lockstep_eligible(Sequential())
    assert not lockstep_eligible(Stateful())
    assert lockstep_eligible(FakeBatchClient())
    stateful = Stateful()
    for client in (Sequential(), stateful):
        run = run_eval(registry, registry.get_eval_set("batched"), traces, client, label_grader, spec, batch_size=8)
        metrics = registry.get_eval_run(run)["metrics"]
        assert metrics["throughput_mode"] == "unbatched"
        assert metrics["n_rows"] == 40
    assert stateful.batch_sizes == [], "a stateful client must never be batched"


def test_progress_counts_every_item_on_the_batched_path(forty):
    registry, traces = forty
    seen: list[tuple[int, int]] = []
    run_eval(registry, registry.get_eval_set("batched"), traces, FakeBatchClient(), label_grader,
             RunSpec(subject="s", eval_set="batched", n_per_task=1), batch_size=7,
             progress=lambda d, t: seen.append((d, t)))
    assert seen[-1] == (40, 40)
    assert [d for d, _ in seen] == list(range(1, 41))


def test_random_batch_sizes_never_change_the_rows(forty):
    registry, traces = forty
    spec = RunSpec(subject="s", eval_set="batched", n_per_task=1, max_turns=6)
    es = registry.get_eval_set("batched")
    reference = _content(registry.eval_results(run_eval(registry, es, traces, Sequential(**KINDS), label_grader,
                                                         spec)))
    rng = random.Random(0)
    for size in rng.sample(range(1, 41), 3):
        run = run_eval(registry, es, traces, FakeBatchClient(**KINDS), label_grader, spec, batch_size=size)
        assert _content(registry.eval_results(run)) == reference, f"batch={size}"


# --------------------------------------------------------------------------------------------------------------
# the report's rule: no saving from an unbatched cost
# --------------------------------------------------------------------------------------------------------------


def _unclaimable(text: str) -> list[str]:
    return [line for line in text.splitlines() if "saving" in line.lower() or "break-even" in line.lower()]


_cascade = st.fixed_dictionaries({
    "threshold": st.floats(0, 1), "success": st.floats(0, 1), "escalation_rate": st.floats(0, 1),
    "cost_per_task": st.floats(0, 1), "saving_frac": st.none() | st.floats(-1, 1),
    "breakeven_tasks_per_day": st.none() | st.floats(0, 1e6), "measured": st.just(True),
})


@given(cascade=_cascade, mode=st.sampled_from(["unbatched", None]), with_code=st.booleans(),
       conditions=st.sampled_from(["sequential eval harness, one request at a time (unbatched; overstates cost)",
                                   "unstated", "gateway replay, concurrency 1"]))
def test_no_renderer_prints_a_saving_or_break_even_from_an_unbatched_cost(cascade, mode, with_code, conditions):
    from agentdistill.report.assemble import ReportData
    from agentdistill.report.html import render
    from agentdistill.report.markdown import full_markdown, results_block

    cost = {"teacher_cost_per_task": 0.02, "student_cost_per_mtok": 1.0, "throughput_tok_per_s": 50.0,
            "throughput_conditions": conditions, "cascade": cascade}
    if mode:
        cost["throughput_mode"] = mode
    report = ReportData(generated_at="t", project="p", eval_set="e", cost=cost,
                        subjects={"student": {"run_id": "ev_s", "success": 0.5, "schema_valid": 1.0,
                                              "divergence_rate": 0.0, "tokens_median": 100}})
    if with_code:
        report.warn("cost_unbatched", "student throughput was measured unbatched")
    for name, text in {"block": results_block(report), "markdown": full_markdown(report),
                       "html": render(report)}.items():
        assert not _unclaimable(text), f"{name} priced an unbatched cascade: {_unclaimable(text)}"
        assert "escalation" in text.lower()
        assert conditions in text or conditions.replace("“", "") in text


def test_a_real_unbatched_student_run_raises_cost_unbatched_and_prints_no_saving(registry, project_config):
    from sqlalchemy import text

    from agentdistill.config import TeacherConfig
    from agentdistill.registry.base import dumps
    from agentdistill.report.assemble import assemble
    from agentdistill.report.html import render
    from agentdistill.report.markdown import full_markdown, results_block
    from tests.test_report import seed

    project_config.eval.eval_set = "holdout"
    project_config.teacher = TeacherConfig(model="frontier-v1", provider="anthropic", input_per_mtok=3.0,
                                           output_per_mtok=15.0)
    seed(registry)
    metrics = registry.get_eval_run("ev_student")["metrics"]
    metrics.update(throughput_mode="unbatched",
                   throughput_conditions="sequential eval harness, one request at a time (unbatched; overstates cost)")
    with registry.engine.begin() as conn:
        conn.execute(text("UPDATE eval_runs SET metrics = :m WHERE id = 'ev_student'"), {"m": dumps(metrics)})

    r = assemble(registry, project_config, tag_glob="gpu-day")
    assert "cost_unbatched" in r.warning_codes
    assert "upper bound from sequential measurement" in r.warnings[r.warning_codes.index("cost_unbatched")]
    assert r.cost["throughput_mode"] == "unbatched"
    assert r.cost["cascade"]["saving_frac"] is None and r.cost["cascade"]["breakeven_tasks_per_day"] is None
    for rendered in (results_block(r), full_markdown(r), render(r)):
        assert not _unclaimable(rendered)
        assert "one request at a time" in rendered


def test_a_batched_student_run_prices_the_cascade(registry, project_config):
    from agentdistill.config import TeacherConfig
    from agentdistill.report.assemble import assemble
    from agentdistill.report.markdown import results_block
    from tests.test_report import seed

    project_config.eval.eval_set = "holdout"
    project_config.teacher = TeacherConfig(model="frontier-v1", provider="anthropic", input_per_mtok=3.0,
                                           output_per_mtok=15.0)
    r = assemble(seed(registry), project_config, tag_glob="gpu-day")
    assert "cost_unbatched" not in r.warning_codes
    assert r.cost["throughput_mode"] == "batched"
    assert "saving" in results_block(r) and "Break-even" in results_block(r)
