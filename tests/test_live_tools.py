"""Live tools: grading a subject on what it did to the world, not on how closely it copied the recording.

The corpus was recorded from a scripted solver. Replay serves that recording's tool results and counts any other
call as a divergence, so a subject that looks a customer up by email where the script used an id fails -- not
because it got the task wrong, but because it is not the script. The example's CRM is local, deterministic and
seeded per task, so every subject can be run against the real thing and graded on the state it actually left.

These are the behaviours the GPU day's validity rests on, so each is named for the claim it protects.
"""

from __future__ import annotations

import pytest

from agentdistill.eval.live import LiveToolProvider, LiveToolsUnavailable, resolve_env_factory
from agentdistill.eval.replay import Divergence, ReplayToolProvider
from agentdistill.eval.runner import RunSpec, compare, eval_mode, run_eval, tool_mode

EXAMPLE_SOURCE = "examples.support_agent.replay_grader:grade_outcome"


def _recorded_traces(n: int = 3) -> list[dict]:
    """Real recorded traces from the example corpus, so the calls and the CRM agree by construction."""
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "examples" / "support_agent" / "eval-holdout.jsonl"
    if not path.exists():
        pytest.skip("the example corpus is generated; run scripts/make_corpus.sh")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [r for r in rows if r.get("success")][:n]


def _replay_the_recorded_calls(trace: dict, provider) -> None:
    """Execute exactly the calls the recording made, in order, against `provider`."""
    for m in trace["messages"]:
        for call in m.get("tool_calls") or []:
            import json

            args = call["function"]["arguments"]
            provider.lookup(call["function"]["name"], json.loads(args) if isinstance(args, str) else args)


def test_live_replays_scripted_calls_to_the_same_outcome():
    """If executing the recording's own calls did not reproduce its outcome, live grading would be measuring a
    different world than the one the corpus was recorded in."""
    from examples.support_agent.replay_grader import grade_outcome  # noqa: F401 - import proves the source loads

    factory = resolve_env_factory(EXAMPLE_SOURCE)
    for trace in _recorded_traces(3):
        provider = LiveToolProvider(trace, factory)
        _replay_the_recorded_calls(trace, provider)
        ok, detail = provider.grade(_final_text(trace))
        assert ok, f"{trace.get('task_id')}: the recording's own calls did not reach its outcome ({detail})"


def _final_text(trace: dict) -> str:
    return next((m.get("content") or "" for m in reversed(trace["messages"]) if m["role"] == "assistant"), "")


def test_live_mode_does_not_penalize_valid_alternative_calls():
    """The whole reason live mode exists: a lookup by email where the script used an id is a valid way to do the
    task, and replay counts it as a divergence."""
    trace = _recorded_traces(1)[0]
    call = next((c for m in trace["messages"] for c in (m.get("tool_calls") or [])), None)
    assert call is not None, "the fixture trace makes no tool call"

    import json

    args = json.loads(call["function"]["arguments"])
    alternative = dict(args)
    # Any argument the recording did not use makes this "not the recorded call" as far as replay is concerned.
    alternative["email"] = alternative.get("email") or "someone.else@example.com"
    if alternative == args:
        pytest.skip("this trace's first call cannot be varied without changing what it asks for")

    with pytest.raises(Divergence):
        ReplayToolProvider(trace, policy="strict").lookup(call["function"]["name"], alternative)

    # Live mode answers it: the environment knows this customer whatever key was used to find them.
    content = LiveToolProvider(trace, resolve_env_factory(EXAMPLE_SOURCE)).lookup(call["function"]["name"],
                                                                                 alternative)
    assert content, "live mode returned an empty result for a call the environment can answer"


def test_live_mode_needs_an_environment_and_says_which_key_configures_it():
    with pytest.raises(LiveToolsUnavailable, match="predicate_source"):
        tool_mode(RunSpec(subject="s", eval_set="e", tools="live", env_source=None), lambda t, o: (True, {}))


def test_a_tool_error_is_content_not_a_divergence():
    """The recording agent saw refusals as tool results; so must the student, or it never learns to handle one."""
    trace = _recorded_traces(1)[0]
    live = LiveToolProvider(trace, resolve_env_factory(EXAMPLE_SOURCE))
    call = next(c for m in trace["messages"] for c in (m.get("tool_calls") or []))
    content = live.lookup(call["function"]["name"], {"order_id": "o_does_not_exist"})
    assert content, "a refused call must come back as content, not as a divergence"


# --------------------------------------------------------------------------------------------------------------
# the mode is recorded, and runs from different modes are never compared
# --------------------------------------------------------------------------------------------------------------


@pytest.fixture
def populated(registry):
    """The eval-runner fixture's corpus: synthetic traces the CRM knows nothing about, which is fine here --
    these tests are about the recorded mode and the refusal to compare across modes, not about the tools."""
    from tests.test_eval_runner import make_eval_trace

    traces = [make_eval_trace(i) for i in range(4)]
    registry.insert_traces(traces)
    registry.insert_eval_set({"id": "es_test", "name": "test-set", "trace_ids": [t["id"] for t in traces],
                              "grader": {"type": "label"}})
    return registry, {t["id"]: t for t in traces}


def _run(registry, traces, mode: str, subject: str) -> str:
    from tests.test_eval_runner import PerTaskRecorded

    return run_eval(registry, registry.get_eval_set("test-set"), traces, PerTaskRecorded(traces),
                    lambda t, o: (True, {"detail": "stub"}),
                    RunSpec(subject=subject, eval_set="test-set", n_per_task=1, tools=mode,
                            env_source=EXAMPLE_SOURCE if mode == "live" else None))


def test_every_run_records_the_mode_it_was_graded_under(populated):
    registry, traces = populated
    run_id = _run(registry, traces, "replay", "a")
    assert registry.get_eval_run(run_id)["metrics"]["eval_mode"] == "replay"
    assert eval_mode(registry.get_eval_run(run_id)) == "replay"


def test_a_run_from_before_tool_modes_reads_as_replay():
    """Replay was the only mode that existed, so a run with no recorded mode is not unknown -- it is replay."""
    assert eval_mode({"metrics": {}}) == "replay"


def test_comparing_a_live_run_against_a_replay_run_is_refused(populated, monkeypatch):
    """The two modes answer different questions. A delta between them is not a delta at all."""
    registry, traces = populated
    replay_run = _run(registry, traces, "replay", "a")
    live_run = _run(registry, traces, "replay", "b")
    # Relabel the second run's mode rather than running live tools against synthetic traces the CRM never saw.
    run = registry.get_eval_run(live_run)
    registry.finish_eval_run(live_run, {**run["metrics"], "eval_mode": "live"})

    result = compare(registry, live_run, replay_run)
    marker = result["incompatible"]
    assert marker["code"] == "eval_mode_mismatch"
    assert {marker["eval_mode_a"], marker["eval_mode_b"]} == {"live", "replay"}
    for key in ("success", "mcnemar", "tokens", "turns", "holm"):
        assert key not in result, f"{key} was computed across two different questions"


def test_no_renderer_prints_statistics_for_an_incompatible_comparison(populated):
    from agentdistill.report.markdown import comparison_text

    registry, traces = populated
    a, b = _run(registry, traces, "replay", "a"), _run(registry, traces, "replay", "b")
    run = registry.get_eval_run(a)
    registry.finish_eval_run(a, {**run["metrics"], "eval_mode": "live"})
    text = comparison_text(compare(registry, a, b))
    assert "p=" not in text and "pp" not in text
    assert "live" in text and "replay" in text


def test_the_corpus_teacher_is_found_at_the_root_of_a_dpo_lineage(registry, tmp_path):
    """A DPO round's adapter trained on a pairs dataset, which records no corpus teacher. The corpus is the SFT
    dataset its parent trained on; the first live rehearsal reported "not recorded" because it stopped at the
    pairs and never looked up the chain."""
    import json

    from agentdistill.report.assemble import _corpus_teacher
    from agentdistill.report.registry_views import lineage

    for ds_id, teacher in (("ds_sft", "scripted/rule-based-teacher"), ("ds_pairs", None)):
        path = tmp_path / ds_id
        path.mkdir()
        (path / "manifest.json").write_text(json.dumps({"corpus_teacher": teacher} if teacher else {}))
        registry.insert_dataset({"id": ds_id, "name": ds_id, "version": 1, "kind": "sft", "filter_config": {},
                                 "n_samples": 1, "n_tokens": 1, "content_hash": ds_id, "path": str(path)})
    for run_id, ds_id in (("tr_sft", "ds_sft"), ("tr_dpo", "ds_pairs")):
        registry.insert_training_run({"id": run_id, "dataset_id": ds_id, "base_model": "m", "method": "sft",
                                      "config": {}, "status": "succeeded",
                                      "started_at": "2026-09-21T00:00:00+00:00"})
    registry.insert_adapter({"id": "ad_sft", "training_run_id": "tr_sft", "name": "s", "version": 1,
                             "base_model": "m", "path": "/tmp/a"})
    registry.insert_adapter({"id": "ad_dpo", "training_run_id": "tr_dpo", "name": "s-dpo", "version": 1,
                             "base_model": "m", "path": "/tmp/b", "parent_adapter_id": "ad_sft"})

    assert _corpus_teacher(registry, lineage(registry, "ad_dpo")) == "scripted/rule-based-teacher"
