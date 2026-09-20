"""The on-policy round in the report.

A discarded round is a result: one round of RFT plus DPO did not beat its starting adapter on this data. If the
report only showed rounds that promoted, a discard would look like a stage that never ran, so both decisions are
rendered, with the recorded reason and the pair statistics that let the cap be checked after the fact.
"""

from __future__ import annotations

import pytest

from agentdistill.config import TeacherConfig
from agentdistill.eval.stats import InsufficientPower
from agentdistill.report.assemble import assemble
from agentdistill.report.html import render
from agentdistill.report.markdown import full_markdown, results_block
from agentdistill.report.registry_views import latest_round
from tests.test_report import seed

PAIR_STATS = {
    "n_pairs": 42, "n_rollout": 30, "n_teacher": 12, "cap_per_task": 3, "teacher_pair_ratio": 1.0,
    "tasks_with_pairs": 18, "max_per_task": 3, "diff_kind": {"tool": 30, "args": 9, "text": 3},
    "per_task_histogram": {"1": 5, "2": 6, "3": 7}, "warnings": ["a pair warning worth reading"],
}


@pytest.fixture
def cfg(project_config):
    project_config.eval.eval_set = "holdout"
    project_config.name = "support-agent"
    project_config.teacher = TeacherConfig(model="frontier-v1", provider="anthropic", input_per_mtok=3.0,
                                           output_per_mtok=15.0)
    return project_config


def add_round(registry, round_id: str, decision: str, reason: str, tag: str = "gpu-day", round_idx: int = 0,
              compare: dict | None = None, candidate: str | None = "ad2") -> None:
    registry.record_round({
        "ids": {"round_id": round_id}, "tag": tag, "round_idx": round_idx, "start_adapter": "ad1",
        "candidate_adapter": candidate, "n_rollouts": 160, "fuzzy_share": 0.21, "pair_kinds": PAIR_STATS,
        "compare": compare or {}, "decision": decision, "reason": reason,
    })


@pytest.fixture
def seeded(registry):
    seed(registry)
    registry.insert_adapter({"id": "ad2", "training_run_id": "tr1", "name": "support-r1", "version": 2,
                             "base_model": "m", "path": "/tmp/ad2", "tag": "gpu-day-r1",
                             "parent_adapter_id": "ad1"})
    return registry


def test_a_discarded_round_is_reported_as_a_finding(seeded, cfg):
    add_round(seeded, "rnd_disc", "discard", "success -2.1 pp, CI [-5.0, +0.8], cost not better")
    r = assemble(seeded, cfg, tag_glob="gpu-day")

    o = r.onpolicy
    assert o["round_id"] == "rnd_disc" and o["round_idx"] == 0
    assert (o["start_adapter"], o["candidate_adapter"]) == ("ad1", "ad2")
    assert o["decision"] == "discard"
    assert o["reason"].startswith("success -2.1 pp")
    assert (o["n_rollouts"], o["fuzzy_share"]) == (160, pytest.approx(0.21))
    assert o["pair_stats"]["n_pairs"] == 42 and o["pair_stats"]["per_task_histogram"] == {"1": 5, "2": 6, "3": 7}
    assert o["pair_stats"]["diff_kind"] == PAIR_STATS["diff_kind"]
    assert o["insufficient_power"] is None

    block, md, html = results_block(r), full_markdown(r), render(r)
    for text in (block, md, html):
        assert "rnd_disc" in text
        assert "discarded" in text
        assert "did not beat its starting adapter" in text
        assert "success -2.1 pp, CI [-5.0, +0.8]" in text
    assert "## On-policy round" in md and "<h2>On-policy round</h2>" in html
    for text in (md, html):
        assert "42" in text and "30 rollout" in text and "12 teacher" in text
        assert "max per task 3" in text and "(cap 3)" in text
        assert "21.0%" in text, "the fuzzy share is the round's honesty check and must be printed"
        assert "a pair warning worth reading" in text


def test_a_promoted_round_is_reported_too(seeded, cfg):
    add_round(seeded, "rnd_prom", "promote", "success up +6.0 pp, CI [+1.0, +11.0] excludes zero")
    r = assemble(seeded, cfg, tag_glob="gpu-day")
    assert r.onpolicy["decision"] == "promote"
    for text in (results_block(r), full_markdown(r), render(r)):
        assert "promoted" in text and "rnd_prom" in text
        assert "did not beat" not in text


def test_an_underpowered_comparison_carries_its_reason(seeded, cfg):
    weak = InsufficientPower(8, 2).as_dict()
    add_round(seeded, "rnd_weak", "discard", "adopted without a comparison", compare={"insufficient_power": weak})
    r = assemble(seeded, cfg, tag_glob="gpu-day")
    assert r.onpolicy["insufficient_power"] == weak["reason"]
    for text in (results_block(r), full_markdown(r), render(r)):
        assert "underpowered" in text and "need at least 20" in text


def test_the_underpowered_reason_is_not_repeated_when_it_is_the_decision_reason(seeded, cfg):
    weak = InsufficientPower(8, 2).as_dict()
    add_round(seeded, "rnd_weak", "discard", weak["reason"], compare={"insufficient_power": weak})
    block = results_block(assemble(seeded, cfg, tag_glob="gpu-day"))
    assert block.count("need at least 20") == 1


def test_a_round_without_pair_stats_says_so(seeded, cfg):
    seeded.record_round({"ids": {"round_id": "rnd_err"}, "tag": "gpu-day", "round_idx": 0,
                         "start_adapter": "ad1", "decision": "error", "reason": "rollouts failed"})
    r = assemble(seeded, cfg, tag_glob="gpu-day")
    assert r.onpolicy["pair_stats"] == {}
    assert "errored" in results_block(r)
    assert "No pair statistics were recorded" in full_markdown(r)


def test_no_round_means_no_section(seeded, cfg):
    r = assemble(seeded, cfg, tag_glob="gpu-day")
    assert r.onpolicy == {}
    assert "On-policy" not in full_markdown(r) and "On-policy" not in render(r)


def test_the_round_is_scoped_to_the_report_tag(seeded, cfg):
    add_round(seeded, "rnd_mine", "discard", "mine", tag="gpu-day")
    add_round(seeded, "rnd_other", "promote", "someone else's", tag="other-session", round_idx=3)
    assert assemble(seeded, cfg, tag_glob="gpu-day").onpolicy["round_id"] == "rnd_mine"
    assert assemble(seeded, cfg, tag_glob="gpu-day*").onpolicy["round_id"] == "rnd_mine"
    assert assemble(seeded, cfg, tag_glob="nothing-here").onpolicy == {}


def test_without_a_tag_the_latest_round_overall_is_taken(seeded, cfg):
    add_round(seeded, "rnd_a", "discard", "first", tag="a", round_idx=0)
    add_round(seeded, "rnd_b", "promote", "second", tag="b", round_idx=1)
    assert latest_round(seeded, None)["id"] == "rnd_b"
    assert assemble(seeded, cfg).onpolicy["round_id"] == "rnd_b"


def test_the_latest_of_several_rounds_in_one_tag_is_taken(seeded, cfg):
    for idx in range(3):
        add_round(seeded, f"rnd_{idx}", "discard", f"round {idx}", round_idx=idx)
    assert latest_round(seeded, "gpu-day")["id"] == "rnd_2"


def test_the_sidecar_carries_the_round(seeded, cfg):
    add_round(seeded, "rnd_disc", "discard", "x")
    data = assemble(seeded, cfg, tag_glob="gpu-day").to_dict()
    assert data["onpolicy"]["round_id"] == "rnd_disc"
    assert data["onpolicy"]["pair_stats"]["max_per_task"] == 3
