"""The on-policy round loop.

`decide` is the part that matters: it is the rule by which one adapter supersedes another, and every number in
the final report rests on the chain it produces. These drive it with scripted comparisons so every branch is
exercised without a GPU.
"""

from __future__ import annotations

import pytest

from agentdistill.train.onpolicy import RoundCfg, Stages, decide, plan, run_round, run_rounds


def cmp_dict(delta: float, lo: float, hi: float, token_delta: float = 0.0) -> dict:
    return {
        "success": {"delta": delta, "ci95": (lo, hi), "mean_a": 0.7, "mean_b": 0.7 - delta},
        "tokens": {"median_delta": token_delta},
        "turns": {"median_delta": 0.0},
    }


def metrics(schema: float = 1.0, divergence: float = 0.05, success: float = 0.7) -> dict:
    return {"schema_valid": schema, "divergence_rate": divergence, "success": success}


class StubStages:
    """Records what the loop asked for, and returns whatever the test scripted."""

    def __init__(self, compare_result: dict, candidate_metrics: dict | None = None, n_pairs: int = 100,
                 fuzzy_share: float = 0.1, n_rollouts: int = 80, fail_at: str | None = None) -> None:
        self.compare_result = compare_result
        self.candidate_metrics = candidate_metrics or metrics()
        self.n_pairs = n_pairs
        self.fuzzy_share = fuzzy_share
        self.n_rollouts = n_rollouts
        self.fail_at = fail_at
        self.calls: list[str] = []
        self.recorded: list[dict] = []
        self.counter = 0

    def _step(self, name: str):
        self.calls.append(name)
        if self.fail_at == name:
            raise RuntimeError(f"{name} blew up")

    def stages(self) -> Stages:
        def collect(adapter, task_ids, k, policy):
            self._step("collect_rollouts")
            return {"rollouts": [{"id": f"r{i}"} for i in range(self.n_rollouts)],
                    "fuzzy_share": self.fuzzy_share, "eval_run_id": "ev_roll"}

        def build_rft(rollouts, cap):
            self._step("build_rft")
            return "ds_rft", 40

        def build_pairs(rollouts, teacher_by_task):
            self._step("build_pairs")
            return "ds_pairs", self.n_pairs, {"rollout": self.n_pairs, "teacher": 0}

        def sft(adapter, dataset_id):
            self._step("train_sft_continue")
            return "tr_sft", f"ad_sft_{self.counter}"

        def merge(adapter):
            self._step("merge")
            return f"/merged/{adapter}"

        def dpo(merged, dataset_id):
            self._step("train_dpo")
            self.counter += 1
            return "tr_dpo", f"ad_cand_{self.counter}"

        def run_eval(adapter):
            self._step("run_eval")
            return f"ev_{adapter}"

        def compare(a, b):
            self._step("compare")
            return self.compare_result

        def get_metrics(eval_run):
            return self.candidate_metrics if eval_run.startswith("ev_ad_cand") else metrics()

        return Stages(
            collect_rollouts=collect, build_rft=build_rft, build_pairs=build_pairs,
            train_sft_continue=sft, merge=merge, train_dpo=dpo, run_eval=run_eval,
            compare=compare, metrics=get_metrics, record=self.recorded.append,
        )


def run_one(stub: StubStages, cfg: RoundCfg | None = None):
    return run_round(0, "ad_start", ["t1", "t2"], {}, "ev_current", stub.stages(), cfg or RoundCfg())


# --------------------------------------------------------------------------------------------------------------
# decide
# --------------------------------------------------------------------------------------------------------------


def test_promote_on_a_clear_improvement():
    decision, reason = decide(cmp_dict(0.08, 0.02, 0.14), metrics(), metrics(), RoundCfg())
    assert decision == "promote"
    assert "excludes zero" in reason


def test_promote_on_equal_success_with_a_cost_saving():
    decision, reason = decide(cmp_dict(-0.002, -0.03, 0.03, token_delta=-40), metrics(), metrics(), RoundCfg())
    assert decision == "promote"
    assert "tokens down" in reason


def test_discard_on_equal_success_without_a_cost_saving():
    """A differently-shaped model for nothing is a link in the lineage nobody can justify."""
    decision, reason = decide(cmp_dict(0.001, -0.03, 0.03, token_delta=+5), metrics(), metrics(), RoundCfg())
    assert decision == "discard"
    assert "cost_better=False" in reason


def test_discard_when_success_dropped_beyond_tolerance():
    decision, _ = decide(cmp_dict(-0.05, -0.09, -0.01, token_delta=-100), metrics(), metrics(), RoundCfg())
    assert decision == "discard"


def test_schema_floor_is_a_hard_gate_no_matter_the_cost_saving():
    """Tool calls that stopped validating are not redeemed by token savings."""
    decision, reason = decide(
        cmp_dict(0.10, 0.05, 0.15, token_delta=-500), metrics(), metrics(schema=0.90), RoundCfg()
    )
    assert decision == "discard"
    assert "schema validity" in reason


def test_divergence_regression_is_a_hard_gate():
    decision, reason = decide(
        cmp_dict(0.10, 0.05, 0.15), metrics(divergence=0.05), metrics(divergence=0.20), RoundCfg()
    )
    assert decision == "discard"
    assert "divergence rate regressed" in reason


def test_small_divergence_regression_is_within_slack():
    decision, _ = decide(
        cmp_dict(0.10, 0.05, 0.15), metrics(divergence=0.05), metrics(divergence=0.08), RoundCfg()
    )
    assert decision == "promote"


# --------------------------------------------------------------------------------------------------------------
# one round
# --------------------------------------------------------------------------------------------------------------


def test_a_promoting_round_runs_every_stage_in_order():
    stub = StubStages(cmp_dict(0.08, 0.02, 0.14))
    r = run_one(stub)
    assert r.decision == "promote"
    assert stub.calls == [
        "collect_rollouts", "build_rft", "build_pairs", "train_sft_continue", "merge", "train_dpo",
        "run_eval", "compare",
    ]
    assert r.candidate_adapter == "ad_cand_1"
    assert r.ids["rft_dataset"] == "ds_rft" and r.ids["dpo_dataset"] == "ds_pairs"


def test_fuzzy_share_aborts_before_any_training():
    """Training on rollouts whose results were mostly approximate is worse than not training."""
    stub = StubStages(cmp_dict(0.08, 0.02, 0.14), fuzzy_share=0.8)
    r = run_one(stub)
    assert r.decision == "discard"
    assert "fuzzy replay share" in r.reason
    assert stub.calls == ["collect_rollouts"], "nothing should be trained"


def test_too_few_pairs_aborts_before_training():
    stub = StubStages(cmp_dict(0.08, 0.02, 0.14), n_pairs=5)
    r = run_one(stub)
    assert r.decision == "discard"
    assert "only 5 usable pairs" in r.reason
    assert "train_dpo" not in stub.calls


def test_schema_floor_discards_after_evaluation():
    stub = StubStages(cmp_dict(0.08, 0.02, 0.14), candidate_metrics=metrics(schema=0.5))
    r = run_one(stub)
    assert r.decision == "discard" and "schema validity" in r.reason


def test_every_round_is_recorded_including_discards():
    stub = StubStages(cmp_dict(0.08, 0.02, 0.14), fuzzy_share=0.9)
    run_one(stub)
    assert len(stub.recorded) == 1
    assert stub.recorded[0]["decision"] == "discard"


def test_an_exception_is_recorded_as_an_error_and_reraised():
    """A round that blew up must never be mistaken for a round that decided something."""
    stub = StubStages(cmp_dict(0.08, 0.02, 0.14), fail_at="train_dpo")
    with pytest.raises(RuntimeError, match="train_dpo blew up"):
        run_one(stub)
    assert len(stub.recorded) == 1
    assert stub.recorded[0]["decision"] == "error"
    assert "RuntimeError" in stub.recorded[0]["reason"]


def test_round_records_pair_kinds():
    stub = StubStages(cmp_dict(0.08, 0.02, 0.14))
    r = run_one(stub)
    assert r.pair_kinds == {"rollout": 100, "teacher": 0}


def test_comparison_is_candidate_versus_current_not_teacher():
    """Positive deltas must favour the candidate, or `decide` reads every sign backwards."""
    seen: list[tuple[str, str]] = []
    stub = StubStages(cmp_dict(0.08, 0.02, 0.14))
    stages = stub.stages()
    original = stages.compare

    def spy(a, b):
        seen.append((a, b))
        return original(a, b)

    stages.compare = spy
    run_round(0, "ad_start", ["t1"], {}, "ev_current", stages, RoundCfg())
    assert seen == [("ev_ad_cand_1", "ev_current")]


# --------------------------------------------------------------------------------------------------------------
# several rounds
# --------------------------------------------------------------------------------------------------------------


def test_round_two_starts_from_round_ones_candidate():
    stub = StubStages(cmp_dict(0.08, 0.02, 0.14))
    results = run_rounds("ad_start", 2, ["t1"], {}, "ev_current", stub.stages(), RoundCfg())
    assert len(results) == 2
    assert results[0].start_adapter == "ad_start"
    assert results[1].start_adapter == results[0].candidate_adapter


def test_a_discarded_round_stops_the_loop():
    """The next round would start from the same adapter against the same tasks and land in the same place."""
    stub = StubStages(cmp_dict(-0.05, -0.10, -0.01))
    results = run_rounds("ad_start", 3, ["t1"], {}, "ev_current", stub.stages(), RoundCfg())
    assert len(results) == 1 and results[0].decision == "discard"


def test_zero_rounds_does_nothing():
    stub = StubStages(cmp_dict(0.08, 0.02, 0.14))
    assert run_rounds("ad_start", 0, ["t1"], {}, "ev", stub.stages(), RoundCfg()) == []
    assert stub.calls == []


# --------------------------------------------------------------------------------------------------------------
# dry run
# --------------------------------------------------------------------------------------------------------------


def test_plan_reports_sizes_and_gates():
    lines = "\n".join(plan("ad_1", 2, [f"t{i}" for i in range(50)], RoundCfg()))
    assert "50 x 8 = 400" in lines
    assert "fuzzy" in lines
    assert "promote if" in lines and "schema validity" in lines
