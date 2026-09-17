"""The on-policy round loop.

`decide` is the part that matters: it is the rule by which one adapter supersedes another, and every number in
the final report rests on the chain it produces. These drive it with scripted comparisons so every branch is
exercised without a GPU.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agentdistill.train.onpolicy import RoundCfg, Stages, decide, plan, run_round, run_rounds


def cmp_dict(delta: float, lo: float, hi: float, token_delta: float = 0.0,
             token_ci: tuple[float, float] | None = None) -> dict:
    """A comparison dict. `token_ci` defaults to an interval that supports the median, so existing cases that
    mean "cost clearly improved" keep meaning that."""
    if token_ci is None:
        token_ci = (token_delta * 2 - 1, token_delta / 2) if token_delta < 0 else (token_delta - 1, token_delta + 1)
    return {
        "success": {"delta": delta, "ci95": (lo, hi), "mean_a": 0.7, "mean_b": 0.7 - delta},
        "tokens": {"median_delta": token_delta, "ci95": list(token_ci)},
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


def test_a_token_saving_whose_interval_includes_zero_does_not_promote():
    """`median_delta < 0` alone promotes on noise: on a small eval set it is negative about half the time."""
    cmp = cmp_dict(0.0, -0.03, 0.03, token_delta=-12, token_ci=(-40.0, +18.0))
    decision, reason = decide(cmp, metrics(), metrics(), RoundCfg())
    assert decision == "discard"
    assert "CI" in reason and "includes zero" in reason


def test_a_token_saving_whose_interval_excludes_zero_promotes():
    cmp = cmp_dict(0.0, -0.03, 0.03, token_delta=-40, token_ci=(-70.0, -12.0))
    decision, reason = decide(cmp, metrics(), metrics(), RoundCfg())
    assert decision == "promote"
    assert "tokens down" in reason


def test_a_missing_token_interval_is_not_a_saving():
    """NaN comparisons are False, so an absent CI must never read as an improvement."""
    cmp = cmp_dict(0.0, -0.03, 0.03, token_delta=-40)
    cmp["tokens"].pop("ci95")
    assert decide(cmp, metrics(), metrics(), RoundCfg())[0] == "discard"


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


# --------------------------------------------------------------------------------------------------------------
# the datasets a round builds must be real
#
# They were fabricated ids: `ds_rft_<hex>` that nothing ever inserted. Recording the round then violated the
# foreign key on `rounds.rft_dataset_id`, so a round could not be written down at all -- and the training stage
# looked the id up, found nothing, and silently fell back to treating it as a path. A round that cannot be
# audited is worse than a round that did not run.
# --------------------------------------------------------------------------------------------------------------


def test_a_pair_set_is_written_as_a_registered_dataset(project_config, registry, tmp_path):
    from agentdistill.data.pairs import write_pairs_dataset

    pairs = [
        {"prompt": [{"role": "user", "content": f"task {i}"}],
         "chosen": [{"role": "assistant", "content": "right"}],
         "rejected": [{"role": "assistant", "content": "wrong"}]}
        for i in range(6)
    ]
    dataset_id = write_pairs_dataset(pairs, project_config, registry, name="demo-pairs", version=1, tag="r1")

    row = registry.get_dataset(dataset_id)
    assert row is not None
    assert row["kind"] == "dpo"
    assert row["n_samples"] == 6
    # Looked up by id, which is what the round records and the trainer reads back.
    assert registry.get_dataset(row["id"])["id"] == dataset_id
    assert (Path(row["path"]) / "pairs.jsonl").exists()


def test_the_same_pairs_produce_the_same_dataset_id(project_config, registry):
    """Content-hashed: two rounds that produced identical pairs are recognisably the same input."""
    from agentdistill.data.pairs import write_pairs_dataset

    pairs = [{"prompt": [{"role": "user", "content": "a"}],
              "chosen": [{"role": "assistant", "content": "x"}],
              "rejected": [{"role": "assistant", "content": "y"}]}]
    first = write_pairs_dataset(pairs, project_config, registry, name="p", version=1)
    second = write_pairs_dataset(pairs, project_config, registry, name="p", version=2)
    assert first == second
    assert len([d for d in registry.list_datasets() if d["id"] == first]) == 1


def test_different_pairs_produce_different_ids(project_config, registry):
    from agentdistill.data.pairs import write_pairs_dataset

    def pairs(answer: str) -> list[dict]:
        return [{"prompt": [{"role": "user", "content": "a"}],
                 "chosen": [{"role": "assistant", "content": answer}],
                 "rejected": [{"role": "assistant", "content": "y"}]}]

    a = write_pairs_dataset(pairs("x"), project_config, registry, name="p", version=1)
    b = write_pairs_dataset(pairs("z"), project_config, registry, name="p", version=2)
    assert a != b


def test_a_round_can_be_recorded_with_the_dataset_ids_it_built(project_config, registry):
    """The foreign keys are the point: a fabricated id made `record_round` fail outright."""
    from agentdistill.data.pairs import write_pairs_dataset
    from agentdistill.registry.base import utcnow

    registry.insert_dataset({"id": "ds_seed", "name": "seed", "version": 1, "kind": "sft", "filter_config": {},
                             "n_samples": 1, "n_tokens": 1, "content_hash": "h", "path": "/tmp/seed"})
    registry.insert_training_run({"id": "tr1", "dataset_id": "ds_seed", "base_model": "m", "method": "sft",
                                  "config": {}, "status": "succeeded", "started_at": utcnow()})
    registry.insert_adapter({"id": "ad1", "training_run_id": "tr1", "name": "a", "version": 1,
                             "base_model": "m", "path": "/tmp/a"})

    pairs_id = write_pairs_dataset(
        [{"prompt": [{"role": "user", "content": "a"}],
          "chosen": [{"role": "assistant", "content": "x"}],
          "rejected": [{"role": "assistant", "content": "y"}]}],
        project_config, registry, name="pp", version=1,
    )

    registry.record_round({
        "round_idx": 1, "start_adapter": "ad1", "tag": "r1",
        "ids": {"round_id": "rd1", "dpo_dataset": pairs_id},
        "decision": "discard", "reason": "rehearsal",
        "started_at": utcnow(),
    })

    from sqlalchemy import text

    with registry.engine.connect() as conn:
        row = conn.execute(text("SELECT * FROM onpolicy_rounds WHERE id = 'rd1'")).mappings().first()
    assert row is not None
    assert row["dpo_dataset_id"] == pairs_id


def test_rollouts_are_registered_with_a_source_that_marks_them_as_the_students_own():
    """A rollout must never be mistaken for a teacher trace, in a report or in a later curation run."""
    from agentdistill.eval.rollouts import register_rollouts

    class FakeRegistry:
        def __init__(self) -> None:
            self.inserted: list[dict] = []

        def insert_traces(self, traces):
            self.inserted.extend(traces)
            return {"added": len(traces), "skipped": 0}

    reg = FakeRegistry()
    rollouts = [
        {"task_id": "t1", "success": True,
         "messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]},
    ]
    prepared = register_rollouts(reg, rollouts, tag="r1")

    assert len(reg.inserted) == 1
    assert reg.inserted[0]["source"] == "rollout"
    assert reg.inserted[0]["metadata"]["rollout"] is True
    assert reg.inserted[0]["metadata"]["tag"] == "r1"
    # `build_dataset` needs both of these, and a rollout arrives with neither.
    assert prepared[0]["id"].startswith("ro_")
    assert len(prepared[0]["content_hash"]) == 64


def test_identical_rollouts_hash_to_the_same_id():
    from agentdistill.eval.rollouts import register_rollouts

    class FakeRegistry:
        def insert_traces(self, traces):
            return {"added": len(traces), "skipped": 0}

    messages = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    a = register_rollouts(FakeRegistry(), [{"task_id": "t", "messages": messages}])
    b = register_rollouts(FakeRegistry(), [{"task_id": "t", "messages": messages}])
    assert a[0]["id"] == b[0]["id"]


def test_curation_never_sees_a_rollout(tmp_path, monkeypatch):
    """The whole reason rollouts carry a distinct source.

    A student curated from its own rollouts is training on itself. The failure is slow, quiet, and very hard to
    attribute weeks later, so it is worth a test that fails loudly rather than a comment.
    """
    import json

    import yaml
    from typer.testing import CliRunner

    from agentdistill.cli import app
    from agentdistill.eval.rollouts import register_rollouts
    from agentdistill.registry import open_registry
    from tests.conftest import TOKENIZER_DIR, make_trace

    monkeypatch.chdir(tmp_path)
    traces = [
        make_trace(f"t{i}", task=f"Order {i} is late and I would like to know where it is",
                   closing=" ".join(f"Point {j} of case {i} is {i * 7 + j}" for j in range(5)))
        for i in range(8)
    ]
    (tmp_path / "traces.jsonl").write_text("\n".join(json.dumps(t) for t in traces) + "\n")
    (tmp_path / "project.yaml").write_text(yaml.safe_dump({
        "name": "demo",
        "registry": "sqlite:///.agentdistill/registry.db",
        "artifacts": "./artifacts",
        "reports": "./reports",
        "dataset": {"max_seq_len": 512},
        "curate": {"clusters": 2, "cap_per_cluster": 50},
        "train": {"base_model": str(TOKENIZER_DIR), "max_seq_len": 512},
    }))

    runner = CliRunner()
    assert runner.invoke(app, ["ingest", "jsonl", "traces.jsonl"]).exit_code == 0

    reg = open_registry("sqlite:///.agentdistill/registry.db", root=tmp_path)
    try:
        # Twenty rollouts, which would swamp eight teacher traces if curation picked them up.
        register_rollouts(reg, [
            {"task_id": f"r{i}", "success": True,
             "messages": [{"role": "user", "content": f"rollout {i} please help with this order now"},
                          {"role": "assistant", "content": f"Rollout answer {i}, which is the student talking"}]}
            for i in range(20)
        ], tag="round-1")
        assert len(reg.list_traces()) == 28
    finally:
        reg.close()

    assert runner.invoke(app, ["curate"]).exit_code == 0

    reg = open_registry("sqlite:///.agentdistill/registry.db", root=tmp_path)
    try:
        dataset = reg.get_dataset("demo")
        assert dataset is not None
        # Eight teacher traces went in. Nothing from the twenty rollouts may come out.
        assert dataset["n_samples"] <= 8, (
            f"curation produced {dataset['n_samples']} samples from 8 teacher traces; rollouts leaked in"
        )
    finally:
        reg.close()


# --------------------------------------------------------------------------------------------------------------
# the real stages against the contract they claim to satisfy
#
# `Stages` is typed, but the loop's tests drive it with stubs -- so the stubs matched the contract while the real
# `_onpolicy_stages` did not. `train_sft_continue` is declared to return (run, adapter) and returned only the
# adapter, which surfaced as "too many values to unpack" partway through a round.
# --------------------------------------------------------------------------------------------------------------


def test_the_real_stage_set_fills_every_slot_in_the_contract(project_config, registry):
    from agentdistill.cli import _onpolicy_stages
    from agentdistill.train.onpolicy import RoundCfg, Stages

    stages = _onpolicy_stages(project_config, registry, "t", RoundCfg(), "hf")
    assert isinstance(stages, Stages)
    for field in Stages.__dataclass_fields__:
        assert callable(getattr(stages, field)), f"{field} is not callable"


def test_train_sft_continue_returns_a_run_and_an_adapter(project_config, registry, tmp_path, monkeypatch):
    """Both, because the round records both: a candidate has to be traceable to the run that made it."""
    from agentdistill.cli import _onpolicy_stages
    from agentdistill.registry.base import utcnow
    from agentdistill.train.onpolicy import RoundCfg

    registry.insert_dataset({"id": "ds_rft", "name": "r", "version": 1, "kind": "rft", "filter_config": {},
                             "n_samples": 4, "n_tokens": 40, "content_hash": "h", "path": str(tmp_path)})
    registry.insert_dataset({"id": "ds_seed", "name": "s", "version": 1, "kind": "sft", "filter_config": {},
                             "n_samples": 4, "n_tokens": 40, "content_hash": "h2", "path": str(tmp_path)})
    registry.insert_training_run({"id": "tr_seed", "dataset_id": "ds_seed", "base_model": "m", "method": "sft",
                                  "config": {}, "status": "succeeded", "started_at": utcnow()})
    registry.insert_adapter({"id": "ad_start", "training_run_id": "tr_seed", "name": "student", "version": 1,
                             "base_model": "m", "path": str(tmp_path / "start")})

    class FakeResult:
        eval_loss = 0.5

        def to_dict(self):
            return {"eval_loss": 0.5, "steps": 1}

    seen = {}

    def fake_train_sft(cfg, dataset_path, out_dir, resume_adapter=None, **kw):
        seen["resume_adapter"] = resume_adapter
        seen["lr"] = cfg["lr"]
        return FakeResult()

    monkeypatch.setattr("agentdistill.train.sft.train_sft", fake_train_sft)

    from agentdistill.config import TrainConfig

    project_config.train = TrainConfig(base_model="m", max_seq_len=project_config.dataset.max_seq_len)

    stages = _onpolicy_stages(project_config, registry, "round-1", RoundCfg(), "hf")
    run_id, adapter_id = stages.train_sft_continue("ad_start", "ds_rft")

    assert registry.get_training_run(run_id) is not None
    row = next(a for a in registry.list_adapters() if a["id"] == adapter_id)
    assert row["parent_adapter_id"] == "ad_start"
    assert row["tag"] == "round-1"
    # Continuation, not a fresh run, at a third of the configured rate.
    assert seen["resume_adapter"] == str(tmp_path / "start")
    assert seen["lr"] == pytest.approx(project_config.train.lr / 3)


def test_train_dpo_returns_a_run_and_an_adapter_from_the_merged_weights(
    project_config, registry, tmp_path, monkeypatch
):
    """DPO trains a fresh LoRA, so it starts from merged weights rather than stacking on the SFT adapter --
    the reference-model maths does not account for one adapter on top of another."""
    from agentdistill.cli import _onpolicy_stages
    from agentdistill.config import TrainConfig
    from agentdistill.registry.base import utcnow
    from agentdistill.train.dpo import DpoResult
    from agentdistill.train.onpolicy import RoundCfg

    registry.insert_dataset({"id": "ds_pairs", "name": "p", "version": 1, "kind": "dpo", "filter_config": {},
                             "n_samples": 4, "n_tokens": 0, "content_hash": "h", "path": str(tmp_path)})
    registry.insert_dataset({"id": "ds_seed", "name": "s", "version": 1, "kind": "sft", "filter_config": {},
                             "n_samples": 4, "n_tokens": 40, "content_hash": "h2", "path": str(tmp_path)})
    registry.insert_training_run({"id": "tr_seed", "dataset_id": "ds_seed", "base_model": "m", "method": "sft",
                                  "config": {}, "status": "succeeded", "started_at": utcnow()})
    registry.insert_adapter({"id": "ad_start", "training_run_id": "tr_seed", "name": "student", "version": 1,
                             "base_model": "m", "path": str(tmp_path / "start")})

    seen = {}

    def fake_train_dpo(cfg, pairs, base_or_merged, out_dir, max_teacher_ratio=1.0):
        seen["base"] = base_or_merged
        seen["cfg_base"] = cfg["base_model"]
        seen["n_pairs"] = len(pairs)
        return DpoResult(adapter_path=str(out_dir), steps=1, n_pairs=len(pairs),
                         final_reward_accuracy=0.8, metrics={"steps": 1})

    monkeypatch.setattr("agentdistill.train.dpo.train_dpo", fake_train_dpo)
    project_config.train = TrainConfig(base_model="m", max_seq_len=project_config.dataset.max_seq_len)

    stages = _onpolicy_stages(project_config, registry, "round-1", RoundCfg(), "hf")
    merged_dir = str(tmp_path / "merged")
    run_id, adapter_id = stages.train_dpo(merged_dir, "ds_pairs")

    assert seen["base"] == merged_dir
    assert seen["cfg_base"] == merged_dir, "DPO must train from the merged weights, not the configured base"
    assert registry.get_training_run(run_id)["method"] == "dpo"
    row = next(a for a in registry.list_adapters() if a["id"] == adapter_id)
    assert row["tag"] == "round-1"
    assert row["base_model"] == merged_dir


def test_a_flat_dpo_reward_is_reported_but_does_not_stop_the_round(
    project_config, registry, tmp_path, monkeypatch, capsys
):
    """The round's own comparison decides. A flat reward means the pairs carried no signal, which is the first
    thing to look at if the round is then discarded -- so it is said out loud, not raised."""
    from agentdistill.cli import _onpolicy_stages
    from agentdistill.config import TrainConfig
    from agentdistill.registry.base import utcnow
    from agentdistill.train.dpo import DpoResult
    from agentdistill.train.onpolicy import RoundCfg

    registry.insert_dataset({"id": "ds_pairs", "name": "p", "version": 1, "kind": "dpo", "filter_config": {},
                             "n_samples": 2, "n_tokens": 0, "content_hash": "h", "path": str(tmp_path)})
    registry.insert_training_run({"id": "tr_seed", "dataset_id": "ds_pairs", "base_model": "m", "method": "sft",
                                  "config": {}, "status": "succeeded", "started_at": utcnow()})

    monkeypatch.setattr(
        "agentdistill.train.dpo.train_dpo",
        lambda cfg, pairs, base, out, max_teacher_ratio=1.0: DpoResult(
            adapter_path=str(out), steps=1, n_pairs=2, final_reward_accuracy=0.5, metrics={},
        ),
    )
    project_config.train = TrainConfig(base_model="m", max_seq_len=project_config.dataset.max_seq_len)

    stages = _onpolicy_stages(project_config, registry, "r", RoundCfg(), "hf")
    run_id, adapter_id = stages.train_dpo(str(tmp_path / "merged"), "ds_pairs")

    assert run_id and adapter_id
    assert "flat" in capsys.readouterr().out.lower()
