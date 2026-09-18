"""Assembling and rendering the report.

The rule the tests enforce: a report with missing pieces is still a report, and it says which claim it could not
make. Silence about a missing calibration or an unmeasured throughput is how a cost number ends up in a README
with nothing behind it.
"""

from __future__ import annotations

import pytest

from agentdistill.report.assemble import ReportData, assemble
from agentdistill.report.cost import (
    breakeven_tasks_per_day,
    cascade_cost_per_task,
    saving_fraction,
    student_cost_per_mtok,
    teacher_cost_per_task,
)
from agentdistill.report.markdown import BEGIN, END, full_markdown, inject, results_block

# --------------------------------------------------------------------------------------------------------------
# the cost model
# --------------------------------------------------------------------------------------------------------------


def test_student_cost_per_mtok_against_a_hand_computed_value():
    # $1.20/h over 2000 tok/s at 60% utilization: 1.20 / (2000 * 0.6 * 3600) * 1e6
    assert student_cost_per_mtok(1.20, 2000, 0.6) == pytest.approx(0.2777778, rel=1e-6)


def test_utilization_scales_the_cost():
    assert student_cost_per_mtok(1.0, 1000, 0.5) == pytest.approx(2 * student_cost_per_mtok(1.0, 1000, 1.0))


def test_zero_throughput_is_infinite_cost():
    assert student_cost_per_mtok(1.0, 0) == float("inf")


def test_teacher_cost_against_a_hand_computed_value():
    # 3000 prompt at $3/Mtok + 300 completion at $15/Mtok = 0.009 + 0.0045
    assert teacher_cost_per_task(3000, 300, 3.0, 15.0) == pytest.approx(0.0135)


def test_prompt_caching_reduces_the_teacher_cost():
    """Agent traffic is unusually cacheable; ignoring it overstates the teacher and flatters the student."""
    uncached = teacher_cost_per_task(3000, 300, 3.0, 15.0)
    cached = teacher_cost_per_task(3000, 300, 3.0, 15.0, cache_hit_frac=0.8, cache_read_per_mtok=0.3)
    assert cached < uncached
    # 600 uncached at 3.0 + 2400 cached at 0.3 + 300 out at 15.0
    assert cached == pytest.approx((600 * 3.0 + 2400 * 0.3 + 300 * 15.0) / 1e6)


def test_cascade_cost_includes_the_discarded_generations():
    """The student generates on escalated turns too; a cascade costed without them looks cheaper than it is."""
    without = cascade_cost_per_task(400, 0.28, 0.25, 0.0135, wasted_student_tokens=0)
    with_waste = cascade_cost_per_task(400, 0.28, 0.25, 0.0135, wasted_student_tokens=200)
    assert with_waste > without


def test_cascade_cost_at_full_escalation_approaches_the_teacher():
    teacher = 0.0135
    cost = cascade_cost_per_task(0, 0.28, 1.0, teacher, wasted_student_tokens=0)
    assert cost == pytest.approx(teacher)


def test_breakeven_against_a_hand_computed_value():
    # $1.20/h * 24 = $28.80/day; saving $0.01/task -> 2880 tasks
    assert breakeven_tasks_per_day(1.20, 0.0135, 0.0035) == pytest.approx(2880)


def test_breakeven_is_infinite_when_the_cascade_costs_more():
    """No volume rescues a variable cost that is already higher than the alternative."""
    assert breakeven_tasks_per_day(1.20, 0.001, 0.002) == float("inf")
    assert breakeven_tasks_per_day(1.20, 0.001, 0.001) == float("inf")


def test_saving_fraction():
    assert saving_fraction(0.01, 0.0025) == pytest.approx(0.75)
    assert saving_fraction(0.0, 0.001) is None


# --------------------------------------------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_tree(monkeypatch):
    """Rows record the git state of whatever tree the tests run in, and a developer's tree is usually dirty. Pin
    it clean so the `dirty_tree` warning appears only in the tests that ask for it."""
    import agentdistill.provenance as prov

    monkeypatch.setattr(prov, "git_state", lambda cwd=None: {"commit": "abc1234", "dirty": False})


@pytest.fixture
def cfg(project_config):
    project_config.eval.eval_set = "holdout"
    project_config.name = "support-agent"
    from agentdistill.config import TeacherConfig

    project_config.teacher = TeacherConfig(
        model="frontier-v1", provider="anthropic", input_per_mtok=3.0, output_per_mtok=15.0
    )
    return project_config


class _Outcome:
    """The shape `write_eval_result` expects, without running the harness."""

    def __init__(self, task_id, repeat_idx, success, tokens, cluster):
        self.task_id, self.repeat_idx, self.success = task_id, repeat_idx, success
        self.tokens, self.cluster = tokens, cluster
        self.messages = []
        self.grader_detail = ""

    def to_row(self):
        return {
            "task_id": self.task_id, "repeat_idx": self.repeat_idx, "final_text": "",
            "n_turns": 4, "n_tool_calls": 2, "schema_valid": True, "diverged": False, "divergence": None,
            "replay_stats": {}, "latency_ms": 10, "completion_tokens_est": self.tokens,
            "stop_reason": "answered", "success": self.success, "grader_detail": "",
            "escalations": 0, "wasted_student_tokens": 0,
        }


def seed(registry, *, with_calibration=True, with_teacher=True, with_throughput=True, with_quantized=False):
    from sqlalchemy import text

    from agentdistill.registry.base import dumps, utcnow

    registry.insert_dataset({"id": "ds1", "name": "support", "version": 1, "kind": "sft",
                             "filter_config": {"filters": ["outcome"]}, "n_samples": 300, "n_tokens": 10000,
                             "content_hash": "abc123def456", "path": "/tmp/ds1"})
    registry.insert_training_run({"id": "tr1", "dataset_id": "ds1", "base_model": "m", "method": "sft",
                                  "config": {}, "status": "succeeded", "started_at": utcnow(),
                                  "command": "agentdistill train sft support"})
    registry.insert_adapter({"id": "ad1", "training_run_id": "tr1", "name": "support", "version": 1,
                             "base_model": "m", "path": "/tmp/ad1", "tag": "gpu-day"})
    registry.insert_eval_set({"id": "es_holdout", "name": "holdout", "trace_ids": [], "grader": {}})

    def add_run(run_id, subject, success, extra=None, tokens=150):
        """Writes per-repeat rows as well as metrics, because `compare` reads the rows."""
        registry.start_eval_run(run_id, "es_holdout", subject, 5)
        for task in range(20):
            for repeat in range(5):
                # A difficulty draw shared across subjects: the same task/repeat is hard for everyone, which
                # is exactly the correlation a paired comparison exploits.
                draw = ((task * 31 + repeat * 17) % 100) / 100.0
                outcome = _Outcome(
                    task_id=f"task{task}", repeat_idx=repeat,
                    success=draw < success,
                    tokens=tokens, cluster=task % 2,
                )
                registry.write_eval_result(run_id, outcome, cluster=task % 2, store_messages=False)
        metrics = {"success": success, "schema_valid": 0.99, "divergence_rate": 0.05,
                   "tokens_est_median": tokens, "turns_median": 4, "n_tasks": 20, "n_per_task": 5,
                   **(extra or {})}
        registry.finish_eval_run(run_id, metrics, {"0": {"n_tasks": 10, "success": success},
                                                   "1": {"n_tasks": 10, "success": max(success - 0.3, 0.0)}})

    add_run("ev_base", "base", 0.40, tokens=180)
    student_extra = {"throughput_tok_per_s": 1800.0, "throughput_conditions": "batched, max_num_seqs=64"} \
        if with_throughput else {}
    add_run("ev_student", "ad1", 0.68, student_extra, tokens=150)
    if with_teacher:
        add_run("ev_teacher", "teacher", 0.84,
                {"prompt_tokens_median": 3000, "completion_tokens_median": 300}, tokens=210)

    if with_calibration:
        with registry.engine.begin() as conn:
            conn.execute(
                text("""INSERT INTO calibrations (id, adapter_id, eval_run_id, features, model_path, threshold,
                                                  target, ece, brier, auroc, escalation_rate, created_at,
                                                  holdout_metrics, reliability_bins, verified, feature_order)
                        VALUES (:id, 'ad1', 'ev_student', :features, '/tmp/cal', 0.62, :target, 0.03, 0.12,
                                0.78, 0.24, :created, :holdout, :bins, :verified, :order)"""),
                {"id": "cal1", "features": dumps(["mean_logprob"]), "target": dumps({}), "created": utcnow(),
                 "holdout": dumps({"auroc": 0.78, "ece": 0.03, "brier": 0.12, "n": 400}),
                 "bins": dumps([{"lo": 0.0, "hi": 0.1, "n": 5}]),
                 "verified": dumps([{"threshold": 0.62, "success": 0.70, "escalation_rate": 0.24,
                                     "wasted_student_tokens": 40, "cost_per_task": 0.004}]),
                 "order": dumps(["mean_logprob"])},
            )

    if with_quantized:
        registry.insert_adapter({"id": "ad1q", "training_run_id": "tr1", "name": "support-awq", "version": 2,
                                 "base_model": "m", "path": "/tmp/ad1q", "quantization": "awq",
                                 "parent_adapter_id": "ad1"})
        add_run("ev_quant", "ad1q", 0.66, tokens=150)
    return registry


def test_a_complete_registry_assembles_without_warnings(registry, cfg):
    # Complete includes the quantized artifact: serving is configured quantized, so its absence is a warning.
    r = assemble(seed(registry, with_quantized=True), cfg, tag_glob="gpu-day")
    assert set(r.subjects) >= {"base", "student", "teacher"}
    assert r.subjects["student"]["success"] == 0.68
    assert r.paired["student_vs_teacher"]["success"]["delta"] < 0
    assert r.calibration["holdout"]["auroc"] == 0.78
    assert r.cost["cascade"]["cost_per_task"] > 0
    assert r.warnings == [], r.warnings
    assert r.warning_codes == []


def test_every_subject_row_carries_its_run_id(registry, cfg):
    r = assemble(seed(registry), cfg, tag_glob="gpu-day")
    for name, s in r.subjects.items():
        assert s["run_id"], f"{name} has no run id"


def test_a_missing_calibration_warns_about_escalating_everything(registry, cfg):
    r = assemble(seed(registry, with_calibration=False), cfg, tag_glob="gpu-day")
    assert any("escalate every turn" in w for w in r.warnings)
    assert "cascade" not in r.cost


def test_a_missing_teacher_run_skips_the_cost_block(registry, cfg):
    r = assemble(seed(registry, with_teacher=False), cfg, tag_glob="gpu-day")
    assert r.cost == {}
    assert any("no baseline cost" in w for w in r.warnings)


def test_unmeasured_throughput_blocks_the_cost_per_task(registry, cfg):
    """An unbatched measurement would overstate the cost several times over, so refuse rather than guess."""
    r = assemble(seed(registry, with_throughput=False), cfg, tag_glob="gpu-day")
    assert "cascade" not in r.cost
    assert r.cost.get("teacher_cost_per_task")
    assert any("throughput was not measured" in w for w in r.warnings)


def test_no_evaluated_adapter_warns_rather_than_raising(registry, cfg):
    registry.insert_eval_set({"id": "es_holdout", "name": "holdout", "trace_ids": [], "grader": {}})
    r = assemble(registry, cfg)
    assert any("no adapter" in w for w in r.warnings)
    assert r.subjects == {} or "student" not in r.subjects


def test_quantization_delta_is_reported(registry, cfg):
    r = assemble(seed(registry, with_quantized=True), cfg, tag_glob="gpu-day")
    assert r.quantization["method"] == "awq"
    assert r.quantization["delta_pp"] == pytest.approx(-2.0)


def test_per_cluster_table_flags_clusters_below_the_floor(registry, cfg):
    cfg.router.floor = 0.55
    r = assemble(seed(registry), cfg, tag_glob="gpu-day")
    assert r.per_cluster
    assert any(row["routing"].startswith("teacher") for row in r.per_cluster)


def test_lineage_carries_the_dataset_hash(registry, cfg):
    r = assemble(seed(registry), cfg, tag_glob="gpu-day")
    assert r.lineage["dataset"]["content_hash"] == "abc123def456"
    assert r.lineage["training_run"]["id"] == "tr1"


def test_commands_are_collected(registry, cfg):
    r = assemble(seed(registry), cfg, tag_glob="gpu-day")
    assert any("train sft" in c["command"] for c in r.commands)


def test_tiny_mode_is_flagged_loudly(registry, cfg, tmp_path):
    cfg.source_path = tmp_path / "project.tiny.yaml"
    r = assemble(seed(registry), cfg, tag_glob="gpu-day")
    assert r.tiny
    assert any("TINY MODE" in w for w in r.warnings)


# --------------------------------------------------------------------------------------------------------------
# markdown
# --------------------------------------------------------------------------------------------------------------


def built(registry, cfg) -> ReportData:
    return assemble(seed(registry), cfg, tag_glob="gpu-day")


def test_results_block_has_markers_and_a_table(registry, cfg):
    block = results_block(built(registry, cfg))
    assert block.startswith(BEGIN) and block.rstrip().endswith(END)
    assert "| Subject |" in block


def test_no_number_appears_without_a_run_id(registry, cfg):
    """The README rule: every figure is traceable to the run that produced it."""
    block = results_block(built(registry, cfg))
    for line in block.splitlines():
        if line.startswith("| ") and "%" in line and "Subject" not in line:
            assert "`ev_" in line, f"row has numbers but no run id: {line}"


def test_injection_is_idempotent(registry, cfg):
    block = results_block(built(registry, cfg))
    once = inject("# Project\n\nIntro text.\n", block)
    twice = inject(once, block)
    assert once == twice
    assert once.count(BEGIN) == 1 and once.count(END) == 1


def test_injection_appends_a_results_section_when_there_are_no_markers(registry, cfg):
    out = inject("# Project\n\nIntro text.\n", results_block(built(registry, cfg)))
    assert "## Results" in out
    assert out.count("## Results") == 1


def test_injection_replaces_stale_numbers(registry, cfg):
    block = results_block(built(registry, cfg))
    stale = inject("# P\n", BEGIN + "\nold numbers 99.9%\n" + END)
    fresh = inject(stale, block)
    assert "old numbers" not in fresh


def test_injection_preserves_surrounding_text(registry, cfg):
    readme = "# Project\n\nBefore.\n\n" + BEGIN + "\nold\n" + END + "\n\nAfter.\n"
    out = inject(readme, results_block(built(registry, cfg)))
    assert "Before." in out and "After." in out


def test_warnings_render_in_the_block(registry, cfg):
    r = assemble(seed(registry, with_calibration=False), cfg, tag_glob="gpu-day")
    assert "**Warnings:**" in results_block(r)


def test_full_markdown_includes_cluster_lineage_and_commands(registry, cfg):
    text = full_markdown(built(registry, cfg))
    assert "## Per cluster" in text
    assert "## Lineage" in text
    assert "## How to reproduce" in text
    assert "abc123def456"[:12] in text


def test_an_empty_report_still_renders():
    r = ReportData(generated_at="2026-09-18T00:00:00+00:00", project="p", eval_set="e")
    block = results_block(r)
    assert BEGIN in block and END in block


# --------------------------------------------------------------------------------------------------------------
# HTML
#
# The report has to survive being attached to an email and opened offline. No JavaScript, no external assets.
# --------------------------------------------------------------------------------------------------------------


def test_html_renders_both_charts(registry, cfg):
    from agentdistill.report.html import render

    r = built(registry, cfg)
    r.cascade["analytic"] = [{"cost_per_turn": 1 + i * 0.3, "cascade_success": 0.5 + i * 0.03} for i in range(6)]
    html = render(r)
    assert html.count("<svg") == 2


def test_html_is_self_contained(registry, cfg):
    from agentdistill.report.html import render

    html = render(built(registry, cfg))
    assert "<script" not in html
    assert "http://" not in html.replace("http://www.w3.org/2000/svg", "")
    assert "https://" not in html


def test_html_escapes_untrusted_text(registry, cfg):
    from agentdistill.report.html import render

    r = built(registry, cfg)
    r.warnings.append("<img src=x onerror=alert(1)>")
    html = render(r)
    assert "<img src=x" not in html
    assert "&lt;img" in html


def test_html_renders_warnings_at_the_top(registry, cfg):
    from agentdistill.report.html import render

    r = assemble(seed(registry, with_calibration=False), cfg, tag_glob="gpu-day")
    html = render(r)
    assert 'class="warn"' in html
    assert html.index('class="warn"') < html.index("<h2>")


def test_html_marks_clusters_below_the_floor(registry, cfg):
    from agentdistill.report.html import render

    cfg.router.floor = 0.55
    html = render(assemble(seed(registry), cfg, tag_glob="gpu-day"))
    assert 'class="below"' in html


def test_html_renders_from_an_empty_report():
    from agentdistill.report.html import render

    html = render(ReportData(generated_at="t", project="p", eval_set="e"))
    assert html.startswith("<!doctype html>") and html.rstrip().endswith("</html>")


def test_html_writes_to_disk(registry, cfg, tmp_path):
    from agentdistill.report.html import write

    path = write(built(registry, cfg), tmp_path / "nested" / "report.html")
    assert path.exists() and "<svg" in path.read_text()
