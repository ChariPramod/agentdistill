"""The insufficient-power marker, from `compare` to every place that reads it.

The rule under test: below the power floor a comparison carries a reason and the raw observed rates, and nothing
downstream may render it as a result or decide on it as though it were one.
"""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from agentdistill.eval.report import render_comparison
from agentdistill.eval.stats import InsufficientPower
from agentdistill.report.assemble import ReportData
from agentdistill.report.html import render
from agentdistill.report.markdown import comparison_line, full_markdown, results_block
from agentdistill.retrain import gate_not_worse_than_prod
from agentdistill.train.onpolicy import RoundCfg, decide


def _refused(n_tasks: int = 5, n_repeats: int = 1, observed: dict | None = None, **extra) -> dict:
    return {
        "subject_a": "student", "subject_b": "teacher", "run_a": "ev_a", "run_b": "ev_b",
        "eval_set_id": "es", "n_shared_tasks": n_tasks, "n_repeats": n_repeats,
        "metrics_a": {}, "metrics_b": {},
        "insufficient_power": InsufficientPower(n_tasks, n_repeats).as_dict(),
        "observed": observed if observed is not None else {"rate_a": 0.4, "rate_b": 0.8},
        **extra,
    }


def test_comparison_line_prints_the_reason_and_observed_rates():
    line = comparison_line("student vs teacher", _refused())
    assert line.startswith("student vs teacher: not enough data for a comparison: 5 tasks (need at least 20)")
    assert "Observed: 40.0% vs 80.0%, not compared." in line
    assert "p=" not in line and "pp" not in line


def test_comparison_line_without_observed_rates():
    line = comparison_line("x", _refused(observed={}))
    assert "Observed" not in line


def test_comparison_line_renders_statistics_above_the_floor():
    cmp = {"success": {"delta": 0.05, "ci95": [-0.01, 0.11]}, "mcnemar": {"p": 0.2},
           "tokens": {"median_delta": -40.0, "ci95": [-60.0, -10.0]}}
    line = comparison_line("s", cmp)
    assert "+5.0 pp [-1.0, +11.0]" in line and "McNemar p=0.2" in line


# Whatever else a malformed or stale comparison dict carries -- including a leftover `success` block with a
# p-value in it -- the marker wins. That is the property: no p-value next to insufficient power, in any renderer.
_junk = st.fixed_dictionaries({}, optional={
    "success": st.just({"delta": 0.1, "ci95": [0.0, 0.2]}),
    "mcnemar": st.just({"p": 0.01}),
    "tokens": st.just({"median_delta": -5.0, "ci95": [-9.0, -1.0], "p": 0.03}),
    "holm": st.just({"success": {"p": 0.01, "p_adjusted": 0.03, "significant": True}}),
})


@given(n_tasks=st.integers(0, 19), n_repeats=st.integers(0, 10), junk=_junk,
       rate_a=st.none() | st.floats(0, 1), rate_b=st.none() | st.floats(0, 1))
def test_no_renderer_prints_a_p_value_next_to_insufficient_power(n_tasks, n_repeats, junk, rate_a, rate_b):
    cmp = _refused(n_tasks, n_repeats, observed={"rate_a": rate_a, "rate_b": rate_b}, **junk)
    report = ReportData(generated_at="t", project="p", eval_set="e",
                        paired={"student_vs_teacher": cmp, "student_vs_base": cmp})
    rendered = {
        "line": comparison_line("x", cmp),
        "cli": render_comparison(cmp),
        "block": results_block(report),
        "markdown": full_markdown(report),
        "html": render(report),
    }
    for name, text in rendered.items():
        for line in text.splitlines():
            assert "p=" not in line, f"{name} printed a p-value beside an insufficient-power marker: {line}"


def test_html_and_markdown_both_show_the_reason():
    report = ReportData(generated_at="t", project="p", eval_set="e", paired={"student_vs_teacher": _refused()})
    assert "need at least 20" in results_block(report)
    assert "need at least 20" in render(report)


# --------------------------------------------------------------------------------------------------------------
# the gates that decide on a comparison
# --------------------------------------------------------------------------------------------------------------


def test_retrain_gate_refuses_an_unpowered_comparison():
    ok, detail = gate_not_worse_than_prod({"comparison": _refused()})
    assert not ok
    assert "need at least 20" in detail


def test_onpolicy_round_is_discarded_on_an_unpowered_comparison():
    decision, reason = decide(_refused(), {"schema_valid": 1.0, "divergence_rate": 0.0},
                              {"schema_valid": 1.0, "divergence_rate": 0.0}, RoundCfg())
    assert decision == "discard"
    assert "not enough data" in reason
