"""Registry selectors.

The GPU-day script substitutes these into the next command. A selector that returns the wrong row trains the
wrong thing; one that returns nothing produces a blank argument and a confusing failure several stages later.
"""

from __future__ import annotations

import pytest

from agentdistill.registry.base import utcnow
from agentdistill.registry.select import (
    Ambiguous,
    NoMatch,
    best_adapter,
    canary_adapter,
    latest_adapter,
    latest_calibration,
    latest_dataset,
    latest_eval,
    latest_training_run,
    prod_adapter,
    rounds_for_tag,
)


@pytest.fixture
def seeded(registry):
    """Two tagged adapters with evals, one untagged, one quantized, plus datasets and a prod/canary pair."""
    for i, (name, kind) in enumerate([("support", "sft"), ("support", "sft"), ("pairs", "dpo")]):
        registry.insert_dataset({
            "id": f"ds{i}", "name": name, "version": i + 1, "kind": kind, "filter_config": {},
            "n_samples": 10 * (i + 1), "n_tokens": 100, "content_hash": f"h{i}", "path": f"/tmp/{i}",
            "created_at": f"2026-09-1{i}T00:00:00+00:00",
        })
    registry.insert_training_run({
        "id": "tr1", "dataset_id": "ds0", "base_model": "m", "method": "sft", "config": {},
        "status": "succeeded", "started_at": "2026-09-10T00:00:00+00:00",
    })
    registry.insert_training_run({
        "id": "tr2", "dataset_id": "ds0", "base_model": "m", "method": "dpo", "config": {},
        "status": "succeeded", "started_at": "2026-09-11T00:00:00+00:00",
    })

    specs = [
        ("ad_sft", "support", 1, "gpu-day", None, "2026-09-10T00:00:00+00:00", "candidate"),
        ("ad_r1", "support", 2, "gpu-day-r1", None, "2026-09-11T00:00:00+00:00", "candidate"),
        ("ad_old", "support", 3, None, None, "2026-09-09T00:00:00+00:00", "retired"),
        ("ad_q", "support", 4, "gpu-day", "awq", "2026-09-12T00:00:00+00:00", "candidate"),
        ("ad_prod", "support", 5, None, None, "2026-09-08T00:00:00+00:00", "prod"),
        ("ad_can", "support", 6, None, None, "2026-09-07T00:00:00+00:00", "canary"),
        # Tagged but never evaluated, so `best` must refuse rather than pick it.
        ("ad_untested", "support", 7, "untested", None, "2026-09-13T00:00:00+00:00", "candidate"),
    ]
    for aid, name, version, tag, quant, created, status in specs:
        registry.insert_adapter({
            "id": aid, "training_run_id": "tr1", "name": name, "version": version, "base_model": "m",
            "path": f"/tmp/{aid}", "tag": tag, "quantization": quant, "created_at": created, "status": status,
        })

    registry.insert_eval_set({"id": "es_hold", "name": "hold", "trace_ids": [], "grader": {}})
    registry.insert_eval_set({"id": "es_unseen", "name": "unseen", "trace_ids": [], "grader": {}})
    for run_id, subject, es, success, tag, started in [
        ("ev_base", "base", "es_hold", 0.40, "gpu-day", "2026-09-10T01:00:00+00:00"),
        ("ev_sft", "ad_sft", "es_hold", 0.62, "gpu-day", "2026-09-10T02:00:00+00:00"),
        ("ev_r1", "ad_r1", "es_hold", 0.71, "gpu-day-r1", "2026-09-11T02:00:00+00:00"),
        ("ev_r1_unseen", "ad_r1", "es_unseen", 0.55, "gpu-day-r1", "2026-09-11T03:00:00+00:00"),
        ("ev_stale", "ad_sft", "es_hold", 0.10, "old", "2026-09-01T00:00:00+00:00"),
    ]:
        registry.start_eval_run(run_id, es, subject, 5, tag=tag)
        registry.finish_eval_run(run_id, {"success": success, "schema_valid": 1.0})
        with registry.engine.begin() as conn:
            from sqlalchemy import text

            conn.execute(text("UPDATE eval_runs SET started_at = :s WHERE id = :i"),
                         {"s": started, "i": run_id})
    return registry


# --------------------------------------------------------------------------------------------------------------
# datasets
# --------------------------------------------------------------------------------------------------------------


def test_latest_dataset_by_name_and_kind(seeded):
    assert latest_dataset(seeded, name="support", kind="sft")["id"] == "ds1"
    assert latest_dataset(seeded, kind="dpo")["id"] == "ds2"


def test_latest_dataset_decodes_filter_config(seeded):
    assert latest_dataset(seeded)["filter_config"] == {}


def test_latest_dataset_no_match_names_the_query(seeded):
    with pytest.raises(NoMatch, match="nope"):
        latest_dataset(seeded, name="nope")


# --------------------------------------------------------------------------------------------------------------
# adapters
# --------------------------------------------------------------------------------------------------------------


def test_latest_adapter_by_exact_tag(seeded):
    assert latest_adapter(seeded, tag="gpu-day-r1")["id"] == "ad_r1"


def test_latest_adapter_tag_glob_spans_rounds(seeded):
    """`gpu-day*` is how the script picks up round-1 adapters as well as the base run's."""
    assert latest_adapter(seeded, tag="gpu-day*")["id"] == "ad_q"


def test_tag_glob_does_not_match_unrelated_tags(seeded):
    assert latest_adapter(seeded, tag="gpu-day*")["tag"].startswith("gpu-day")


def test_latest_adapter_quantized_filter(seeded):
    assert latest_adapter(seeded, quantized=True)["id"] == "ad_q"
    assert latest_adapter(seeded, quantized=False)["id"] != "ad_q"


def test_latest_adapter_by_status(seeded):
    assert latest_adapter(seeded, status="prod")["id"] == "ad_prod"


def test_latest_adapter_no_match(seeded):
    with pytest.raises(NoMatch, match="no adapter"):
        latest_adapter(seeded, tag="never-used")


def test_best_adapter_ranks_by_measured_success(seeded):
    assert best_adapter(seeded, tag="gpu-day*", eval_set="es_hold")["id"] == "ad_r1"


def test_best_adapter_ignores_an_unevaluated_adapter(seeded):
    """ad_q is newest and tagged, but has no eval; promoting it would put an unmeasured model in the report."""
    assert best_adapter(seeded, tag="gpu-day*", eval_set="es_hold")["id"] != "ad_q"


def test_best_adapter_is_scoped_to_the_eval_set(seeded):
    """Success on the unseen set is a different number and must not be mixed with holdout rankings."""
    assert best_adapter(seeded, tag="gpu-day-r1", eval_set="es_unseen")["id"] == "ad_r1"


def test_best_adapter_refuses_when_nothing_is_evaluated(seeded):
    """The tag matches an adapter; it simply has no measurement, so there is nothing to rank."""
    with pytest.raises(NoMatch, match="has an eval run"):
        best_adapter(seeded, tag="untested", eval_set="es_hold")


def test_best_adapter_refuses_an_unknown_tag(seeded):
    with pytest.raises(NoMatch, match="no adapter"):
        best_adapter(seeded, tag="never-used", eval_set="es_hold")


def test_prod_and_canary(seeded):
    assert prod_adapter(seeded)["id"] == "ad_prod"
    assert canary_adapter(seeded)["id"] == "ad_can"


def test_two_prod_adapters_is_an_error(seeded):
    seeded.insert_adapter({
        "id": "ad_prod2", "training_run_id": "tr1", "name": "support", "version": 9, "base_model": "m",
        "path": "/tmp/x", "status": "prod",
    })
    with pytest.raises(Ambiguous, match="exactly one"):
        prod_adapter(seeded)


# --------------------------------------------------------------------------------------------------------------
# eval runs
# --------------------------------------------------------------------------------------------------------------


def test_latest_eval_by_subject_prefers_the_newest(seeded):
    """A stale run from a previous session must not win just because it matches the subject."""
    assert latest_eval(seeded, subject="ad_sft")["id"] == "ev_sft"


def test_latest_eval_by_tag(seeded):
    assert latest_eval(seeded, tag="gpu-day-r1", eval_set="es_hold")["id"] == "ev_r1"


def test_latest_eval_by_eval_set_accepts_a_bare_name(seeded):
    """The script passes `support-holdout-v1`; the registry stores `es_support-holdout-v1`."""
    assert latest_eval(seeded, subject="ad_r1", eval_set="unseen")["id"] == "ev_r1_unseen"


def test_latest_eval_decodes_metrics(seeded):
    assert latest_eval(seeded, subject="base")["metrics"]["success"] == 0.40


def test_latest_eval_no_match(seeded):
    with pytest.raises(NoMatch, match="no eval run"):
        latest_eval(seeded, subject="nobody")


# --------------------------------------------------------------------------------------------------------------
# training runs, rounds, calibrations
# --------------------------------------------------------------------------------------------------------------


def test_latest_training_run_by_method(seeded):
    assert latest_training_run(seeded, method="dpo")["id"] == "tr2"
    assert latest_training_run(seeded)["id"] == "tr2"


def test_latest_calibration_requires_one(seeded):
    with pytest.raises(NoMatch, match="no calibration"):
        latest_calibration(seeded, adapter_id="ad_sft")


def test_rounds_for_tag(seeded):
    from sqlalchemy import text

    with seeded.engine.begin() as conn:
        for i, decision in enumerate(["promote", "discard"]):
            conn.execute(
                text("""INSERT INTO onpolicy_rounds (id, tag, round_idx, start_adapter_id, decision, started_at)
                        VALUES (:id, 'gpu-day', :idx, 'ad_sft', :d, :t)"""),
                {"id": f"r{i}", "idx": i, "d": decision, "t": utcnow()},
            )
    rows = rounds_for_tag(seeded, "gpu-day")
    assert [r["round_idx"] for r in rows] == [0, 1]
    assert [r["decision"] for r in rows] == ["promote", "discard"]


def test_rounds_for_an_unknown_tag_is_empty(seeded):
    assert rounds_for_tag(seeded, "nope") == []
