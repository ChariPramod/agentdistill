"""Comparing two adapters on live traffic.

Live traffic is not a randomized trial, so the comparison is paired within clusters: if the canary happened to
draw more of an easy cluster, an unpaired comparison would credit it for the mix rather than the model. The
other rule is that too little data returns `None` rather than a wide interval, because a caller that sees a
number acts on it.
"""

from __future__ import annotations

import numpy as np
import pytest

from agentdistill.router.compare_live import compare_live

PROD, CANARY = "prod-v1", "canary-v2"


def rows(
    clusters: int = 4,
    per_arm: int = 40,
    prod_rate: float = 0.70,
    lift: float = 0.05,
    seed: int = 0,
    **over,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    for c in range(clusters):
        for adapter, rate in ((PROD, prod_rate), (CANARY, prod_rate + lift)):
            for _ in range(per_arm):
                out.append({
                    "adapter_id": adapter, "cluster_id": c, "arm": "student",
                    "outcome": bool(rng.random() < rate), "fallback": False, **over,
                })
    return out


def test_a_real_lift_shows_up_as_a_positive_delta_with_a_ci_above_zero():
    result = compare_live(rows(clusters=6, per_arm=200, lift=0.05), PROD, CANARY)
    assert result is not None
    assert result["success"]["delta"] > 0
    assert result["success"]["ci95"][0] > 0, result["success"]


def test_no_real_difference_gives_a_ci_that_contains_zero():
    result = compare_live(rows(clusters=6, per_arm=200, lift=0.0, seed=3), PROD, CANARY)
    assert result is not None
    lo, hi = result["success"]["ci95"]
    assert lo < 0 < hi, result["success"]


def test_fewer_than_three_qualifying_clusters_returns_none():
    assert compare_live(rows(clusters=2), PROD, CANARY) is None


def test_a_cluster_with_too_few_observations_does_not_qualify():
    # Three clusters, but one of them has four observations on the canary side.
    data = rows(clusters=3, per_arm=20)
    data = [r for r in data if not (r["cluster_id"] == 2 and r["adapter_id"] == CANARY)]
    data += [{"adapter_id": CANARY, "cluster_id": 2, "arm": "student", "outcome": True, "fallback": False}] * 4
    assert compare_live(data, PROD, CANARY) is None

    # One more observation and it qualifies.
    data += [{"adapter_id": CANARY, "cluster_id": 2, "arm": "student", "outcome": True, "fallback": False}]
    result = compare_live(data, PROD, CANARY)
    assert result is not None and result["n_clusters"] == 3


def test_fallback_rows_are_ignored():
    """A fallback served the teacher. Counting it as a canary observation would score the teacher's work as
    the canary's."""
    data = rows(clusters=4, per_arm=40, lift=0.0, seed=7)
    poisoned = data + [
        {"adapter_id": CANARY, "cluster_id": c, "arm": "student", "outcome": True, "fallback": True}
        for c in range(4) for _ in range(100)
    ]
    clean = compare_live(data, PROD, CANARY)
    with_fallbacks = compare_live(poisoned, PROD, CANARY)
    assert clean == with_fallbacks


def test_rows_without_an_outcome_or_cluster_are_ignored():
    data = rows(clusters=4, per_arm=40)
    noise = [
        {"adapter_id": CANARY, "cluster_id": 0, "arm": "student", "outcome": None, "fallback": False},
        {"adapter_id": CANARY, "cluster_id": None, "arm": "student", "outcome": True, "fallback": False},
        {"adapter_id": CANARY, "cluster_id": 0, "arm": "teacher", "outcome": True, "fallback": False},
        {"adapter_id": "some-third-adapter", "cluster_id": 0, "arm": "student",
         "outcome": True, "fallback": False},
    ]
    assert compare_live(data, PROD, CANARY) == compare_live(data + noise, PROD, CANARY)


def test_an_unbalanced_cluster_mix_does_not_fake_a_lift():
    """The pairing's whole job: the canary draws mostly the easy cluster and still scores as no better."""
    rng = np.random.default_rng(11)
    rates = {0: 0.95, 1: 0.60, 2: 0.55, 3: 0.50}
    data = []
    for c, rate in rates.items():
        # The canary gets far more of cluster 0 and less of the hard ones.
        n_canary = 300 if c == 0 else 20
        n_prod = 20 if c == 0 else 300
        for adapter, n in ((PROD, n_prod), (CANARY, n_canary)):
            for _ in range(n):
                data.append({"adapter_id": adapter, "cluster_id": c, "arm": "student",
                             "outcome": bool(rng.random() < rate), "fallback": False})

    naive = (
        np.mean([r["outcome"] for r in data if r["adapter_id"] == CANARY])
        - np.mean([r["outcome"] for r in data if r["adapter_id"] == PROD])
    )
    result = compare_live(data, PROD, CANARY)

    assert naive > 0.15, "the unpaired comparison should be badly fooled by the mix"
    assert result is not None
    assert abs(result["success"]["delta"]) < 0.05, result["success"]


def test_the_result_reports_what_it_was_computed_from():
    result = compare_live(rows(clusters=4, per_arm=40), PROD, CANARY)
    assert result is not None
    assert result["n_clusters"] == 4
    assert result["n_prod"] == 160
    assert result["n_canary"] == 160
    assert set(result["per_cluster"]) == {"0", "1", "2", "3"}
    assert result["clusters"] == [0, 1, 2, 3]


def test_it_is_deterministic_for_a_given_seed():
    """Same seed, same interval -- a report that moves when rerun is not a report.

    The converse is not asserted: with a handful of clusters the resample space is small enough that two seeds
    legitimately land on the same percentiles.
    """
    data = rows(clusters=5, per_arm=50)
    assert compare_live(data, PROD, CANARY, seed=4) == compare_live(data, PROD, CANARY, seed=4)


def test_the_point_estimate_does_not_depend_on_the_bootstrap_seed():
    data = rows(clusters=5, per_arm=50)
    a = compare_live(data, PROD, CANARY, seed=4)
    b = compare_live(data, PROD, CANARY, seed=99)
    assert a is not None and b is not None
    assert a["success"]["delta"] == b["success"]["delta"]


def test_empty_traffic_returns_none():
    assert compare_live([], PROD, CANARY) is None


def test_thinner_arm_weights_the_cluster():
    """A cluster measured on five observations should not outweigh one measured on five hundred."""
    data = rows(clusters=3, per_arm=400, lift=0.0, seed=2)
    # A fourth cluster, tiny, where the canary happens to win every one of its five.
    for adapter, outcome in ((PROD, False), (CANARY, True)):
        data += [{"adapter_id": adapter, "cluster_id": 3, "arm": "student",
                  "outcome": outcome, "fallback": False}] * 5

    result = compare_live(data, PROD, CANARY)
    assert result is not None
    # The tiny cluster's +1.0 difference is real but carries weight 5 against 3 x 400.
    assert result["per_cluster"]["3"]["delta"] == pytest.approx(1.0)
    assert result["success"]["delta"] < 0.02, result["success"]
