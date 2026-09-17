"""The router's arm selection.

Textbook Thompson sampling is not quite what production wants. The two departures -- a hard floor and decayed
posteriors -- are what these tests pin down, because both exist to stop the router doing something defensible in
a simulation and expensive with real customers.
"""

from __future__ import annotations

import pytest

from agentdistill.router.thompson import ArmState, ThompsonRouter


def run(router: ThompsonRouter, cluster: int, n: int) -> dict[str, int]:
    counts = {"student": 0, "teacher": 0}
    for _ in range(n):
        counts[router.choose(cluster)] += 1
    return counts


def test_a_clearly_better_student_gets_the_traffic():
    r = ThompsonRouter(seed=1)
    for _ in range(80):
        r.update(0, "student", True)
        r.update(0, "teacher", False)
    counts = run(r, 0, 500)
    assert counts["student"] > 450


def test_the_floor_stops_traffic_to_a_bad_student():
    """Pure Thompson sampling keeps a trickle going to a known-bad arm forever. Here each sample is a customer."""
    r = ThompsonRouter(floor=0.55, min_observations=10, seed=2)
    for _ in range(40):
        r.update(0, "student", False)
    assert run(r, 0, 300)["student"] == 0


def test_the_floor_waits_for_evidence():
    """Three failures is not evidence. Cutting a cluster off on three observations would strand a good adapter."""
    r = ThompsonRouter(floor=0.55, min_observations=10, seed=3)
    for _ in range(3):
        r.update(0, "student", False)
    assert run(r, 0, 300)["student"] > 0


def test_a_recovered_student_is_routed_to_again():
    """The floor is a state, not a sentence. Feedback from teacher-served traffic can lift a cluster back."""
    r = ThompsonRouter(floor=0.55, min_observations=10, decay=0.95, seed=4)
    for _ in range(20):
        r.update(0, "student", False)
    assert run(r, 0, 200)["student"] == 0
    for _ in range(60):
        r.update(0, "student", True)
    assert run(r, 0, 200)["student"] > 150


def test_decay_lets_a_retrained_adapter_escape_its_predecessors_record():
    """Without decay, 500 old observations take 500 new ones to overturn."""
    decayed = ThompsonRouter(decay=0.99, seed=5)
    undecayed = ThompsonRouter(decay=1.0, seed=5)
    for r in (decayed, undecayed):
        for _ in range(500):
            r.update(0, "student", False)
        for _ in range(60):
            r.update(0, "student", True)

    # 60 successes against a half-life of ~69 updates does not fully flip the decayed arm, but it moves it an
    # order of magnitude; the undecayed arm is still buried under its predecessor's record.
    assert decayed.arm(0, "student").mean > 0.4
    assert undecayed.arm(0, "student").mean < 0.15
    assert decayed.arm(0, "student").mean > 3 * undecayed.arm(0, "student").mean


def test_cost_breaks_a_tie_toward_the_cheaper_arm():
    equal = {"student": 0.0, "teacher": 0.0}
    priced = {"student": 0.0002, "teacher": 0.01}
    for _ in range(1):
        a = ThompsonRouter(cost=equal, lam=20.0, seed=6)
        b = ThompsonRouter(cost=priced, lam=20.0, seed=6)
        for r in (a, b):
            for _ in range(60):
                r.update(0, "student", True)
                r.update(0, "teacher", True)
        assert run(b, 0, 300)["student"] >= run(a, 0, 300)["student"]


def test_cost_does_not_override_a_floor_breach():
    """No price makes a broken student worth routing to."""
    r = ThompsonRouter(cost={"student": 0.0, "teacher": 1.0}, lam=100.0, floor=0.55, min_observations=10, seed=7)
    for _ in range(30):
        r.update(0, "student", False)
    assert run(r, 0, 200)["student"] == 0


def test_clusters_are_learned_independently():
    r = ThompsonRouter(seed=8)
    for _ in range(60):
        r.update(0, "student", True)
        r.update(1, "student", False)
        r.update(1, "teacher", True)
    assert run(r, 0, 200)["student"] > 150
    assert run(r, 1, 200)["student"] < 50


def test_warm_start_seeds_posteriors_from_counts():
    r = ThompsonRouter(seed=9)
    r.warm_start({(0, "student"): (18, 2), (0, "teacher"): (19, 1)})
    assert r.arm(0, "student").mean == pytest.approx(19 / 22)
    assert r.arm(0, "student").observations == pytest.approx(20)


def test_the_prior_is_not_counted_as_evidence():
    """Beta(1,1) contributes two pseudo-observations. Counting them would trip the floor's threshold early."""
    assert ArmState().observations == 0
    r = ThompsonRouter(min_observations=10, seed=10)
    for _ in range(9):
        r.update(0, "student", False)
    assert r.arm(0, "student").observations < 10
    assert run(r, 0, 100)["student"] > 0


def test_an_unseen_cluster_starts_neutral_and_still_chooses():
    r = ThompsonRouter(seed=11)
    assert r.state_mean(99, "student") == pytest.approx(0.5)
    assert run(r, 99, 200)["student"] > 40


def test_exploration_is_capped_once_both_arms_are_settled():
    r = ThompsonRouter(explore_cap=0.10, exploit_after=30, seed=12)
    for _ in range(200):
        r.update(0, "student", True)
        r.update(0, "teacher", True)
    # Both arms near 1.0; the tie goes to the student, and the teacher only appears via the explore cap.
    counts = run(r, 0, 2000)
    assert counts["teacher"] < 0.2 * 2000


def test_snapshot_round_trips_through_state():
    r = ThompsonRouter(seed=13)
    r.update(0, "student", True)
    r.update(1, "teacher", False)
    restored = ThompsonRouter(state={(c, a): ArmState(al, be) for c, a, al, be in r.snapshot()}, seed=13)
    assert restored.snapshot() == r.snapshot()


def test_summary_names_the_clusters_below_the_floor():
    r = ThompsonRouter(floor=0.55, min_observations=10, seed=14)
    for _ in range(30):
        r.update(0, "student", False)
        r.update(1, "student", True)
    summary = r.summary()
    assert summary["below_floor"] == [0]
    assert summary["clusters"] == 2


def test_a_decay_that_would_disable_the_floor_is_refused():
    """Decay caps the evidence an arm can hold. A floor that can never engage is a floor that is not there."""
    with pytest.raises(ValueError, match="could never engage"):
        ThompsonRouter(decay=0.9, min_observations=10)

    # The defaults leave plenty of room: 1/(1-0.995) - 2 = 198 observations against a threshold of 10.
    ThompsonRouter()
