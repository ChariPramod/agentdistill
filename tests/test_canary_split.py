"""The canary split.

Two properties matter. It must be stable -- a retried request lands on the same adapter, or a retry would
silently compare two adapters on one task and muddy the comparison the canary exists to produce. And it must hit
the configured share, because the share is what bounds the blast radius of a bad canary.

The second property is where the plan's sketch was wrong; see `test_the_plans_two_hex_digit_split_misses_its_share`.
"""

from __future__ import annotations

import uuid

import pytest

from agentdistill.router.canary import bucket, choose_adapter, use_canary


def _ids(n: int) -> list[str]:
    return [uuid.UUID(int=i).hex for i in range(n)]


@pytest.mark.parametrize("share", [0.05, 0.1, 0.25, 0.5])
def test_the_share_is_hit_within_a_point(share):
    ids = _ids(10_000)
    hit = sum(use_canary(i, share, "canary-v1") for i in ids) / len(ids)
    assert abs(hit - share) < 0.01, f"share {share} came out at {hit}"


def test_the_same_id_always_maps_to_the_same_adapter():
    for request_id in _ids(500):
        first = choose_adapter(request_id, "prod-v1", "canary-v1", 0.3)
        for _ in range(3):
            assert choose_adapter(request_id, "prod-v1", "canary-v1", 0.3) == first


def test_no_canary_adapter_means_no_canary_traffic():
    assert not any(use_canary(i, 0.5, None) for i in _ids(200))
    assert choose_adapter("abc", "prod-v1", None, 0.5) == ("prod-v1", False)


def test_zero_share_sends_nothing_and_full_share_sends_everything():
    ids = _ids(200)
    assert not any(use_canary(i, 0.0, "canary-v1") for i in ids)
    assert all(use_canary(i, 1.0, "canary-v1") for i in ids)


def test_choose_adapter_reports_which_arm_it_picked():
    ids = _ids(400)
    picks = [choose_adapter(i, "prod-v1", "canary-v1", 0.25) for i in ids]
    assert all((name == "canary-v1") == is_canary for name, is_canary in picks)
    assert {name for name, _ in picks} == {"prod-v1", "canary-v1"}


def test_buckets_are_spread_over_the_whole_range():
    counts = [0] * 100
    for i in _ids(20_000):
        counts[bucket(i)] += 1
    # 20,000 ids over 100 buckets: 200 each in expectation, and nothing empty or wildly over.
    assert min(counts) > 120
    assert max(counts) < 300


def test_the_plans_two_hex_digit_split_misses_its_share():
    """The plan specified `int(request_id[-2:], 16) % 100 < share * 100`. It does not hold the share.

    Two hex digits give 256 values, and 256 does not divide by 100: buckets 0-55 collect three source values
    each and 56-99 collect two. A 10% share therefore takes buckets 0-9, which is 30/256 = 11.7% of traffic --
    outside the plan's own 9-11% acceptance band. It is also sensitive to the shape of the id: request ids are
    often sequential or timestamped, and their last byte then correlates with arrival time rather than being
    uniform. Hashing first fixes both.
    """
    ids = _ids(10_000)
    plan_share = sum(int(i[-2:], 16) % 100 < 10 for i in ids) / len(ids)
    ours = sum(use_canary(i, 0.1, "canary-v1") for i in ids) / len(ids)

    assert plan_share > 0.11, "the 256/100 rounding bias should be visible"
    assert abs(ours - 0.1) < 0.01
