"""The canary split.

A deterministic share of student-routed traffic goes to the canary adapter. Deterministic on the request id, so
a retried request lands on the same adapter -- otherwise a retry would silently compare two adapters on one
task and muddy the very comparison the canary exists to produce.
"""

from __future__ import annotations

import hashlib


def bucket(request_id: str) -> int:
    """A stable 0-99 bucket for a request id.

    A digest rather than the id's last bytes: request ids are often sequential or timestamped, and slicing those
    produces buckets that correlate with time of day.
    """
    return int(hashlib.sha256(request_id.encode("utf-8")).hexdigest()[:8], 16) % 100


def use_canary(request_id: str, canary_share: float, canary_adapter: str | None) -> bool:
    if not canary_adapter or canary_share <= 0:
        return False
    if canary_share >= 1:
        return True
    return bucket(request_id) < canary_share * 100


def choose_adapter(
    request_id: str, prod_adapter: str | None, canary_adapter: str | None, canary_share: float
) -> tuple[str | None, bool]:
    """Returns (adapter, is_canary)."""
    if use_canary(request_id, canary_share, canary_adapter):
        return canary_adapter, True
    return prod_adapter, False
