"""Loading and persisting the router's posteriors."""

from __future__ import annotations

import logging
from typing import Any

from agentdistill.report.registry_views import (
    per_cluster_counts,
    reset_router_state,
    router_state,
    upsert_router_state,
)
from agentdistill.router.thompson import ArmState, ThompsonRouter

logger = logging.getLogger(__name__)


def load_router(registry: Any, cfg: Any, cost: dict[str, float] | None = None) -> ThompsonRouter:
    """Load the router, warm-starting from eval counts when there is no stored state.

    The warm start only happens on an empty table. Re-running it over live posteriors would throw away
    everything the router had learned from real traffic in favour of an offline eval.
    """
    rows = router_state(registry)
    state = {(int(r["cluster_id"]), r["arm"]): ArmState(float(r["alpha"]), float(r["beta"])) for r in rows}
    router = ThompsonRouter(
        state=state,
        cost=cost or {"student": 0.0, "teacher": 0.0},
        lam=cfg.router.lambda_per_usd,
        floor=cfg.router.floor,
        explore_cap=cfg.router.explore_cap,
        decay=cfg.router.decay,
        min_observations=cfg.router.min_observations,
        seed=cfg.router.seed,
    )
    if not state:
        counts = _warm_start_counts(registry, cfg)
        if counts:
            router.warm_start(counts)
            flush_router(registry, router)
            logger.info("router warm-started from %d cluster/arm eval counts", len(counts))
        else:
            logger.warning(
                "no router state and no per-cluster eval counts to warm-start from; the router begins blind "
                "and will spend its first requests rediscovering what an eval would have told it"
            )
    return router


def _warm_start_counts(registry: Any, cfg: Any) -> dict[tuple[int, str], tuple[int, int]]:
    """Per-cluster counts for both arms, from the prod adapter's and the teacher's latest eval runs."""
    from agentdistill.registry.select import NoMatch, latest_eval, prod_adapter

    counts: dict[tuple[int, str], tuple[int, int]] = {}
    prod = prod_adapter(registry)
    for arm, subject in (("student", prod["id"] if prod else None), ("teacher", "teacher")):
        if not subject:
            continue
        try:
            run = latest_eval(registry, subject=subject, eval_set=cfg.eval.eval_set)
        except NoMatch:
            continue
        for (cluster, _), value in per_cluster_counts(registry, run["id"]).items():
            counts[(cluster, arm)] = value
    return counts


def flush_router(registry: Any, router: ThompsonRouter) -> None:
    upsert_router_state(registry, router.snapshot())


def reset_router(registry: Any) -> None:
    """Forget everything.

    Called when a new adapter reaches prod: the old adapter's record is not evidence about the new one, and
    inheriting it would take hundreds of requests to unlearn.
    """
    reset_router_state(registry)


class RouterStore:
    """The gateway's handle on persistence.

    Flushing after every feedback call is a write per request, which is affordable at the traffic a distilled
    agent sees and is the only way a gateway restart does not lose what the router learned. If that ever becomes
    the bottleneck, batch here rather than in the request path.
    """

    def __init__(self, registry: Any) -> None:
        self.registry = registry

    def flush(self, router: ThompsonRouter) -> None:
        flush_router(self.registry, router)

    def reset(self) -> None:
        reset_router(self.registry)
