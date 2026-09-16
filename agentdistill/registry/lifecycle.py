"""Adapter lifecycle.

`candidate` → `canary` → `prod`, with every transition gated on measured evidence and recorded as an event
carrying the checks that justified it. Months later, "why is this adapter in prod" has an answer.

The checks are deliberately the kind that can fail. A promotion path where everything always passes is
decoration; these compare against the current prod adapter on a frozen eval set and refuse when the comparison
is missing, not just when it is bad. `--force` exists because sometimes you know better than the checks, and it
records that you overrode them.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import text

from agentdistill.registry.base import dumps, loads, utcnow

#: Legal moves. Everything else is refused, including the ones that look harmless: `prod` back to `canary` would
#: leave two adapters claiming live traffic.
TRANSITIONS: set[tuple[str, str]] = {
    ("candidate", "canary"),
    ("candidate", "prod"),
    ("candidate", "retired"),
    ("canary", "prod"),
    ("canary", "retired"),
    ("prod", "retired"),
}

#: Success may not regress by more than this, measured at the low end of the interval.
MAX_SUCCESS_REGRESSION_PP = 1.0
MIN_SCHEMA_VALID = 0.99
MAX_HOLDOUT_ECE = 0.05


class IllegalTransition(ValueError):
    pass


@dataclass
class Check:
    ok: bool
    detail: str = ""
    value: Any = None

    def to_dict(self) -> dict:
        return {"ok": self.ok, "detail": self.detail, "value": self.value}


@dataclass
class PromotionResult:
    ok: bool
    checks: dict[str, Check] = field(default_factory=dict)
    forced: bool = False
    from_status: str | None = None
    to_status: str | None = None

    @property
    def failed(self) -> list[str]:
        return [name for name, c in self.checks.items() if not c.ok]

    def to_dict(self) -> dict:
        return {
            "ok": self.ok, "forced": self.forced, "from_status": self.from_status, "to_status": self.to_status,
            "checks": {k: v.to_dict() for k, v in self.checks.items()},
        }


def adapter(registry: Any, adapter_id: str) -> dict:
    with registry.engine.connect() as conn:
        row = conn.execute(
            text("SELECT * FROM adapters WHERE id = :i OR name = :i ORDER BY version DESC LIMIT 1"),
            {"i": adapter_id},
        ).mappings().first()
    if row is None:
        raise LookupError(f"no adapter {adapter_id!r}")
    return dict(row)


def promotion_checks(registry: Any, adapter_id: str, to: str, cfg: Any) -> dict[str, Check]:
    """Everything that must be true before an adapter serves traffic."""
    from agentdistill.eval.runner import compare
    from agentdistill.registry.select import NoMatch, latest_calibration, latest_eval, prod_adapter

    row = adapter(registry, adapter_id)
    eval_set = cfg.eval.eval_set
    checks: dict[str, Check] = {}

    try:
        run = latest_eval(registry, subject=row["id"], eval_set=eval_set)
    except NoMatch:
        try:
            run = latest_eval(registry, subject=row["name"], eval_set=eval_set)
        except NoMatch:
            run = None

    checks["has_eval"] = Check(
        ok=run is not None,
        detail=run["id"] if run else f"no eval run on the frozen set {eval_set!r}",
    )
    if run is None:
        return checks

    metrics = run["metrics"] or {}
    schema = metrics.get("schema_valid")
    checks["schema_valid"] = Check(
        ok=schema is not None and schema >= MIN_SCHEMA_VALID,
        detail=f"tool calls must validate at least {MIN_SCHEMA_VALID:.0%} of the time",
        value=schema,
    )

    current = prod_adapter(registry)
    if current and current["id"] != row["id"]:
        try:
            current_run = latest_eval(registry, subject=current["id"], eval_set=eval_set)
        except NoMatch:
            current_run = None
        if current_run is None:
            checks["not_worse_than_prod"] = Check(
                False, f"the current prod adapter has no eval on {eval_set!r}, so there is nothing to compare to"
            )
        else:
            try:
                cmp = compare(registry, run["id"], current_run["id"])
            except Exception as e:
                checks["not_worse_than_prod"] = Check(False, f"comparison failed: {type(e).__name__}: {e}")
            else:
                lo = cmp["success"]["ci95"][0] * 100
                checks["not_worse_than_prod"] = Check(
                    ok=lo >= -MAX_SUCCESS_REGRESSION_PP,
                    detail=f"success CI low end {lo:+.1f} pp (must be >= -{MAX_SUCCESS_REGRESSION_PP})",
                    value=lo,
                )
                token_delta = cmp["tokens"]["median_delta"]
                checks["cost_not_worse"] = Check(
                    ok=token_delta <= 0,
                    detail="median per-task token difference against prod",
                    value=token_delta,
                )
    else:
        checks["not_worse_than_prod"] = Check(True, "no incumbent to compare against")

    if to in ("canary", "prod"):
        try:
            cal = latest_calibration(registry, adapter_id=row["id"])
        except NoMatch:
            cal = None
        ece = ((cal or {}).get("holdout_metrics") or {}).get("ece") if cal else None
        checks["calibrated"] = Check(
            ok=cal is not None and ece is not None and ece <= MAX_HOLDOUT_ECE,
            detail=(
                f"holdout ECE must be at most {MAX_HOLDOUT_ECE}"
                if cal else "no calibration; the cascade would escalate everything"
            ),
            value=ece,
        )

    if to == "prod" and row["status"] == "canary":
        live = compare_live(registry, current["id"] if current else None, row["id"],
                            since_days=getattr(cfg.serve, "canary_days", 7))
        checks["live_not_worse"] = Check(
            ok=live is not None and live.get("ci95", [None])[0] is not None
            and live["ci95"][0] * 100 >= -MAX_SUCCESS_REGRESSION_PP,
            detail="paired-by-cluster comparison from the request log",
            value=live,
        )

    return checks


def transition(
    registry: Any, adapter_id: str, to: str, cfg: Any, actor: str = "cli", force: bool = False
) -> PromotionResult:
    """Move an adapter, if the evidence supports it."""
    row = adapter(registry, adapter_id)
    current_status = row["status"]
    if (current_status, to) not in TRANSITIONS:
        raise IllegalTransition(
            f"{current_status} -> {to} is not a legal transition. Legal moves from {current_status}: "
            f"{sorted(t for f, t in TRANSITIONS if f == current_status) or 'none'}"
        )

    checks = promotion_checks(registry, row["id"], to, cfg)
    ok = all(c.ok for c in checks.values())
    result = PromotionResult(ok=ok or force, checks=checks, forced=bool(force and not ok),
                             from_status=current_status, to_status=to)
    if not result.ok:
        return result

    if to == "prod":
        # Exactly one adapter serves prod traffic. The old one is retired in the same breath, with the reason.
        for other in _with_status(registry, "prod"):
            if other["id"] != row["id"]:
                _set_status(registry, other["id"], "retired",
                            {"reason": f"superseded by {row['id']}"}, actor)

    _set_status(registry, row["id"], to, result.to_dict(), actor)
    return result


def _with_status(registry: Any, status: str) -> list[dict]:
    with registry.engine.connect() as conn:
        return [dict(r) for r in conn.execute(
            text("SELECT * FROM adapters WHERE status = :s"), {"s": status}
        ).mappings()]


def _set_status(registry: Any, adapter_id: str, to: str, checks: dict, actor: str) -> None:
    with registry.engine.begin() as conn:
        row = conn.execute(text("SELECT status FROM adapters WHERE id = :i"), {"i": adapter_id}).first()
        from_status = row[0] if row else None
        conn.execute(text("UPDATE adapters SET status = :s WHERE id = :i"), {"s": to, "i": adapter_id})
        conn.execute(
            text(
                """INSERT INTO adapter_events (id, adapter_id, from_status, to_status, checks, actor, created_at)
                   VALUES (:id, :a, :f, :t, :c, :actor, :created)"""
            ),
            {"id": f"ae_{uuid.uuid4().hex[:16]}", "a": adapter_id, "f": from_status, "t": to,
             "c": dumps(checks), "actor": actor, "created": utcnow()},
        )


def events(registry: Any, adapter_id: str) -> list[dict]:
    with registry.engine.connect() as conn:
        rows = conn.execute(
            text("SELECT * FROM adapter_events WHERE adapter_id = :a ORDER BY created_at"), {"a": adapter_id}
        ).mappings().fetchall()
    out = []
    for r in rows:
        e = dict(r)
        e["checks"] = loads(e["checks"])
        out.append(e)
    return out


def compare_live(registry: Any, prod_id: str | None, canary_id: str, since_days: int = 7) -> dict | None:
    """Paired-by-cluster success difference from the request log.

    The evidence for promoting a canary to prod: not a frozen eval set, but the traffic each actually served.
    Returns None when either side has too few graded requests to say anything.
    """

    from agentdistill.eval.stats import cluster_bootstrap_diff

    if not prod_id:
        return None
    with registry.engine.connect() as conn:
        rows = [dict(r) for r in conn.execute(
            text(
                """SELECT adapter_id, cluster_id, outcome FROM requests
                   WHERE outcome IS NOT NULL AND cluster_id IS NOT NULL
                     AND received_at >= :since"""
            ),
            {"since": _days_ago(since_days)},
        ).mappings()]

    by_adapter: dict[str, dict[str, list[float]]] = {}
    for r in rows:
        bucket = by_adapter.setdefault(str(r["adapter_id"]), {})
        bucket.setdefault(str(r["cluster_id"]), []).append(float(bool(r["outcome"])))

    a, b = by_adapter.get(canary_id, {}), by_adapter.get(prod_id, {})
    shared = set(a) & set(b)
    if len(shared) < 2:
        return None
    try:
        cmp = cluster_bootstrap_diff({k: a[k] for k in shared}, {k: b[k] for k in shared}, iters=2000)
    except Exception:
        return None
    return {
        "delta": cmp.delta, "ci95": list(cmp.ci95), "n_clusters": cmp.n_tasks,
        "canary_requests": int(sum(len(v) for v in a.values())),
        "prod_requests": int(sum(len(v) for v in b.values())),
        "note": "paired by cluster over graded requests from the log",
    }


def _days_ago(days: int) -> str:
    from datetime import UTC, datetime, timedelta

    return (datetime.now(UTC) - timedelta(days=days)).isoformat()
