"""Registry selectors.

Every `$(agentdistill ... latest --tag ...)` in the GPU-day script is one of these. They exist as a separate,
tested module because the script substitutes their output straight into the next command: a selector that returns
the wrong row, or an empty string, silently trains or evaluates the wrong thing.

All of them raise `NoMatch` rather than returning None. On the GPU box a clear failure costs a rerun; a blank
substitution costs the session.
"""

from __future__ import annotations

import fnmatch
from typing import Any

from sqlalchemy import text

from agentdistill.registry.base import loads


class NoMatch(LookupError):
    """No row matched. The message says what was searched for."""


class Ambiguous(LookupError):
    """More than one row matched where exactly one was required."""


def _rows(registry: Any, sql: str, params: dict | None = None) -> list[dict]:
    with registry.engine.connect() as conn:
        return [dict(r) for r in conn.execute(text(sql), params or {}).mappings()]


def _tag_filter(rows: list[dict], tag: str | None) -> list[dict]:
    """Exact tag, or a glob when the tag contains `*` (the script uses `gpu-day*` to span rounds)."""
    if tag is None:
        return rows
    if any(ch in tag for ch in "*?["):
        return [r for r in rows if r.get("tag") and fnmatch.fnmatch(r["tag"], tag)]
    return [r for r in rows if r.get("tag") == tag]


# --------------------------------------------------------------------------------------------------------------
# datasets
# --------------------------------------------------------------------------------------------------------------


def latest_dataset(registry: Any, name: str | None = None, kind: str | None = None) -> dict:
    """The newest dataset version, optionally filtered by name and kind."""
    rows = _rows(registry, "SELECT * FROM datasets ORDER BY created_at DESC, version DESC")
    if name:
        rows = [r for r in rows if r["name"] == name]
    if kind:
        rows = [r for r in rows if r["kind"] == kind]
    if not rows:
        raise NoMatch(f"no dataset matching name={name!r} kind={kind!r}; run `agentdistill curate` first")
    row = rows[0]
    row["filter_config"] = loads(row["filter_config"])
    return row


# --------------------------------------------------------------------------------------------------------------
# adapters
# --------------------------------------------------------------------------------------------------------------


def latest_adapter(
    registry: Any, tag: str | None = None, status: str | None = None, quantized: bool | None = None
) -> dict:
    rows = _tag_filter(
        _rows(registry, "SELECT * FROM adapters ORDER BY created_at DESC, version DESC"), tag
    )
    if status:
        rows = [r for r in rows if r["status"] == status]
    if quantized is not None:
        rows = [r for r in rows if bool(r.get("quantization")) is quantized]
    if not rows:
        raise NoMatch(f"no adapter matching tag={tag!r} status={status!r} quantized={quantized!r}")
    return rows[0]


def best_adapter(registry: Any, tag: str | None = None, eval_set: str | None = None) -> dict:
    """The adapter with the highest measured success on `eval_set`.

    "Best" means best *measured*, so an adapter with no eval on that set is not a candidate however new it is.
    That is deliberate: the GPU script uses this to pick what to calibrate and quantize, and picking an
    unevaluated adapter would put an unmeasured model into the report.
    """
    adapters = _tag_filter(_rows(registry, "SELECT * FROM adapters"), tag)
    if not adapters:
        raise NoMatch(f"no adapter matching tag={tag!r}")

    scored: list[tuple[float, str, dict]] = []
    for a in adapters:
        run = latest_eval_for_subject(registry, a["id"], eval_set=eval_set, missing_ok=True) or \
            latest_eval_for_subject(registry, a["name"], eval_set=eval_set, missing_ok=True)
        if not run:
            continue
        metrics = run["metrics"] or {}
        if metrics.get("success") is None:
            continue
        scored.append((float(metrics["success"]), run["started_at"] or "", a))
    if not scored:
        raise NoMatch(
            f"no adapter matching tag={tag!r} has an eval run"
            + (f" on {eval_set}" if eval_set else "")
            + "; `best` ranks by measured success, so evaluate one first"
        )
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return scored[0][2]


# --------------------------------------------------------------------------------------------------------------
# eval runs
# --------------------------------------------------------------------------------------------------------------


def _eval_runs(registry: Any) -> list[dict]:
    rows = _rows(registry, "SELECT * FROM eval_runs ORDER BY started_at DESC")
    for r in rows:
        r["metrics"] = loads(r["metrics"])
        r["per_cluster"] = loads(r.get("per_cluster"))
    return rows


def latest_eval(
    registry: Any, subject: str | None = None, tag: str | None = None, eval_set: str | None = None
) -> dict:
    rows = _eval_runs(registry)
    if subject:
        rows = [r for r in rows if r["subject"] == subject]
    rows = _tag_filter(rows, tag)
    if eval_set:
        rows = [r for r in rows if r["eval_set_id"] in (eval_set, f"es_{eval_set}")]
    if not rows:
        raise NoMatch(f"no eval run matching subject={subject!r} tag={tag!r} eval_set={eval_set!r}")
    return rows[0]


def latest_eval_for_subject(
    registry: Any, subject: str, eval_set: str | None = None, missing_ok: bool = False
) -> dict | None:
    try:
        return latest_eval(registry, subject=subject, eval_set=eval_set)
    except NoMatch:
        if missing_ok:
            return None
        raise


# --------------------------------------------------------------------------------------------------------------
# training runs, rounds, calibrations
# --------------------------------------------------------------------------------------------------------------


def latest_training_run(registry: Any, method: str | None = None) -> dict:
    rows = _rows(registry, "SELECT * FROM training_runs ORDER BY started_at DESC")
    if method:
        rows = [r for r in rows if r["method"] == method]
    if not rows:
        raise NoMatch(f"no training run matching method={method!r}")
    row = rows[0]
    row["config"] = loads(row["config"])
    row["metrics"] = loads(row["metrics"])
    return row


def latest_calibration(registry: Any, adapter_id: str | None = None) -> dict:
    rows = _rows(registry, "SELECT * FROM calibrations ORDER BY created_at DESC")
    if adapter_id:
        rows = [r for r in rows if r["adapter_id"] == adapter_id]
    if not rows:
        raise NoMatch(f"no calibration for adapter={adapter_id!r}")
    row = rows[0]
    for key in ("target", "holdout_metrics", "reliability_bins", "verified", "features", "feature_order"):
        if key in row:
            row[key] = loads(row[key])
    return row


def rounds_for_tag(registry: Any, tag: str) -> list[dict]:
    rows = _tag_filter(_rows(registry, "SELECT * FROM onpolicy_rounds ORDER BY round_idx"), tag)
    for r in rows:
        r["compare"] = loads(r.get("compare"))
    return rows


def prod_adapter(registry: Any) -> dict | None:
    rows = _rows(registry, "SELECT * FROM adapters WHERE status = 'prod' ORDER BY created_at DESC")
    if len(rows) > 1:
        raise Ambiguous(f"{len(rows)} adapters are marked prod; exactly one may be")
    return rows[0] if rows else None


def canary_adapter(registry: Any) -> dict | None:
    rows = _rows(registry, "SELECT * FROM adapters WHERE status = 'canary' ORDER BY created_at DESC")
    return rows[0] if rows else None


#: `agentdistill <thing> latest|best` dispatches through here, so the CLI and the GPU script cannot drift apart.
SELECTORS = {
    "dataset": latest_dataset,
    "adapter": latest_adapter,
    "eval": latest_eval,
    "training-run": latest_training_run,
    "calibration": latest_calibration,
}
