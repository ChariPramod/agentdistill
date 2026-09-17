"""The retrain loop's stages, wired to real work.

Each stage runs the same `agentdistill` subcommand a person would run by hand, then reads the registry for what
it produced. Two reasons for going through the CLI rather than calling the library directly:

- The registry records the exact invocation on every row it writes, so a retrain leaves behind a sequence of
  commands that reproduces it. A pipeline that called internal functions would leave rows nobody could rerun.
- The unattended path and the manual path are then the same path. A retrain that exercised different code from
  the one an engineer debugs with is a retrain whose failures cannot be reproduced.

The runner is injected, so the pipeline is testable end to end without training anything.
"""

from __future__ import annotations

import logging
import shlex
import subprocess
from collections.abc import Callable
from typing import Any

from agentdistill.retrain import GATES, Stage

logger = logging.getLogger(__name__)

Runner = Callable[[list[str]], str]


def subprocess_runner(argv: list[str]) -> str:
    """Run a command, stream nothing, return stdout, and raise with stderr on failure."""
    logger.info("$ %s", shlex.join(argv))
    proc = subprocess.run(argv, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RetrainStageFailed(
            f"`{shlex.join(argv)}` exited {proc.returncode}\n{proc.stderr.strip()[-2000:]}"
        )
    return proc.stdout


class RetrainStageFailed(RuntimeError):
    """A stage's command failed. Distinct from a gate refusing to continue, which is a normal outcome."""


def build_stages(cfg: Any, registry: Any, runner: Runner | None = None, since: str = "7d") -> list[Stage]:
    """The eight stages, in order, each with its gate from `retrain.GATES`."""
    run = runner or subprocess_runner
    ad = ["agentdistill"]
    config = ["--config", str(getattr(cfg, "source_path", None) or "project.yaml")]

    def ingest_gateway(ctx: dict) -> dict:
        run([*ad, "ingest", "gateway", "--since", since, *config])
        ctx["n_requests"] = _count_graded(registry, since)
        return ctx

    def curate(ctx: dict) -> dict:
        before = _dataset_samples(registry, cfg)
        run([*ad, "curate", *config])
        after = _dataset_samples(registry, cfg)
        ctx["n_new_samples"] = max(0, after - before)
        ctx["dataset_samples"] = after
        return ctx

    def train_sft_continue(ctx: dict) -> dict:
        # One epoch at a third of the learning rate, continuing from what is already serving. A retrain is an
        # update, not a fresh run: restarting from the base model throws away everything the prod adapter knows
        # and re-learns it from a corpus that is mostly the same.
        prod = _prod(registry)
        dataset = _latest_dataset_name(registry, cfg)
        argv = [*ad, "train", "sft", dataset, "--epochs", "1", "--lr-scale", "0.333", *config]
        if prod:
            argv += ["--from-adapter", prod["id"]]
        run(argv)
        run_row = _latest_training_run(registry)
        ctx["training_run_id"] = (run_row or {}).get("id")
        ctx["eval_loss"] = ((run_row or {}).get("metrics") or {}).get("eval_loss")
        ctx["candidate_adapter"] = (_latest_adapter(registry) or {}).get("id")
        return ctx

    def onpolicy(ctx: dict) -> dict:
        run([*ad, "train", "onpolicy", str(ctx.get("candidate_adapter")), "--rounds", "1", *config])
        round_row = _latest_round(registry) or {}
        ctx["round_decision"] = round_row.get("decision")
        ctx["round_id"] = round_row.get("id")
        if round_row.get("candidate_adapter"):
            ctx["candidate_adapter"] = round_row["candidate_adapter"]
        return ctx

    def evaluate(ctx: dict) -> dict:
        candidate = ctx.get("candidate_adapter")
        run([*ad, "eval", "run", str(candidate), "--n", str(cfg.eval.n_per_task), *config])
        ctx["comparison"] = _compare_with_prod(registry, cfg, candidate)
        return ctx

    def calibrate(ctx: dict) -> dict:
        run([*ad, "calibrate", str(ctx.get("candidate_adapter")), *config])
        ctx["calibration"] = _holdout_metrics(registry, ctx.get("candidate_adapter"))
        return ctx

    def quantize(ctx: dict) -> dict:
        method = cfg.serve.quantization
        if not method:
            ctx["quantization_drop_pp"] = None
            return ctx
        run([*ad, "adapter", "quantize", str(ctx.get("candidate_adapter")), "--method", method, *config])
        quantized = _latest_adapter(registry, quantization=method)
        if quantized:
            run([*ad, "eval", "run", quantized["id"], "--n", str(cfg.eval.n_per_task), *config])
            ctx["quantized_adapter"] = quantized["id"]
            ctx["quantization_drop_pp"] = _success_drop_pp(
                registry, cfg, ctx.get("candidate_adapter"), quantized["id"]
            )
        return ctx

    def promote_canary(ctx: dict) -> dict:
        target = ctx.get("quantized_adapter") or ctx.get("candidate_adapter")
        ctx["promotion_checks"] = _promotion_checks(registry, target, cfg)
        if all(c.get("ok") for c in ctx["promotion_checks"].values()):
            run([*ad, "adapter", "promote", str(target), "--to", "canary", "--actor", "retrain", *config])
            ctx["promoted"] = target
        return ctx

    described = [
        ("ingest_gateway", ingest_gateway, lambda c: f"import graded requests since {since}"),
        ("curate", curate, lambda c: "rebuild the SFT dataset with the configured filters"),
        ("train_sft_continue", train_sft_continue,
         lambda c: "one epoch at lr/3, continuing from the prod adapter"),
        ("onpolicy", onpolicy, lambda c: "one on-policy round from the new adapter"),
        ("eval", evaluate, lambda c: f"eval on {cfg.eval.eval_set} at N={cfg.eval.n_per_task}"),
        ("calibrate", calibrate, lambda c: "fit the confidence gate and verify it on a disjoint split"),
        ("quantize", quantize, lambda c: f"quantize as {cfg.serve.quantization or 'nothing (not configured)'}"),
        ("promote_canary", promote_canary, lambda c: "promote to canary if every lifecycle check is green"),
    ]
    return [Stage(name=n, run=f, gate=GATES[n], describe=d) for n, f, d in described]


# ------------------------------------------------------------------------------------------------------------
# registry reads
#
# Every one of these tolerates an empty registry and returns None or zero rather than raising. A stage that
# raised here would report "retrain crashed" where the truth is "there was nothing to read", and those are
# different problems.
# ------------------------------------------------------------------------------------------------------------


def _count_graded(registry: Any, since: str) -> int:
    from sqlalchemy import text

    from agentdistill.router.compare_live import _iso_since

    with registry.engine.connect() as conn:
        return int(conn.execute(
            text("SELECT COUNT(*) FROM requests WHERE outcome IS NOT NULL AND received_at >= :s"),
            {"s": _iso_since(since)},
        ).scalar() or 0)


def _latest_dataset_name(registry: Any, cfg: Any) -> str:
    """The dataset `curate` just wrote. Named rather than pathed so the registry resolves the latest version."""
    from agentdistill.registry.select import NoMatch, latest_dataset

    try:
        return str(latest_dataset(registry, kind="sft")["name"])
    except (NoMatch, KeyError, TypeError):
        return cfg.name


def _dataset_samples(registry: Any, cfg: Any) -> int:
    from agentdistill.registry.select import NoMatch, latest_dataset

    try:
        return int(latest_dataset(registry, kind="sft")["n_samples"])
    except (NoMatch, KeyError, TypeError):
        return 0


def _prod(registry: Any) -> dict | None:
    from agentdistill.registry.select import prod_adapter

    try:
        return prod_adapter(registry)
    except LookupError:
        return None


def _latest_training_run(registry: Any) -> dict | None:
    from agentdistill.registry.select import NoMatch, latest_training_run

    try:
        return latest_training_run(registry)
    except NoMatch:
        return None


def _latest_adapter(registry: Any, quantization: str | None = None) -> dict | None:
    rows = registry.list_adapters()
    if quantization:
        rows = [a for a in rows if a.get("quantization") == quantization]
    return sorted(rows, key=lambda a: a["created_at"])[-1] if rows else None


def _latest_round(registry: Any) -> dict | None:
    from sqlalchemy import text

    with registry.engine.connect() as conn:
        row = conn.execute(
            text("SELECT * FROM onpolicy_rounds ORDER BY started_at DESC LIMIT 1")
        ).mappings().first()
    return dict(row) if row else None


def _compare_with_prod(registry: Any, cfg: Any, candidate: str | None) -> dict | None:
    from agentdistill.eval.runner import compare
    from agentdistill.registry.select import NoMatch, latest_eval

    prod = _prod(registry)
    if not prod or not candidate:
        return None
    try:
        a = latest_eval(registry, subject=candidate, eval_set=cfg.eval.eval_set)
        b = latest_eval(registry, subject=prod["id"], eval_set=cfg.eval.eval_set)
    except NoMatch:
        return None
    try:
        return compare(registry, a["id"], b["id"])
    except Exception as e:  # a comparison that cannot be made is not a comparison that passed
        logger.warning("paired comparison failed: %s", e)
        return None


def _holdout_metrics(registry: Any, adapter_id: str | None) -> dict | None:
    from agentdistill.report.registry_views import calibration_for

    if not adapter_id:
        return None
    row = calibration_for(registry, adapter_id)
    return (row or {}).get("holdout_metrics")


def _success_drop_pp(registry: Any, cfg: Any, bf16: str | None, quantized: str | None) -> float | None:
    from agentdistill.registry.select import NoMatch, latest_eval

    if not bf16 or not quantized:
        return None
    try:
        a = latest_eval(registry, subject=bf16, eval_set=cfg.eval.eval_set)
        b = latest_eval(registry, subject=quantized, eval_set=cfg.eval.eval_set)
    except NoMatch:
        return None
    before = (a.get("metrics") or {}).get("success")
    after = (b.get("metrics") or {}).get("success")
    if before is None or after is None:
        return None
    return (float(before) - float(after)) * 100


def _promotion_checks(registry: Any, adapter_id: str | None, cfg: Any) -> dict:
    from agentdistill.registry.lifecycle import promotion_checks

    if not adapter_id:
        return {}
    try:
        checks = promotion_checks(registry, adapter_id, "canary", cfg)
    except Exception as e:
        return {"checks_ran": {"ok": False, "detail": f"{type(e).__name__}: {e}"}}
    return {name: {"ok": c.ok, "detail": c.detail} for name, c in checks.items()}
