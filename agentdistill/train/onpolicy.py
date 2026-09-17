"""The on-policy round loop.

One round: roll the current student out on training tasks, keep what worked as new SFT data, pair what did not
against what did, train, evaluate, and decide whether the result is worth keeping.

Stages are injected so the loop's decisions can be tested without a GPU. The real wiring lives in `cli.py`; what
lives here is the sequencing and, more importantly, `decide` -- the promotion rule, which is the part that has to
be right. A loop that promotes on a loss curve or on an unmeasured hunch produces a chain of adapters nobody can
justify.

The comparison is candidate versus **current adapter**, never versus the teacher. The teacher comparison belongs
in the report; here the only question is whether this round improved on the last one.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RoundCfg:
    k_rollouts: int = 8
    rft_cap_per_task: int = 2
    #: Rollouts need fuzzy replay: an on-policy trajectory drifts from the teacher's argument phrasing, and
    #: strict mode would stop most rollouts at the first turn.
    replay_policy: str = "fuzzy"
    #: Above this, too many results were served by approximate match for the successes to mean anything.
    max_fuzzy_share: float = 0.5
    min_pairs: int = 40
    #: Promote on equal success if cost improved, within this tolerance.
    success_tolerance_pp: float = 1.0
    schema_floor: float = 0.99
    divergence_slack_pp: float = 5.0
    max_teacher_ratio: float = 1.0


@dataclass
class RoundResult:
    round_idx: int
    start_adapter: str
    decision: str = "error"
    reason: str = ""
    n_rollouts: int = 0
    fuzzy_share: float = 0.0
    n_rft: int = 0
    n_pairs: int = 0
    pair_kinds: dict = field(default_factory=dict)
    candidate_adapter: str | None = None
    compare: dict = field(default_factory=dict)
    ids: dict = field(default_factory=dict)

    @property
    def promoted(self) -> bool:
        return self.decision == "promote"


@dataclass
class Stages:
    """Everything the loop needs from the outside world, so it can be driven by stubs in tests."""

    collect_rollouts: Callable[[str, list[str], int, str], dict]
    build_rft: Callable[[list[dict], int], tuple[str, int]]
    build_pairs: Callable[[list[dict], dict[str, dict]], tuple[str, int, dict]]
    train_sft_continue: Callable[[str, str], tuple[str, str]]
    merge: Callable[[str], str]
    train_dpo: Callable[[str, str], tuple[str, str]]
    run_eval: Callable[[str], str]
    compare: Callable[[str, str], dict]
    metrics: Callable[[str], dict]
    record: Callable[[dict], None]


def decide(cmp: dict, current: dict, candidate: dict, cfg: RoundCfg) -> tuple[str, str]:
    """Keep the round, or throw it away.

    Two hard gates first, because they are not trade-offs: a student whose tool calls stopped validating, or that
    wanders off the recording far more than its predecessor, is worse in a way no token saving redeems.

    Then: a success improvement whose interval excludes zero is enough on its own. Equal success is enough only
    if it came with a cost saving -- otherwise the round produced a differently-shaped model for nothing, and
    keeping it just adds a link to the lineage.
    """
    if candidate.get("schema_valid", 1.0) < cfg.schema_floor:
        return "discard", (
            f"schema validity {candidate['schema_valid']:.3f} is below the {cfg.schema_floor} floor"
        )

    divergence_delta = (candidate.get("divergence_rate", 0.0) - current.get("divergence_rate", 0.0)) * 100
    if divergence_delta > cfg.divergence_slack_pp:
        return "discard", (
            f"divergence rate regressed by {divergence_delta:+.1f} pp, above the {cfg.divergence_slack_pp} pp slack"
        )

    success = cmp.get("success") or {}
    lo, hi = success.get("ci95", (0.0, 0.0))
    delta_pp = success.get("delta", 0.0) * 100

    tokens = cmp.get("tokens") or {}
    token_delta = tokens.get("median_delta", 0.0)
    # A saving has to be one the interval supports. `median_delta < 0` promotes on noise: on a small eval set the
    # median per-task token difference is negative about half the time by chance.
    token_ci = tokens.get("ci95") or [float("nan"), float("nan")]
    cost_better = bool(token_ci[1] < 0)

    if lo > 0:
        return "promote", f"success up {delta_pp:+.1f} pp, CI [{lo * 100:+.1f}, {hi * 100:+.1f}] excludes zero"
    if lo <= 0 <= hi and delta_pp > -cfg.success_tolerance_pp and cost_better:
        return "promote", (
            f"success unchanged within tolerance ({delta_pp:+.1f} pp) and tokens down {token_delta:.0f} "
            f"[CI {token_ci[0]:+.0f}, {token_ci[1]:+.0f}]"
        )
    if lo <= 0 <= hi and delta_pp > -cfg.success_tolerance_pp and token_delta < 0 and not cost_better:
        return "discard", (
            f"success unchanged ({delta_pp:+.1f} pp) and the median token delta is {token_delta:.0f}, but its "
            f"CI [{token_ci[0]:+.0f}, {token_ci[1]:+.0f}] includes zero, so the saving is not established"
        )
    return "discard", (
        f"success {delta_pp:+.1f} pp with CI [{lo * 100:+.1f}, {hi * 100:+.1f}], cost_better={cost_better}"
    )


def run_round(
    round_idx: int,
    adapter: str,
    train_task_ids: list[str],
    teacher_by_task: dict[str, dict],
    current_eval_run: str,
    st: Stages,
    cfg: RoundCfg,
) -> RoundResult:
    """One round. Always recorded, including when it fails."""
    r = RoundResult(round_idx=round_idx, start_adapter=adapter, ids={"round_id": f"rd_{uuid.uuid4().hex[:16]}"})
    try:
        roll = st.collect_rollouts(adapter, train_task_ids, cfg.k_rollouts, cfg.replay_policy)
        rollouts = roll["rollouts"]
        r.n_rollouts = len(rollouts)
        r.fuzzy_share = float(roll.get("fuzzy_share", 0.0))
        r.ids["rollout_eval_run"] = roll.get("eval_run_id")

        if r.fuzzy_share > cfg.max_fuzzy_share:
            r.decision, r.reason = "discard", (
                f"fuzzy replay share {r.fuzzy_share:.0%} is above {cfg.max_fuzzy_share:.0%}; too many tool "
                f"results were served by approximate match for these successes to mean anything"
            )
            return r

        rft_id, r.n_rft = st.build_rft(rollouts, cfg.rft_cap_per_task)
        pairs_id, r.n_pairs, r.pair_kinds = st.build_pairs(rollouts, teacher_by_task)
        r.ids.update(rft_dataset=rft_id, dpo_dataset=pairs_id)

        if r.n_pairs < cfg.min_pairs:
            r.decision, r.reason = "discard", (
                f"only {r.n_pairs} usable pairs, below the minimum of {cfg.min_pairs}; "
                f"DPO on this few would be noise"
            )
            return r

        sft_run, sft_adapter = st.train_sft_continue(adapter, rft_id)
        merged = st.merge(sft_adapter)
        dpo_run, candidate = st.train_dpo(merged, pairs_id)
        r.candidate_adapter = candidate
        r.ids.update(sft_run=sft_run, sft_adapter=sft_adapter, dpo_run=dpo_run, merged=merged)

        candidate_eval = st.run_eval(candidate)
        r.ids["eval_run"] = candidate_eval
        # a = candidate, b = current, so a positive delta favours the candidate.
        r.compare = st.compare(candidate_eval, current_eval_run)
        r.decision, r.reason = decide(
            r.compare, st.metrics(current_eval_run), st.metrics(candidate_eval), cfg
        )
        return r
    except Exception as e:
        # Recorded as an error and re-raised: a round that blew up must never look like a round that decided.
        r.decision, r.reason = "error", f"{type(e).__name__}: {e}"
        raise
    finally:
        st.record(asdict(r))


def run_rounds(
    adapter: str,
    rounds: int,
    train_task_ids: list[str],
    teacher_by_task: dict[str, dict],
    current_eval_run: str,
    st: Stages,
    cfg: RoundCfg,
) -> list[RoundResult]:
    """Run rounds until one is discarded.

    Stopping on a discard is deliberate: the next round would start from the same adapter against the same tasks
    and reach the same place, at the same cost.
    """
    out: list[RoundResult] = []
    current_adapter, current_eval = adapter, current_eval_run
    for i in range(rounds):
        r = run_round(i, current_adapter, train_task_ids, teacher_by_task, current_eval, st, cfg)
        out.append(r)
        if not r.promoted or not r.candidate_adapter:
            logger.info("round %d %s: %s; stopping", i, r.decision, r.reason)
            break
        current_adapter, current_eval = r.candidate_adapter, r.ids["eval_run"]
    return out


def plan(adapter: str, rounds: int, train_task_ids: list[str], cfg: RoundCfg) -> list[str]:
    """What `--dry-run` prints: the stages and their sizes, with nothing executed."""
    n = len(train_task_ids)
    return [
        f"start adapter      {adapter}",
        f"rounds             {rounds}",
        f"training tasks     {n}",
        f"rollouts           {n} x {cfg.k_rollouts} = {n * cfg.k_rollouts} per round ({cfg.replay_policy} replay)",
        f"RFT cap            {cfg.rft_cap_per_task} per task",
        f"minimum pairs      {cfg.min_pairs} (teacher pairs capped at {cfg.max_teacher_ratio:g}x rollout pairs)",
        f"abort if           fuzzy share > {cfg.max_fuzzy_share:.0%}",
        f"promote if         success CI excludes zero, or within {cfg.success_tolerance_pp} pp with fewer tokens",
        f"hard floors        schema validity >= {cfg.schema_floor}, "
        f"divergence not worse by more than {cfg.divergence_slack_pp} pp",
    ]
