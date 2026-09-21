"""What the teacher will cost on the GPU day, stage by stage, before the box is rented.

The number this prints is the input to one decision: the hard spend cap on the teacher's API key. So it is built
as an **upper bound** and says where each part of it comes from.

Three things it gets right that a back-of-envelope does not:

- **Prompt tokens grow with the turn index.** An agent turn resends the tools, the system prompt and every
  earlier message. Multiplying a median prompt by the turn count understates a five-turn task by roughly half,
  so this sums the prefix length over each trace's turns instead.
- **The tool schemas are in every prompt.** They are resent on every call and are not small.
- **Every escalating stage is costed as if every turn escalated.** A cascade whose gate works escalates a
  fraction of turns; the cap has to survive a gate that does not work.

Token counts come from the corpus traces the eval sets are drawn from, at four characters per token of compact
JSON, times a 1.3 safety factor for a real model being wordier than the scripted solver that recorded them. The
factor applies to prompts as well as completions, because a wordier turn is resent in every later prompt.

Prompt caching is deliberately not modelled. It can only lower the real figure, and a cap computed from a cache
hit rate nobody has measured yet is a cap that fails at the wrong moment.

    python -m agentdistill.ops.spend --config examples/support_agent/project.yaml
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from typing import Any

#: Characters per token. A rough, model-independent estimate, deliberately not a tokenizer: the tokenizer for the
#: pinned base model is not the teacher's, and downloading one to estimate a cap is a dependency for no accuracy.
CHARS_PER_TOKEN = 4

#: A real model is wordier than the scripted solver whose traces these are. Applied to prompts too, because a
#: longer assistant turn is resent in every prompt after it.
SAFETY_FACTOR = 1.3

#: `scripts/serve_smoke.sh` drives the example agent twice, five tasks each, through both dialects.
SERVE_SMOKE_TASKS = 10

#: `s_cascade_ver` runs `eval run cascade:…:auto --verify-threshold`, which measures the threshold search's
#: candidates. Budget three, each with every turn escalated.
CASCADE_THRESHOLDS = 3


def tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def _compact(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True, default=str)


def message_tokens(message: dict) -> int:
    """Tokens for one message, from its compact JSON.

    The whole message, not only its text: tool calls, tool results and their ids are all sent to the teacher and
    a count that skips them understates every agent trace.
    """
    return tokens(_compact(message))


@dataclass
class TraceProfile:
    """One task's teacher cost shape: turns, and the prefix-summed prompt tokens across them."""

    turns: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


def profile_trace(trace: dict) -> TraceProfile:
    """Turn count and prefix-summed token totals for one recorded trace.

    Every assistant message is one call to the model. Its prompt is the tool schemas plus every message before
    it; its completion is the message itself. Summing those over the trace is what a subject costs for this task.
    """
    messages = [m for m in (trace.get("messages") or []) if isinstance(m, dict)]
    tools_tokens = tokens(_compact(trace.get("tools") or []))
    prof = TraceProfile()
    prefix = tools_tokens
    for m in messages:
        if m.get("role") == "assistant":
            prof.turns += 1
            prof.prompt_tokens += prefix
            prof.completion_tokens += message_tokens(m)
        prefix += message_tokens(m)
    return prof


@dataclass
class StageEstimate:
    name: str
    what: str
    tasks: int
    repeats: int
    #: Multiplies tasks x repeats: three threshold candidates, two dialects, and so on.
    passes: int = 1
    turns: float = 0.0
    prompt_tokens: float = 0.0
    completion_tokens: float = 0.0
    usd: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def calls(self) -> float:
        return self.tasks * self.repeats * self.passes


def _corpus_profile(traces: list[dict]) -> tuple[int, TraceProfile]:
    """(task count, per-task means) for a set of traces, with the safety factor applied."""
    profiles = [profile_trace(t) for t in traces]
    n = len(profiles) or 1
    mean = TraceProfile(
        turns=sum(p.turns for p in profiles) / n,  # type: ignore[arg-type]
        prompt_tokens=sum(p.prompt_tokens for p in profiles) * SAFETY_FACTOR / n,  # type: ignore[arg-type]
        completion_tokens=sum(p.completion_tokens for p in profiles) * SAFETY_FACTOR / n,  # type: ignore[arg-type]
    )
    return len(profiles), mean


def _price(cfg: Any, registry: Any) -> dict:
    """The teacher's price row, from the registry. Raises when there is none: a cap guessed from the config's
    defaults is not a cap, and the pre-flight has a check for exactly this."""
    if cfg.teacher is None:
        raise RuntimeError("this project has no `teacher` section, so no teacher spend can be estimated")
    rows = [
        r for r in registry.list_pricing()
        if r["model"] == cfg.teacher.model and r["provider"] == cfg.teacher.provider
    ]
    if not rows:
        raise RuntimeError(
            f"no pricing row for {cfg.teacher.provider}/{cfg.teacher.model}; seed one with "
            f"`agentdistill pricing set {cfg.teacher.model} --provider {cfg.teacher.provider} "
            f"--input <usd> --output <usd>` (docs/gpu-day.md, pre-flight step 3b)"
        )
    return max(rows, key=lambda r: r["effective_from"])


def eval_set_traces(registry: Any, name: str) -> list[dict]:
    """The traces of one registered eval set, or [] when it is not registered."""
    es = registry.get_eval_set(name)
    if not es:
        return []
    ids = set(es["trace_ids"])
    return [t for t in registry.list_traces() if t["id"] in ids]


def estimate(
    cfg: Any,
    registry: Any,
    holdout: str | None = None,
    unseen: str = "support-unseen-v1",
    n_per_task: int | None = None,
) -> dict:
    """Every teacher-calling stage of `scripts/gpu_day.sh`, costed."""
    price = _price(cfg, registry)
    in_rate, out_rate = float(price["input_per_mtok"]), float(price["output_per_mtok"])

    holdout = holdout or cfg.eval.eval_set or "support-holdout-v1"
    n_eval = n_per_task if n_per_task is not None else cfg.eval.n_per_task

    holdout_traces = eval_set_traces(registry, holdout)
    unseen_traces = eval_set_traces(registry, unseen)
    if not holdout_traces:
        raise RuntimeError(
            f"eval set {holdout!r} is not registered, so there is nothing to estimate from; run "
            f"`agentdistill evalset add {holdout} …` first"
        )
    n_holdout, per_holdout = _corpus_profile(holdout_traces)
    n_unseen, per_unseen = _corpus_profile(unseen_traces or holdout_traces)

    stages: list[StageEstimate] = [
        StageEstimate(
            "eval_teach", f"teacher on {holdout}, live tools", n_holdout, n_eval,
            notes=["gpu_day.sh `s_eval_teach`"],
        ),
        StageEstimate(
            "eval_teach_replay", f"teacher on {holdout}, replay tools", n_holdout, n_eval,
            notes=["the second teacher eval WP1 adds, which measures replay's grading distortion"],
        ),
        StageEstimate(
            "unseen_teach", f"teacher on {unseen}", n_unseen, n_eval,
            notes=[
                "a provision: gpu_day.sh currently evaluates only the student on the unseen set, so this is "
                "budgeted for the teacher line that makes that comparison readable" if unseen_traces
                else f"eval set {unseen!r} is not registered; costed at the holdout's shape as a provision",
            ],
        ),
        StageEstimate(
            "cascade_ver", "cascade verification, every turn escalated", n_holdout, 3,
            passes=CASCADE_THRESHOLDS,
            notes=[f"gpu_day.sh `s_cascade_ver` at --n 3, {CASCADE_THRESHOLDS} thresholds, upper bound"],
        ),
        StageEstimate(
            "serve_smoke", "gateway smoke test, every turn escalated", SERVE_SMOKE_TASKS, 1,
            notes=["scripts/serve_smoke.sh drives 5 tasks through each dialect as cascade::auto"],
        ),
    ]

    for stage in stages:
        per = per_unseen if stage.name == "unseen_teach" else per_holdout
        stage.turns = per.turns * stage.calls
        stage.prompt_tokens = per.prompt_tokens * stage.calls
        stage.completion_tokens = per.completion_tokens * stage.calls
        stage.usd = (stage.prompt_tokens * in_rate + stage.completion_tokens * out_rate) / 1e6

    total = sum(s.usd for s in stages)
    return {
        "teacher": f"{cfg.teacher.provider}/{cfg.teacher.model}",
        "price": {"input_per_mtok": in_rate, "output_per_mtok": out_rate,
                  "effective_from": price["effective_from"],
                  "cache_read_per_mtok": price.get("cache_read_per_mtok")},
        "holdout": {"name": holdout, "tasks": n_holdout, "turns_per_task": per_holdout.turns,
                    "prompt_tokens_per_task": per_holdout.prompt_tokens,
                    "completion_tokens_per_task": per_holdout.completion_tokens},
        "unseen": {"name": unseen, "tasks": n_unseen, "registered": bool(unseen_traces)},
        "n_per_task": n_eval,
        "stages": stages,
        "total_usd": total,
        "recommended_cap_usd": 2 * total,
    }


def render(result: dict) -> str:
    """The printed estimate. Plain text, because it is pasted into a console's spend-limit field."""
    price = result["price"]
    lines = [
        f"Teacher spend estimate for the GPU day — {result['teacher']}",
        f"  priced from the registry row effective {price['effective_from']}: "
        f"${price['input_per_mtok']}/Mtok in, ${price['output_per_mtok']}/Mtok out",
        f"  token counts from the corpus traces at {CHARS_PER_TOKEN} chars/token, prefix-summed over each "
        f"trace's turns, times a {SAFETY_FACTOR} safety factor",
        f"  eval set {result['holdout']['name']}: {result['holdout']['tasks']} tasks, "
        f"{result['holdout']['turns_per_task']:.1f} turns per task, "
        f"{result['holdout']['prompt_tokens_per_task']:,.0f} prompt + "
        f"{result['holdout']['completion_tokens_per_task']:,.0f} completion tokens per task",
        "",
        f"{'stage':<18} {'tasks':>6} {'x reps':>7} {'x pass':>7} {'turns':>8} {'prompt tok':>13} "
        f"{'completion':>12} {'USD':>9}",
        "-" * 88,
    ]
    for s in result["stages"]:
        lines.append(
            f"{s.name:<18} {s.tasks:>6} {s.repeats:>7} {s.passes:>7} {s.turns:>8,.0f} "
            f"{s.prompt_tokens:>13,.0f} {s.completion_tokens:>12,.0f} {s.usd:>9,.2f}"
        )
        lines.append(f"{'':<18} {s.what}")
        for note in s.notes:
            if note:
                lines.append(f"{'':<18} ({note})")
    lines += [
        "-" * 88,
        f"{'TOTAL':<18} {'':>6} {'':>7} {'':>7} {'':>8} "
        f"{sum(s.prompt_tokens for s in result['stages']):>13,.0f} "
        f"{sum(s.completion_tokens for s in result['stages']):>12,.0f} {result['total_usd']:>9,.2f}",
        "",
        f"Total: ${result['total_usd']:,.2f}.  RECOMMENDED CAP: ${result['recommended_cap_usd']:,.2f} "
        f"(twice the estimate), set on the teacher key's own workspace.",
        "",
        "This is an upper bound. Every escalating stage is costed as if the gate escalated every turn, and "
        "prompt caching is not modelled:",
        "caching can only lower the real figure, never raise it. The on-policy rounds call no teacher — their "
        "pairs come from recorded traces —",
        "and the student's own evaluations run locally.",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="project.yaml")
    ap.add_argument("--eval-set", default=None, help="Holdout eval set. Defaults to eval.eval_set.")
    ap.add_argument("--unseen-set", default="support-unseen-v1", help="Unseen eval set.")
    ap.add_argument("--n", type=int, default=None, help="Repeats per task. Defaults to eval.n_per_task.")
    args = ap.parse_args(argv)

    from agentdistill.config import ProjectConfig
    from agentdistill.registry import open_registry

    cfg = ProjectConfig.load(args.config)
    registry = open_registry(cfg.registry, root=cfg.root)
    try:
        result = estimate(cfg, registry, holdout=args.eval_set, unseen=args.unseen_set, n_per_task=args.n)
    except RuntimeError as e:
        print(f"cannot estimate: {e}")
        return 1
    print(render(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
