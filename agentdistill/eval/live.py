"""Live tools for evaluation.

Replay serves the tool results the recording happened to contain, and counts anything else as a divergence. That
is safe when the tools are a real service -- nothing gets refunded twice because an eval ran twice -- and it is
the wrong instrument when the tools are a local, deterministic sandbox. A subject that looks a customer up by
email where the recorded solver used the id has not gone anywhere the environment cannot follow; it has made a
different, equally valid call. Replay fails it anyway, so the comparison measures how closely a subject imitates
the recording, which is not the question the report asks.

In live mode every subject runs against a fresh environment built from the task's own seed, every call is
executed for real, and the grader reads the **final state** rather than reconstructing one from the calls. Tool
errors come back as `{"error": "..."}` content, exactly as the recording agent saw them, because recovering from
a refusal is part of the trajectory. Nothing here ever raises `Divergence`: in live mode there is no recording to
diverge from.

## The project seam

The core must not know about anyone's CRM. What it needs from a project is one function:

    build_live_env(trace) -> environment

where the environment has `call(tool, args)` and, for grading, a `predicate(state, final_text)`. Two ways to
provide it, both lazy so nothing in `examples/` is imported unless live mode is actually used:

1. **The module already named in config.** `eval.grader.predicate_source` is a `module:callable`; live mode
   imports that *module* and takes `build_live_env` if it defines one, or `task_for_trace` -- a callable
   returning an object with `fresh_crm()` and `predicate` -- if it does not. The example project needs no new
   file: `examples/support_agent/replay_grader.py` already exposes `task_for_trace`.
2. **A code registry hook**, for a project that would rather register than be discovered:
   `register_env_factory("my-env", fn)` and then name `my-env` as the source.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

#: A factory turns a trace into the environment its task runs in. One per (task, repeat): the state is mutated.
EnvFactory = Callable[[dict], "LiveEnv"]

#: Config-free hook. A project that does not want to be discovered by module convention registers a factory here
#: under a name and sets that name as the source.
ENV_FACTORIES: dict[str, EnvFactory] = {}

#: The attribute names looked for on the module named by `eval.grader.predicate_source`, in order.
ENV_BUILDERS = ("build_live_env", "live_env_for_trace")
#: The fallback convention: a callable returning a task object with `fresh_crm()` and `predicate`.
TASK_BUILDERS = ("task_for_trace",)


class LiveToolsUnavailable(RuntimeError):
    """Live mode was asked for and the project's environment could not be resolved. The message says what to add."""


class LiveGradingUnavailable(RuntimeError):
    """The live environment has no predicate, so the final state cannot be graded."""


def register_env_factory(name: str, factory: EnvFactory) -> None:
    """Register a live-environment factory under a name usable as `eval.grader.predicate_source`."""
    ENV_FACTORIES[name] = factory


@dataclass
class LiveEnv:
    """One task's live environment.

    `state` is whatever the project's tools mutate -- the CRM in the example -- and is what the grader reads at
    the end. `call` executes one tool call against it. `predicate` grades `(state, final_text)`; without one,
    live mode can still run tools but cannot grade, and says so rather than falling back to something weaker.
    """

    state: Any
    call: Callable[[str, dict], Any]
    predicate: Callable[[Any, str], tuple[bool, str]] | None = None
    #: Free-form identification of the task, carried into the grader detail for debugging.
    label: str = ""

    @classmethod
    def from_task(cls, task: Any) -> LiveEnv:
        """Build from a project task object exposing `fresh_crm()` and `predicate`.

        This is the convention `examples/support_agent` already follows, which is why live mode needs no new file
        in the example: a scenario Task builds its own seeded database and carries the predicate that grades it.
        """
        state = task.fresh_crm()
        return cls(
            state=state,
            call=state.call,
            predicate=getattr(task, "predicate", None),
            label=str(getattr(task, "task_id", "") or getattr(task, "scenario", "")),
        )


def resolve_env_factory(source: str | None) -> EnvFactory:
    """Find the project's live-environment factory, or explain exactly what is missing.

    `source` is `eval.grader.predicate_source`: either a registered name or a `module:callable`. The callable
    half names the grader, not the environment, so it is the *module* that is searched -- see the module
    docstring for the two names it looks for.
    """
    if not source:
        raise LiveToolsUnavailable(
            "tools mode is `live` but no environment source is configured. Set eval.grader.predicate_source to "
            "`module:callable` for a module that defines `build_live_env(trace)` (or `task_for_trace(trace)`), "
            "or register one with agentdistill.eval.live.register_env_factory."
        )
    if source in ENV_FACTORIES:
        return ENV_FACTORIES[source]

    module_name = source.partition(":")[0]
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise LiveToolsUnavailable(
            f"tools mode is `live` and the environment module {module_name!r} could not be imported: {e}"
        ) from e

    for attr in ENV_BUILDERS:
        fn = getattr(module, attr, None)
        if callable(fn):
            return fn
    for attr in TASK_BUILDERS:
        fn = getattr(module, attr, None)
        if callable(fn):
            def from_task(trace: dict, _build: Callable[[dict], Any] = fn) -> LiveEnv:
                return LiveEnv.from_task(_build(trace))

            return from_task
    raise LiveToolsUnavailable(
        f"tools mode is `live` but {module_name!r} defines none of "
        f"{', '.join(ENV_BUILDERS + TASK_BUILDERS)}. Live mode needs one function that turns a trace into a "
        f"fresh environment with `call(tool, args)` and a `predicate(state, final_text)`."
    )


class LiveToolProvider:
    """Serves real tool results for one task, against a fresh environment built from the task's own seed.

    Same interface as `ReplayToolProvider`: `lookup(tool, args) -> str` and `summary() -> dict`. It never raises
    `Divergence`, so a trajectory only ends when the subject answers or runs out of turns, and it keeps the
    environment afterwards so the grader can read the final state.
    """

    #: What the run records as its `eval_mode`.
    mode = "live"

    def __init__(self, trace: dict, factory: EnvFactory) -> None:
        self.trace = trace
        self.env = factory(trace)
        self.stats: dict[str, Any] = {"calls": 0, "tool_errors": 0, "by_tool": {}}

    @property
    def state(self) -> Any:
        """The environment the calls were executed against. The grader's input."""
        return self.env.state

    @property
    def can_grade(self) -> bool:
        return self.env.predicate is not None

    def lookup(self, tool: str, args: dict) -> str:
        """Execute the call and return the tool message content.

        A refusal is a result, not an exception: the recording agent fed `{"error": ...}` back to the model and
        the recovery is the behaviour worth measuring, so live mode does the same. The error text is the
        environment's own, not a harness paraphrase.
        """
        self.stats["calls"] += 1
        self.stats["by_tool"][tool] = self.stats["by_tool"].get(tool, 0) + 1
        try:
            result = self.env.call(tool, dict(args))
        except Exception as e:  # any tool refusal is content, exactly as the recorder saw it
            self.stats["tool_errors"] += 1
            return json.dumps({"error": str(e)})
        return json.dumps(result)

    def grade(self, final_text: str) -> tuple[bool, str]:
        """Grade the final state plus the final assistant message."""
        if self.env.predicate is None:
            raise LiveGradingUnavailable(
                "the live environment carries no predicate, so the final state cannot be graded. Give "
                "`build_live_env` a `predicate(state, final_text)` or grade with a configured judge instead."
            )
        ok, detail = self.env.predicate(self.env.state, final_text)
        return bool(ok), str(detail)

    def state_hash(self) -> str | None:
        fn = getattr(self.env.state, "state_hash", None)
        return str(fn()) if callable(fn) else None

    def summary(self) -> dict:
        """The per-task statistics the run row stores.

        The replay counters are present and zero on purpose: nothing was replayed, nothing was served by fuzzy
        match, and nothing diverged, because there was no recording in the loop. `mode` is what tells a reader
        that those zeros are a fact about live mode rather than a measurement that went missing.
        """
        return {
            "mode": "live",
            "calls": self.stats["calls"],
            "tool_errors": self.stats["tool_errors"],
            "by_tool": dict(self.stats["by_tool"]),
            "replayed": 0,
            "fuzzy": 0,
            "diverged": 0,
            "fuzzy_share": 0.0,
            "min_fuzzy_score": None,
            "n_recorded": 0,
        }


@dataclass
class LiveStats:
    """Aggregated live-tool statistics across many tasks, for the run report."""

    calls: int = 0
    tool_errors: int = 0
    by_tool: dict[str, int] = field(default_factory=dict)

    def add(self, summary: dict) -> None:
        self.calls += int(summary.get("calls") or 0)
        self.tool_errors += int(summary.get("tool_errors") or 0)
        for tool, n in (summary.get("by_tool") or {}).items():
            self.by_tool[tool] = self.by_tool.get(tool, 0) + int(n)

    def to_dict(self) -> dict:
        return {"mode": "live", "calls": self.calls, "tool_errors": self.tool_errors,
                "tool_error_rate": (self.tool_errors / self.calls) if self.calls else 0.0,
                "by_tool": dict(self.by_tool)}


def live_grader(fallback: Any = None) -> Callable[[dict, Any], tuple[bool, dict]]:
    """A grader with the runner's signature that reads the live provider's final state.

    It does not reconstruct state from the calls, which is what the replay grader has to do and what makes replay
    grading an approximation. The provider travels on the outcome, so the grader can reach the environment the
    trajectory actually ran against.

    `fallback` is the configured grader, used only when the live environment has no predicate of its own (a judge
    project, say). Without a fallback, an ungradeable live environment raises rather than scoring zero.
    """

    def grade(trace: dict, outcome: Any) -> tuple[bool, dict]:
        provider = getattr(outcome, "provider", None)
        if not isinstance(provider, LiveToolProvider):
            raise LiveGradingUnavailable(
                "live grading was asked for but the outcome carries no live provider; the task was run against "
                "replayed tools."
            )
        if not provider.can_grade:
            if fallback is None:
                raise LiveGradingUnavailable(
                    f"the live environment for task {outcome.task_id!r} has no predicate and no fallback grader "
                    f"is configured"
                )
            success, detail = fallback(trace, outcome)
            return success, {**detail, "eval_mode": "live", "graded_by": "fallback"}
        success, detail = provider.grade(outcome.final_text)
        stats = provider.summary()
        return success, {
            "detail": detail,
            "eval_mode": "live",
            "graded_by": "live_predicate",
            "final_state_hash": provider.state_hash(),
            "n_calls": stats["calls"],
            "n_tool_errors": stats["tool_errors"],
            "task": provider.env.label,
        }

    return grade
