"""Record trajectories and label them with end-state predicates.

Output is JSONL that `agentdistill ingest jsonl` accepts directly.

    # Pipeline test, no API key, no network. Traces are NOT training data -- see scripted_teacher.py.
    python -m examples.support_agent.record --scripted --n 60 --out traces.jsonl

    # A real corpus. Any LiteLLM-supported model. Read docs/tos.md first.
    python -m examples.support_agent.record --model openai/gpt-4.1 --n 400 --out traces.jsonl

    # Against a running agentdistill gateway, in either dialect. Used by scripts/serve_smoke.sh.
    python -m examples.support_agent.record --model openai/cascade::auto \
        --base-url http://127.0.0.1:8710/v1 --n 5 --out smoke.jsonl

Labels come from predicates over the final database state plus the final message -- no judge, no rubric, no model
grading a model. That is what makes the success rate here worth comparing against.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from examples.support_agent import scenarios
from examples.support_agent.agent import final_assistant_text, run_agent
from examples.support_agent.crm import TOOLS

HERE = Path(__file__).resolve().parent


def gateway_completion(base_url: str) -> Any:
    """A `completion` callable that talks to a gateway instead of a provider.

    The model prefix picks the dialect -- `openai/...` or `anthropic/...` -- because the gateway speaks both and
    the point of driving it through the example agent is to exercise the dialect translation, not to bypass it.

    An API key is required by the SDKs but not by the gateway, which authenticates nothing and must not be
    exposed. A placeholder keeps the SDK happy without putting a real credential on a local connection.
    """
    import litellm

    def completion(**kwargs: Any) -> Any:
        kwargs.setdefault("api_base", base_url)
        kwargs.setdefault("api_key", "agentdistill-gateway-local")
        return litellm.completion(**kwargs)

    return completion


def record_one(task: Any, model: str, completion: Any = None, max_turns: int = 12,
               temperature: float = 0.2) -> dict:
    """Run one task and return a normalized, labelled trace."""
    crm = task.fresh_crm()
    run = run_agent(task, crm, TOOLS, model, max_turns=max_turns, temperature=temperature, completion=completion)
    final_text = final_assistant_text(run["messages"])
    success, detail = task.predicate(crm, final_text)

    payload = json.dumps(run["messages"], sort_keys=True).encode()
    return {
        "id": f"tr_{hashlib.sha256(payload).hexdigest()[:20]}",
        "task_id": task.task_id,
        "task_input": {"text": task.user_message, "scenario": task.scenario},
        "messages": run["messages"],
        "tools": TOOLS,
        "teacher_model": model,
        "success": bool(success),
        "grader": "predicate",
        "score": 1.0 if success else 0.0,
        "prompt_tokens": run["usage"]["prompt_tokens"],
        "completion_tokens": run["usage"]["completion_tokens"],
        "metadata": {
            "scenario": task.scenario,
            "db_seed": task.db_seed,
            "latency_ms": run["latency_ms"],
            "stop_reason": run["stop_reason"],
            "system_prompt_version": run["system_prompt_version"],
            "predicate_detail": detail,
            "n_tool_calls_made": len(crm.calls),
            "final_state_hash": crm.state_hash(),
        },
    }


def record(
    tasks: list[Any],
    model: str,
    completion: Any = None,
    max_turns: int = 12,
    temperature: float = 0.2,
    progress: bool = False,
) -> list[dict]:
    traces = []
    for i, task in enumerate(tasks, 1):
        try:
            traces.append(record_one(task, model, completion=completion, max_turns=max_turns,
                                     temperature=temperature))
        except Exception as e:
            # One bad episode must not lose the rest of a paid recording run.
            print(f"  [{i}/{len(tasks)}] {task.task_id}: FAILED {type(e).__name__}: {e}", file=sys.stderr)
            continue
        if progress and i % 10 == 0:
            rate = sum(t["success"] for t in traces) / len(traces)
            print(f"  [{i}/{len(tasks)}] success so far {rate:.0%}", file=sys.stderr)
    return traces


def summarize(traces: list[dict]) -> dict:
    """What to read before trusting a corpus."""
    if not traces:
        return {"n": 0}
    by_scenario: dict[str, list[bool]] = {}
    for t in traces:
        by_scenario.setdefault(t["metadata"]["scenario"], []).append(t["success"])
    return {
        "n": len(traces),
        "success_rate": sum(t["success"] for t in traces) / len(traces),
        "prompt_tokens": sum(t["prompt_tokens"] for t in traces),
        "completion_tokens": sum(t["completion_tokens"] for t in traces),
        "by_scenario": {
            k: {"n": len(v), "success": sum(v) / len(v)} for k, v in sorted(by_scenario.items())
        },
    }


def print_summary(summary: dict) -> None:
    print(f"\nrecorded {summary['n']} traces, teacher success {summary['success_rate']:.1%}")
    print(f"tokens: {summary['prompt_tokens']:,} prompt, {summary['completion_tokens']:,} completion")
    print("\nby scenario:")
    for name, s in summary["by_scenario"].items():
        flag = ""
        if s["success"] >= 0.95:
            flag = "  <- too easy; add a wrinkle"
        elif s["success"] <= 0.2:
            flag = "  <- too hard or the predicate is wrong; read a trace by hand"
        print(f"  {name:32} n={s['n']:<4} success={s['success']:.0%}{flag}")

    rate = summary["success_rate"]
    if rate >= 0.95:
        print(
            "\nWARNING: teacher success is above 95%. The scenarios cannot separate a student from the teacher; "
            "add wrinkles before recording a full corpus."
        )
    elif rate < 0.6:
        print(
            "\nWARNING: teacher success is below 60%. Either the scenarios are unreasonable or a predicate is "
            "wrong. Read ten traces by hand before spending money on a full run."
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None, help="Any LiteLLM model id, e.g. openai/gpt-4.1.")
    ap.add_argument("--scripted", action="store_true",
                    help="Use the rule-based teacher. No API key. NOT training data.")
    ap.add_argument("--error-rate", type=float, default=0.15,
                    help="Scripted teacher only: how often it takes a wrong action, so the corpus has failures.")
    ap.add_argument("--n", type=int, default=60, help="Number of tasks.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=HERE / "traces.jsonl")
    ap.add_argument("--scenarios", nargs="*", default=None, help="Restrict to these scenario names.")
    ap.add_argument("--max-turns", type=int, default=12)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--base-url", default=None,
                    help="Send requests here instead of to the provider, e.g. a running agentdistill gateway. "
                         "The model prefix (openai/ or anthropic/) picks the dialect.")
    args = ap.parse_args(argv)

    if not args.scripted and not args.model:
        ap.error("pass --model <litellm-model-id>, or --scripted to run without a teacher")
    if args.base_url and args.scripted:
        ap.error("--base-url and --scripted are mutually exclusive; the scripted teacher makes no requests")
    if args.base_url and not (args.model or "").startswith(("openai/", "anthropic/")):
        ap.error(
            "--base-url needs a model prefixed with the dialect to speak, e.g. openai/cascade::auto or "
            "anthropic/cascade::auto"
        )

    completion = None
    model = args.model or "scripted/rule-based-teacher"
    if args.scripted:
        from examples.support_agent.scripted_teacher import ScriptedTeacher

        completion = ScriptedTeacher(error_rate=args.error_rate, seed=args.seed)
        print(
            "Using the scripted teacher. These traces exercise the pipeline; they are NOT a teacher's traces and "
            "must not be used to train a student or to publish a number.",
            file=sys.stderr,
        )
    elif args.base_url:
        completion = gateway_completion(args.base_url)
        print(f"sending requests to {args.base_url} as {model}", file=sys.stderr)

    tasks = scenarios.sample(args.n, seed=args.seed, scenarios=args.scenarios)
    print(f"recording {len(tasks)} tasks with {model}", file=sys.stderr)
    traces = record(tasks, model, completion=completion, max_turns=args.max_turns,
                    temperature=args.temperature, progress=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(json.dumps(t) for t in traces) + "\n")
    print(f"wrote {len(traces)} traces -> {args.out}", file=sys.stderr)
    print_summary(summarize(traces))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
