"""Merging a LoRA adapter into its base weights.

vLLM can serve LoRA adapters directly, so merging is not required. It is worth doing when the adapter is going
to be quantized -- quantizers work on a single set of weights -- and when the serving stack's LoRA support is
slower than a merged model.

Merging is also where a training configuration error becomes visible for the first time. A `target_modules`
list that missed a projection, or an adapter trained against a different base revision, produces a merged model
that loads cleanly and behaves differently. `verify_merge` is the check that catches it, and `merge_adapter`
refuses to hand back a merge that fails it.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: How far merged and unmerged next-action agreement may diverge, in percentage points.
MAX_FULL_MATCH_DRIFT_PP = 2.0


class MergeVerificationFailed(RuntimeError):
    """The merged weights do not reproduce the adapter's behaviour."""


def merge_adapter(
    base_model: str, adapter_path: str, out_dir: str, dtype: str = "bfloat16", device_map: str | None = None
) -> dict:
    """Merge `adapter_path` into `base_model` and write the result to `out_dir`.

    Always into bf16, never into a quantized base. Merging into 4-bit weights means dequantizing, adding the
    LoRA delta, and requantizing, and the round trip loses more than the adapter contributed. That failure is
    silent: the model loads, generates fluent text, and is worse.

    `device_map` defaults to `auto` on CUDA and to CPU everywhere else. A merge is arithmetic on weights, not
    generation: it does not need an accelerator, and `auto` on a machine with MPS or a partial accelerate setup
    segfaults rather than falling back. The rehearsal died here.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved_dtype = getattr(torch, dtype, None)
    if resolved_dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError(f"refusing to merge into {dtype}; merge into a float dtype and quantize afterwards")

    if device_map is None:
        device_map = "auto" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    # `dtype`, not `torch_dtype`: transformers 5.x deprecated the old name.
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=resolved_dtype, device_map=device_map)
    merged = PeftModel.from_pretrained(model, adapter_path).merge_and_unload()

    os.makedirs(out_dir, exist_ok=True)
    merged.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)
    write_marker(out_dir, {"base_model": base_model, "adapter_path": adapter_path, "dtype": dtype,
                           "device_map": device_map})
    return {"out_dir": out_dir, "dtype": dtype, "base_model": base_model, "adapter_path": adapter_path,
            "device_map": device_map}


def write_marker(out_dir: str, payload: dict) -> Path:
    """Record what produced these weights, next to the weights.

    A directory of safetensors with no provenance is unusable six weeks later: nothing in the files says which
    adapter or which base revision they came from.
    """
    path = Path(out_dir) / "agentdistill_merge.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def read_marker(out_dir: str) -> dict | None:
    path = Path(out_dir) / "agentdistill_merge.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def paired_full_match(merged: list, unmerged: list, iters: int = 2000, seed: int = 0) -> dict:
    """Paired per-turn agreement between two sets of `TurnResult`s, with a task-clustered interval.

    Paired because the two models saw identical prefixes: the same turn either matched under both or differed,
    and comparing the two rates unpaired throws away that structure and widens the interval for no reason.

    The interval is reported alongside the gate rather than instead of it. On fifty turns a two-point tolerance
    is inside the noise, and a reviewer should be able to see that rather than infer it.
    """
    import numpy as np

    if not merged or not unmerged:
        return {"drift_pp": float("nan"), "ci95_pp": (float("nan"), float("nan")), "n_turns": 0, "n_tasks": 0}
    if len(merged) != len(unmerged):
        raise ValueError(
            f"paired comparison needs the same turns on both sides, got {len(merged)} and {len(unmerged)}"
        )

    by_task: dict[str, list[float]] = {}
    for a, b in zip(merged, unmerged, strict=True):
        if a.task_id != b.task_id or a.turn_idx != b.turn_idx:
            raise ValueError("paired comparison requires both sides in the same turn order")
        by_task.setdefault(a.task_id, []).append(float(a.full_match) - float(b.full_match))

    tasks = sorted(by_task)
    per_task = np.array([float(np.mean(by_task[t])) for t in tasks], dtype=float)
    delta = float(per_task.mean()) * 100

    if len(per_task) < 2:
        ci = (float("nan"), float("nan"))
    else:
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(per_task), size=(iters, len(per_task)))
        lo, hi = np.percentile(per_task[idx].mean(axis=1) * 100, [2.5, 97.5])
        ci = (float(lo), float(hi))

    return {
        "drift_pp": delta,
        "ci95_pp": ci,
        "n_turns": len(merged),
        "n_tasks": len(tasks),
    }


def verify_merge(
    merged_dir: str,
    adapter_path: str,
    base_model: str,
    traces: list[dict],
    client_factory: Any,
    max_turns_per_trace: int | None = None,
) -> dict:
    """Run teacher-forced next-action prediction through both paths and compare.

    Teacher-forced rather than free-running: the two models are given identical prefixes, so any difference in
    the predicted next action comes from the weights rather than from divergent trajectories. A free-running
    comparison would disagree constantly for reasons that have nothing to do with the merge.

    `client_factory(spec) -> TurnClient` is injected so this stays testable without a GPU, and so the same code
    serves whether the turns come from vLLM, from transformers, or from a stub.
    """
    from agentdistill.eval.teacher_forced import summarize, teacher_forced_batched

    if not traces:
        return {"ok": False, "reason": "no held-out traces to verify against", "n_turns": 0}

    merged = teacher_forced_batched(
        traces, client_factory({"kind": "merged", "path": merged_dir}), max_turns_per_trace
    )
    unmerged = teacher_forced_batched(
        traces,
        client_factory({"kind": "lora", "base": base_model, "adapter": adapter_path}),
        max_turns_per_trace,
    )
    if not merged:
        return {"ok": False, "reason": "the held-out traces produced no scorable turns", "n_turns": 0}

    paired = paired_full_match(merged, unmerged)
    merged_rate = summarize(merged)["full_match"][0]
    unmerged_rate = summarize(unmerged)["full_match"][0]

    result = {
        "merged_full_match": merged_rate,
        "unmerged_full_match": unmerged_rate,
        "drift_pp": paired["drift_pp"],
        "ci95_pp": list(paired["ci95_pp"]),
        "n_turns": paired["n_turns"],
        "n_tasks": paired["n_tasks"],
        "tolerance_pp": MAX_FULL_MATCH_DRIFT_PP,
        "ok": abs(paired["drift_pp"]) <= MAX_FULL_MATCH_DRIFT_PP,
    }
    # The plan specified 50 held-out turns against a 2 pp tolerance. One disagreeing turn out of 50 is 2 pp, so
    # at that size the gate admits only an exact reproduction -- which is a defensible thing to require, but not
    # what "within 2 points" sounds like. Say so rather than let a reader assume there is slack.
    granularity = 100.0 / result["n_turns"] if result["n_turns"] else float("inf")
    if granularity > MAX_FULL_MATCH_DRIFT_PP:
        result["note"] = (
            f"{result['n_turns']} turns give a granularity of {granularity:.1f} pp per turn, coarser than the "
            f"{MAX_FULL_MATCH_DRIFT_PP} pp tolerance: at this size the check passes only an exact match. Verify "
            f"on at least {int(100 / MAX_FULL_MATCH_DRIFT_PP)} turns for the tolerance to mean anything."
        )

    if not result["ok"]:
        result["reason"] = (
            f"merged next-action agreement differs from the adapter by {paired['drift_pp']:+.1f} pp "
            f"(95% CI [{paired['ci95_pp'][0]:+.1f}, {paired['ci95_pp'][1]:+.1f}]) over {paired['n_turns']} "
            f"turns, more than the {MAX_FULL_MATCH_DRIFT_PP} pp tolerance. The usual causes are a "
            f"`target_modules` list that missed a projection, a base model revision that moved under the "
            f"adapter, or a merge into a quantized base."
        )
    return result


def merge_and_verify(
    base_model: str,
    adapter_path: str,
    out_dir: str,
    traces: list[dict],
    client_factory: Any,
    dtype: str = "bfloat16",
    max_turns_per_trace: int | None = None,
) -> dict:
    """Merge, then verify, and raise rather than return a merge that does not reproduce the adapter."""
    info = merge_adapter(base_model, adapter_path, out_dir, dtype=dtype)
    verification = verify_merge(
        out_dir, adapter_path, base_model, traces, client_factory, max_turns_per_trace
    )
    info["verification"] = verification
    write_marker(out_dir, {**info, "verification": verification})
    if not verification["ok"]:
        raise MergeVerificationFailed(verification.get("reason", "merge verification failed"))
    return info
