"""Eval-set decontamination.

A training trace that is also an eval task inflates every number downstream. This filter drops any trace whose
task input exactly matches an eval task, or whose 8-gram overlap with one exceeds the configured share.

The overlap is measured against the *training* trace's n-grams: a short training task fully contained in a long
eval task is contamination, and normalizing by the eval task's length would hide it.
"""

from __future__ import annotations

import hashlib
import re

_WORD = re.compile(r"\w+")


def task_text(task_input: dict | str | None) -> str:
    if task_input is None:
        return ""
    if isinstance(task_input, str):
        return task_input
    if isinstance(task_input, dict):
        # The system prompt is shared across every task in a project; including it would make every pair look
        # contaminated. Only the user-posed part identifies a task.
        user = task_input.get("user")
        if user is not None:
            return str(user)
        return " ".join(str(v) for v in task_input.values())
    return str(task_input)


def normalize_text(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def ngrams(tokens: list[str], n: int) -> set[str]:
    if len(tokens) < n:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def exact_key(text: str) -> str:
    return hashlib.sha256(" ".join(normalize_text(text)).encode("utf-8")).hexdigest()


class Decontaminator:
    """Built once from the eval sets, then queried per trace."""

    def __init__(self, eval_task_inputs: list[dict | str | None], n: int = 8, overlap: float = 0.5) -> None:
        self.n = n
        self.overlap = overlap
        self.exact: set[str] = set()
        self.eval_ngrams: list[set[str]] = []
        for ti in eval_task_inputs:
            text = task_text(ti)
            if not text.strip():
                continue
            self.exact.add(exact_key(text))
            self.eval_ngrams.append(ngrams(normalize_text(text), n))
        # One flat set makes the common "no overlap at all" case a single cheap intersection.
        self.all_ngrams: set[str] = set().union(*self.eval_ngrams) if self.eval_ngrams else set()

    def __bool__(self) -> bool:
        return bool(self.exact or self.eval_ngrams)

    def check(self, task_input: dict | str | None) -> tuple[bool, str]:
        """Return (contaminated, reason)."""
        text = task_text(task_input)
        if not text.strip():
            return False, ""
        if exact_key(text) in self.exact:
            return True, "exact match with an eval task"
        train = ngrams(normalize_text(text), self.n)
        if not train or not self.all_ngrams:
            return False, ""
        if not (train & self.all_ngrams):
            return False, ""
        best = max((len(train & ev) / len(train) for ev in self.eval_ngrams), default=0.0)
        if best >= self.overlap:
            return True, f"{best:.0%} {self.n}-gram overlap with an eval task"
        return False, ""


def contaminated_ids(
    traces: list[dict], eval_task_inputs: list[dict | str | None], n: int = 8, overlap: float = 0.5
) -> dict[str, str]:
    """Map trace id -> reason, for every contaminated trace."""
    dec = Decontaminator(eval_task_inputs, n=n, overlap=overlap)
    if not dec:
        return {}
    out: dict[str, str] = {}
    for t in traces:
        hit, reason = dec.check(t.get("task_input"))
        if hit:
            out[t["id"]] = reason
    return out
