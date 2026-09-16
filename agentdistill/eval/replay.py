"""Replay tool provider.

During evaluation the student never touches a live service. Tool results come from the recorded teacher
trajectory, looked up by canonical argument hash. That keeps eval deterministic, free, and safe: nothing is
refunded twice because an eval ran twice.

The cost is that the recording only covers the calls the teacher made. When the student calls something that was
never recorded, there is no result to serve, and that is a **divergence** -- the trajectory stops. Divergence is
reported separately from failure, because a student that diverged did not necessarily do anything wrong; it went
somewhere the recording cannot follow.

Two policies:

- `strict` — only an exact canonical-hash match is served. Any other call diverges.
- `fuzzy` — the nearest recorded call for the same tool is served when similarity clears a threshold. Necessary
  for on-policy rollouts, where the student's phrasing drifts from the teacher's, and dangerous everywhere else:
  a fuzzily served result is not the result the student's call would really have produced. The fuzzy-hit share is
  always reported so a fuzzy success is never mistaken for a real one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from agentdistill.canonical import args_hash, canonical_json, normalize, rules_for


@dataclass
class Recorded:
    tool: str
    args: dict
    args_hash: str
    result: str  # the tool message content, exactly as recorded


class Divergence(Exception):
    """The student called something the recording cannot answer."""

    def __init__(self, tool: str, args: dict, hash_: str, nearest: Recorded | None, score: float) -> None:
        super().__init__(f"divergence on {tool} (hash {hash_[:12]})")
        self.tool = tool
        # Not `self.args`: BaseException.args is a tuple, and shadowing it breaks exception handling in
        # subtle ways.
        self.call_args = args
        self.hash = hash_
        self.nearest = nearest
        self.score = score

    def to_dict(self) -> dict:
        return {
            "tool": self.tool,
            "args": self.call_args,
            "hash": self.hash,
            "nearest_args": self.nearest.args if self.nearest else None,
            "nearest_score": round(self.score, 4),
        }


def _key_value_tokens(tool: str, args: Any) -> set[str]:
    """Flatten normalized arguments into comparable `path=value` tokens.

    Splitting canonical JSON on commas would treat `{"a":{"b":1}}` as one token and compare nothing useful, so
    the structure is walked instead.
    """
    tokens: set[str] = set()

    def walk(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for k, v in value.items():
                walk(v, f"{path}.{k}" if path else str(k))
        elif isinstance(value, list):
            for i, v in enumerate(value):
                walk(v, f"{path}[{i}]")
        else:
            tokens.add(f"{path}={canonical_json(value)}")

    walk(normalize(args, rules_for(tool)), "")
    return tokens


def similarity(tool: str, a: Any, b: Any) -> float:
    """Jaccard over `path=value` tokens. 1.0 means identical after normalization."""
    ta, tb = _key_value_tokens(tool, a), _key_value_tokens(tool, b)
    if not ta and not tb:
        return 1.0
    union = ta | tb
    return len(ta & tb) / len(union) if union else 0.0


class ReplayToolProvider:
    """Serves recorded tool results for one task."""

    def __init__(
        self,
        trace: dict,
        policy: str = "strict",
        fuzzy_threshold: float = 0.92,
        embedder: Any = None,
    ) -> None:
        if policy not in ("strict", "fuzzy"):
            raise ValueError(f"policy must be 'strict' or 'fuzzy', got {policy!r}")
        self.policy = policy
        self.threshold = fuzzy_threshold
        self.embedder = embedder
        self.index: dict[tuple[str, str], Recorded] = {}
        self.by_tool: dict[str, list[Recorded]] = {}
        self.stats: dict[str, Any] = {"replayed": 0, "fuzzy": 0, "diverged": 0, "fuzzy_scores": []}

        calls_by_id: dict[str, tuple[str, Any]] = {}
        for m in trace["messages"]:
            for c in m.get("tool_calls") or []:
                raw = c["function"]["arguments"]
                try:
                    args = json.loads(raw) if isinstance(raw, str) else raw
                except (json.JSONDecodeError, TypeError):
                    args = {}
                calls_by_id[c["id"]] = (c["function"]["name"], args)

        for m in trace["messages"]:
            if m["role"] != "tool":
                continue
            pair = calls_by_id.get(m.get("tool_call_id", ""))
            if pair is None:
                # A result with no matching call. The trace is malformed; skip rather than index it under a
                # tool name we would be guessing at.
                continue
            name, args = pair
            record = Recorded(name, args, args_hash(name, args), m.get("content") or "")
            # First occurrence wins: a tool called twice with the same arguments and different results is
            # non-deterministic, and replaying the first is the only reproducible choice.
            self.index.setdefault((name, record.args_hash), record)
            self.by_tool.setdefault(name, []).append(record)

    @property
    def n_recorded(self) -> int:
        return len(self.index)

    def lookup(self, tool: str, args: dict) -> str:
        """Return the recorded result, or raise `Divergence`."""
        h = args_hash(tool, args)
        hit = self.index.get((tool, h))
        if hit is not None:
            self.stats["replayed"] += 1
            return hit.result

        nearest, score = self.nearest(tool, args)
        if self.policy == "fuzzy" and nearest is not None and score >= self.threshold:
            self.stats["fuzzy"] += 1
            self.stats["fuzzy_scores"].append(round(score, 4))
            return nearest.result

        self.stats["diverged"] += 1
        raise Divergence(tool, args, h, nearest, score)

    def nearest(self, tool: str, args: dict) -> tuple[Recorded | None, float]:
        """The most similar recorded call for this tool, and its score.

        Reported on every divergence: a high score means an argument-phrasing gap that belongs in a per-tool
        normalization rule, and a low score means the student genuinely went somewhere else.
        """
        records = self.by_tool.get(tool, [])
        if not records:
            return None, 0.0
        if self.embedder is None:
            best, best_score = None, -1.0
            for r in records:
                s = similarity(tool, args, r.args)
                if s > best_score:
                    best, best_score = r, s
            return best, max(best_score, 0.0)

        import numpy as np

        vectors = self.embedder.embed(
            [canonical_json(normalize(r.args, rules_for(tool))) for r in records]
        )
        query = self.embedder.embed([canonical_json(normalize(args, rules_for(tool)))])[0]
        norms = np.linalg.norm(vectors, axis=1) * np.linalg.norm(query) + 1e-9
        sims = vectors @ query / norms
        i = int(sims.argmax())
        return records[i], float(sims[i])

    def summary(self) -> dict:
        served = self.stats["replayed"] + self.stats["fuzzy"]
        scores = self.stats["fuzzy_scores"]
        return {
            "replayed": self.stats["replayed"],
            "fuzzy": self.stats["fuzzy"],
            "diverged": self.stats["diverged"],
            "fuzzy_share": (self.stats["fuzzy"] / served) if served else 0.0,
            "min_fuzzy_score": min(scores) if scores else None,
            "n_recorded": self.n_recorded,
        }


@dataclass
class ReplayStats:
    """Aggregated replay statistics across many tasks, for the run report."""

    replayed: int = 0
    fuzzy: int = 0
    diverged: int = 0
    fuzzy_scores: list[float] = field(default_factory=list)

    def add(self, summary: dict) -> None:
        self.replayed += summary["replayed"]
        self.fuzzy += summary["fuzzy"]
        self.diverged += summary["diverged"]
        if summary.get("min_fuzzy_score") is not None:
            self.fuzzy_scores.append(summary["min_fuzzy_score"])

    def to_dict(self) -> dict:
        served = self.replayed + self.fuzzy
        return {
            "replayed": self.replayed,
            "fuzzy": self.fuzzy,
            "diverged": self.diverged,
            "fuzzy_share": (self.fuzzy / served) if served else 0.0,
            "min_fuzzy_score": min(self.fuzzy_scores) if self.fuzzy_scores else None,
        }
