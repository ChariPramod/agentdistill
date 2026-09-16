"""Features for the confidence gate.

Everything here is computed from one generated turn plus a few extra samples of it. The point is to predict
whether *this turn* is good before its tool calls execute, cheaply enough that doing so is worth it.

Two design notes that matter:

- **Argument features are separate from whole-turn features.** A model is frequently confident about which tool
  to call and much less confident about what to put in it. Averaging over the whole turn hides exactly the
  uncertainty that predicts a bad call.
- **Missing features are NaN, not zero.** A turn with no tool call has no argument logprobs, and zero is a
  perfectly plausible logprob value. Encoding "absent" as a real number teaches the gate that absence means
  high confidence.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

#: The order is fixed by `cascade.features` in project.yaml and stored on the calibration row. The runtime
#: asserts the same order before scoring, because a reordered vector scores silently and wrongly.
DEFAULT_FEATURES: tuple[str, ...] = (
    "mean_logprob",
    "min_logprob",
    "p10_logprob",
    "arg_mean_logprob",
    "arg_min_logprob",
    "first_tool_token_entropy",
    "n_tokens",
    "n_tool_calls",
    "has_tool_call",
    "agreement",
    "cluster_prior",
    "turn_idx",
    "prefix_tokens",
)


@dataclass
class TurnFeatures:
    mean_logprob: float
    min_logprob: float
    p10_logprob: float
    arg_mean_logprob: float  # nan when the turn made no tool call
    arg_min_logprob: float
    first_tool_token_entropy: float
    n_tokens: int
    n_tool_calls: int
    has_tool_call: int
    agreement: float  # nan when no extra samples were drawn
    cluster_prior: float
    turn_idx: int
    prefix_tokens: int

    def to_dict(self) -> dict:
        return asdict(self)


def _entropy(top_logprobs: list[dict]) -> float:
    """Entropy over the top-k alternatives at one position.

    Computed over the renormalized top-k rather than the full vocabulary, because that is what a server returns.
    It is a lower bound on the true entropy, which is fine: it is a feature, not a measurement.
    """
    if not top_logprobs:
        return 0.0
    logprobs = np.array([t["logprob"] for t in top_logprobs], dtype=float)
    probs = np.exp(logprobs - logprobs.max())
    probs = probs / probs.sum()
    return float(-(probs * np.log(probs + 1e-12)).sum())


def call_signature(message: dict) -> str:
    """What a turn decided, canonically, for comparing samples to each other."""
    from agentdistill.canonical import args_hash

    calls = []
    for c in message.get("tool_calls") or []:
        raw = c["function"]["arguments"]
        try:
            args = json.loads(raw) if isinstance(raw, str) else raw
        except (json.JSONDecodeError, TypeError):
            args = {"__unparseable__": str(raw)}
        calls.append(args_hash(c["function"]["name"], args))
    return json.dumps([message.get("content") or "", sorted(calls)], sort_keys=True)


def self_consistency(primary: dict, samples: list[dict]) -> float:
    """Share of extra samples that made the same decision.

    The single most informative feature in practice, and the most expensive: it costs k extra samples. With
    prefix caching those samples share nearly the whole prompt, so the marginal cost is small -- but the
    ablation in the calibration report shows what it is worth on a given project.
    """
    if not samples:
        return float("nan")
    base = call_signature(primary["message"])
    return float(np.mean([call_signature(s["message"]) == base for s in samples]))


def turn_features(
    choice: dict,
    arg_token_mask: list[bool],
    extra_samples: list[dict],
    cluster_prior: float,
    turn_idx: int,
    prefix_tokens: int,
) -> TurnFeatures:
    """Build the feature vector for one generated turn."""
    content = (choice.get("logprobs") or {}).get("content") or []
    logprobs = np.array([t["logprob"] for t in content], dtype=float)
    if logprobs.size == 0:
        logprobs = np.array([float("nan")])

    mask = np.array(arg_token_mask, dtype=bool) if arg_token_mask else np.zeros(len(content), dtype=bool)
    if mask.size != logprobs.size:
        # A mask that does not line up with the tokens describes different tokens; better to have no argument
        # features than wrong ones.
        mask = np.zeros(logprobs.size, dtype=bool)
    arg_logprobs = logprobs[mask] if mask.any() else np.array([float("nan")])

    first_arg_token = next((t for t, m in zip(content, mask, strict=False) if m), None)
    entropy = _entropy(first_arg_token.get("top_logprobs") or []) if first_arg_token else 0.0

    calls = choice["message"].get("tool_calls") or []
    return TurnFeatures(
        mean_logprob=_nanmean(logprobs),
        min_logprob=_nanmin(logprobs),
        p10_logprob=_nanpercentile(logprobs, 10),
        arg_mean_logprob=_nanmean(arg_logprobs),
        arg_min_logprob=_nanmin(arg_logprobs),
        first_tool_token_entropy=entropy,
        n_tokens=len(content),
        n_tool_calls=len(calls),
        has_tool_call=int(bool(calls)),
        agreement=self_consistency(choice, extra_samples),
        cluster_prior=float(cluster_prior),
        turn_idx=int(turn_idx),
        prefix_tokens=int(prefix_tokens),
    )


# numpy warns on an all-NaN slice and returns NaN, which is the answer we want; these return it quietly, so a
# turn with no tool call does not fill the logs.
def _finite(values: np.ndarray) -> np.ndarray:
    return values[~np.isnan(values)]


def _nanmean(values: np.ndarray) -> float:
    finite = _finite(values)
    return float(finite.mean()) if finite.size else float("nan")


def _nanmin(values: np.ndarray) -> float:
    finite = _finite(values)
    return float(finite.min()) if finite.size else float("nan")


def _nanpercentile(values: np.ndarray, q: float) -> float:
    finite = _finite(values)
    return float(np.percentile(finite, q)) if finite.size else float("nan")


def as_vector(features: TurnFeatures | dict, names: list[str] | tuple[str, ...]) -> np.ndarray:
    """Project features onto the configured order.

    NaN becomes a sentinel here rather than earlier: `HistGradientBoostingClassifier` handles NaN natively, so
    the imputation only happens for models that cannot.
    """
    d = features.to_dict() if isinstance(features, TurnFeatures) else dict(features)
    missing = [n for n in names if n not in d]
    if missing:
        raise KeyError(f"feature vector is missing {missing}; configured order is {list(names)}")
    return np.array([float(d[n]) for n in names], dtype=float)


def matrix(rows: list[TurnFeatures | dict], names: list[str] | tuple[str, ...]) -> np.ndarray:
    return np.vstack([as_vector(r, names) for r in rows]) if rows else np.zeros((0, len(names)))


def describe_missing(rows: list[TurnFeatures | dict], names: list[str] | tuple[str, ...]) -> dict[str, float]:
    """Share of NaN per feature. A feature that is mostly missing is not carrying signal."""
    X = matrix(rows, names)
    if X.size == 0:
        return dict.fromkeys(names, 0.0)
    return {name: float(np.isnan(X[:, i]).mean()) for i, name in enumerate(names)}


def is_missing(value: Any) -> bool:
    return isinstance(value, float) and math.isnan(value)
