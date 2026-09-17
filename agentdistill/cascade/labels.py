"""Turn labels for the confidence gate.

The gate predicts whether *this turn* is good, so it needs per-turn labels, and task success alone is too coarse:
a task can succeed despite a bad turn that a later turn repaired, and labelling that turn "good" teaches the gate
to be confident about exactly the mistakes it exists to catch.

Three rules, in order of how much they are worth:

1. **teacher_match / teacher_mismatch** — the teacher faced this same prefix and we know what it did. The
   strongest signal available.
2. **corrected_later** — no teacher reference, but the trajectory itself shows the turn was wrong: the tool
   errored, or a later turn redid the same call differently.
3. **uncorrected** — no teacher reference and nothing visibly went wrong. The weakest label, and effectively an
   assumption.

`how` is recorded on every label so the calibration report can show the mix. A label set that is mostly
`uncorrected` is weak, and the gate's AUROC should be read with that in mind rather than taken at face value.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from agentdistill.canonical import args_hash

#: How far after a turn a tool error still counts as that turn's fault, in messages per issued call.
ERROR_WINDOW_PER_CALL = 2


def call_signature(message: dict) -> frozenset[str]:
    """Canonical hashes of a turn's tool calls."""
    out = set()
    for c in message.get("tool_calls") or []:
        raw = c["function"]["arguments"]
        try:
            args = json.loads(raw) if isinstance(raw, str) else raw
        except (json.JSONDecodeError, TypeError):
            args = {"__unparseable__": str(raw)}
        out.add(args_hash(c["function"]["name"], args))
    return frozenset(out)


def prefix_key(messages: list[dict], upto: int) -> str:
    """Identity of the context a turn was produced in.

    Includes tool names and canonical argument hashes, not just roles and text: a call with different arguments
    leads to a different state, so the teacher's next turn is not a valid reference for it. Matching on roles
    alone would compare the student's turn against a teacher turn taken in a different situation.
    """
    return json.dumps(
        [(m["role"], m.get("content"), sorted(call_signature(m))) for m in messages[:upto]],
        sort_keys=True,
    )


def _is_error_result(message: dict) -> bool:
    content = (message.get("content") or "").lower()
    return '"error"' in content or content.startswith(("error:", "error "))


def label_turns(
    rollout_messages: list[dict], task_success: bool, teacher_messages: list[dict] | None
) -> list[tuple[int, bool, str]]:
    """Return `(turn_index, good, how)` for every assistant turn in the rollout."""
    teacher_by_prefix: dict[str, dict] = {}
    if teacher_messages:
        for i, m in enumerate(teacher_messages):
            if m["role"] == "assistant":
                teacher_by_prefix[prefix_key(teacher_messages, i)] = m

    out: list[tuple[int, bool, str]] = []
    for i, m in enumerate(rollout_messages):
        if m["role"] != "assistant":
            continue
        if not task_success:
            # The task failed. Nothing in it is evidence of a good turn, whatever the individual turn looked like.
            out.append((i, False, "task_failed"))
            continue

        teacher_turn = teacher_by_prefix.get(prefix_key(rollout_messages, i))
        if teacher_turn is not None:
            same = (
                call_signature(m) == call_signature(teacher_turn)
                and bool(m.get("tool_calls")) == bool(teacher_turn.get("tool_calls"))
            )
            out.append((i, same, "teacher_match" if same else "teacher_mismatch"))
            continue

        out.append((i, not _corrected_later(rollout_messages, i), "corrected_later"
                    if _corrected_later(rollout_messages, i) else "uncorrected"))
    return out


def _corrected_later(messages: list[dict], index: int) -> bool:
    """Did the trajectory itself show this turn was wrong?"""
    turn = messages[index]
    names = {c["function"]["name"] for c in (turn.get("tool_calls") or [])}
    window = ERROR_WINDOW_PER_CALL * max(len(names), 1)
    signature = call_signature(turn)

    for offset, later in enumerate(messages[index + 1 :]):
        # A tool error close behind the turn is that turn's fault.
        if later["role"] == "tool" and offset < window and _is_error_result(later):
            return True
        # The same tool called again with different arguments is a retry, which means the first attempt was wrong.
        if later["role"] == "assistant" and names:
            later_names = {c["function"]["name"] for c in (later.get("tool_calls") or [])}
            if (later_names & names) and call_signature(later) != signature:
                return True
    return False


def label_mix(labels: list[tuple[int, bool, str]]) -> dict[str, int]:
    """How many labels came from each rule. Printed with the calibration so the gate's AUROC can be read fairly."""
    return dict(Counter(how for _, _, how in labels))


def weak_label_share(labels: list[tuple[int, bool, str]]) -> float:
    """Share of labels resting on an assumption rather than evidence."""
    if not labels:
        return 0.0
    return sum(1 for _, _, how in labels if how == "uncorrected") / len(labels)


def label_rollouts(
    rollouts: list[dict], teacher_by_task: dict[str, dict]
) -> tuple[list[dict], dict[str, Any]]:
    """Label every rollout's turns, returning flat records plus the label mix."""
    records: list[dict] = []
    all_labels: list[tuple[int, bool, str]] = []
    for rollout in rollouts:
        task_id = rollout.get("task_id") or rollout["id"]
        teacher = teacher_by_task.get(task_id)
        labels = label_turns(
            rollout["messages"], bool(rollout.get("success")), teacher["messages"] if teacher else None
        )
        all_labels.extend(labels)
        for turn_index, good, how in labels:
            records.append({
                "task_id": task_id,
                "rollout_id": rollout.get("id"),
                "turn_index": turn_index,
                "good": good,
                "how": how,
                "message": rollout["messages"][turn_index],
                # The messages this turn was generated from. The gate has a `prefix_tokens` feature, and a
                # turn's position in a long trajectory is one of the things that predicts whether it is right.
                "prefix": rollout["messages"][:turn_index],
            })
    return records, {
        "mix": label_mix(all_labels),
        "n": len(all_labels),
        "weak_share": weak_label_share(all_labels),
        "positive_rate": (sum(1 for _, g, _ in all_labels if g) / len(all_labels)) if all_labels else 0.0,
    }


def as_choice(message: dict) -> tuple[dict, list[dict]]:
    """Split a stored eval message into the (choice, extra samples) pair the feature extractor expects.

    The eval harness stores an assistant message with `logprobs` and `samples` hung off it; the gate's feature
    code works on an OpenAI-style *choice*, `{message, logprobs, text}`. This is the adapter between them, and
    it exists in one place so that a gate fitted offline and a gate serving in the gateway are fitted and
    evaluated on identically shaped input. Two conversions would eventually disagree, and the disagreement would
    show up as a calibrated threshold that behaves differently in production.
    """
    extras = {"logprobs", "samples", "text"}
    core = {k: v for k, v in message.items() if k not in extras}
    tokens = [t["token"] for t in (message.get("logprobs") or {}).get("content") or []]
    choice = {
        "message": core,
        "logprobs": message.get("logprobs"),
        "text": message.get("text") or "".join(tokens),
    }
    samples = [
        {"message": {k: v for k, v in s.items() if k not in extras}}
        for s in (message.get("samples") or [])
    ]
    return choice, samples


def features_for(
    message: dict, cluster_prior: float = 0.5, turn_idx: int = 0, prefix_tokens: int = 0
) -> Any:
    """The gate's feature vector for one stored turn."""
    from agentdistill.cascade.arg_mask import arg_token_mask
    from agentdistill.cascade.features import turn_features

    choice, samples = as_choice(message)
    tokens = [t["token"] for t in (choice.get("logprobs") or {}).get("content") or []]
    mask = arg_token_mask(tokens, choice["text"], choice["message"].get("tool_calls") or [])
    return turn_features(choice, mask, samples, cluster_prior, turn_idx, prefix_tokens)


def has_logprobs(message: dict) -> bool:
    return bool((message.get("logprobs") or {}).get("content"))


def attach_features(records: list[dict], cluster_priors: dict[str, float] | None = None) -> int:
    """Compute and attach a feature vector to every record whose turn carries logprobs.

    Returns how many were attached. Records without logprobs are left alone rather than given a vector of NaN:
    a row that is entirely missing teaches the calibrator nothing and dilutes every metric computed over it.
    """
    from agentdistill.cascade.client import prefix_token_estimate

    attached = 0
    for record in records:
        message = record.get("message") or {}
        if not has_logprobs(message):
            continue
        prior = (cluster_priors or {}).get(record.get("task_id", ""), 0.5)
        record["features"] = features_for(
            message,
            cluster_prior=prior,
            turn_idx=record.get("turn_index", 0),
            prefix_tokens=prefix_token_estimate(record.get("prefix") or []),
        )
        attached += 1
    return attached
