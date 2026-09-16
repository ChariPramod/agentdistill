"""Model-name policies.

The agent does not change. It points `base_url` at the gateway and keeps sending the model name it always sent,
and that name decides what happens:

| requested model            | behaviour |
|----------------------------|-----------|
| the teacher's own name     | the router decides: cascade or teacher passthrough |
| `teacher`                  | passthrough, for paired evals |
| `student` / `student:<a>`  | student only, no escalation, for evals |
| `cascade:<a>:<tau>`        | a fixed cascade at that threshold, for evals |

The eval names exist so a report can measure each arm in isolation. Only the agent's own model name goes through
the router, because that is the one whose behaviour has to stay invisible to the caller.
"""

from __future__ import annotations

from dataclasses import dataclass


class UnknownModel(ValueError):
    """The requested model name matches no policy."""


@dataclass
class Route:
    mode: str  # 'teacher' | 'student' | 'cascade' | 'router'
    adapter: str | None = None
    threshold: float | None = None

    def to_dict(self) -> dict:
        return {"mode": self.mode, "adapter": self.adapter, "threshold": self.threshold}


def resolve(
    model: str,
    prod_adapter: str | None,
    prod_threshold: float | None,
    teacher_names: set[str],
) -> Route:
    """Map a requested model name onto a route."""
    if model == "teacher":
        return Route("teacher")

    if model == "student" or model.startswith("student:"):
        _, _, adapter = model.partition(":")
        return Route("student", adapter or prod_adapter)

    if model.startswith("cascade:"):
        parts = model.split(":")
        if len(parts) != 3:
            raise UnknownModel(f"cascade model names look like cascade:<adapter>:<tau|auto>, got {model!r}")
        _, adapter, tau = parts
        threshold = prod_threshold if tau in ("", "auto") else _parse_threshold(tau, model)
        return Route("cascade", adapter or prod_adapter, threshold)

    if model in teacher_names:
        # The agent's own model name. With no student deployed there is nothing to route to, so it passes
        # through -- the gateway must never be the reason an agent stops working.
        if prod_adapter is None:
            return Route("teacher")
        return Route("router", prod_adapter, prod_threshold)

    raise UnknownModel(
        f"unknown model name {model!r}. Known: 'teacher', 'student[:adapter]', 'cascade:<adapter>:<tau>', "
        f"or one of the configured teacher models {sorted(teacher_names)}."
    )


def _parse_threshold(tau: str, model: str) -> float:
    try:
        value = float(tau)
    except ValueError as e:
        raise UnknownModel(f"{model!r} has a non-numeric threshold {tau!r}") from e
    if not 0.0 <= value <= 1.0:
        raise UnknownModel(f"{model!r} has a threshold outside [0, 1]")
    return value
