"""What the gateway holds while it serves.

Loaded once at boot: the prod and canary adapters, the calibrator and threshold for the prod adapter, the cluster
model, the router, the backends, and the request log.

The defaults are conservative on purpose. No calibration means escalate everything; no prod adapter means
passthrough. The gateway sits in front of a working agent, and it must never be the reason that agent starts
behaving differently.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class GatewayState:
    student: Any = None
    teacher: Any = None
    log: Any = None
    registry: Any = None
    prod_adapter: str | None = None
    canary_adapter: str | None = None
    canary_share: float = 0.0
    prod_threshold: float | None = None
    calibrator: Any = None
    feature_names: list[str] = field(default_factory=list)
    clusters: Any = None
    router: Any = None
    #: Persists router posteriors; set once the router lands (milestone 7).
    router_store: Any = None
    teacher_names: set[str] = field(default_factory=set)
    k_samples: int = 2
    notes: list[str] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return self.student is not None or self.teacher is not None

    @property
    def cascade_available(self) -> bool:
        """A cascade needs a fitted gate and a threshold. Without both it degrades to the teacher."""
        return self.calibrator is not None and self.prod_threshold is not None

    @classmethod
    def uninitialized(cls) -> GatewayState:
        return cls()

    def health(self) -> dict:
        return {
            "ok": self.ready,
            "prod_adapter": self.prod_adapter,
            "canary_adapter": self.canary_adapter,
            "canary_share": self.canary_share,
            "threshold": self.prod_threshold,
            "cascade_available": self.cascade_available,
            "notes": self.notes,
        }

    async def cascade_turn(
        self, messages: list[dict], tools: list[dict], adapter: str | None, threshold: float | None,
        cluster_prior: float = 0.5, temperature: float = 0.0,
    ) -> tuple[dict, dict, dict]:
        """One cascaded turn: student first, teacher if the gate says so.

        Async all the way down rather than bridging the synchronous harness client onto the event loop -- that
        bridge would have had to run a coroutine on the loop it was already inside.

        The scoring itself is shared with the offline client, so the gate that gets measured is the gate that
        serves.
        """
        from agentdistill.cascade.client import prefix_token_estimate, score_turn, turn_index

        threshold = threshold if threshold is not None else self.prod_threshold
        if not self.cascade_available or threshold is None:
            data = await self.teacher.chat(messages, tools, temperature=temperature)
            choice = data["choices"][0]
            return choice, data.get("usage", {}), {
                "arm": "teacher", "escalated": True, "confidence": None,
                "teacher_tokens": data.get("usage", {}).get("completion_tokens", 0),
                "reason": "no usable calibration; escalating every turn",
            }

        student_data = await self.student.chat(
            messages, tools, model=_student_model(adapter), n=1 + self.k_samples,
            logprobs=True, temperature=temperature,
        )
        choices = student_data["choices"]
        primary, extra = choices[0], list(choices[1:])
        student_tokens = len((primary.get("logprobs") or {}).get("content") or [])
        p = score_turn(
            primary, extra, self.calibrator, self.feature_names, cluster_prior,
            turn_index(messages), prefix_token_estimate(messages),
        )

        if p >= threshold:
            return primary, student_data.get("usage", {}), {
                "arm": "student", "escalated": False, "confidence": p,
                "adapter": adapter, "student_tokens": student_tokens,
            }

        teacher_data = await self.teacher.chat(messages, tools, temperature=temperature)
        teacher_choice = teacher_data["choices"][0]
        return teacher_choice, teacher_data.get("usage", {}), {
            "arm": "teacher", "escalated": True, "confidence": p, "adapter": adapter,
            # Generated and discarded, but still paid for.
            "student_tokens": student_tokens,
            "wasted_student_tokens": student_tokens,
            "teacher_tokens": teacher_data.get("usage", {}).get("completion_tokens", 0),
        }


def _student_model(adapter: str | None) -> str:
    """vLLM serves each LoRA under `student:<name>`; the base model is plain `student`."""
    return f"student:{adapter}" if adapter else "student"


def load_state(cfg: Any, registry: Any, student: Any = None, teacher: Any = None) -> GatewayState:
    """Assemble the gateway's state from config and the registry."""
    from agentdistill.gateway.backends import StudentBackend, TeacherBackend
    from agentdistill.gateway.log import RequestLog
    from agentdistill.registry.select import canary_adapter, prod_adapter

    notes: list[str] = []
    state = GatewayState(
        student=student or StudentBackend(cfg.serve.vllm_url),
        teacher=teacher or (TeacherBackend(cfg.teacher.model) if cfg.teacher else None),
        registry=registry,
        log=RequestLog(registry),
        feature_names=list(cfg.cascade.features),
        k_samples=cfg.cascade.k_samples,
        teacher_names={cfg.teacher.model} if cfg.teacher else set(),
    )

    prod = prod_adapter(registry)
    if prod is None:
        notes.append("no adapter is in prod, so the agent's own model name passes through to the teacher")
    else:
        state.prod_adapter = prod["name"]
        calibration = _load_calibration(cfg, registry, prod["id"], notes)
        if calibration:
            state.calibrator, state.prod_threshold = calibration
        else:
            notes.append(
                "the prod adapter has no usable calibration, so the cascade escalates every turn. "
                "Run `agentdistill calibrate` before claiming any cost saving."
            )

    canary = canary_adapter(registry)
    if canary:
        state.canary_adapter = canary["name"]
        state.canary_share = cfg.serve.canary_share

    state.notes = notes
    for note in notes:
        logger.warning("gateway: %s", note)
    return state


def _load_calibration(cfg: Any, registry: Any, adapter_id: str, notes: list[str]):
    from agentdistill.cascade.calibrate import assert_feature_order, load
    from agentdistill.registry.select import NoMatch, latest_calibration

    try:
        row = latest_calibration(registry, adapter_id=adapter_id)
    except NoMatch:
        return None
    try:
        model, report = load(row["model_path"])
    except (FileNotFoundError, OSError) as e:
        notes.append(f"calibration artifact missing at {row['model_path']}: {e}")
        return None
    if model is None or not report.get("usable", False):
        return None
    try:
        assert_feature_order(report, cfg.cascade.features)
    except ValueError as e:
        notes.append(str(e))
        return None
    return model, float(row["threshold"])
