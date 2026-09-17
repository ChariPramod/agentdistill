"""Thompson sampling with a floor.

The cascade decides per turn. The router decides per task, before the first turn, whether this class of task
should go to the student at all. Clusters are the context.

Two departures from textbook Thompson sampling, both about not learning the wrong thing in production:

- **A hard floor.** Once there is evidence that the student is bad at a cluster, exploration there stops. Pure
  Thompson sampling keeps sending a trickle of traffic to a known-bad arm forever, which is fine in a simulation
  and not fine when each sample is a customer.
- **Decay.** Posteriors are discounted so they stay responsive after a retrain. Without it, a new adapter
  inherits hundreds of observations about its predecessor and takes just as many to escape them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ArmState:
    alpha: float = 1.0
    beta: float = 1.0

    @property
    def mean(self) -> float:
        total = self.alpha + self.beta
        return self.alpha / total if total else 0.5

    @property
    def observations(self) -> float:
        # The Beta(1,1) prior contributes two pseudo-observations that are not evidence.
        return max(0.0, self.alpha + self.beta - 2.0)


class ThompsonRouter:
    def __init__(
        self,
        state: dict[tuple[int, str], ArmState] | None = None,
        cost: dict[str, float] | None = None,
        lam: float = 20.0,
        floor: float = 0.55,
        explore_cap: float = 0.10,
        decay: float = 0.995,
        min_observations: int = 10,
        exploit_after: int = 30,
        seed: int = 0,
    ) -> None:
        self.state: dict[tuple[int, str], ArmState] = dict(state or {})
        self.cost = cost or {"student": 0.0, "teacher": 0.0}
        self.lam = lam
        self.floor = floor
        self.explore_cap = explore_cap
        self.decay = decay
        self.min_observations = min_observations
        self.exploit_after = exploit_after
        self.rng = np.random.default_rng(seed)
        self._check_floor_is_reachable()

    def _check_floor_is_reachable(self) -> None:
        """Decay caps how much evidence an arm can ever hold, and the floor needs evidence to engage.

        Each update multiplies alpha+beta by `decay` and adds one, so the total converges to 1/(1-decay) no
        matter how much traffic flows. If `min_observations` sits above that ceiling the floor never fires, and
        the router keeps sending traffic to a student it has already watched fail -- the exact failure the floor
        exists to prevent. Silently disabling a safety floor is worse than refusing to start.
        """
        if self.decay >= 1.0:
            return
        ceiling = 1.0 / (1.0 - self.decay) - 2.0
        if self.min_observations >= ceiling:
            raise ValueError(
                f"decay={self.decay} caps evidence at {ceiling:.1f} observations, so a floor needing "
                f"min_observations={self.min_observations} could never engage. Raise decay or lower "
                f"min_observations."
            )

    def arm(self, cluster_id: int, name: str) -> ArmState:
        return self.state.setdefault((cluster_id, name), ArmState())

    def state_mean(self, cluster_id: int, name: str) -> float:
        return self.arm(cluster_id, name).mean

    def choose(self, cluster_id: int) -> str:
        student = self.arm(cluster_id, "student")
        teacher = self.arm(cluster_id, "teacher")

        # The floor: with evidence that the student is bad here, stop sending it traffic.
        if student.observations >= self.min_observations and student.mean < self.floor:
            return "teacher"

        both_settled = (
            student.observations >= self.exploit_after and teacher.observations >= self.exploit_after
        )
        if both_settled and self.rng.random() > self.explore_cap:
            # Exploit on posterior means, without sampling noise.
            return self._better(student.mean, teacher.mean)

        return self._better(
            float(self.rng.beta(student.alpha, student.beta)),
            float(self.rng.beta(teacher.alpha, teacher.beta)),
        )

    def _better(self, student_value: float, teacher_value: float) -> str:
        """Compare arms on success minus the dollar cost of using them."""
        student_score = student_value - self.lam * self.cost.get("student", 0.0)
        teacher_score = teacher_value - self.lam * self.cost.get("teacher", 0.0)
        return "student" if student_score >= teacher_score else "teacher"

    def update(self, cluster_id: int, arm: str, success: bool) -> None:
        state = self.arm(cluster_id, arm)
        state.alpha = state.alpha * self.decay + (1.0 if success else 0.0)
        state.beta = state.beta * self.decay + (0.0 if success else 1.0)

    def warm_start(self, counts: dict[tuple[int, str], tuple[int, int]]) -> None:
        """Seed posteriors from eval counts so the router never starts blind.

        Starting from a flat prior means the first hundred production requests are spent rediscovering what the
        eval set already measured, and paying for the mistakes.
        """
        for key, (successes, failures) in counts.items():
            self.state[key] = ArmState(alpha=1.0 + successes, beta=1.0 + failures)

    def snapshot(self) -> list[tuple[int, str, float, float]]:
        return [(cluster, arm, s.alpha, s.beta) for (cluster, arm), s in sorted(self.state.items())]

    def summary(self) -> dict:
        clusters = sorted({c for c, _ in self.state})
        below = [
            c for c in clusters
            if self.arm(c, "student").observations >= self.min_observations
            and self.arm(c, "student").mean < self.floor
        ]
        return {
            "clusters": len(clusters),
            "below_floor": below,
            "floor": self.floor,
            "arms": {f"{c}:{a}": {"mean": s.mean, "n": s.observations}
                     for (c, a), s in sorted(self.state.items())},
        }
