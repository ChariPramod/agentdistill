"""Training callbacks.

Loss is a proxy. Next-action accuracy is the first metric that reflects what the agent actually has to do, so it
runs at every eval step and lands in the registry alongside the loss.
"""

from __future__ import annotations

import logging
from typing import Any

from transformers import TrainerCallback

from agentdistill.eval.teacher_forced import format_summary, summarize, teacher_forced

logger = logging.getLogger(__name__)


class NextActionCallback(TrainerCallback):
    """Teacher-forced next-action accuracy at each eval step.

    Generation under 4-bit with gradient checkpointing is slow, so this is deliberately small: a couple of turns
    from each of a few held-out traces. It is a trend line, not the final number -- the full run happens once at
    the end, and the real number comes from the harness.
    """

    def __init__(
        self,
        traces: list[dict],
        tok: Any,
        parser_name: str | None = None,
        family: str | None = None,
        max_turns: int = 50,
        turns_per_trace: int = 2,
    ) -> None:
        self.traces = traces
        self.tok = tok
        self.parser_name = parser_name
        self.family = family
        self.max_turns = max_turns
        self.turns_per_trace = turns_per_trace
        #: The most recent summary, read by `train_sft` for the final registry metrics.
        self.last_summary: dict | None = None
        self.history: list[tuple[int, float]] = []

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if model is None or not self.traces:
            return
        from agentdistill.eval.clients import HfTurnClient

        was_training = model.training
        model.eval()
        try:
            client = HfTurnClient(model, self.tok, self.parser_name, self.family)
            results = teacher_forced(self.traces, client, max_turns_per_trace=self.turns_per_trace)
            results = results[: self.max_turns]
            summary = summarize(results)
        except Exception as e:
            # A broken metric must never kill a training run that is otherwise fine.
            logger.warning("next-action callback failed at step %s: %s: %s", state.global_step, type(e).__name__, e)
            return
        finally:
            if was_training:
                model.train()

        self.last_summary = summary
        self.history.append((int(state.global_step), float(summary["full_match"][0])))
        if getattr(state, "is_world_process_zero", True):
            logger.info("step=%s %s", state.global_step, format_summary(summary))
        logs = kwargs.get("logs")
        if isinstance(logs, dict):
            logs["next_action_full_match"] = summary["full_match"][0]
            logs["next_action_name_match"] = summary["name_match_on_tool_turns"]
            logs["next_action_args_match"] = summary["args_match_on_tool_turns"]
