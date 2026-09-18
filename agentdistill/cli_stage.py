"""A stage that produces nothing must not look like a stage that produced something uninteresting.

Every pipeline command that is supposed to write a registry row reports what it wrote through `stage_guard`.
Three outcomes, and they must look different at a glance in a log:

    [stage eval_teach] ok: run ev_1234 wrote 50 rows
    [stage eval_teach] SKIPPED: eval.skip_teacher set
    [stage eval_teach] wrote no row: teacher backend returned no completions

A skip is only ever declared by a config value, never inferred. Silent emptiness exits 3, which `gpu_day.sh`
treats as "do not write the .done marker", so a rerun retries the stage instead of skipping past a hole.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

#: The exit code for a stage that completed but wrote nothing. Distinct from 1 (usage or input error) so a
#: runner can tell "you called it wrong" from "it ran and produced a hole".
EXIT_EMPTY = 3


class StageEmpty(RuntimeError):
    """A stage completed but wrote no row. Exit code 3."""


@dataclass
class StageOutcome:
    wrote: bool
    detail: str
    skipped_reason: str | None = None


def stage_guard(name: str, fn: Callable[[], StageOutcome], allow_skip: bool = False) -> int:
    """Exit 0 on a row written, 0 with a SKIPPED log on a declared skip, raise `StageEmpty` on silent emptiness."""
    out = fn()
    if out.wrote:
        print(f"[stage {name}] ok: {out.detail}")
        return 0
    if out.skipped_reason and allow_skip:
        print(f"[stage {name}] SKIPPED: {out.skipped_reason}")
        return 0
    raise StageEmpty(f"[stage {name}] wrote no row: {out.detail}")
