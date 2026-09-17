"""The cost model.

Four functions, each small enough to check by hand, because every number in the report comes out of them and a
cost claim nobody can reproduce is worse than no cost claim.

The one that matters most is `breakeven_tasks_per_day`. A GPU is a fixed cost and the teacher is a variable one,
so distillation only pays above a volume, and that volume is the first thing an honest report should state.
"""

from __future__ import annotations


def student_cost_per_mtok(gpu_usd_per_hour: float, tokens_per_second: float, utilization: float = 0.6) -> float:
    """Dollars per million generated tokens on a GPU you are renting.

    `utilization` is the share of wall-clock the GPU is actually generating. Assuming 1.0 is how a cost claim
    ends up a factor of two out: a served model spends real time idle between requests.
    """
    if tokens_per_second <= 0 or utilization <= 0:
        return float("inf")
    return gpu_usd_per_hour / (tokens_per_second * utilization * 3600) * 1e6


def teacher_cost_per_task(
    prompt_tokens: float,
    completion_tokens: float,
    input_per_mtok: float,
    output_per_mtok: float,
    cache_hit_frac: float = 0.0,
    cache_read_per_mtok: float | None = None,
) -> float:
    """Dollars for one task from the teacher.

    Prompt caching is included because agent traffic is unusually cacheable -- a long system prompt and tool
    schemas repeat on every turn -- and ignoring it overstates the teacher's cost, which would flatter the
    student.
    """
    cached = prompt_tokens * cache_hit_frac
    uncached = prompt_tokens - cached
    read_rate = cache_read_per_mtok if cache_read_per_mtok is not None else input_per_mtok
    return (uncached * input_per_mtok + cached * read_rate + completion_tokens * output_per_mtok) / 1e6


def cascade_cost_per_task(
    student_tokens: float,
    student_per_mtok: float,
    escalation_rate: float,
    teacher_task_cost: float,
    wasted_student_tokens: float = 0.0,
) -> float:
    """Dollars for one task under the cascade.

    `wasted_student_tokens` is not optional in spirit: the student generates on every turn, including the ones
    the gate discards, and a cascade costed without them looks cheaper than it is.
    """
    student = (student_tokens + wasted_student_tokens) * student_per_mtok / 1e6
    return student + escalation_rate * teacher_task_cost


def breakeven_tasks_per_day(
    gpu_usd_per_hour: float, teacher_task_cost: float, cascade_variable_cost: float
) -> float:
    """Tasks per day at which the GPU pays for itself.

    Infinity when the cascade costs more per task than the teacher, which is the honest answer: no volume
    rescues a variable cost that is already higher.
    """
    saving = teacher_task_cost - cascade_variable_cost
    if saving <= 0:
        return float("inf")
    return gpu_usd_per_hour * 24 / saving


def saving_fraction(teacher_task_cost: float, cascade_cost: float) -> float | None:
    """Share of the teacher's per-task cost saved. None when the teacher cost is unknown or zero."""
    if not teacher_task_cost:
        return None
    return 1.0 - cascade_cost / teacher_task_cost
