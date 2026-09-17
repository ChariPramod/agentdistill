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


def observed_token_means(registry: object, limit: int = 5000) -> dict[str, float] | None:
    """Mean tokens per request for each arm, from the gateway's own log.

    Measured rather than assumed. A token profile guessed from the config is the kind of input that makes a cost
    model unfalsifiable; if the gateway has not served enough traffic to measure one, this returns `None` and the
    caller says so.

    Only requests that reached the teacher contribute to the teacher averages, and only requests that reached the
    student to the student average. Averaging over all rows would divide the teacher's tokens by the student's
    traffic and quietly make the teacher look cheap.
    """
    from sqlalchemy import text

    with registry.engine.connect() as conn:  # type: ignore[attr-defined]
        row = conn.execute(
            text(
                """SELECT
                       AVG(CASE WHEN student_tokens IS NOT NULL THEN student_tokens END)   AS student_out,
                       AVG(CASE WHEN teacher_tokens IS NOT NULL THEN teacher_tokens END)   AS teacher_out,
                       AVG(CASE WHEN teacher_tokens IS NOT NULL THEN prompt_tokens END)    AS teacher_in,
                       AVG(CASE WHEN teacher_tokens IS NOT NULL
                                THEN CAST(cached_prompt_tokens AS REAL) / prompt_tokens END) AS cache_frac,
                       COUNT(*)                                                            AS n,
                       SUM(CASE WHEN teacher_tokens IS NOT NULL THEN 1 ELSE 0 END)         AS n_teacher
                   FROM (SELECT student_tokens, teacher_tokens, prompt_tokens, cached_prompt_tokens
                         FROM requests WHERE fallback = 0 OR fallback IS NULL
                         ORDER BY received_at DESC LIMIT :lim) q"""
            ),
            {"lim": limit},
        ).mappings().first()
    if not row or int(row["n"] or 0) < 50 or int(row["n_teacher"] or 0) < 10:
        return None
    if row["teacher_in"] is None:
        # Rows predating the prompt-token columns. Pricing the teacher on completions alone would understate it.
        return None
    return {
        "student_tokens": float(row["student_out"] or 0.0),
        "teacher_prompt_tokens": float(row["teacher_in"]),
        "teacher_completion_tokens": float(row["teacher_out"] or 0.0),
        "cache_hit_frac": float(row["cache_frac"] or 0.0),
        "n": float(row["n"]),
    }


def arm_costs(cfg: object, registry: object | None = None) -> dict[str, float]:
    """Dollars per request for each router arm.

    The router trades success against cost at `lambda_per_usd`, so these must be real dollars or the trade is
    meaningless. When either side cannot be priced -- no teacher prices on file, or too little traffic to measure
    a token profile -- both arms are returned at zero, which makes the router maximize success alone. That is the
    conservative failure: it may route to the teacher more often than a cost-aware router would, and it will
    never route to the student for a saving that was never verified.
    """
    zero = {"student": 0.0, "teacher": 0.0}
    teacher = getattr(cfg, "teacher", None)
    serve = getattr(cfg, "serve", None)
    if teacher is None or serve is None or registry is None:
        return zero
    if teacher.input_per_mtok is None or teacher.output_per_mtok is None:
        return zero

    means = observed_token_means(registry)
    if means is None:
        return zero

    student_rate = student_cost_per_mtok(
        serve.gpu_usd_per_hour, tokens_per_second=_tokens_per_second(serve)
    )
    teacher_cost = teacher_cost_per_task(
        prompt_tokens=means["teacher_prompt_tokens"],
        completion_tokens=means["teacher_completion_tokens"],
        input_per_mtok=teacher.input_per_mtok,
        output_per_mtok=teacher.output_per_mtok,
        cache_hit_frac=means["cache_hit_frac"],
        cache_read_per_mtok=teacher.cache_read_per_mtok,
    )
    return {
        "student": means["student_tokens"] * student_rate / 1e6,
        "teacher": teacher_cost,
    }


def _tokens_per_second(serve: object) -> float:
    """Generation throughput, from config when declared and a conservative default otherwise."""
    return float(getattr(serve, "tokens_per_second", 0.0) or 800.0)
