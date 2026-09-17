# Results

Numbers appear here only via `agentdistill report --format md --inject docs/results.md`.

The tables below have headers and no rows. That is deliberate: this project has not run on a GPU yet, and a
README with plausible numbers in it is worse than one with none. Everything the pipeline can produce on a
laptop is a smoke test, and the report labels it as such.

<!-- agentdistill:results:begin -->
<!-- agentdistill:results:end -->

## What each column will mean

| Subject | Success | Schema valid | Divergence | Tokens/task | Run |
|---|---:|---:|---:|---:|---|

- **Subject** — `base` is mandatory. Without it you cannot tell training from the base model already being
  good at the task.
- **Success** — the grader's verdict. For the example project that is a state predicate over the CRM, not a
  model's opinion.
- **Schema valid** — share of tool calls that validate against the tool's schema. A student that succeeds while
  emitting invalid calls is succeeding by luck.
- **Divergence** — share of trajectories that reached a tool call the recording cannot answer. Reported
  separately from failure, because a diverged trajectory did not necessarily do anything wrong.
- **Tokens/task** — median, not mean. A few long trajectories otherwise dominate.
- **Run** — the eval run id, so every number can be traced to the command that produced it.

## The cost block

| | Teacher | Cascade |
|---|---:|---:|
| Cost per task | | |
| Saving | | |
| Break-even tasks/day | | |

Break-even is the number that matters first. A GPU is a fixed cost and the teacher is a variable one, so
distillation only pays above a volume, and an honest report states that volume before it states a percentage.

Where the report can compute both, it shows an analytic estimate faintly and the verified measurement boldly.
Where it can only estimate, it says so, and where it can do neither it says which claim it could not make. A
report with warnings is still a report; a report with unsupported numbers is not.

## Paired comparison

| Metric | Delta | 95% CI | p (Holm) |
|---|---:|---|---:|

Deltas are paired by task and bootstrapped over task clusters, because repeats of one task are correlated and a
row-level bootstrap gives an interval several times too narrow. Significance is Holm-corrected across the
metrics compared, so three tests do not become three chances to find an effect.

An interval that includes zero means this run did not show a difference. It does not mean the two are
equivalent, and the verdict line says which one it is.
