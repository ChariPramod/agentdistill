# The confidence cascade

The student answers; a gate scores that answer; low-confidence turns go to the teacher. The saving comes from
the turns that never reach the teacher, and the whole thing rests on the gate being honest about what it knows.

## Features

Per turn, from the student's own generation:

| feature | what it is |
|---|---|
| `mean_logprob`, `min_logprob`, `p10_logprob` | over the whole generated turn |
| `arg_mean_logprob`, `arg_min_logprob` | over tokens inside tool-call arguments only |
| `first_tool_token_entropy` | entropy of the distribution at the first argument token |
| `n_tokens`, `n_tool_calls`, `has_tool_call` | shape of the turn |
| `agreement` | share of `k` extra samples that made the same call |
| `cluster_prior` | the router's posterior for this cluster |
| `turn_idx`, `prefix_tokens` | how deep into the trajectory this turn is |

**Missing is NaN, never zero.** A turn with no tool call has no argument logprobs, and zero is a confident
statement about a value that does not exist. `HistGradientBoostingClassifier` handles NaN natively; for models
that cannot, `as_vector` substitutes a sentinel at the last moment.

`agreement` is usually the strongest feature and the only expensive one: it costs `k` extra samples per turn.
With prefix caching those samples share nearly the whole prompt, so the marginal cost is small, but the
calibration report's ablation shows what it is worth on your project before you pay for it. At
`cascade.k_samples: 0` the column is dropped entirely and the calibration notes say so.

## Labels

A turn is labelled good or bad by `label_turns`, and *how* it was labelled matters as much as the label:

- `teacher_match` — the turn matches the teacher's recorded call. Strong.
- `teacher_mismatch` — it does not. Strong.
- `task_failed` — the task failed and the turn is assumed to have contributed. Weak.
- `uncorrected` — the task succeeded and the turn is assumed fine. Weak.

`calibrate` prints the mix, and warns when weak labels exceed half. A gate fitted mostly on assumptions has an
AUROC that describes those assumptions.

## Calibration

```bash
agentdistill eval run <adapter> --eval-set support-calib-v1 --n 3 --policy fuzzy --logprobs --samples 3
agentdistill calibrate <adapter> --from-eval <run-id>
```

Fitted on a **task-disjoint** split and reported on the half it never saw. In-sample calibration error is
optimistic by construction, and a gate whose ECE is quoted from its own training turns is not a measurement.
The artifact that ships is refit on everything — the holdout numbers exist for honesty, not to ship a weaker
model.

The gate refuses to fit rather than fit badly. Under 100 labelled turns, or a single outcome class, or a
holdout AUROC that does not clear the bar: `usable: false`, and the cascade escalates every turn. **A cascade
with a meaningless gate is worse than no cascade**, because it converts the teacher's reliability into the
student's at no saving.

## Threshold

`choose_threshold` picks the highest escalation rate that keeps the predicted success drop inside
`--max-drop-pp`. The predicted drop is an estimate on turns where the student answered alone.

**Verify it.** In a real cascade the teacher answers the low-confidence turns, so later turns build on a
prefix the student did not write, and the student then faces a different distribution:

```bash
agentdistill eval run cascade:<adapter>:auto --eval-set support-holdout-v1 --n 3 --policy fuzzy \
  --verify-threshold
```

The two rates diverging is expected, not a bug. It is also the difference between a saving that was estimated
and one that was measured, and the report quotes the measured one when it exists.

## Escalation and fallback are different things

- **Escalation** is the gate deciding. It is the cascade working, and it is counted as a cascade cost.
- **Fallback** is the student being unreachable. The teacher answers, the request is flagged `fallback`, and
  the router learns nothing from it — crediting either arm would be teaching the router from an outage.

`/healthz` fails when the fallback rate over the recent window exceeds 20% on at least five requests. A silent
fallback is how a broken vLLM becomes a quiet 100% teacher bill, and counting it is the whole point.

## What the cascade costs when it escalates

An escalated turn costs both models: the student generated tokens that were discarded, and the teacher
generated the ones that were used. `cascade_cost_per_task` takes `wasted_student_tokens` for exactly this
reason. A cascade costed without them looks cheaper than it is.
