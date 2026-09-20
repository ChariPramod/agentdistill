# Evaluation

The eval harness is the ground truth. Nothing is promoted on a loss curve; every claim is a paired comparison
against a baseline on held-out tasks, with an interval.

## What a run does

```bash
agentdistill eval run <subject> --eval-set support-holdout-v1 --n 5 --policy strict
```

For each task in the eval set, `n` times: start from the recorded system and user messages, let the subject
produce turns, serve tool results from the recorded trace, stop when it answers, diverges, or hits `max_turns`.
Grade the result. Store every repeat.

Subjects:

| subject | meaning |
|---|---|
| `recorded` | replays the recorded turns. The control — it must reproduce the recording exactly |
| `base` | the base model with no adapter. **Every report must include this**; without it you cannot tell training from the base model already being good |
| an adapter name | base + that LoRA |
| `http:<model>@<url>` | any OpenAI-compatible endpoint — a teacher API, a vLLM server, the gateway later |

### Batched runs and throughput

`run_eval(..., batch_size=N)` runs an eval in lockstep. With a batch size, every (task, repeat) runs in lockstep: up to that many are in flight, and each step asks the
model for the next turn of all of them in one batched call. Finished tasks leave, queued ones join. Each turn
goes through the same per-turn rules as the sequential harness, and a test holds the two to identical
trajectories, so batching changes the throughput and nothing else.

Throughput is recorded with its mode. `throughput_mode: batched` means completion tokens over the seconds spent
inside batched generate calls, at the stated batch size. Anything else, including a client that cannot batch
(transformers, a cascade, a teacher API) is `unbatched`: one request at a time, a floor on serving throughput.
The report prices the cascade against the teacher only from a batched figure; from an unbatched one it raises
`cost_unbatched` and prints the escalation rate and the measurement conditions instead of a saving.

## Replay, and why divergence is its own metric

The student never touches a live service. Tool results come from the recording, keyed by canonical argument hash
([`canonical-json.md`](canonical-json.md)). Eval is therefore deterministic, free, and safe — nothing gets
refunded twice because an eval ran twice.

The cost: the recording only covers the calls the teacher made. A call that was never recorded cannot be
answered, so the trajectory stops. That is a **divergence**, and it is reported separately from failure, because:

- **Low success, low divergence** — the student follows the teacher's path and gets the task wrong. A quality
  problem.
- **Low success, high divergence** — the student goes somewhere else entirely. It may even be right; you cannot
  tell from this eval, because the environment could not follow it.

Those need different fixes, and a single success number hides which one you have.

### Reading a divergence

```bash
agentdistill eval show <run> --divergences
```

Every divergence carries the nearest recorded call and a similarity score.

- **High score** (a near-match differing in one key or an optional filter) — an argument-phrasing gap. Add a
  per-tool rule in `canonical.py` and re-run.
- **Low score** — the student genuinely went elsewhere. That is a real behavioural difference. Leave it.

Loosening the hash rules to make a divergence rate look better is how a cost claim becomes fiction.

### strict and fuzzy

`strict` serves only exact canonical matches. `fuzzy` serves the nearest recorded call above a threshold.

Use `strict` for every reported number. Use `fuzzy` for on-policy rollouts, where the student's phrasing drifts
and strict mode would stop most trajectories at the first turn. A fuzzily served result is **not** the result the
student's call would really have produced, so the fuzzy-hit share is reported on every run and any number built
on it must carry that caveat.

## Grading

| grader | what it does |
|---|---|
| `predicate` / `replay_predicate` | a pure function of the reconstructed end state plus the final message. No model in the loop. The strongest option |
| `label` | exact final-text match against the recording. Weak, and the report says so |
| `llm_judge` | a judge model against a rubric. Never reported as a bare number — see below |

The replay predicate rebuilds a fresh environment from the task's seed and applies the student's own tool calls to
it. Because results were served from the recording, a student that made the same calls reaches the same state; a
student that made different calls is graded on what its own calls produced, which is the right thing to judge.

`tests/test_eval_runner.py::test_replay_predicate_matches_live` asserts the reconstruction reproduces the label
recorded live, across the whole example corpus. It has already caught one real bug: tracking numbers built from
Python's per-process-randomized `hash()`, which made every rebuilt state differ from the recording.

### Judge grading

**This is a secondary path.** The example project grades with predicates — pure functions of the final state,
with no model in the loop — and `eval run` never silently falls back to a judge when a predicate is configured.
Judge grading exists for projects that have no state to check. If you have state, check it.

A judge is a measuring instrument with its own error rate, and that rate is rarely symmetric: most judges call a
mediocre trajectory a success far more readily than they call a good one a failure. A raw judge score therefore
carries a bias that propagates into everything downstream.

So a judge number never appears alone. Calibrate against trusted labels on the same tasks:

```bash
agentdistill eval run student --eval-set holdout          # graded by the judge
agentdistill eval run student --eval-set holdout-labelled # graded by predicate or human labels
agentdistill eval calibrate-judge <judge-run> <truth-run>
```

That prints, and stores, the judge's agreement, false-positive and false-negative rates, its bias, and a
Rogan-Gladen corrected estimate:

```
judge success 75.0%  (corrected 63.5%; agreement 88.5% on n=200, false-positive 31.5%,
                      false-negative 0.0%, bias +11.5%)
```

Two refusals are deliberate. Below 30 labelled items no correction is applied, because it would be noise
presented as precision. And if the judge is near-random (sensitivity + specificity ≤ 1) the correction is
undefined and says so rather than producing a number. With no calibration at all, the line reads `UNCALIBRATED`
and states that the score should not be compared against anything.

**The holdout check is the one that means something.** Estimating the judge's error rates on a set and then
correcting that same set's rate is a tautology: Rogan-Gladen inverts exactly, so the error is zero by
construction however bad the judge is. `calibrate-judge` therefore also fits on one split and corrects a disjoint
one, and reports that error with a bootstrap interval next to what the uncorrected rate would have been:

```
holdout check: correcting a disjoint split lands +6.9% from truth [95% CI -0.4%, +14.1%] on n=200
               (uncorrected would be +8.0%)
```

Read those two numbers together. If the corrected error is not clearly smaller than the uncorrected one, the
correction is not buying anything. Below 100 labelled items the interval is wide enough that it usually is not,
and the output says so.

A judge that returns unparseable output is recorded as `judge_error`, not as a task failure — failing closed
would bias the score downward and hide that the instrument broke.

## Comparing

```bash
agentdistill eval compare <run_a> <run_b>
```

Positive deltas favour `run_a`. The report gives success with a 95% interval and McNemar, tokens and turns with
Wilcoxon, all Holm-corrected, plus schema validity, divergence rate, and the weakest clusters.

Four choices worth knowing about:

**Below the floor there are no statistics.** Fewer than 20 shared tasks or fewer than 3 repeats per task, and
`compare` returns an `insufficient_power` marker with the reason and the raw observed rates, not a delta, an
interval or a p-value. So does a comparison where every task came out the same under both subjects: its
interval has zero width, which is not a precise measurement of no difference. Every renderer prints the reason
instead of a result, and every promotion gate (lifecycle, retrain, on-policy) treats the marker as a failure,
because parity that was never measured is not parity. `eval compare` still exits 0: a comparison that cannot
be made is not an error.

**Intervals resample tasks, not rows.** 40 tasks run 5 times is 200 rows and nowhere near 200 independent
observations — repeats of one task are highly correlated. Resampling rows would report an interval several times
too narrow.

**McNemar is the exact binomial on discordant tasks.** Only tasks where the two subjects disagree carry
information. This has a floor: with `d` discordant tasks the smallest achievable p is `2 / 2**d`, so with four
disagreements you cannot get below 0.125 no matter how large the effect. If a comparison looks underpowered,
that is usually why, and the answer is more tasks, not more repeats.

**Holm correction across the family.** Reporting success, tokens, and turns and announcing whichever cleared 0.05
is how noise becomes a finding.

An inconclusive result says so explicitly, and distinguishes "we did not detect a difference" from "they are
equivalent" — with a small eval set those are very different claims.

## Reading failures

```bash
agentdistill eval show <run> --failures-only
```

Per-task outcomes with the first failure detail. Reading failures by hand is the only way to learn why a number
moved; a per-cluster table tells you where to look, not what happened.

## What this does not measure

Teacher-forced next-action accuracy (`agentdistill.eval.teacher_forced`) scores the student's choice given the
*teacher's* prefix. It is cheap, runs during training, and is the first real quality signal. It is **not** a
substitute for end-to-end success: at inference the prefix is the student's own, mistakes included, and that gap
is exposure bias. Expect next-action accuracy to look better than end-to-end success. Report both; closing the
gap is what the on-policy round in Milestone 4 is for.
