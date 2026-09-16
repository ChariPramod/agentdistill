# Progress and divergences

What is built, what is measured, and every place the code deliberately differs from the plan.

## Status

| Milestone | State |
|---|---|
| M1 — ingest, curate, dataset | **done** |
| M2 — SFT | **code complete, unmeasured.** No GPU run; no throughput or next-action number exists. |
| M2.5 — example agent | **built and tested.** Corpus recorded from a scripted solver, not a teacher. |
| M3 — eval harness | **done.** Control test passes; `eval run` / `compare` / `show` work end to end. |
| M4 — on-policy | bridge built (`eval/rollouts.py`); the round loop is not |
| M5–M8 | not started; those commands exit 2 naming their milestone |

**The sentence this phase exists to produce is still not true.** It requires a student, and a student requires a
GPU run this environment cannot do. What exists is every piece around it: the harness reproduces recordings
exactly, the statistics are simulation-tested, and a paired comparison between two subjects prints a correct
report with an interval. Point it at a trained adapter and the number appears.

Adapters can only reach `candidate`. No cost or quality claim has been measured.

## Divergences from the implementation plan

### near_dedupe verifies LSH candidates (plan §3.2)

`MinHashLSH.query` returns candidates from banding, not verified matches — in practice pairs at Jaccard 0.74
against a 0.85 threshold. Every candidate is now verified with the MinHash estimate before a drop.

**Why:** dropping candidates unverified discards traces the configured rule says to keep, silently shrinking the
dataset in a way no report would reveal.

### near_dedupe_normalize_literals defaults off (plan §3.2)

The plan's default (0.85 raw Jaccard on 5-gram shingles) cannot satisfy the plan's own test that "a trace and its
copy with one changed number are near-duplicates" — on a short trace, one changed token costs ~0.3 Jaccard.
Masking ids before shingling fixes that case and breaks a worse one: on a corpus whose agent answers in
templates, every trace of a shape becomes identical once ids are masked, collapsing 479 of 526 example traces.

**Resolution:** raw shingling stays the default. It catches real near-duplicates on realistic traces (one changed
number leaves Jaccard ~0.92); the earlier failure was an artifact of unrealistically short fixtures. Normalization
is a documented opt-in.

### Loss masking uses start-inside, not containment (plan §4.3)

A token is a target if its *start* offset falls inside an assistant span.

**Why:** a BPE token can straddle a turn boundary, and the token most likely to straddle is the end-of-turn
marker. Requiring full containment dropped it, producing a student that never learns to stop. Guarded by
`tests/test_mask_invariants.py` across three templates.

### `warmup_ratio` is converted, not dropped (next-phase §2.3)

transformers 5.x removed `warmup_ratio` in favour of the absolute `warmup_steps`. The compat shim converts the
configured ratio using the estimated total step count rather than dropping the field.

**Why:** dropping it silently removes warmup from every run. This was caught by the shim before any GPU time was
spent, which is what the shim is for.

### Round-trip parsing falls back to regex when vLLM is absent (next-phase §2.1)

vLLM is Linux/CUDA only. `base-check` uses vLLM's real tool parser when importable and a per-family regex
otherwise, reporting which one ran. `parse_with_vllm` returns `None` for "could not check", never `[]`.

**Why:** conflating "no parser available" with "no tool call found" would turn a missing dependency into a
passing check. The fallback is explicitly labelled as weaker assurance.

### The pii filter and agents whose tools take personal data (found in M2.5)

The `pii` filter dropped 100% of the example corpus. It was behaving exactly as specified: the rule is "drop any
trace where redaction changed a tool argument", the agent's core tool is `get_customer(email)`, so every trace
qualifies.

The rule is right — a student that learns `<EMAIL>` is a valid argument will send that placeholder to a real API.
But it means **an agent whose tools legitimately take personal data cannot use masking-based redaction at all.**
It needs pseudonymization that preserves referential integrity: a stable fake email per real one, so the
trajectory still makes sense and no real address survives. That is not built; the example disables the filter and
says why inline.

### Eval traces were being curated as training data (found in M2.5)

`curate` reads every trace in the registry, and eval traces live in the same table. Decontamination caught them —
they match themselves exactly — but only after counting them as training candidates, so the report read as a
contaminated corpus rather than the eval set being seen twice. They are now excluded before the filters run.

### Nondeterministic state in the example CRM (found in M3)

Tracking numbers were built from Python's built-in `hash()`, which is randomized per process. The replay grader
rebuilds state in a different process from the recorder, so every tracking-related predicate failed on
reconstruction. Fixed with a sha256 digest; the equivalence test now agrees 260/260.

This is the bug `test_replay_predicate_matches_live` exists to catch, and it would have silently corrupted every
eval number involving a tracking number.

## Blocked on hardware or credentials

| Item | Blocker |
|---|---|
| Real GPU run (next-phase §2.6) | No GPU. Code path is CPU-smoke-tested on a tiny model; throughput, eval loss, and next-action numbers are unmeasured. |
| vLLM parser path, `VllmOfflineTurnClient` | vLLM does not install on macOS ARM. Import-guarded and skipped, not stubbed. |
| Recording real teacher traces (next-phase §3) | Needs a teacher endpoint. The agent, CRM, scenarios, and graders are built and tested; `record.py --model <id>` needs only credentials. |
| The M3 definition of done (`base` vs adapter vs teacher on one GPU) | Needs a trained adapter. The harness, graders, statistics, and report are done and tested; the missing input is a student. |

## Where the example corpus falls short of the plan

The next-phase plan asks for 40 scenarios × 10 instances = 400 tasks recorded from a real teacher. What exists is
13 scenario shapes × 20 instances = 260 tasks recorded from a rule-based solver.

- **13 shapes, not 40.** Each shape carries a real wrinkle (ineligible orders, wrong ids, two orders where only
  one qualifies, requests whose right answer is to refuse). More shapes would improve cluster diversity and the
  generalization measurement; these were chosen over 40 shallow ones.
- **A scripted solver, not a teacher.** `scripted_teacher.py` reacts to real tool results from a real stateful
  database, so trajectories have the right shape — but it is a generator, and the plan is right that a student
  would learn the generator. These traces are for exercising the pipeline. **They must not be trained on or used
  to publish a number.**
- **Error injection is uneven.** It perturbs refund actions only, so read-only scenarios sit at 100% and the
  recorder correctly flags them as unable to separate a student from the teacher.
