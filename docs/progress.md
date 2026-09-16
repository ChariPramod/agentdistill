# Progress and divergences

What is built, what is measured, and every place the code deliberately differs from the plan.

## Status

| Milestone | State |
|---|---|
| M1 — ingest, curate, dataset | **done** |
| M2 — SFT | **code complete, unmeasured.** No GPU run yet; no throughput or next-action number exists. |
| M2.5 — real traces | scaffolding built; recording needs a teacher endpoint |
| M3 — eval harness | in progress |
| M4–M8 | not started; those commands exit 2 naming their milestone |

**No cost or quality claim has been measured.** Adapters can only reach `candidate`. The sentence this phase
exists to produce — "on N real held-out tasks the student scores X% versus the teacher's Y%, delta with a 95% CI"
— is not yet true.

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

## Blocked on hardware or credentials

| Item | Blocker |
|---|---|
| Real GPU run (next-phase §2.6) | No GPU. Code path is CPU-smoke-tested on a tiny model; throughput, eval loss, and next-action numbers are unmeasured. |
| vLLM parser path, `VllmOfflineTurnClient` | vLLM does not install on macOS ARM. Import-guarded and skipped, not stubbed. |
| Recording real teacher traces (next-phase §3) | Needs a teacher endpoint. The agent, CRM, scenarios, and graders are built and tested against a scripted teacher; `record.py` needs only a model name and credentials. |
