# Progress and divergences

What is built, what is measured, and every place the code deliberately differs from the plan.

## Status

| Milestone | State |
|---|---|
| M1 — ingest, curate, dataset | **done** |
| M2 — SFT | **code complete, unmeasured.** No GPU run; no throughput or next-action number exists. |
| M2.5 — example agent | **built and tested.** Corpus recorded from a scripted teacher, not a teacher. |
| M3 — eval harness | **done.** Control test passes; `eval run` / `compare` / `show` work end to end. Judge grading is calibrated. |
| M4 — on-policy | **done.** The full round runs end to end on CPU: rollouts, RFT dataset, SFT continuation, verified merge, DPO, eval, decide. |
| M5 — cascade | **code complete, unmeasured.** Gate, calibration, threshold search, and verification all run; no real logprobs have been fitted on. |
| M6 — gateway, router, serving | **done.** Both dialects verified against the real SDKs; router, canary split, and live comparison tested. |
| M7 — retrain loop | **done.** Eight stages with gates; the workflow is committed as a draft until `serve_smoke.sh` passes on real hardware. |
| M8 — cost model and report | **done.** HTML and markdown, idempotent injection, no number without a run id. |

The only remaining `_not_built` command is `ingest otel`. Everything else runs.

**The sentence this phase exists to produce is still not true.** It requires a student, and a student requires a
GPU run this environment cannot do. What exists is every piece around it: the harness reproduces recordings
exactly, the statistics are simulation-tested, and a paired comparison between two subjects prints a correct
report with an interval. Point it at a trained adapter and the number appears.

Adapters can only reach `candidate`. No cost or quality claim has been measured. `README.md` and
`docs/results.md` carry the report's injection markers with nothing between them, and a test keeps them empty.

### What the CPU rehearsal covers, and what it does not

`AGENTDISTILL_TINY=1 bash scripts/gpu_day.sh` runs every stage on a laptop against a randomly-initialized
2-layer model. It has found eight real defects so far, listed in `docs/gpu-day.md`. It cannot cover vLLM
itself — no real tool parser, no LoRA loading, no template handling — or anything CUDA. Those are what
`scripts/serve_smoke.sh` on the real box is for.

**Tiny-mode numbers measure nothing.** The model is random. The report carries a tiny-mode warning saying so.

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

### The canary split hashes the request id (phase 3c §3.2)

The plan specified `int(request_id[-2:], 16) % 100 < share * 100`. Two hex digits give 256 values and 256 does
not divide by 100, so buckets 0–55 collect three source values each and 56–99 collect two: a 10% share takes
11.7% of traffic, outside the plan's own 9–11% acceptance band. It is also sensitive to the shape of the id —
request ids are often sequential or timestamped, and their last byte then correlates with arrival time.

**Resolution:** hash first. `test_canary_split.py` pins the old behaviour so the reason stays on record.

### A decay that would disable the router's floor is refused (phase 3c §3.1)

Each update multiplies `alpha + beta` by `decay` and adds one, so evidence converges to `1/(1-decay)` however
much traffic flows. At `decay: 0.9` the ceiling is 8 observations, below the default floor threshold of 10 — the
floor could never engage, and the router would keep sending traffic to a student it had already watched fail.
Refused at config load, where the fix is cheap, rather than at gateway boot, where the only safe response is to
serve without a router.

### Two retrain gates are stricter than the plan (phase 3c §3.4)

The eval gate reads the CI low end rather than the point estimate: a candidate 0.5 pp below prod with an
eight-point interval has not been shown at parity, it has been measured badly. Calibration requires AUROC as
well as ECE, because ECE alone passes a calibrator that outputs the base rate for every turn — perfectly
calibrated, and useless.

### The merge tolerance reports its own granularity (phase 3c §4.1)

The plan asked for 50 held-out turns against a 2 pp tolerance. One disagreeing turn out of 50 is 2 pp, so at
that size the gate passes only an exact reproduction. That is a defensible bar but not what "within 2 points"
sounds like, so the verification result says which one applies and how many turns the tolerance would need.

### Tiny mode uses ten-task eval sets, not five (phase 3c §5)

Originally because `eval compare` refused below eight tasks. Since phase 3d the floor is 20 tasks at 3 repeats and
a comparison below it returns an insufficient-power marker rather than refusing, so a tiny rehearsal's
comparison stage runs and reports why it cannot compare. The ten-task sets stay.

### `eval run teacher` was evaluating the base model

Not a plan divergence but the most consequential bug this phase found. `_resolve_client` excluded `teacher`
from the adapter lookup and then fell through to the local-model branch, loading `train.base_model` and
labelling the run "teacher". Every student-against-teacher comparison would have been a student-against-base
comparison, and it would have been believed. `teacher` now resolves to a LiteLLM client on `teacher.model` and
refuses when none is configured.

### Phase 3d: what the plan assumed existed, and did not

The plan's code sketches assumed several pieces were already wired. Three were not, and each was the same bug it
set out to fix -- a stage that produced nothing and passed:

- **`calibrate` never wrote a calibration row.** It saved `calibration.json` to disk and exited 0, so the report
  and the gateway, which read the `calibrations` table, always saw "no calibration". It now writes the row with a
  verdict, and exits 3 when there is nothing to write.
- **`--verify-threshold` never recorded its measurement,** and the runner never computed a run-level escalation
  rate from the per-row counts, so verification always printed "not a cascade". Both are fixed; the measured
  point lands on the calibration row the report's cost block reads.
- **Curation discarded its k-means centroids,** and the gateway never loaded a cluster model, so every request
  reached the router unplaced. Centroids are now a `cluster_models` row; the gateway loads them or reports why not.

Two more gaps sat under the cost block: nothing recorded the teacher's prompt tokens, and nothing measured
student throughput. Both are now run metrics. Throughput is measured by the sequential harness and labelled
unbatched, so the cost it implies is an upper bound and the report prints the conditions beside it.

### The power floor is 20 tasks at 3 repeats, and below it there are no statistics (phase 3d §2.1)

As the plan says. Two consequences worth knowing: a comparison where every task came out identically under both
subjects is also refused (a zero-width interval is not a precise zero), and every promotion gate treats the
refusal as a failure. Tiny mode runs 10 tasks at N=1, so its comparisons always report the reason.

### The pair builder's cap now binds (phase 3d §2.4)

The 532 pairs were two bugs, not one: teacher pairs had no per-task cap, and `pairs_against_teacher` never set
`pair_kind`, so `balance_kinds` filed them as rollout pairs and the teacher ratio never applied either.

### The replay teacher is stateless (phase 3d §3.1)

The plan's stub binds a cursor per task. In a cascade the teacher is wrapped and called mid-conversation on a
prefix the student built, where nothing binds it, so the stub finds the task from the conversation and replays
the recorded turn at the current turn index instead.

### The gate verdict has four values (phase 3d §3.2)

`uninformative` (holdout AUROC below 0.55), `unreliable` (too few turns, AUROC below 0.6, or ECE above 0.05),
`no_threshold` (a good gate with no threshold inside the budget), and `usable`. Only `usable` is loaded by the
gateway. The threshold search runs whatever the verdict, so its code path executes on every calibration.

### Two skips that are recorded decisions rather than config (phase 3d §2.2)

The plan says a skip is only ever a config value. Two exceptions, both keyed on a decision already written to a
registry row rather than inferred from absence:

- `eval_r1` skips when the on-policy round kept no candidate. The round row records the discard and its reason;
  there is simply no adapter to evaluate.
- `calibrate` writes a row with verdict `uninformative`, rather than exiting 3, when every labelled turn has the
  same outcome. That is a measurement -- the gate has nothing to discriminate -- and it is what a random tiny model
  produces. Too few turns, no logprobs, no features, or a fitter crash still exit 3.

### The rehearsal found three more on its first clean run

- `cmp_sft` selected its left side with `eval latest --tag`, which returns the newest tagged run. Once tiny mode had
  a teacher row, that was the teacher, and the stage compared teacher against base under the name `cmp_sft`.
- The report picked "best adapter" without the script's tag and could pick the quantized artifact, which on a
  noisy eval outscored its parent -- so every calibration and quantization row hung off a different adapter.
  Quantized artifacts are no longer candidates, and the report stage passes the tag.
- With pairs capped, a tiny round has 15 pairs and discarded before DPO, so the rehearsal never trained DPO. Tiny
  mode now sets `onpolicy.min_pairs: 10`.

### The boot-time feature check allows an ordered subset (phase 3e §4.2)

The plan says `cascade.features` must *equal* the calibration's features, in order. The gateway instead accepts
any stored order that is a subsequence of the configured one and scores with the stored order, because
calibration legitimately drops features that had no values; requiring equality would refuse every such gate.
What it refuses is what scores wrongly: a fitted feature the config no longer produces, a different relative
order, or a registry row whose `feature_order` disagrees with its `calibration.json`. Before this the gateway ran
the subset check but then scored with the full configured list, so a narrowed calibration crashed or, worse,
scored misaligned columns. The rule lives in `cascade.client.scoring_order` and `from_calibration` uses it too.

### The tokenizer guard checks the model id on legacy manifests too (phase 3e §2.1 step 6)

Manifests now record `base_model_revision` (the pin actually applied at build time, `null` when unpinned). A
manifest without that key predates the recording, and is a failure only when the config pins a revision, a
warning otherwise, as planned. The model id, though, has been in every manifest as `tokenizer` since the first
schema, so it is compared even on a legacy manifest: that is precisely the fixture-tokenized registry dataset
meeting a real base model. The revision enters the content hash only when set, so unpinned datasets keep their
hashes, and a pinned rebuild of otherwise identical samples is a new dataset rather than a reuse of the old one
with its old manifest -- which would fail the guard forever. `test_sft_smoke`'s two-step test now trains on a
dataset recorded for the tiny model it trains, rather than for the fixture tokenizer it happens to share.

### The batched runner: what it batches, and how order is checked (phase 3e §4.1)

The lockstep runner (`eval/lockstep.py`) and the sequential `run_task` both drive one `TaskStepper`, so the
per-turn rules exist once. Four choices the plan left open or stated differently:

- `VllmOfflineTurnClient.next_turns_batch` calls `LLM.generate` on prompts rendered with the project tokenizer,
  not `LLM.chat`. `chat` takes one tools list for the whole batch and renders with vLLM's copy of the template;
  either would make a batched turn differ from a sequential one. `next_turn` is now a batch of one, so both
  share rendering, sampling and parsing. Order is asserted against each output's `prompt`, not trusted.
- `run_eval(..., batch_size=N)` uses the lockstep runner only for a client that has `next_turns_batch` and no
  per-task state (`reset`, `summary`, `usage`): interleaving tasks would scramble a cascade's gate summary and a
  teacher's per-task usage. Everything else runs sequentially and is recorded `throughput_mode: unbatched`. The
  runner itself still drives a non-batching client one `next_turn` per item (never labelled batched), for tests.
- A generic batched client cannot prove its order to the runner; the runner checks count and shape, and the
  order check lives in the vLLM client, the one place that can see which prompt an output belongs to.
- Under `cost_unbatched` the cost dict also carries `saving_frac` and `breakeven_tasks_per_day` as `null` (not
  only hidden by the renderers), with `cost_is_upper_bound: true`; `student_cost_per_mtok` stays, as an upper
  bound. Every run now records `throughput_mode`; a run without it reads as unbatched. `tests/test_report.py`'s
  seed marks its throughput batched, since it stands for a measured serving figure.

Tiny mode's student eval runs on the transformers client, which does not batch, so a tiny report carries
`cost_unbatched`: the rehearsal's allow list needs it, and the GPU day should forbid it.

### Base-model selection, and the bug base-check found (phase 3e §2.1)

`Qwen/Qwen2.5-7B-Instruct`, pinned at `a09a35458c702b33eeacc393d103063234e8bc28`, tool parser `hermes`. It is the
only candidate checked so far; the others in the plan's list are unevaluated, and a later change of base should
start by recording its `base-check` output here.

| Candidate | has_template | accepts_tools | prefix_stable | tool_call_roundtrip |
|---|---|---|---|---|
| Qwen/Qwen2.5-7B-Instruct | PASS | PASS | PASS | PASS (hermes; after the fix below) |

The first run failed the round trip, and the cause was ours rather than the model's. Traces store tool-call
arguments the way the OpenAI wire format does, as a JSON **string**. Chat templates serialize whatever they are
given, so Qwen's template emitted `"arguments": "{\"customer_id\": \"x\"}"` -- a quoted, escaped string -- and
the hermes parser recovered a string rather than the call's arguments. A student trained on that text would emit
tool calls the serving stack drops, silently. `render` now converts arguments to objects at the single point where
a chat template is applied, which is what the HF convention expects.

It went unnoticed because all four fixture templates interpolated `{{ c.function.arguments }}` raw, which only
works when the value is already a serialized string. Real templates apply `tojson`. The fixtures now do too, so
they model the templates that exist rather than the one shape that hid this.

### The example's real config needs the network; tiny mode still does not (phase 3e §2.1)

`examples/support_agent/project.yaml` now names a Hub base model and a real teacher, so building the real dataset
downloads a tokenizer. The rehearsal path is unchanged: `project.tiny.yaml` keeps the locally generated tiny model
and the replay teacher, and the whole clean rehearsal still runs offline.

### A batched reply now names the prompt it answers (phase 3f §4, WP2)

The phase 3e entry above says "a generic batched client cannot prove its order to the runner; the runner checks
count and shape, and the order check lives in the vLLM client". That is no longer true, and it was the weaker
arrangement: the runner zips replies against live items, so an out-of-order batch hands one task another task's
turn and every trajectory after that point is fiction — a failure with no symptom, in the one place the numbers
come from.

Every reply from `next_turns_batch` now carries `lockstep.REPLY_INDEX` (`"_index"`), the index of the prompt it
answers. `check_replies` refuses a reply whose index is not its position, naming both, and refuses an unindexed
batch outright rather than guessing. The key is underscore-prefixed, so `TaskStepper.step` strips it with the
other transport keys and it never reaches a trajectory. The vLLM client keeps its own `_in_order` check — it
compares each output against the prompt it was generated from, evidence that exists nowhere else — and fills the
index in from request order afterwards. The two checks are independent on purpose.

No recorded number moves: the only production batched client is `VllmOfflineTurnClient`, which already asserted
its order. What changes is that a client which *cannot* prove its order now fails loudly instead of quietly.

Three smaller things from the same review:

- **The oracle's divergence payload follows the harness, not the handbook.** `tests/oracle/reference_harness.py`
  is Appendix A verbatim (imports aside) except for `DIVERGED`, which the appendix gives as
  `{"error": "replay divergence"}`. The harness writes `{"error": "replay divergence: this call was not
  recorded"}`, and the production string is the contract. `BAD_JSON` already matched.
- **`run_lockstep` takes an injectable `clock`.** Default `time.perf_counter`; nothing in production passes it.
  It exists so a test can make replay lookups expensive and show `generate_seconds` does not move — the figure
  is the denominator of the batched throughput number, and "time spent generating" has to mean that and only
  that. The timing was already correct; it was untestable.
- **The lockstep test corpus lives in `tests/lockstep_corpus.py`.** `docs/ownership.md` gives WP2
  `tests/test_lockstep*.py` and `tests/oracle/`, which does not cover a shared, non-collected helper. The
  synthetic tasks and the stateless student are shared by the oracle tests and the `run_eval` tests, and
  duplicating them would let the two drift. The lead should widen WP2's glob to include it.

### Decision A: the corpus is not re-recorded with Opus this round (owner, phase 3f §1.2)

The training corpus stays the scripted solver's. This round ships the operational claim -- a student with an Opus
fallback, compared with all-Opus, at this cost and this success rate -- and the report says plainly that the
student imitates a scripted solver: the dataset manifest records `corpus_teacher`, the lineage prints it beside
the serving teacher, and `corpus_teacher_differs` is a standing, allowed disclosure. Re-recording with Opus is real
distillation but costs teacher spend, a new corpus and a terms review; it goes to the top of the next phase.

### Decision B: retraining does not ingest teacher-written turns (owner, phase 3f §1.2)

`ingest.exclude_teacher_turns` stays `true` until Anthropic's commercial terms on using model outputs to train
other models have been read and a decision recorded here. Excluding them is safe whichever way that decision goes,
and reversible with one config key. The rule is the simple one: a gateway request is excluded entirely if any of
its assistant turns came from the teacher arm, and the count excluded is logged and recorded on the ingest row.

### Three work packages were cut off mid-edit; the lead finished them (phase 3f)

WP1, WP3 and WP4 hit a session limit partway through and never reported; WP2 finished. What had landed was
integrated and the gaps closed by the lead, so some of this phase's code was not written by the package that owns
it. What was missing, and is now in: migration 008 was never registered; three retirement tests used a cutoff
equal to the rows' own timestamps (`--built-before` is exclusive, as its name says -- the tests were wrong, not the
code); `corpus_teacher` was never written to the manifest; the CLI registrations; `bootstrap_box.sh`; the export
stage, exit trap and timing table in `gpu_day.sh`; live tools in tiny mode; and the tests for live mode, the
pre-flight, the lock and the export. WP1 recorded `tools_mode` twice -- on the round result and inside the pair
statistics, which broke a test that pins those statistics -- so it now has one home, and rides in the round row's
stats JSON only at persistence time.

### The report pairs subjects in the configured tool mode (phase 3f WP1)

The GPU day evaluates the teacher twice on purpose, live and replay, to measure replay's distortion. "The latest
teacher run" is then the replay one, and the first live rehearsal paired it with a live student: `compare`
correctly refused, and the report carried `eval_mode_mismatch`, which the gate forbids. `assemble` now selects
every subject in `eval.tools`; the other mode's runs feed only the evaluation-mode section.

### Retirement uses the time the fix entered the tree, not the commit that recorded it (phase 3f WP3)

The render-boundary fix was applied, the real dataset rebuilt with it at 07:04Z, and the fix committed as 87e8653
at 07:12Z. `--built-before 87e8653` would have retired that valid dataset and left `dataset latest` empty for the
GPU day. The local registry was retired with `--built-before 2026-09-20T07:00:00Z`, which retires only the
fixture-tokenizer dataset `ds_99a6c87b9a49b745`.

### The export's registry was a macOS resource fork (phase 3f WP4)

The first export failed verification: the verifier opened `._registry.tiny.db`, a 163-byte AppleDouble header
macOS `tar` adds beside every file and hides from its own listing, because it too ends in `.db`. The export now sets
`COPYFILE_DISABLE=1` and the verifier skips `._` members. Linux boxes never produce these, which is why only the
laptop rehearsal could find it.

### mypy had gone red during phases 3d-3f

CI runs `mypy agentdistill`; it passed at the start of phase 3d and had 21 errors by phase 3f, none caught because
no phase ran it. All were narrowing and annotation gaps, not behaviour; it is clean again.

### The lock caught the laptop's registry drifting from the committed recipe (phase 3f, go criterion 3)

The first lock was written from the laptop's registry and pinned dataset `dca202f381da` (342 samples). A fresh
clone following the same recipe built `7a4bd4ffe877` (334 samples) with different eval-set hashes, while the
corpus hash matched. The cause was the laptop: its `support-*-v1` eval sets had been registered and frozen on
Sep 16 from the old, unreproducible corpus, and when the corpus was regenerated from `make_corpus.sh` the freeze
correctly refused to replace them -- so curation decontaminated against stale eval sets and kept eight traces a
fresh build rejects. The GPU box would have trained on a dataset that did not match the lock.

The fresh clone is the reference. The laptop's registry and dataset directory were moved aside (to
`registry.db.stale-20260921` and `artifacts/datasets.stale-20260921`, not deleted), rebuilt from the committed
recipe, and the lock rewritten; it now pins `7a4bd4ffe877451c` and a fresh clone reproduces it. The general
lesson: a lock written from a long-lived registry records that registry's history, so the lock is only ever
written from a registry rebuilt from nothing.

## Blocked on hardware or credentials

| Item | Blocker |
|---|---|
| Real GPU run (next-phase §2.6) | No GPU. Code path is CPU-smoke-tested on a tiny model; throughput, eval loss, and next-action numbers are unmeasured. |
| vLLM parser path, `VllmOfflineTurnClient` | vLLM does not install on macOS ARM. Import-guarded and skipped, not stubbed. |
| Recording real teacher traces (next-phase §3) | Needs a teacher endpoint. The agent, CRM, scenarios, and graders are built and tested; `record.py --model <id>` needs only credentials. |
| The M3 definition of done (`base` vs adapter vs teacher on one GPU) | Needs a trained adapter. The harness, graders, statistics, and report are done and tested; the missing input is a student. |
| A fitted confidence gate | Needs real per-turn logprobs from a model that can do the task. `eval run --logprobs --samples` and `calibrate` run end to end on CPU, but a random model produces a single outcome class and the gate correctly refuses to fit. |
| `adapter quantize --method awq` | Needs `llmcompressor` and a GPU. `fp8` writes its marker and runs anywhere. |
| Renaming `.github/workflows/retrain.yml.draft` | Gated on `serve_smoke.sh` passing on the self-hosted runner. Scheduling an unverified pipeline to train and promote models weekly is the wrong order. |

## Where the example corpus falls short of the plan

The plan asks for 40 scenarios × 10 instances recorded from a real teacher. What exists is **40 scenario shapes ×
15 instances = 600 tasks**, recorded from a rule-based teacher.

- **The shape count and volume now meet the plan.** 40 shapes, 600 tasks, 408 training traces, 296 kept after
  curation (the plan's bar was 250), and two frozen eval sets. Solver success is 84%, inside the plan's 60–90%
  band.
- **A scripted teacher, not a teacher.** `scripted_teacher.py` reacts to real tool results from a real stateful
  database, so trajectories have the right shape — but it is a generator, and the plan is right that a student
  would learn the generator. These traces are for exercising the pipeline. **They must not be trained on or used
  to publish a number.**
- **A handful of scenarios still sit at 100%.** Error injection now covers every intent rather than refunds
  alone, but the "refuse and explain" shapes have no wrong action to inject. The recorder flags them, honestly,
  as unable to separate a student from the teacher.
