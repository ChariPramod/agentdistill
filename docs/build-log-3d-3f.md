# Build log: phases 3d, 3e and 3f

What was done between commits `28ac6e4` and `41d7935`, why, and what it cost. Ten commits, 140 files, roughly
14,600 lines added. The test suite went from 1,114 to **1,365 passing** (5 skipped); `ruff` and `mypy` are clean.

This log exists because most of the work was *finding* things rather than building them. The plans for these
three phases asked for a set of features; what the features turned up was a pipeline that could report success
while producing nothing, and inputs no second machine could reproduce. Those findings are the substance here, so
each one says what was wrong, how it surfaced, and what it would have cost on a rented GPU.

Related documents: `docs/progress.md` (every divergence from the plans, in the repo's standing format),
`docs/gpu-day-go.md` (the go/no-go with evidence per criterion), `docs/ownership.md` (who owned which files while
four work packages ran in parallel).

---

## 1. Where it started, and the single idea

At `28ac6e4` the pipeline ran end to end on a laptop and produced a report. But the rehearsal exercised about two
thirds of the code: the teacher eval, the cost block, the calibration fit, the cascade verification and the
quantization delta never executed, **because the stages feeding them produced nothing and exited 0**.

One sentence drove all three phases:

> A stage that produces nothing must not look like a stage that produced something uninteresting.

Everything below is either an application of that rule or something the rule uncovered.

---

## 2. Phase 3d — honesty invariants and tiny-mode coverage

Commit `d689b57`, plus `24157c3` and `fe7d692`.

### Built

| Area | What changed |
|---|---|
| Statistics | `compare` returns an `insufficient_power` marker below 20 tasks × 3 repeats, or on a zero-width interval, instead of statistics. No renderer may print a p-value beside it. |
| Stage guard | `eval run`, `calibrate`, `train onpolicy`, `adapter quantize` and `report` exit **3** when they write no registry row. `gpu_day.sh` writes no `.done` marker on exit 3, so a rerun retries that stage. |
| Declared skips | A skip must cite a config key (`eval.skip_teacher`) and logs differently from emptiness: `SKIPPED: …` versus `wrote no row: …`. |
| Routing | Unassigned requests route on the pooled posterior *without* the floor, are logged with `routing_reason`, and are counted on `/healthz` beside the cluster-model and calibration states. |
| Pairs | DPO pairs are capped per task and sampled rather than enumerated. |
| Provenance | Every registry row records the command, commit, dirty flag, config path and config hash. |
| Tiny mode | A replay teacher so the cost block, calibration and cascade all execute on a laptop; `report.json` sidecar; `assert_report` gating the rehearsal on stable warning codes. |

### Found

1. **`calibrate` never wrote a calibration row.** It saved `calibration.json` and exited 0. The report and the
   gateway both read the `calibrations` table, so both correctly said "no calibration" while a third component
   believed it had written one. No test caught it because no test asserted the row existed.
2. **`--verify-threshold` never recorded its measurement,** and the runner never computed a run-level escalation
   rate, so verification always printed "this run recorded no escalation rate".
3. **Curation discarded its k-means centroids** and the gateway never loaded a cluster model, so every request
   reached the router unplaced and the router could never learn.
4. **Nothing recorded the teacher's prompt tokens or the student's throughput,** so the cost block could never be
   computed at all.
5. **Teacher-versus-student DPO pairs were mislabelled,** so the teacher cap never applied: 532 pairs from a tiny
   corpus, which is a cross product, not a dataset. After the fix, 15.
6. **The corpus could not be rebuilt.** The example corpus is generated and gitignored, and the command that
   produced it was never written down; 189 of its 800 traces no longer reproduced from any commit. Every
   rehearsal to that point had run on inputs no clone could recreate. `scripts/make_corpus.sh` is now the
   committed, deterministic recipe (800 tasks, seed 7, error rate 0.25).

### Decisions

Two skips are keyed on a *recorded decision* rather than a config flag, which the plan did not allow but which is
stronger evidence: `eval_r1` skips when the on-policy round row says it kept no candidate, and `calibrate` writes
a row with verdict `degenerate_labels` rather than exiting 3 when every labelled turn has the same outcome —
that is a finding about the data, not an empty stage.

---

## 3. Phase 3e — batched throughput, the GPU-day config, stage inputs

Commit `87e8653`.

### Built

- **Batched lockstep eval runner** sharing one per-turn stepper with the sequential path, with `throughput_mode`
  recorded on every run. When throughput came from sequential measurement, the report prints no saving
  percentage — only the escalation rate and the measurement conditions.
- **On-policy report section**, rendered whether the round promoted or discarded. A discarded round is a finding
  ("one round of RFT plus DPO did not beat SFT on this data"), not a missing stage.
- **Stage-input declarations** on the retrain loop, with two tests: a static walk checking each stage's declared
  inputs against what earlier stages provide, and a dynamic run recording what each stage actually reads.
- **Calibration verdicts**: `usable`, `no_threshold`, `unreliable`, `uninformative`, `degenerate_labels`. Only
  `usable` is loaded by the gateway. AUROC is stored as `None`, never NaN.
- **Train-time tokenizer guard**: `train sft` refuses a dataset tokenized for a different base model or revision,
  printing both values.
- **Feature-order assertion at gateway boot**, and a **warning-code audit** test: every code the report can emit
  must appear in the allow or forbid list of `clean_rehearsal.sh`, because a code in neither passes silently.

### Found

1. **The base model's template double-encoded every tool call.** Traces store tool-call arguments as a JSON
   *string* (the OpenAI wire format); real chat templates serialize what they are given, so Qwen rendered
   `"arguments": "{\"customer_id\": \"x\"}"` and the hermes parser recovered a string instead of a call. A student
   trained on that text emits tool calls the serving stack silently drops. `base-check` caught it on the first
   candidate. It had hidden because all four fixture templates interpolated arguments raw — the one shape no real
   template uses. The fixtures now apply `tojson`, like real templates.
2. **`calibrate` would have crashed on the first usable gate** whose fit dropped an empty feature column, because
   the threshold search scored with the configured feature list rather than the fitted one. `agreement` is empty
   whenever `k_samples` is 0.
3. **Retrain's calibrate stage had always exited 1**: it called `agentdistill calibrate <adapter>` with no
   `--from-eval`, and no earlier stage produced a calibration run. Flag checks passed because the flag is
   optional to the parser and required at runtime.
4. **CI's example job would have broken offline** once the config named a Hub base model.

### Configured for the GPU day

Teacher `claude-opus-5`; base `Qwen/Qwen2.5-7B-Instruct` pinned at `a09a35458c702b33eeacc393d103063234e8bc28`
(a SHA, never a tag); `hermes` parser verified by `base-check`; prices seeded through a new `pricing set`
command; the real dataset rebuilt with the pinned tokenizer.

---

## 4. Phase 3f — evaluation validity, and four packages in parallel

Commits `30cbaec`, `44a8b63`, `4d77276`.

Four work packages ran as parallel agents with disjoint file sets (`docs/ownership.md`). **Three of the four hit a
session limit mid-edit and never reported.** WP2 finished. What had landed was integrated and the gaps closed by
hand; `docs/progress.md` records which parts were not written by the package that owns them.

### Built

- **Live tools (WP1).** The corpus was recorded from a scripted solver, and replay grading counted any call the
  script did not make as a divergence — so the evaluation partly measured how closely a subject imitates the
  script. Evaluations and rollouts now run against the real, seeded CRM and are graded on the final state. Every
  run records `eval_mode`; `compare` refuses to pair runs from different modes; the report adds an
  evaluation-mode section measuring replay's distortion from a second teacher eval.
- **Teacher provenance (WP1).** The dataset manifest records `corpus_teacher`; the report prints it beside
  `serving_teacher` and raises `corpus_teacher_differs` when they differ — the student imitates the corpus
  teacher, so the comparison is operational rather than distillation.
- **Retirement (WP3).** Rows built before the render-boundary fix are marked retired; `dataset latest`,
  `adapter best|latest`, the prod and canary selectors and the report's selectors all skip them. The rows stay:
  the registry is the record of what was run.
- **Lockstep oracle (WP2).** The equivalence test had been comparing the shared stepper with itself and passing
  by construction. There is now an independent oracle that imports nothing from the package, plus order, timing,
  refill, repeat and isolation checks. A batched reply now carries the index of the prompt it answers.
- **Operations (WP4).** `gpu-day.lock.json`, `preflight.sh`, `bootstrap_box.sh`, a teacher spend estimate, and an
  export that also runs from an `EXIT` trap when a stage fails, with a laptop-side verifier.

### Found

1. **The lock pinned a dataset no fresh clone could build.** Written from the laptop's registry, whose eval sets
   had been frozen from the older corpus, it pinned 342 samples where a fresh clone builds 334. The GPU box would
   have trained on data that did not match the lock. The registry was rebuilt from the committed recipe and the
   lock rewritten; a fresh clone now reproduces every hash. The general lesson is in `progress.md`: a lock is only
   ever written from a registry rebuilt from nothing.
2. **Retiring by commit timestamp would have retired the good dataset.** The fix was applied at 07:04Z and
   committed at 07:12Z; `--built-before 87e8653` would have retired the dataset built with it, leaving
   `dataset latest` empty. Retirement used the time the fix entered the tree instead.
3. **The report paired a replay teacher run with a live student.** The GPU day evaluates the teacher twice on
   purpose, so "the latest teacher run" is the replay one. `compare` correctly refused and the report carried a
   forbidden code. `assemble` now selects every subject in the configured mode.
4. **The export did not verify.** The verifier opened `._registry.tiny.db` — a 163-byte macOS AppleDouble header
   that also ends in `.db` and which `tar` hides from its own listing. Linux boxes never produce these, so only a
   laptop rehearsal could find it.
5. **The corpus teacher was never found.** After an on-policy round the best adapter's own dataset is the DPO
   pairs, which record no corpus teacher, so the lineage printed "not recorded". It now walks the parent chain.
6. **mypy had gone red** across 3d–3f: clean at the start of 3d, 21 errors by 3f, none caught because no phase ran
   it. All were narrowing and annotation gaps, not behaviour.

### Owner decisions recorded

- **A:** the corpus is not re-recorded with Opus this round. The claim shipped is operational, and the report says
  the student imitates a scripted solver.
- **B:** retraining does not ingest teacher-written turns (`ingest.exclude_teacher_turns: true`) until the
  provider's terms on training from model outputs have been read.

---

## 5. Published

| What | Where | Notes |
|---|---|---|
| Source | [github.com/ChariPramod/agentdistill](https://github.com/ChariPramod/agentdistill) | Public; default branch `phase3d`; tag `pre-gpu-day`. History scanned for credentials before pushing. |
| Report site | [charipramod.github.io/agentdistill](https://charipramod.github.io/agentdistill/) | `scripts/build_site.sh --publish` copies the report the pipeline wrote to `gh-pages`. The page restates no number. |
| Gateway demo | [agentdistill.vercel.app](https://agentdistill.vercel.app) | The real FastAPI gateway as a serverless function. |

The demo runs the gateway, both API dialects, model-name resolution, the cascade's escalation rule, the request
log, feedback and `/healthz`. **The student and the teacher are canned replies**: a serverless function has no
GPU, and a public URL must not spend a teacher budget. The registry is created in `/tmp` per instance. With no
adapter, no calibration and no cluster model, `/healthz` reports each as missing and every cascade turn
escalates — the documented default, not a defect.

Two deployment problems, both fixed in `vercel.json`: Vercel found the root `pyproject.toml` and tried to install
the whole training stack (the function now installs only `api/requirements.txt` — seven packages, 84 MB against a
250 MB limit, with the package bundled from the repo); and an internal rewrite made every path 404, because
Vercel now routes rewrites by the *rewritten* path, handing the app the literal string `/api/index`.

---

## 6. Where it stands

`docs/gpu-day-go.md` reads **GO**: all nine criteria green, each with the command and output line that proves it.
`scripts/preflight.sh` is 13 PASS, 1 FAIL, 3 SKIP — the failure is the teacher API key, which by design exists
only in the shell on the rented box, and the skips are the hardware checks.

The teacher spend estimate is **$73.42** as an upper bound (every escalating stage costed as if every turn
escalated, a 1.3 wordiness factor, prompt caching not modelled). **Recommended workspace cap: $147.**

Still to happen, and only you can do it:

1. Create a Console workspace for this project, generate the key inside it, and set its spend limit to $147.
2. Rent the box (L40S or A100; Ubuntu 22.04/24.04; CUDA 12.x; 100 GB disk), export `ANTHROPIC_API_KEY` yourself in
   its shell, and run `bash scripts/bootstrap_box.sh`. It ends with the pre-flight; any FAIL means stop.
3. Run `bash scripts/gpu_day.sh`, watching the four checkpoints in `docs/gpu-day.md` §5.2. The one call only you
   make: if next-action accuracy is flat against base at the end of `sft`, stop — everything downstream would be
   measuring a student that did not learn.
4. Verify the export on the laptop **before** terminating the box, then write `docs/results.md`.

What no amount of laptop work can establish: whether the student is any good. Every number in the published
report comes from a randomly-initialised two-layer model and a replay stub, and says so.

---

## 7. Checking any of this yourself

```bash
bash scripts/clean_rehearsal.sh      # destroys every artifact, reruns all 19 stages, asserts the report
python -m pytest -q                  # 1365 passed, 5 skipped
ruff check agentdistill tests && mypy agentdistill
bash scripts/preflight.sh            # 13 PASS, 1 FAIL (the key), 3 SKIP (hardware)
agentdistill ops lock check --config examples/support_agent/project.yaml
agentdistill ops estimate-spend --config examples/support_agent/project.yaml
```

The rehearsal is the one that matters. It deletes the registry, the datasets, the adapters, the markers and the
generated eval sets, rebuilds the corpus from its recipe, runs every stage in sequence, and then asserts the
report it produced: the subjects present, the sections populated, and no warning outside the ones tiny mode is
expected to raise. It is also a nightly CI job, from a fresh checkout.
