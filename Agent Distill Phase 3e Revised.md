# agentdistill: Phase 3e Plan, Revised

Supersedes `Agent Distill Phase 3e Plan.md` in your repo (that file is mine; this replaces it, and its Appendix A still stands)
Version 0.3e-r1, September 21, 2026
Builds on: the phase 3d completion report (clean tiny rehearsal green from scratch, 1217 tests, lint clean, nothing committed)

---

## 0. What changed, and what this revision drops

The rehearsal is green from nothing, every stage writes a row, every report section fills. Seven latent bugs surfaced, and six of them would have hit on the GPU day with a rented meter running. That is exactly what the rehearsal was for.

Of these, the calibration one is the most useful finding, because it was invisible: `calibrate` exited 0, wrote a JSON file, and the report and gateway both read the table and correctly reported "no calibration". Two components agreeing that something is missing, while a third believes it wrote it, is the failure mode no test catches unless a test asserts the row exists. Now one does.

The promotion-check finding deserves a second look too. Below the power floor, one check crashed and another **let the candidate through**. The crash is a nuisance; the pass is the one that would have promoted an unmeasured adapter to prod. Worth a regression test named after the behavior, not the function.

**Dropped from the original 3e plan, now done:** warning codes and `assert_report` against `report.json`, deterministic eval-set selection, the migration schema snapshot, the health states, pair capping, provenance, revision plumbing.

**Downgraded:** the `cli_parts` split. Three agents worked in parallel on disjoint file sets this time and did not collide, which means the ownership protocol solved the problem the refactor was meant to solve structurally. Keep the split on the list, do it when `cli.py` next resists an edit, and do not spend a day on it now.

**Still open, in order:** commit the work, complete `project.yaml`, fix the retrain calibrate call, batched throughput, then the GPU day.

---

## 1. Commit today, before anything else

Yes, commit it, and do it before you read the rest of this.

1217 tests of uncommitted work from three agents on one laptop is the largest risk in the project right now, larger than anything on the GPU day. A branch is fine, but do not leave it uncommitted overnight to "tidy it first".

```
git checkout -b phase3d
git add -A
git status --short | head -50          # read this; an unexplained file is the point of the step
git commit -m "phase 3d: honesty invariants, tiny-mode coverage, clean-run determinism"
git push -u origin phase3d
```

Two things to resolve while reading `git status`, both carried from the last phase and neither yet closed:

- **`project.tiny.yaml` shows as modified and nobody claims it.** `git diff project.tiny.yaml`. Decide and either commit it with an explanation in the message or revert it. An unexplained diff in a config the rehearsal depends on is a reproducibility hole, and it will be baked into the branch otherwise.
- **The leftover `schema_version_002` table on a fresh database.** If the snapshot test now enshrines it, the test will resist the fix later. Either fix the migration and regenerate the snapshot now, or add `# known: see progress.md` next to it and a note. Do not leave it undecided.

Then run the clean rehearsal once more from a fresh clone, without `CLEAN_ALLOW_DIRTY`, and confirm the only difference is that the dirty-tree warning is gone. That is the run that proves the determinism claim, because it is the first one whose inputs are all committed.

---

## 2. Blockers before the GPU day

### 2.1 `project.yaml` has no teacher and no pinned base

This is now the real blocker, and it is bigger than the base-model pin I flagged last time. With no `teacher:` section:

- `eval run teacher` has no model to call, so there is no teacher row.
- With no teacher row, `cost_block` has no baseline, so no cost figure exists.
- With no cost figure, the cascade's saving cannot be computed, which is the number the whole project is for.
- `model_pricing` has no row to look up, so even a teacher row would not price.

Four things to add, all on the laptop, none needing a GPU:

```yaml
# examples/support_agent/project.yaml
teacher:
  model: <provider/model-id>
  provider: <provider>
  pricing_from_registry: true

train:
  base_model: <org>/<model-id>            # a Hub id, not ../../tests/fixtures/tokenizer
  base_model_revision: <40-char commit sha>   # a SHA, never a tag; tags move
  tool_parser:
    vllm: <parser name that base-check passed with>
    family: <fallback regex family>
```

Sequence, because each step can invalidate the next:

1. `agentdistill base-check <candidate>` for two or three candidates. Tokenizer download only. It checks template tool support, prefix stability, the mask invariants, and the round trip through the real parser. Record all candidates' results in `docs/progress.md`, not just the winner: the next person to change base models will want to know what was already rejected and why.
2. Pin the winner by SHA. Turn the revision skip into a hard failure for any non-local `base_model`.
3. Seed `model_pricing` for the teacher with an effective date.
4. Add the teacher key with a hard spend cap at the provider, not just an environment variable. The rollout and eval stages call it in a loop; a bug there is a bill.
5. Rebuild the real SFT dataset with the pinned tokenizer. The dataset in the registry now was tokenized by the fixture tokenizer and is not usable for the GPU day. Record the new hash.
6. Guard it: `train sft` compares the dataset manifest's tokenizer id and revision against the config and exits 3 with both values printed on a mismatch. Without this, step 5 being forgotten looks exactly like a normal run.

Also confirm the mask invariant tests pass against the real template. They were written against three fixture templates; the real one is a fourth, and it is the only one that matters.

### 2.2 Retrain's calibrate stage always exits 1

You found it; fix it in the same pass, because retrain is the one path where a stage failure is invisible until a week later. The stage needs the eval run it should calibrate from:

```python
# in the retrain stage list
Stage("calibrate",
      run=lambda ctx: {**ctx, "calibration_id": calibrate(adapter=ctx["candidate_adapter"], from_eval=ctx["calib_eval_run"])},
      gate=lambda ctx: gate_calibration(ctx["calibration_id"]))
```

which means the `eval` stage before it must produce and pass `calib_eval_run` (an eval run on the calibration set with `--logprobs`), not only the holdout run. Add a test that walks the whole stage list with stubs and asserts every stage's declared inputs are produced by an earlier stage. That test would have caught this without anyone running retrain, and it will catch the next one.

---

## 3. Your two decisions: both right, both need one tightening

### 3.1 Uninformative calibration saves a row instead of exiting 3

Right, and my original rule was wrong here. The rule is "a stage that produces nothing fails". A calibration that fitted, measured, and concluded "this gate carries no signal" produced something: a durable, correct, negative result. Exiting 3 would discard a finding.

Tightening: distinguish the cause, because the causes have different fixes.

```python
# agentdistill/cascade/calibrate.py  (verdict)
from __future__ import annotations


def verdict_for(n_turns: int, n_positive: int, auroc: float | None, ece: float | None,
                min_turns: int, min_auroc: float = 0.55, reliable_auroc: float = 0.6, max_ece: float = 0.05) -> tuple[str, str]:
    """Returns (verdict, reason). Only 'usable' is loaded by the gateway."""
    n_negative = n_turns - n_positive
    if n_turns < min_turns:
        return "unreliable", f"{n_turns} labelled turns below the minimum of {min_turns}"
    if n_positive == 0 or n_negative == 0:
        only = "good" if n_negative == 0 else "bad"
        return "degenerate_labels", f"every one of {n_turns} labelled turns is {only}; AUROC is undefined"
    if auroc is None:
        return "unreliable", "AUROC could not be computed"
    if auroc < min_auroc:
        return "uninformative", f"holdout AUROC {auroc:.3f} below {min_auroc}"
    if auroc < reliable_auroc or (ece is not None and ece > max_ece):
        return "unreliable", f"AUROC {auroc:.3f}, ECE {ece if ece is None else round(ece, 3)}"
    return "usable", f"AUROC {auroc:.3f}, ECE {round(ece, 3) if ece is not None else 'n/a'}"
```

`degenerate_labels` says "your tasks did not separate", which is a data problem. `uninformative` says "the features carry no signal", which is a feature problem. On a random tiny model every turn is bad, so tiny mode will report `degenerate_labels`, which is the honest label. AUROC must be recorded as `None`, never 0.5 or a silent NaN. Give it its own warning code (`GATE_DEGENERATE`) so `assert_report` can allow it in tiny mode and forbid it on the GPU day.

### 3.2 `eval_r1` skips when the round kept no candidate

Right, and it improves on my rule. I said skips must come from config; a decision recorded in the registry is stronger evidence than a config flag, because it is a fact about this run rather than an intention set beforehand.

Two tightenings:

- The skip line must cite the round row id and its reason: `[stage eval_r1] SKIPPED: round rnd_8f3a decided discard (success -2.1 pp, CI [-5.0, +0.8], cost not better)`. A skip that does not say which recorded decision justified it is indistinguishable from a silent pass, which is the thing this whole phase is about.
- On the GPU day, a discarded on-policy round is a **finding**, not a nuisance. It means one round of RFT plus DPO did not beat SFT on your data, which is a real and publishable result. It must appear in the report as a section, not only as a missing eval run. Add `onpolicy` to `ReportData` with the round's decision, reason, pair stats, and fuzzy share, and render it whether the decision was promote or discard.

---

## 4. Remaining engineering

### 4.1 Batched throughput (the one substantial item left)

Still the top item, unchanged from the original plan and now the only thing standing between you and a defensible cost number. Sequential throughput gives a cost per token roughly 5 to 20 times a served deployment's, which makes the cascade's saving look far worse than it is.

The working, tested implementation is in Appendix A of the 3e plan file already sitting in your repo. It is a lockstep runner: one generate call per turn index across all live tasks. It was checked against the sequential harness on 40 synthetic tasks and produced identical messages, turn counts, tool-call counts, divergence flags, and token estimates, in 6 generate calls instead of 40, with a divergence in one task not affecting its siblings.

It pays twice: the eval stages on the GPU day drop three to five fold, and `tokens_per_s` becomes a serving figure at a stated batch size.

Rules that go with it:

- `cost` carries `throughput_mode` in `{batched, unbatched}`.
- `assemble` emits `COST_UNBATCHED` when unbatched, and the markdown block refuses to print a saving percentage under that code; it prints the escalation rate and the measurement conditions instead.
- The equivalence test goes in CI: 40 synthetic tasks through both runners, per-task equality; one divergence isolated; `batch=4` over 15 items yields 15 outcomes at `max_inflight == 4`.
- `next_turns_batch` must preserve input order, since the runner zips replies against live tasks. `LLM.generate` does; assert it rather than trust it.

### 4.2 Small items

- **Feature-order assertion at gateway boot**: config `cascade.features` must equal the calibration row's `features`, in order, or the gateway refuses to load the calibration and says so on `/healthz`. You found the crash-on-dropped-feature path; this is the other half of it.
- **Promotion regression test**, named for the behavior: `test_insufficient_power_never_promotes`. Below the floor, every promotion check fails closed. That is the one that would have shipped an unmeasured adapter.
- **Stage-input test** from section 2.2, over every orchestrated stage list, not just retrain's.
- **Warning code audit**: list every code `assemble` can emit and confirm each appears in `CODES` and in either the allow or forbid list of `clean_rehearsal.sh`. A code in neither list passes silently.

---

## 5. Roles, revised

The protocol worked: disjoint file sets, no collisions, and you reviewed before merging. Keep it, with three changes.

| Role | Owns | This phase |
|---|---|---|
| **A. Rehearsal** | `scripts/`, `assert_report`, `project.tiny.yaml`, tiny eval sets, tiny tests | second clean run from a fresh clone post-commit; warning-code audit; stage timing table |
| **B. Eval and cost** | `agentdistill/eval/`, `agentdistill/report/` | batched runner and `throughput_mode`; `onpolicy` section in the report; the promotion regression test |
| **C. Serving** | `agentdistill/gateway/`, `router/`, `cascade/`, `serve_*.sh` | feature-order assertion at boot; the calibration verdict from section 3.1; `serve_smoke` |
| **D. Data and registry** | `registry/`, `migrations/`, `data/`, `curate/`, `provenance.py` | `schema_version_002`; the real dataset rebuild; the train-time tokenizer guard; pricing seed |
| **E. Training and config** | `train/`, `project.yaml`, `base-check` | base-model selection and SHA pin; teacher section; mask invariants on the real template |
| **Integrator (you)** | `cli.py`, `config.py`, `retrain.py`, CI, `docs/progress.md` | the commit; the retrain calibrate fix and stage-input test; one ruff pass per merge; GPU go/no-go |

Changes from last time:

1. **`project.yaml` gets an owner (E).** It had none, which is how it reached this point with no teacher section.
2. **`retrain.py` belongs to the integrator**, not to a feature role. It is the one file that reads every other subsystem's outputs, so it should be owned by whoever sees all of them.
3. **The `cli_parts` split is deferred.** Revisit if two roles need the same part of `cli.py` in one phase.

Unchanged and worth restating: one ruff pass per merge by the integrator, feature roles never run `--fix` outside their own files, cross-role edits carry `Owner-ack:` in the commit message, and a role that diverges from the plan appends to `docs/progress.md` in the same commit as the code. That last habit is why this phase's report was reviewable at all.

---

## 6. The GPU day

### 6.1 Go criteria

Every one must be true. Renting a GPU to debug configuration is the most expensive way to find a missing YAML key.

1. Everything committed and pushed; clean rehearsal green from a fresh clone with no dirty-tree warning.
2. `project.yaml` has a teacher section, a Hub base model pinned by SHA, a verified tool parser, and pricing seeded.
3. The real SFT dataset built with the pinned tokenizer, hash recorded, train-time guard passing.
4. `base-check` green and mask invariants green on the real template.
5. Batched throughput landed, or an explicit decision to publish no saving percentage this round.
6. `serve_smoke` green against the fake vLLM; retrain's stage-input test passing.
7. `requirements-gpu.txt` frozen; teacher key spend-capped; 60 GB free disk; `git tag pre-gpu-day` pushed.

### 6.2 During

The stage timing table from the rehearsal is your first instrument: a stage at three times its rehearsal-scaled estimate means stop and look, not wait.

| Watch | Abort or investigate if |
|---|---|
| `sft` eval loss and next-action accuracy | next-action flat against base at the end; stop and inspect the mask and template before spending more hours |
| `eval_sft` divergence rate | above 40 percent strict; note it, finish the run, fix per-tool canonicalization after |
| `onpolicy` pair stats | `max_per_task` above the cap, or fuzzy share above 0.5 |
| `calibrate` verdict | `degenerate_labels` on real data means the calibration set does not separate; that is a data finding worth writing down |
| `cmp_*` | any statistics printed below the power floor, which would mean the guard regressed |
| `/healthz` during `serve_smoke` | the gateway exits at boot for any reason |

### 6.3 Same day, after

Write `docs/results.md` from `report.json` while the numbers are in front of you: headline table, the cascade point with its verdict, the on-policy round's decision either way, per-cluster weak spots, and one paragraph on what the numbers do not show. Inject the README block. Commit the logs branch.

Then one decision, from the per-cluster table: if three or more clusters sit below the router floor, the next phase is data, not the router. Write the decision and the number it rests on into `docs/results.md`, so it does not get relitigated in a month.

---

## 7. Risks

| Risk | Mitigation |
|---|---|
| The laptop dies tonight | Section 1, first |
| Base template fails the round trip and the base changes late | Two or three candidates checked on day one, results recorded |
| Dataset rebuild forgotten after the pin | The train-time tokenizer guard makes it loud instead of silent |
| Teacher spend runs away in the rollout loop | Provider-side hard cap, not an env var |
| A discarded on-policy round reads as a broken pipeline | It is a report section with the recorded decision, not a missing stage |
| A new warning code ships in neither the allow nor the forbid list | The warning-code audit in section 4.2 |
| Batched runner changes outcomes | Equivalence test in CI |

---

## 8. Definition of done

- Everything committed and pushed; `project.tiny.yaml` diff and `schema_version_002` both resolved or explicitly documented.
- `project.yaml` complete: teacher, pricing, base model pinned by SHA, tool parser verified.
- Real dataset rebuilt with the pinned tokenizer and guarded at train time.
- Retrain's calibrate call fixed; the stage-input test passing over every stage list.
- Calibration verdicts distinguish `degenerate_labels` from `uninformative`; `GATE_DEGENERATE` in the code list and in the rehearsal gate.
- `eval_r1`'s skip cites the round row; the report has an `onpolicy` section either way.
- Batched throughput landed, or the no-saving-claim decision recorded.
- `test_insufficient_power_never_promotes` passing; feature-order asserted at gateway boot.
- GPU day run; `docs/results.md` written with the per-cluster decision and the number behind it.
