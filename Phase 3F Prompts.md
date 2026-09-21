# agentdistill Phase 3f: Your Tasks and Agent Prompts

Evaluation validity, GPU-day readiness, and the GPU day itself
September 20, 2026

How to use this file: section 1 is yours alone, because no agent can do it. Section 2 is a rules block you paste at the top of every agent prompt. Sections 3 to 5 are paste-ready prompts: the lead, then four work packages that run in parallel. Section 6 is for the agent on the rented box. Section 7 is the write-up afterwards. Appendix A is a tested oracle harness that WP2 drops into the test tree.

Run order:

1. Tonight: section 1.1 (push the repo, create the key).
2. Start the lead (section 3). It hands WP1 to WP4 to separate agents.
3. WP1 to WP4 run in parallel; their file sets do not overlap. The lead merges them in the order WP1, WP3, WP2, WP4.
4. When the lead reports every go criterion green, set the spend cap from its estimate, rent the box, and run section 6.
5. Section 7 the same day.

### One correction to my last message

I said to keep replay for training rollouts. That was wrong for this example. With a local, deterministic, seeded CRM, rollouts should run against live tools too: the student then sees the true result of every call it makes, instead of the nearest recorded result, and the fuzzy-share gate stops mattering. Replay stays in the product as the fallback for users who have no sandbox. WP1 below implements live tools for both evaluation and rollouts.

---

## 1. Your tasks

### 1.1 Tonight (about 20 minutes)

**Push the repo somewhere.** Nothing is pushed, so a lost or broken laptop still takes everything with it. Pick one:

- Private GitHub repo: create it empty (no README, no license), then
  `git remote add origin <url> && git push -u origin --all && git push --tags`
- No remote wanted: `git bundle create ~/agentdistill-$(date +%Y%m%d).bundle --all`, then copy the bundle to cloud storage.

Tell the lead which one you chose; the pre-flight script checks it.

**Create the teacher key, and cap it before the GPU day.** In the Claude Console, create a workspace used only by this project and generate the API key inside it, so any limit on that workspace applies to nothing else. Set the workspace's spend limit once WP4 reports its spend estimate: about twice the estimate. If your account shows no limit at the workspace level, set the organization's monthly limit instead. The key never goes into the repo, a config file, or an agent prompt. On the box you export it yourself as `ANTHROPIC_API_KEY`.

### 1.2 Two decisions (write them into `docs/progress.md` yourself, one paragraph each)

**Decision A: re-record the corpus with Opus, or not this round.** My recommendation: not this round. Ship the operational claim ("student with Opus fallback, compared with all-Opus, at this cost and this success rate"), and state plainly in the report that the student imitates a scripted solver. Re-recording with Opus makes it real distillation, but it costs teacher spend, a new corpus, and a terms review. Put it at the top of the next phase.

**Decision B: whether retraining may ingest teacher-written turns.** The retrain loop ingests gateway traffic, and the turns that escalated to Opus were written by Opus. Before allowing that, read Anthropic's commercial terms on using model outputs to train other models. Until you have, WP1 makes `ingest.exclude_teacher_turns: true` the default, which is safe whichever way you decide.

### 1.3 Before and on the GPU day

**Rent the box.** One L40S (48 GB) or A100 (40 or 80 GB), Ubuntu 22.04 or 24.04, CUDA 12.x, at least 100 GB disk. An L4 or A10G (24 GB) works with QLoRA but takes about twice as long. Book eight hours for four to five hours of compute.

**Run the pre-flight on the box** (`bash scripts/preflight.sh`, built by WP4). If any line says FAIL, stop and hand the output to the lead. Do not debug on the rented meter.

**The one call only you make during the run:** if next-action accuracy at the end of `sft` is flat against the base model, stop. Everything downstream would be measuring a student that did not learn.

**Before terminating the box,** confirm the export tarball is on your laptop and `scripts/verify_export.sh` passes. The registry is gitignored and the box is disposable; if the results stay on a machine you are about to delete, the day produced nothing.

---

## 2. Shared rules (paste at the top of every agent prompt)

```text
PROJECT RULES (apply to everything below)

Repo: agentdistill, branch phase3d. Read docs/progress.md and docs/ownership.md before touching anything.

Ownership
- Edit only the files your work package lists as yours. If you need a change elsewhere, stop and write the
  exact change, the file, and why, as a request to the lead. Do not make it yourself.
- cli.py belongs to the lead. If you add a command, implement its body in a module you own and give the lead
  the few lines of Typer registration to add at merge time.
- Never run ruff --fix on files you do not own. Run ruff check and mypy on your own files only.
- An edit the lead approves in someone else's file carries "Owner-ack: <role>" in the commit message.

Honesty invariants (non-negotiable; they are the reason this project exists)
- A stage that writes nothing exits 3. A skip must cite a config key or a registry row id and its reason.
- No statistics below 20 tasks x 3 repeats, and none on degenerate data. No renderer may print a p-value next
  to an insufficient-power marker.
- Every number that reaches a report or the README carries a run id. Missing data becomes a warning code, never
  a silent omission and never a zero.
- A new warning code goes into agentdistill/report/warnings.py CODES and into either the allow list or the
  forbid list in scripts/clean_rehearsal.sh. A code in neither list is a bug.

Working practice
- When fixing a bug, write the failing test first. Name the test after the behavior, not the function.
- If this prompt is wrong or conflicts with the code, do not quietly work around it. Implement the correct
  thing and append a divergence entry to docs/progress.md, in the existing format, in the same commit.
- Never put secrets in code, configs, commits, logs, or fixtures. ANTHROPIC_API_KEY comes from the environment
  only. Tests that need a teacher use the replay teacher or a stub.
- When you finish, report: files changed, tests added, the exact test command and its result, divergences
  recorded, and anything you could not do with the reason. Do not claim anything you did not run.
```

---

## 3. Lead prompt (integrator)

```text
[paste section 2 here]

ROLE: Lead and integrator for phase 3f.

You own: agentdistill/cli.py, agentdistill/config.py, agentdistill/retrain.py, CI config, .gitignore,
docs/ownership.md, the structure of docs/progress.md, docs/gpu-day-go.md (new), and the merge order.

GOAL
Make the GPU day safe to run and its results valid. Phase 3f ends when every go criterion below is green on a
fresh clone and docs/gpu-day-go.md records the evidence for each.

WORK PACKAGES (one agent each; their file sets do not overlap)
- WP1 Evaluation validity: live tools for evaluation and rollouts in the example project, evaluation mode on
  every run, teacher provenance in the report, and a guard against ingesting teacher-written turns.
- WP2 Lockstep review: an independent oracle for the equivalence tests, plus order, timing, refill, repeat,
  and isolation checks.
- WP3 Render boundary audit: every consumer of tool-call arguments goes through the tojson boundary; rows built
  before the fix are retired and can never be selected.
- WP4 Operations: lock file, pre-flight, box bootstrap, teacher spend estimate, results export with a trap on
  exit, stage timing table.

MERGE ORDER: WP1, WP3, WP2, WP4. WP1 changes numbers and WP3 changes rendering, so later packages must be
tested against the tree that has both. After each merge: add the Typer registrations the package requested,
one ruff pass over the whole repo, the full suite, and resolve any import collisions yourself.

YOUR OWN TASKS
1. Add the four file sets to docs/ownership.md exactly as their prompts list them.
2. If CI exists, add a nightly job running scripts/clean_rehearsal.sh from a fresh checkout in tiny mode. If
   there is no CI, record that in progress.md as a known gap. Do not invent a CI system.
3. After all four merges, clone into a new directory and run scripts/clean_rehearsal.sh with no dirty-tree
   allowance. It must pass with allowed warning codes only.
4. Run the full suite and record the count.
5. Write docs/gpu-day-go.md: each go criterion, green or red, with its evidence (the command and the line of
   output that proves it).
6. Commit, push (or bundle, per the owner's choice), and tag pre-gpu-day.

GO CRITERIA (all must be green)
1. HEAD committed and pushed or bundled; clean rehearsal green from a fresh clone.
2. project.yaml has a teacher section, a Hub base model pinned by a 40-character SHA, a tool parser verified
   by base-check, and pricing seeded by the documented command.
3. The real SFT dataset rebuilt with the pinned tokenizer; the train-time tokenizer guard passes; the lock file
   (WP4) records its hash and a fresh clone reproduces it.
4. The example project's evaluations and rollouts run with live tools (WP1); every eval run records its mode;
   compare refuses two runs with different modes.
5. The render-boundary test passes on the pinned Qwen template (WP3), and no retired row can be selected.
6. Lockstep equivalence tested against the independent oracle (WP2).
7. scripts/preflight.sh passes locally for every check that does not need a GPU (WP4).
8. gpu_day.sh ends with the export stage, and the export also runs when a stage fails (WP4).
9. The teacher spend estimate is printed and recorded in docs/gpu-day-go.md, so the owner can set the cap.

STOP CONDITIONS
- If two packages need the same file, stop both and decide its owner before either continues.
- If a merge breaks the clean rehearsal, revert that merge, send the failure to its owner, and continue with
  the others.

REPORT
Go or no-go, the table from docs/gpu-day-go.md, the test count, the rehearsal result, the spend estimate and
the recommended cap, divergences recorded, and anything left for the owner.
```

---

## 4. Work package prompts

### WP1: Evaluation validity

```text
[paste section 2 here]

ROLE: WP1, evaluation validity.

YOU OWN: agentdistill/eval/live.py (new), agentdistill/eval/runner.py, agentdistill/eval/harness.py,
agentdistill/train/onpolicy.py (rollout tool mode only), agentdistill/report/assemble.py,
agentdistill/report/warnings.py, agentdistill/ingest/gateway_source.py, examples/support_agent/project.yaml,
and the tests you add. Anything in cli.py goes to the lead as a request.

WHY THIS WORK EXISTS
The training corpus was recorded from a scripted solver. The evaluation replays tool results from those
scripted traces. Opus makes valid calls with different arguments than the script (a lookup by email instead of
by id, an extra verification call), and strict replay counts each one as a divergence and fails the task. The
student, trained on the script's exact calls, diverges less. So the current comparison partly measures how well
a subject imitates the script, not whether it solves the task. The example CRM is local, deterministic, and
seeded per task, so every subject can be run against the real CRM and graded on the real final state.

TASKS
1. Live tools. Create agentdistill/eval/live.py with LiveToolProvider, exposing the same lookup(tool, args)
   interface as ReplayToolProvider. It builds a fresh CRM from the task's db_seed, executes each call against it,
   returns the JSON result, never raises Divergence, and keeps the CRM for the grader. Tool errors come back as
   {"error": "..."} content, exactly as the recording agent saw them.
2. Live grading. In live mode the predicate grader reads the provider's final CRM state and the final assistant
   text. It does not reconstruct state from the calls.
3. Mode config. eval.tools: live | replay and onpolicy.tools: live | replay, both overridable on the command
   line (give the lead the flag definitions). Record eval_mode on every eval run's metrics and tools_mode on
   every on-policy round row. Set both to live in examples/support_agent/project.yaml. Ask WP4 to do the same
   in project.tiny.yaml after you merge.
4. Rollouts. collect_rollouts uses LiveToolProvider when onpolicy.tools is live. The fuzzy-share gate then reads
   zero; keep the gate for replay mode and make the round row record which mode applied.
5. Mode mismatch. compare() given runs with different eval modes returns an incompatible marker naming both
   modes, and never statistics. Add warning code EVAL_MODE_MISMATCH to the forbid list.
6. Replay distortion. The teacher is evaluated on the holdout in both modes on the GPU day. Give WP4 the exact
   gpu_day.sh stage line for a second teacher eval with --tools replay. assemble() adds an "evaluation mode"
   section: for each subject with runs in both modes, live success, replay success, and replay divergence rate,
   labelled as the measured distortion of replay grading.
7. Teacher provenance. The dataset manifest records corpus_teacher (from the traces' teacher_model field, or
   the solver's name). The report's lineage prints corpus_teacher and serving_teacher (config teacher.model)
   side by side. When they differ, emit CORPUS_TEACHER_DIFFERS, on the allow list, whose message says the
   student imitates the corpus teacher and the comparison is operational, not distillation.
8. Ingest guard. ingest gateway excludes turns written by the teacher, controlled by
   ingest.exclude_teacher_turns, default true. Exclude a request entirely if any of its assistant turns came
   from the teacher arm (the simple rule; record that choice in progress.md). Log how many were excluded, and
   record the count on the ingest's registry row.

TESTS
- test_live_replays_scripted_calls_to_the_same_outcome: the solver's recorded calls, executed live on three
  fixture tasks, reach the predicate outcome the recording reached.
- test_live_mode_does_not_penalize_valid_alternative_calls: a subject that looks a customer up by email where
  the script used the id fails under strict replay and passes live.
- compare() on one live and one replay run returns the incompatible marker, and no renderer prints statistics.
- Rollouts in live mode record fuzzy_share 0 and tools_mode live.
- Ingest excludes teacher-written requests by default and includes them when the flag is false.
- CORPUS_TEACHER_DIFFERS appears when manifest and config disagree, and lineage prints both fields.

DONE WHEN
All tests pass, and your report includes the gpu_day.sh stage line for WP4 and the flag definitions for the
lead. The lead runs the tiny rehearsal after merging.
```

### WP2: Lockstep review

```text
[paste section 2 here]

ROLE: WP2, lockstep runner review.

YOU OWN: agentdistill/eval/lockstep.py, tests/test_lockstep*.py, tests/oracle/ (new).

WHY THIS WORK EXISTS
lockstep.py was rebuilt from prose after the reference implementation was lost, and it shares one per-turn
stepper with the sequential path. So the equivalence test now compares the stepper with itself and passes by
construction. It no longer tests anything.

TASKS
1. Oracle. Add tests/oracle/reference_harness.py from Appendix A of the phase 3f handbook, unchanged except for
   imports. It depends on nothing in agentdistill. Keep its header comment. If its error payloads differ from
   the production harness's, the production strings are the contract: change the oracle's two constants to
   match and record that in your report.
2. Equivalence against the oracle. For 40 synthetic tasks, run the oracle and the batched lockstep runner with
   a deterministic batch client, and assert per-task equality of messages, turn count, tool-call count,
   divergence flag, and token estimate. Include tasks that diverge, tasks with malformed arguments, and tasks
   that hit max_turns. Add two hand-written expected transcripts for two fixture tasks and assert both runners
   reproduce them exactly.
3. Order. Change the batch client contract so each reply carries the index of the prompt it answers (the vLLM
   client fills it from request order or request_id). The runner asserts reply i answers prompt i and raises,
   naming both, if not. Test with a client that shuffles its replies.
4. Timing. generate_s covers only the generate calls. Test with an injected clock where replay lookups advance
   the clock: generate_s must not change.
5. Refill. batch=4 over 15 items gives 15 outcomes and max_inflight == 4, and after the first completion the
   next step's prompt count is back to 4 (admission happens in the same step).
6. Repeats. Two repeats of the same trace in one batch each get their own provider. Test with a provider that
   counts lookups: the counts are per item.
7. Isolation. One task diverging mid-batch leaves every other task's outcome equal to the oracle's.

DONE WHEN
All seven hold. For each, your report says whether the existing code already passed or what you changed, and
whether any recorded eval number could have been affected by the change.
```

### WP3: Render boundary audit

```text
[paste section 2 here]

ROLE: WP3, render boundary audit.

YOU OWN: agentdistill/data/, agentdistill/train/dpo_data.py, agentdistill/cascade/arg_mask.py,
agentdistill/eval/clients.py (render calls only), agentdistill/registry/retire.py (new), and the tests you add.

WHY THIS WORK EXISTS
Traces store tool-call arguments as a JSON string, the OpenAI wire format. Real chat templates apply tojson, so
passing the string rendered "arguments": "{\"customer_id\": ...}" and the parser recovered a string instead of
a call. A student trained on that emits tool calls the serving stack silently drops. The fix went in at one
render boundary for training. Other consumers render or search argument strings and must go through the same
boundary. Anything built before the fix is invalid.

TASKS
1. Find every place that renders messages through a chat template or searches generated text for arguments.
   List each in your report with file and line. Expect at least: dataset building, DPO pair rendering,
   teacher-forced eval prompts, the vLLM and HF turn clients, and arg_char_spans in cascade/arg_mask.py.
2. Route every template render through the one boundary function. No call site passes raw wire-format argument
   strings to apply_chat_template.
3. arg_char_spans: generated text now contains arguments as JSON objects. Match the serialized object in the
   form the pinned template emits (derive the form from a real render; do not hardcode spacing), keep the
   existing fallbacks, and test on a Qwen-rendered tool call.
4. test_rendered_prompts_never_contain_stringified_arguments: render three real traces with the pinned Qwen
   tokenizer, through every path from task 1, and assert no output contains "arguments": "{ . It must run
   against the pinned revision, not a fixture.
5. Round trip per path: parse each rendered assistant turn back with the hermes parser (vLLM if installed,
   the regex fallback otherwise) and assert the tool name and arguments equal the original.
6. Retirement. Implement the body of `agentdistill registry retire --built-before <iso time or commit>
   --reason "..."` in registry/retire.py and give the lead its registration. It marks datasets and adapters built
   before the fix as retired, recording the reason as an adapter event. adapter best, dataset latest, and every
   report selector must skip retired rows; add a test for each. Run it on the local registry and list what it
   retired in your report.

DONE WHEN
Every path is listed and routed, tests 4 and 5 pass on the pinned template, no selector can return a retired
row, and your report lists the retired ids.
```

### WP4: Operations

```text
[paste section 2 here]

ROLE: WP4, operations and rehearsal.

YOU OWN: scripts/ (all of it), project.tiny.yaml, examples/support_agent/gpu-day.lock.json (new),
agentdistill/tools/, agentdistill/ops/ (new), tests/test_preflight.py, tests/test_gpu_day_script.py,
tests/test_estimate_spend.py, tests/test_export.py. Command registrations go to the lead.

TASKS
1. Lock file. examples/support_agent/gpu-day.lock.json records what the GPU day must reproduce: base_model,
   base_model_revision, tool parser, corpus hash from make_corpus.sh, SFT dataset content_hash, and the three
   eval-set hashes. Provide `agentdistill ops lock write` (writes it from the current registry and config) and
   `agentdistill ops lock check` (exit 1 with a diff on any mismatch). The lock is committed. Hashes are never
   parsed out of progress.md.
2. Pre-flight. scripts/preflight.sh prints PASS, FAIL, or SKIP per line and exits non-zero on any FAIL. Cheapest
   checks first:
   - git: working tree clean; HEAD pushed to a remote, or --bundle-ok when the owner chose a bundle
   - env: ANTHROPIC_API_KEY set (print only "set", never any part of the value); HF_HOME writable
   - config: teacher.model set; base_model is a Hub id; base_model_revision is 40 hex characters; tool
     parser set; eval.tools and onpolicy.tools are live
   - registry: a pricing row exists for the teacher
   - lock: `agentdistill ops lock check` passes after make_corpus.sh and curate
   - tokenizer guard: the dataset manifest's tokenizer and revision equal the config's
   - python: installed versions match requirements-gpu.txt
   - hardware (SKIP when nvidia-smi is absent): a GPU present, VRAM enough for the configured method (at least
     40 GB for bf16 LoRA on an 8B-class model, 22 GB for QLoRA), at least 60 GB free disk
3. Box bootstrap. scripts/bootstrap_box.sh for a fresh machine: install requirements-gpu.txt plus the package,
   run make_corpus.sh, run the pricing set command, run curate, then run preflight. Safe to run twice.
4. Spend estimate. `agentdistill ops estimate-spend` prints each teacher-calling stage in gpu_day.sh (holdout
   eval live, holdout eval replay, unseen eval, cascade verification at three thresholds with every turn
   assumed escalated as the upper bound, serve_smoke) with tasks x repeats x turns and the token totals. Prompt
   tokens per call grow with the turn index, so sum the prefix length over each trace's turns rather than
   multiplying one median by the turn count. Take turn counts and lengths from the corpus traces and apply a
   1.3 safety factor for a real model being wordier than the script. Price with the seeded row. Print the total
   and a recommended cap at twice the total, and state that prompt caching can only lower the real figure.
5. Export. Add a final stage to gpu_day.sh, plus a trap on EXIT so it also runs when a stage fails or the
   script is interrupted:
   - a tarball holding the registry database, artifacts/gpu_day/report.json and report.html, all logs, LoRA
     adapter directories (not merged or quantized full weights), the pip freeze, git rev-parse HEAD, the lock
     file, and the config files
   - its sha256 next to it
   - the scp command to copy both off, and, in capitals, that the box must not be terminated until
     verify_export.sh passes on the laptop
   - scripts/verify_export.sh, run on the laptop: recomputes the checksum, lists the contents, and confirms the
     registry opens and contains the report's run ids
6. Stage timing. At the end, gpu_day.sh prints each stage's duration beside an expected duration taken from the
   rehearsal and scaled by a factor in the config. Flag any stage over 3x.
7. After WP1 merges: apply the requested stage line to gpu_day.sh and set eval.tools and onpolicy.tools to live
   in project.tiny.yaml. Rerun the clean rehearsal and report the result.

TESTS
- preflight.sh on fixture repo states reports each failure it should, and never prints any part of the key.
- The export trap fires when a stub stage exits 3, and the tarball contains the registry.
- estimate-spend on a fixture corpus and pricing row equals a hand-computed value, including the prefix sum.
- gpu_day.sh dry run still resolves every subcommand it calls.
- `ops lock check` fails with a readable diff when one hash is changed.

DONE WHEN
All tests pass, preflight passes locally with the hardware checks SKIPped, and your report includes the spend
estimate with its breakdown and the recommended cap.
```

---

## 5. Lead: final integration check (after all four merges)

```text
[paste section 2 here]

ROLE: Lead, final integration check.

1. Full suite; record the count.
2. Clone into a new directory and run bash scripts/clean_rehearsal.sh with no dirty-tree allowance.
3. That run's report.json must contain an evaluation mode section, both teacher fields in lineage,
   CORPUS_TEACHER_DIFFERS as a disclosure, tools_mode live on the on-policy round, and no forbidden codes.
4. bash scripts/preflight.sh locally: every non-hardware check PASS.
5. agentdistill ops lock check passes in the fresh clone after make_corpus.sh and curate.
6. agentdistill ops estimate-spend: record the total and the recommended cap in docs/gpu-day-go.md.
7. Fill in docs/gpu-day-go.md: the nine go criteria, each with its evidence.
8. Commit, push (or bundle), and tag pre-gpu-day.
9. Report to the owner: go or no-go, and the cap to set on the teacher workspace.
```

---

## 6. GPU day prompt (for the agent on the rented box)

Before starting this agent: SSH in, `export ANTHROPIC_API_KEY=...` yourself in the shell the agent will use, and confirm the workspace spend limit is set.

```text
[paste section 2 here]

ROLE: GPU-day operator on a rented machine. You are running a prepared pipeline, not developing one.

RULES FOR TODAY
- Change code only when a stage fails for a reason that is clearly a bug and the fix is under 20 lines. Commit
  any fix on a branch named gpu-day-fixes with a progress.md divergence entry, and tell the owner immediately.
- Never print, log, or echo ANTHROPIC_API_KEY or any part of it.
- Nothing is terminated or deleted until the owner confirms the export verified on the laptop.

STEPS
1. git clone <remote> agentdistill && cd agentdistill && git checkout pre-gpu-day
2. bash scripts/bootstrap_box.sh. It ends with the pre-flight. If any line is FAIL, stop and report the whole
   pre-flight output. Do not start the run.
3. bash scripts/gpu_day.sh 2>&1 | tee logs/gpu_day.console.log
4. Report to the owner at each checkpoint:
   - after sft: final eval loss, and next-action full match against base. If flat, stop and wait for the
     owner's decision.
   - after eval_sft: success, schema validity, and eval_mode (must be live).
   - after onpolicy: the round's decision and reason, pair count, max pairs per task, tools_mode. A discard is a
     legitimate result, not a failure.
   - after calibrate: the verdict and its reason.
   - after serve_smoke: the /healthz output.
   - any stage over 3x its expected duration in the timing table: report at once.
5. If a stage exits 3, read its log, report the cause, and once the cause is addressed rerun gpu_day.sh;
   completed stages are skipped by their markers.
6. At the end, or on any abort, the export runs. Report the scp command and the checksum, then wait for the
   owner to confirm verify_export.sh passed on the laptop.
7. Only after that confirmation, report that the box can be terminated.

REPORT
Stage timings, the five checkpoint readings, the report's headline lines, every warning code present, any
fixes made, and the export checksum.
```

---

## 7. Results write-up prompt (same day, on the laptop)

```text
[paste section 2 here]

ROLE: Results author.

INPUT: the verified export tarball. Unpack it into artifacts/gpu_day_<date>/.

TASKS
1. Write docs/results.md from report.json only, every number carrying its run id:
   - headline table: base, student, teacher (live mode), and the unseen set
   - evaluation mode: replay against live for the teacher, labelled as the measured distortion of replay
   - on-policy round: decision and reason, whichever way it went
   - calibration verdict and reason; the cascade point if the verdict is usable, the reason if not
   - cost, stated with its throughput mode; no saving percentage if COST_UNBATCHED is present
   - per-cluster table and the weak clusters
   - provenance: corpus teacher against serving teacher, and one sentence on what that means for the claims
   - one paragraph on what these numbers do not show
2. Write the decision with the number it rests on: if three or more clusters sit below the router floor, the
   next phase is data; otherwise the router and canary work proceeds.
3. Inject the README results block with the report markdown command. Numbers reach the README only this way.
4. Commit the logs and docs/results.md on a branch named results-<date>. No weights, no registry.

REPORT: the headline, the decision and its number, and each warning code present with one line of meaning.
```

---

## Appendix A: independent oracle for WP2

Tested before handing over: it reproduces a recorded trace exactly (3 turns, 2 calls), stops on a divergence with the divergence payload, feeds malformed arguments back to the model instead of failing, and caps a looping model at `max_turns`. It imports nothing from `agentdistill`, which is the point. `recorded_lookup` uses sorted-key JSON rather than canonical hashing, so use it only on fixtures whose arguments are already canonical; for anything else, wrap `ReplayToolProvider.lookup` in a function that returns `(False, "")` on `Divergence`.

```python
# tests/oracle/reference_harness.py
"""TEST ORACLE for the lockstep and sequential runners.

Deliberately simple, one task at a time, and independent of agentdistill.eval's shared stepper.
Do not refactor this to share code with the production runners: its only value is that it does not.
It reproduces the harness's documented conventions (the two error payloads below), which are the
contract, not an implementation detail.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

BAD_JSON = json.dumps({"error": "arguments were not valid JSON"})
DIVERGED = json.dumps({"error": "replay divergence"})


@dataclass
class OracleOutcome:
    task_id: str
    messages: list[dict]
    n_turns: int
    n_tool_calls: int
    diverged: bool
    token_estimate: int


def run_reference(trace: dict, next_turn: Callable[[list[dict], list[dict]], dict],
                  lookup: Callable[[str, dict], tuple[bool, str]], max_turns: int = 12,
                  token_estimate: Callable[[str], int] = lambda s: len(s) // 4) -> OracleOutcome:
    """next_turn(messages, tools) -> assistant message. lookup(tool, args) -> (found, content)."""
    msgs = [m for m in trace["messages"][:2] if m["role"] in ("system", "user")]
    n_calls, est, diverged = 0, 0, False
    for _ in range(max_turns):
        reply = next_turn([dict(m) for m in msgs], trace["tools"])
        a = {k: v for k, v in reply.items() if not k.startswith("_")}
        msgs.append(a)
        est += token_estimate((a.get("content") or "") + json.dumps(a.get("tool_calls") or []))
        calls = a.get("tool_calls") or []
        if not calls:
            break
        for c in calls:
            n_calls += 1
            raw = c["function"]["arguments"]
            try:
                args = json.loads(raw) if isinstance(raw, str) else raw
            except json.JSONDecodeError:
                msgs.append({"role": "tool", "tool_call_id": c["id"], "content": BAD_JSON})
                continue
            found, content = lookup(c["function"]["name"], args)
            if not found:
                diverged = True
                msgs.append({"role": "tool", "tool_call_id": c["id"], "content": DIVERGED})
                break
            msgs.append({"role": "tool", "tool_call_id": c["id"], "content": content})
        if diverged:
            break
    return OracleOutcome(trace["task_id"], msgs, sum(1 for m in msgs if m["role"] == "assistant"), n_calls, diverged, est)


def recorded_lookup(trace: dict) -> Callable[[str, dict], tuple[bool, str]]:
    """Exact-match replay keyed on (tool, sorted-key JSON args). Intentionally cruder than canonical hashing,
    so use it only on fixtures whose arguments are already canonical."""
    by_id = {c["id"]: c for m in trace["messages"] for c in (m.get("tool_calls") or [])}
    table: dict[tuple[str, str], str] = {}
    for m in trace["messages"]:
        if m["role"] == "tool":
            c = by_id[m["tool_call_id"]]
            a = c["function"]["arguments"]
            key = (c["function"]["name"], json.dumps(json.loads(a) if isinstance(a, str) else a, sort_keys=True))
            table.setdefault(key, m["content"])

    def lookup(tool: str, args: dict) -> tuple[bool, str]:
        hit = table.get((tool, json.dumps(args, sort_keys=True)))
        return (hit is not None, hit or "")
    return lookup
```
