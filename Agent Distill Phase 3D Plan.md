# agentdistill: Phase 3d Plan

Closing the rehearsal findings, proving the two untested paths, and running the GPU day
Version 0.3d, September 19, 2026
Builds on: v0.3c and the first tiny-mode report

---

## 0. Where you are and what this phase produces

The pipeline runs end to end on a laptop and emits a report that names what it cannot claim. That is the right shape. But the rehearsal exercised roughly two thirds of the code: the teacher eval, the cost block, the calibration fit, the cascade verification, and the quantization delta never executed, because the stages that feed them produced nothing and passed anyway.

This phase ends when a clean tiny run produces a report with **no missing-data warnings** and with every section populated, including cost and cascade. At that point the GPU day is a rerun with different config values, and the failure modes left are hardware and model quality, not plumbing.

Four workstreams:

1. **Honesty invariants** (section 2): the minimum-N guard wired into `compare`, empty-stage detection, unassigned-cluster as a health state, and the pair cross-product cap. These are correctness fixes that matter in production, not tiny-mode cosmetics.
2. **Tiny-mode teacher and full coverage** (section 3): a deterministic fake teacher so the cost and cascade paths execute on a laptop.
3. **Clean-run determinism** (section 4): first-run correctness and reproducible generated eval sets.
4. **GPU day** (section 5): pre-flight, the run, and what to write down the same day.

Five days, then the GPU day.

---

## 1. Grading the rehearsal

What the report got right, and should not regress: every number carries a run id; lineage chains through DPO to the original adapter; the four unclaimable things are printed; the `argv[0]` normalization refuses to guess. Add tests that pin each of those so a later refactor cannot quietly drop them.

What it got wrong is all one class of error: **a stage that produces nothing is indistinguishable from a stage that produced something uninteresting.** The statistics line printed on an empty comparison, the cost block skipped silently, the calibration returned empty, and the router fell back to the teacher on a missing cluster model. Each is the same bug wearing different clothes, and each would behave identically on the GPU day with a half-failed stage.

---

## 2. Honesty invariants

### 2.1 The guard belongs in `compare`, not the renderer

```python
# agentdistill/eval/stats.py  (replace minimum_n_guard)
from __future__ import annotations

from dataclasses import dataclass

MIN_TASKS = 20
MIN_REPEATS = 3


@dataclass(frozen=True)
class InsufficientPower:
    n_tasks: int
    n_repeats: int
    min_tasks: int = MIN_TASKS
    min_repeats: int = MIN_REPEATS

    @property
    def reason(self) -> str:
        parts = []
        if self.n_tasks < self.min_tasks:
            parts.append(f"{self.n_tasks} tasks (need at least {self.min_tasks})")
        if self.n_repeats < self.min_repeats:
            parts.append(f"{self.n_repeats} repeats per task (need at least {self.min_repeats})")
        return "not enough data for a comparison: " + ", ".join(parts)

    def as_dict(self) -> dict:
        return {"insufficient_power": True, "n_tasks": self.n_tasks, "n_repeats": self.n_repeats,
                "min_tasks": self.min_tasks, "min_repeats": self.min_repeats, "reason": self.reason}


def power_check(n_tasks: int, n_repeats: int) -> InsufficientPower | None:
    if n_tasks < MIN_TASKS or n_repeats < MIN_REPEATS:
        return InsufficientPower(n_tasks, n_repeats)
    return None
```

`compare` returns the marker instead of statistics; it never raises, because a report that fails to render is worse than one that says it cannot compare.

```python
# agentdistill/eval/runner.py  (compare, head)
def compare(registry, run_a: str, run_b: str, alpha: float = 0.05) -> dict:
    a_rows, b_rows = registry.eval_results(run_a), registry.eval_results(run_b)
    oa, ob = outcomes(a_rows), outcomes(b_rows)
    common = sorted(set(oa) & set(ob))
    n_repeats = min([len(oa[t]) for t in common] + [len(ob[t]) for t in common], default=0)
    weak = power_check(len(common), n_repeats)
    if weak:
        return {"insufficient_power": weak.as_dict(), "n_tasks": len(common), "n_repeats": n_repeats,
                "observed": {"rate_a": rate(oa, common), "rate_b": rate(ob, common)},
                "generated_at": now_iso()}
    ...
```

`observed` carries the raw rates so the report can still show what happened; it is labelled as observed, never as a delta with a CI.

Renderer, in `results_block` and the HTML template:

```python
# agentdistill/report/markdown.py  (comparison line)
def comparison_line(name: str, p: dict) -> str:
    w = p.get("insufficient_power")
    if w:
        o = p.get("observed", {})
        rates = f" Observed: {o.get('rate_a', float('nan')) * 100:.1f}% vs {o.get('rate_b', float('nan')) * 100:.1f}%, not compared." if o else ""
        return f"{name}: {w['reason']}.{rates}"
    return (f"{name}: {p['success']['delta'] * 100:+.1f} pp{ci(p['success']['ci95'])}, McNemar p={p['mcnemar']['p']:.3f}, "
            f"tokens median {p['tokens']['median_delta']:+.0f}{ci_raw(p['tokens']['ci95'])}.")
```

Tests: `compare` on 5 tasks at N=1 returns the marker with the reason and no `success` key; at 20 tasks and N=3 returns statistics; the markdown renderer prints the reason and never a p-value alongside it; a property test that no rendered line contains "p=" when `insufficient_power` is set. Also assert that a zero-width CI can never reach the renderer: if `ci95[0] == ci95[1]` and the delta is zero, that is degenerate data, so `compare` treats it as insufficient regardless of counts and says so.

### 2.2 Stages that produce nothing must fail

```python
# agentdistill/cli_stage.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


class StageEmpty(RuntimeError):
    """A stage completed but wrote no row. Exit code 3."""


@dataclass
class StageOutcome:
    wrote: bool
    detail: str
    skipped_reason: str | None = None


def stage_guard(name: str, fn: Callable[[], StageOutcome], allow_skip: bool = False) -> int:
    """Wrap a CLI entrypoint. Exit 0 on a row written, 0 with a SKIPPED log on a declared skip, 3 on silent emptiness."""
    out = fn()
    if out.wrote:
        print(f"[stage {name}] ok: {out.detail}")
        return 0
    if out.skipped_reason and allow_skip:
        print(f"[stage {name}] SKIPPED: {out.skipped_reason}")
        return 0
    raise StageEmpty(f"[stage {name}] wrote no row: {out.detail}")
```

Applied to: `eval run` (no `eval_results` rows), `calibrate` (fewer than `cascade.min_turns`, default 200, labelled turns, or no calibration row), `train onpolicy` (no round row), `adapter quantize` (no adapter row), `report` (no subjects at all). A declared skip is a config value, never an inference: `eval.skip_teacher: true` produces `SKIPPED: eval.skip_teacher set`, and the report prints that as a warning distinct from "no run found".

`gpu_day.sh` gains `set -o pipefail` (already there) plus: exit 3 from a stage does not write the `.done` marker and stops the script, so a rerun retries it. Exit 0 with `SKIPPED` writes the marker.

The two log lines must look different at a glance. `[stage eval_teach] SKIPPED: eval.skip_teacher set` versus `[stage eval_teach] wrote no row: teacher backend returned no completions`.

### 2.3 Unassigned cluster is a health state

```python
# agentdistill/router/clusters.py  (additions)
UNASSIGNED = -1


class ClusterAssigner:
    ...
    def assign(self, messages: list[dict]) -> int:
        if self.c is None or len(self.c) == 0:
            return UNASSIGNED
        return self._assign_text(self.task_text(messages))


class NoClusterModel(ClusterAssigner):
    """Explicit null object: the gateway loaded no cluster model."""

    def __init__(self, reason: str):
        self.reason = reason
        self.c, self.embed, self.labels = None, None, {}

    def assign(self, messages: list[dict]) -> int:
        return UNASSIGNED
```

Gateway behavior on `UNASSIGNED`: route by the global posterior (all clusters pooled) rather than by the floor, mark `requests.cluster_id = NULL` and `requests.routing_reason = 'no_cluster_model'`, and surface it. The floor exists to protect clusters the student is measurably bad at; applying it to "we do not know" turns a missing file into a teacher bill.

```python
# agentdistill/gateway/health.py
from __future__ import annotations

import time
from collections import deque


class HealthTracker:
    def __init__(self, window_s: int = 300, fallback_alert: float = 0.2, unassigned_alert: float = 0.5):
        self.window_s, self.fallback_alert, self.unassigned_alert = window_s, fallback_alert, unassigned_alert
        self.events: deque[tuple[float, bool, bool]] = deque()     # (ts, fallback, unassigned)

    def record(self, fallback: bool, unassigned: bool, now: float | None = None) -> None:
        t = now if now is not None else time.time()
        self.events.append((t, fallback, unassigned))
        cut = t - self.window_s
        while self.events and self.events[0][0] < cut:
            self.events.popleft()

    def snapshot(self, now: float | None = None) -> dict:
        t = now if now is not None else time.time()
        cut = t - self.window_s
        rows = [e for e in self.events if e[0] >= cut]
        n = len(rows)
        fb = sum(1 for e in rows if e[1]) / n if n else 0.0
        un = sum(1 for e in rows if e[2]) / n if n else 0.0
        problems = []
        if n >= 20 and fb > self.fallback_alert:
            problems.append(f"student fallback rate {fb:.0%} over {self.window_s}s")
        if n >= 20 and un > self.unassigned_alert:
            problems.append(f"unassigned cluster rate {un:.0%}; cluster model may be missing")
        return {"ok": not problems, "requests": n, "fallback_rate": round(fb, 3), "unassigned_rate": round(un, 3), "problems": problems}
```

`/healthz` returns the snapshot plus `cluster_model` (`loaded` with k and id, or `missing` with the reason) and `calibration` (`loaded` with the threshold, or `missing`). A gateway serving with no calibration and no cluster model should say so on the first request anyone makes, not after the invoice.

### 2.4 The pair cross-product

532 pairs from a tiny corpus is a cross-product. Cap and sample.

```python
# agentdistill/train/pairs.py  (build_pairs)
from __future__ import annotations

import random
from collections import Counter


def build_pairs(rollouts_by_task: dict[str, list[dict]], teacher_by_task: dict[str, dict],
                cap_per_task: int = 3, teacher_pair_ratio: float = 1.0, seed: int = 0) -> tuple[list[dict], dict]:
    rng = random.Random(seed)
    rollout_pairs: list[dict] = []
    teacher_pairs: list[dict] = []
    per_task: Counter[str] = Counter()
    for task_id, rolls in sorted(rollouts_by_task.items()):
        good = [r for r in rolls if r["success"]]
        bad = [r for r in rolls if not r["success"]]
        cands: list[dict] = []
        if good and bad:
            # sample pairs rather than enumerating the cross product
            for _ in range(min(cap_per_task * 3, len(good) * len(bad))):
                p = first_divergent_pair(rng.choice(good), rng.choice(bad))
                if p and pair_is_valid(p)[0]:
                    cands.append({**p, "pair_kind": "rollout", "task_id": task_id})
                if len(cands) >= cap_per_task:
                    break
        elif bad and task_id in teacher_by_task:
            for r in rng.sample(bad, min(cap_per_task, len(bad))):
                p = first_divergent_pair(teacher_by_task[task_id], r)
                if p and pair_is_valid(p)[0]:
                    cands.append({**p, "pair_kind": "teacher", "task_id": task_id})
        cands = dedupe_pairs(cands)[:cap_per_task]
        per_task[task_id] = len(cands)
        for c in cands:
            (rollout_pairs if c["pair_kind"] == "rollout" else teacher_pairs).append(c)
    max_teacher = int(len(rollout_pairs) * teacher_pair_ratio) if rollout_pairs else cap_per_task * 5
    teacher_pairs = rng.sample(teacher_pairs, min(len(teacher_pairs), max_teacher))
    pairs = rollout_pairs + teacher_pairs
    stats = {"n_pairs": len(pairs), "n_rollout": len(rollout_pairs), "n_teacher": len(teacher_pairs),
             "tasks_with_pairs": sum(1 for v in per_task.values() if v), "max_per_task": max(per_task.values(), default=0),
             "diff_kind": dict(Counter(p["diff_kind"] for p in pairs)), "per_task_histogram": dict(Counter(per_task.values()))}
    return pairs, stats


def dedupe_pairs(pairs: list[dict]) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    out = []
    for p in pairs:
        key = (pair_hash(p["chosen"]), pair_hash(p["rejected"]))
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out
```

`stats` is logged by `train onpolicy` and stored on the round row. An invariant test: `n_pairs <= cap_per_task * n_tasks * (1 + teacher_pair_ratio)`, and `max_per_task <= cap_per_task`. A second test on the tiny corpus asserts the count lands in the low tens, not the hundreds.

While in this file, confirm the `diff_kind` distribution is recorded and that a set which is more than 70 percent `text` raises a warning on the round row; that was the v0.3c note and the histogram makes it checkable.

### 2.5 Reproduction block completeness

Add to the recorded command: git commit SHA, a dirty flag, the config path, and the config content hash.

```python
# agentdistill/provenance.py
from __future__ import annotations

import hashlib
import os
import subprocess
import sys

KNOWN_ENTRYPOINTS = {"cli.py": "agentdistill", "__main__.py": "agentdistill"}


def normalized_argv() -> list[str]:
    argv = list(sys.argv)
    base = os.path.basename(argv[0]) if argv else ""
    if base in KNOWN_ENTRYPOINTS:
        argv[0] = KNOWN_ENTRYPOINTS[base]
    return argv


def git_state() -> dict:
    def run(*a: str) -> str | None:
        try:
            return subprocess.run(a, capture_output=True, text=True, timeout=5, check=True).stdout.strip()
        except Exception:
            return None
    sha = run("git", "rev-parse", "HEAD")
    status = run("git", "status", "--porcelain")
    return {"commit": sha, "dirty": bool(status) if status is not None else None}


def config_state(path: str | None) -> dict:
    if not path or not os.path.exists(path):
        return {"config_path": path, "config_hash": None}
    with open(path, "rb") as f:
        return {"config_path": path, "config_hash": hashlib.sha256(f.read()).hexdigest()[:12]}


def provenance(config_path: str | None) -> dict:
    return {"command": " ".join(normalized_argv()), **git_state(), **config_state(config_path)}
```

Every CLI entrypoint stores `provenance(cfg_path)` on its registry row. The report's reproduce block prints the command, and beneath it `commit <sha>[ (dirty)] config <path>@<hash>`. A run recorded from a dirty tree is flagged in the report warnings: a number you cannot get back to is a number you cannot defend.

---

## 3. Tiny mode: covering the two dark paths

### 3.1 A deterministic fake teacher

The goal is not a good teacher; it is a teacher-shaped row so `cost_block`, `calibrate`, and the cascade verification execute.

```python
# agentdistill/eval/fake_teacher.py
from __future__ import annotations

import json


class ReplayTeacherClient:
    """A TurnClient that replays the recorded trace's own assistant turns.
    On the holdout set this is a perfect teacher by construction, which is exactly what tiny mode needs:
    a populated row with realistic token counts, never a quality claim."""

    def __init__(self, traces_by_task: dict[str, dict], prompt_tokens: int = 1200, completion_tokens: int = 90):
        self.by_task = traces_by_task
        self.prompt_tokens, self.completion_tokens = prompt_tokens, completion_tokens
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0}
        self._cursor: dict[str, int] = {}

    def bind(self, task_id: str) -> None:
        self._task, self._cursor[task_id] = task_id, 0

    def next_turn(self, messages: list[dict], tools: list[dict]) -> dict:
        trace = self.by_task[self._task]
        turns = [m for m in trace["messages"] if m["role"] == "assistant"]
        i = self._cursor[self._task]
        self._cursor[self._task] = i + 1
        self.usage["prompt_tokens"] += self.prompt_tokens
        self.usage["completion_tokens"] += self.completion_tokens
        if i >= len(turns):
            return {"role": "assistant", "content": "Done.", "tool_calls": None}
        return json.loads(json.dumps(turns[i]))
```

`eval run teacher --backend replay` uses it in tiny mode. The eval run row carries `metrics.teacher_backend = "replay"`, and `assemble` adds the warning `teacher metrics come from a replay stub (tiny mode); cost figures are structural, not measured`. The report must never show a replay-teacher cost without that line.

Token medians for the cost block come from the stub's fixed per-turn counts, which is enough to exercise `teacher_cost_per_task`, `cascade_cost_per_task`, and `breakeven_tasks_per_day`.

### 3.2 Calibration with enough turns

Tiny mode's calibration set is five tasks, so no real gate can be fit. Two changes:

- `calibrate` exits 3 below `cascade.min_turns` (default 200) with the count in the message.
- Tiny mode sets `cascade.min_turns: 20` and generates a `support-calib-tiny` set of 20 tasks at N=3 with `--logprobs --samples 1`. With a tiny model the AUROC will be near chance; that is fine and the calibration row records it. `choose_threshold` will likely return "no threshold meets the budget; escalate everything", and the report prints exactly that. The point is that `fit_calibrator`, `expected_calibration_error`, the reliability bins, `choose_threshold`, and the `--verify-threshold` harness pass all execute.

Add a guard in `calibration_report`: if AUROC is below 0.55, set `verdict = "uninformative"`, and have the gateway refuse to load an uninformative calibration (escalate everything, say so on `/healthz`). That rule protects the GPU day too.

### 3.3 Quantization in tiny mode

`adapter quantize --method fp8` writes the marker and registers the row, then `eval run` on it produces the delta. That path already works; add the assertion that the quantized eval row exists before `assemble` reads it, so a skipped quantize shows as a warning rather than a `None` in the table.

### 3.4 Definition of done for tiny coverage

A clean tiny run produces a report where:

- `subjects` has base, student, and teacher (replay), all with run ids.
- `paired` shows either statistics or the insufficient-power reason, never a p-value on degenerate data.
- `calibration` has holdout metrics and a verdict.
- `cascade` has verified points (possibly "escalate everything") and a threshold.
- `cost` has teacher cost per task, student cost per million tokens, and either a cascade cost or a named reason.
- `quantization` has a delta.
- `warnings` contains only tiny-mode disclosures (replay teacher, tiny config, uninformative gate), and no "no run found" entries.

---

## 4. Clean-run determinism

The clean run is the real rehearsal. Make the first-run failures impossible rather than discovering them one at a time.

### 4.1 Deterministic generated eval sets

Tiny mode generates its eval sets rather than freezing them, so pin them:

```python
# agentdistill/evalsets/generate.py
from __future__ import annotations

import hashlib
import json


def stable_task_ids(scenarios: list[dict], n: int, salt: str) -> list[str]:
    """Deterministic selection: rank by hash of (salt, scenario, instance index), take n. No RNG, no set iteration order."""
    ranked = sorted(
        ((hashlib.sha256(f"{salt}|{s['scenario']}|{s['instance']}".encode()).hexdigest(), s["task_id"]) for s in scenarios),
        key=lambda t: t[0],
    )
    return [task_id for _, task_id in ranked[:n]]


def eval_set_hash(task_ids: list[str], grader: dict) -> str:
    return hashlib.sha256(json.dumps({"tasks": sorted(task_ids), "grader": grader}, sort_keys=True).encode()).hexdigest()[:12]
```

Test: two clean runs with the same seed produce identical `task_ids` and identical `eval_set_hash`. A separate test asserts the dataset `content_hash` is stable across two clean builds even though row ids differ, which is the invariant that already exists for curate and now gets a clean-run version.

Sources of nondeterminism to eliminate while here: `set` iteration in any id-producing path, `dict` ordering from JSON parsing (fine in Python, but sort before hashing anyway), `random` without an explicit seed, and `datetime.now()` inside anything hashed.

### 4.2 First-run correctness checklist

- Migrations run in order against an empty database; a test creates a fresh SQLite file and applies 001 through 004, then asserts the schema matches a checked-in snapshot of `PRAGMA table_info` for every table.
- Every directory the pipeline writes to is created with `os.makedirs(..., exist_ok=True)` at the point of use, not assumed by an earlier stage. Grep for `open(` with a write mode and check each.
- The tiny model is pinned by revision: `train.base_model_revision` in the config, passed to `from_pretrained(..., revision=...)`. A test asserts the revision is set in tiny mode, so an upstream retag cannot change the rehearsal.
- `HF_HOME` and any cache path are set in `gpu_day.sh` so a clean run does not silently re-download into a temporary directory.
- `registry/select.py` queries return `None` rather than raising on an empty table, and every caller handles `None` (the report's warnings prove it does; add the same for the gateway's boot path).

### 4.3 The clean-run script

```bash
#!/usr/bin/env bash
# scripts/clean_rehearsal.sh   destroys every artifact and reruns from nothing
set -euo pipefail
cd "$(dirname "$0")/.."
: "${AGENTDISTILL_TINY:=1}"; export AGENTDISTILL_TINY
rm -rf artifacts/gpu_day logs/gpu_day.* .agentdistill/tiny.db artifacts/tiny
git clean -ndx artifacts .agentdistill | sed 's/^/would remove: /'
bash scripts/gpu_day.sh
python -m agentdistill.tools.assert_report artifacts/gpu_day/report.html \
  --require-subjects base,student,teacher \
  --require-sections calibration,cascade,cost,quantization \
  --forbid-warning "no run found" --forbid-warning "no calibration" \
  --allow-warning "tiny mode" --allow-warning "replay stub" --allow-warning "uninformative"
echo "clean rehearsal ok"
```

`assert_report` reads the JSON sidecar that `report` writes next to the HTML (`report.json`, the `asdict(ReportData)`), so the assertions are against data, not parsed HTML. Add that sidecar if it does not exist; it is also what `report --format md` consumes, which keeps the three renderers reading one source.

---

## 5. The GPU day

### 5.1 Pre-flight (the day before)

1. `bash scripts/clean_rehearsal.sh` green from a fresh clone into a new directory, not just a clean working tree.
2. `agentdistill base-check <real base model>` reports `ok` for the round trip, with `train.base_model_revision` pinned.
3. `requirements-gpu.txt` from `pip freeze` for torch, transformers, trl, peft, vllm, llmcompressor, accelerate, bitsandbytes.
4. Teacher API key with a hard spend cap; confirm `eval.skip_teacher` is false and the real teacher model name is in the config.
5. Disk: 60 GB free (base weights, merged bf16, AWQ, rollouts, logs).
6. `git tag pre-gpu-day` and push.
7. Record the expected durations from v0.3 section 2.2 in a scratch file so you notice a stage running 3x long while it is happening, not afterwards.

### 5.2 During

Run `bash scripts/gpu_day.sh` and watch four things:

- **`sft`**: eval loss falling, next-action accuracy above the base model by the end. If next-action is flat, stop and inspect the mask and the template before spending more hours.
- **`eval_sft` divergence rate**: above 40 percent in strict mode means canonicalization is fighting the student's phrasing; note it, finish the run, and fix per-tool rules afterwards rather than mid-day.
- **`onpolicy` fuzzy share and pair stats**: the new per-task histogram is the check that section 2.4 worked. `max_per_task` must equal the cap.
- **`calibrate` AUROC**: below 0.55 means the gate is uninformative and the cascade section will say "escalate everything". That is a real result, not a failure; the report states it.

### 5.3 After, the same day

Write `docs/results.md` while the numbers are in front of you: the headline table, the cascade point, the calibration verdict, the per-cluster weak spots, and one paragraph of interpretation including what the numbers do not show. Commit the logs branch. Inject the README block with `report --format md --inject`.

Then decide one thing: whether the student is good enough to justify Milestone 7's router and canary work, or whether the next move is more data in the weak clusters. The per-cluster table answers it. If three or more clusters sit below the floor, data comes first.

---

## 6. Sequence

**Day 1**: `power_check` and `InsufficientPower` wired into `compare`; renderer changes; degenerate-CI rule; tests including the no-p-value property test.
**Day 2**: `cli_stage.stage_guard` applied to the five commands; exit-3 handling in `gpu_day.sh`; skip-versus-empty log distinction; tests.
**Day 3**: `UNASSIGNED` and `NoClusterModel`; gateway routing by pooled posterior; `HealthTracker` and the expanded `/healthz`; uninformative-calibration refusal; tests.
**Day 4**: pair capping and sampling with `stats`; invariant tests; `provenance.py` and the reproduce block; `report.json` sidecar and `assert_report`.
**Day 5**: `ReplayTeacherClient` and `--backend replay`; tiny `support-calib-tiny` generation; deterministic eval-set generation with hash tests; first-run checklist items; `clean_rehearsal.sh`; run it from a fresh clone until green.
**Day 6**: pre-flight; rent the GPU; run; write `docs/results.md`.

---

## 7. Things that will go wrong

| Symptom | Likely cause | Fix |
|---|---|---|
| Clean run fails on migration 004 only | A migration written against an already-migrated database | The fresh-database schema snapshot test |
| Two clean runs produce different eval sets | `set` iteration or unseeded RNG in selection | `stable_task_ids`; the hash equality test |
| `assert_report` fails on a warning you expected | Warning text drifted | Assert on warning codes, not prose: give each warning a stable `code` field and match on that |
| Replay teacher gives 100 percent success and the report reads as a real result | Missing disclosure | `teacher_backend = replay` forces the warning; a test asserts the warning is present whenever the backend is replay |
| Calibration passes with AUROC 0.52 | No verdict gate | `verdict = uninformative` below 0.55; gateway refuses to load it |
| Pair count still large after capping | Dedupe running before the cap, or the cap applied per kind instead of per task | The invariant test on `max_per_task` |
| GPU day stage rerun repeats work | Marker written on an exit-3 path | Markers only on exit 0; the stage guard test covers it |
| `/healthz` says ok while everything escalates | Health tracker not fed on the fallback path | Record on every request, including fallbacks; test with a stub gateway |

---

## 8. Definition of done

- `compare` never returns statistics below the guard, and no renderer can print a p-value next to an insufficient-power marker.
- Every stage that writes no row exits 3 unless a config value declared the skip; the two cases log differently.
- Unassigned clusters route by pooled posterior, are counted, and appear on `/healthz` with the cluster-model state; an uninformative calibration is refused at load.
- Pair building is capped and sampled, with per-task statistics on the round row and an invariant test.
- Reproduce blocks carry commit, dirty flag, config path and hash; dirty runs are flagged in the report.
- A clean tiny rehearsal from a fresh clone passes `assert_report` with only tiny-mode warnings.
- The GPU day has run; `docs/results.md` and the README block are written from `report.json`.
