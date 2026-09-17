# agentdistill: Phase 3c Plan

Finishing Half A: report, router remainder, serving artifacts, docs, and a CPU rehearsal of the GPU day
Version 0.3c, September 18, 2026
Builds on: v0.3 (phase 3) and docs/progress.md after Half A sections 1 to 7 (794 tests, eight commits)

---

## 0. Where you are and what this step produces

Built: the corrections from v0.3 section 1, `scripts/gpu_day.sh` with 18 resumable stages and a test that every subcommand exists, migration 003, `registry/select.py` with `adapter best` ranked by measured success, DPO and the round loop, confidence features, calibration, the cascade with a native async path, the gateway with both dialects and streaming contract tests, the adapter lifecycle.

Remaining in Half A: the cost model and report (section 8), the router store, canary, live comparison, and retrain orchestration (the rest of section 7), `adapter merge`, `adapter quantize`, and the serve scripts (section 6.7), docs.

This step ends when `scripts/gpu_day.sh` runs to completion on a laptop in tiny mode and produces a report with run ids. The numbers will be meaningless; the point is that every stage, every registry query, every file path, and the report template have executed once before you pay for a GPU. Then the GPU day is a rerun with a different model name.

Six days, then the GPU day.

---

## 1. Review of the judgement calls

All five are right. Two need tightening.

**DPO pair validity.** Correct, and the text-only exception is correct: a final-answer turn's prose is the decision. Add the mirror check for text-only pairs: chosen and rejected must differ by more than whitespace, casing, or punctuation, or the pair teaches a style preference under a decision loss. Also record, per pair, `diff_kind in {tool_choice, tool_args, tool_vs_text, text}` so the report can show what the DPO set was made of; a set that is 90 percent `text` pairs is a warning sign.

**`decide` promotes on equal success only if cost improved.** Right, and my v0.3 code was too loose: `cost_better = median_delta < 0` promotes on noise. Require the Wilcoxon CI on the token delta to exclude zero (`ci95[1] < 0`), not just a negative median. A promotion should never rest on a point estimate. Same for the "success up" branch, which already uses the CI.

**NaN for missing features.** Right. `HistGradientBoostingClassifier` handles NaN natively, so nothing else changes, but pin the invariant with a test: the feature vector for a text-only turn has NaN in every argument-logprob slot and the calibrator scores it without error. And assert at gateway startup that `cascade.features` in config equals `calibrations.features` on the loaded row, in order.

**Gateway never breaks a working agent.** Right, with one addition: silent fallback is how a broken vLLM turns into a quiet 100 percent teacher bill. Count it. Every fallback sets `requests.arm = 'teacher'` and a new `requests.fallback = true`; `/healthz` reports the rolling fallback rate; the gateway logs at error level when it exceeds 20 percent over five minutes. Fallback requests must not update the student arm's router posterior (they never ran the student), and they should not update the teacher arm either, since the teacher was not chosen on merit.

**Native async cascade.** Right. The sync bridge would have deadlocked the first time the gateway called it from inside its own loop.

**The shared canonical module.** Fine that it arrived from the mcpgate side; not fine to keep two implementations. One hash function in the repo: make `agentdistill/canonical.py` import and re-export from `canonical_shared.py` (or the reverse), delete the duplicate body, and keep one test that runs all 19 vectors. Two canonicalizers that agree today are two that disagree after the next edit.

---

## 2. Section 8: cost model and report

The cost functions from v0.1 section 11 exist as spec; the work here is assembling the report from the registry and rendering it.

### 2.1 Assembly

```python
# agentdistill/report/assemble.py
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from agentdistill.report.cost import breakeven_tasks_per_day, cascade_cost_per_task, student_cost_per_mtok, teacher_cost_per_task


@dataclass
class ReportData:
    generated_at: str
    project: str
    eval_set: str
    unseen_set: str | None
    subjects: dict = field(default_factory=dict)          # name -> {run_id, success, ci, schema_valid, divergence_rate, tokens_median, n_tasks, n_per_task}
    paired: dict = field(default_factory=dict)            # "student_vs_teacher" -> compare dict
    cascade: dict = field(default_factory=dict)           # threshold, verified points, analytic curve
    calibration: dict = field(default_factory=dict)       # holdout metrics, bins, label mix
    per_cluster: list = field(default_factory=list)       # rows: cluster, label, n, base, student, teacher, routing
    quantization: dict = field(default_factory=dict)      # method, bf16 success, quantized success, delta
    cost: dict = field(default_factory=dict)
    lineage: dict = field(default_factory=dict)
    commands: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


def assemble(registry, cfg: dict, tag_glob: str, now_iso: str) -> ReportData:
    r = ReportData(generated_at=now_iso, project=cfg["name"], eval_set=cfg["eval"]["eval_set"], unseen_set=cfg["eval"].get("unseen_set"))
    holdout = cfg["eval"]["eval_set"]

    def subject(name: str, run):
        if run is None:
            r.warnings.append(f"no eval run for {name}")
            return
        m = run["metrics"]
        r.subjects[name] = {"run_id": run["id"], "success": m["success"], "schema_valid": m["schema_valid"], "divergence_rate": m["divergence_rate"],
                            "tokens_median": m["tokens_est_median"], "n_tasks": m["n_tasks"], "n_per_task": m["n_per_task"], "subject": run["subject"]}

    base = registry.eval_latest(subject="base", eval_set=holdout)
    teacher = registry.eval_latest(subject="teacher", eval_set=holdout)
    best = registry.adapter_best(tag_glob=tag_glob, eval_set=holdout)          # ranked by measured success; None if nothing evaluated
    student = registry.eval_latest(subject=best["id"], eval_set=holdout) if best else None
    subject("base", base); subject("teacher", teacher); subject("student", student)
    if best and r.unseen_set:
        subject("student_unseen", registry.eval_latest(subject=best["id"], eval_set=r.unseen_set))
        subject("teacher_unseen", registry.eval_latest(subject="teacher", eval_set=r.unseen_set))

    if student and teacher:
        r.paired["student_vs_teacher"] = registry.compare(teacher["id"], student["id"])
    if student and base:
        r.paired["student_vs_base"] = registry.compare(base["id"], student["id"])

    cal = registry.latest_calibration(best["id"]) if best else None
    if cal:
        r.calibration = {"id": cal["id"], "holdout": cal["holdout_metrics"], "bins": cal["reliability_bins"], "threshold": cal["threshold"],
                         "label_mix": cal.get("label_mix", {}), "features": cal["features"]}
        r.cascade = {"threshold": cal["threshold"], "verified": cal.get("verified") or [], "analytic": cal.get("analytic_curve") or []}
        if not r.cascade["verified"]:
            r.warnings.append("cascade threshold not verified by the harness; analytic estimate only")
    elif best:
        r.warnings.append("no calibration for the best adapter; gateway will escalate everything")

    if best:
        r.per_cluster = registry.per_cluster_table(base_run=base["id"] if base else None, student_run=student["id"] if student else None,
                                                   teacher_run=teacher["id"] if teacher else None, floor=cfg["router"]["floor"])
        q = registry.latest_quantized(best["id"])
        if q and q.get("eval_run_id"):
            qe = registry.eval(q["eval_run_id"])
            r.quantization = {"method": q["quantization"], "bf16_success": student["metrics"]["success"] if student else None,
                              "quantized_success": qe["metrics"]["success"], "delta_pp": (qe["metrics"]["success"] - student["metrics"]["success"]) * 100 if student else None,
                              "run_id": qe["id"]}

    r.cost = cost_block(cfg, teacher, student, r.cascade, registry)
    r.lineage = registry.lineage(best["id"]) if best else {}
    r.commands = registry.commands_for(tag_glob)
    return r


def cost_block(cfg: dict, teacher, student, cascade: dict, registry) -> dict:
    if not teacher:
        return {"warning": "no teacher eval; cost block skipped"}
    tm = teacher["metrics"]
    price = registry.pricing(cfg["teacher"]["provider"], cfg["teacher"]["model"])
    teacher_task = teacher_cost_per_task(tm["prompt_tokens_median"], tm["completion_tokens_median"], price["input_per_mtok"], price["output_per_mtok"],
                                         cache_hit_frac=tm.get("cache_hit_frac", 0.0), cache_read_per_mtok=price.get("cache_read_per_mtok"))
    out = {"teacher_cost_per_task": teacher_task, "pricing": price}
    if student:
        sm = student["metrics"]
        tps = sm.get("throughput_tok_per_s")
        if not tps:
            out["warning"] = "student throughput not measured; cost per task unavailable"
            return out
        s_mtok = student_cost_per_mtok(cfg["serve"]["gpu_usd_per_hour"], tps, cfg["serve"].get("utilization", 0.6))
        out["student_cost_per_mtok"] = s_mtok
        out["student_only_cost_per_task"] = sm["tokens_est_median"] * s_mtok / 1e6
        v = cascade.get("verified") or []
        chosen = next((p for p in v if abs(p["threshold"] - cascade.get("threshold", -1)) < 1e-6), None) or (v[len(v) // 2] if v else None)
        if chosen:
            c = cascade_cost_per_task(sm["tokens_est_median"], s_mtok, chosen["escalation_rate"], teacher_task, chosen.get("wasted_student_tokens", 0))
            out["cascade"] = {"threshold": chosen["threshold"], "cost_per_task": c, "success": chosen["success"], "escalation_rate": chosen["escalation_rate"],
                              "saving_frac": 1 - c / teacher_task if teacher_task else None,
                              "breakeven_tasks_per_day": breakeven_tasks_per_day(cfg["serve"]["gpu_usd_per_hour"], teacher_task, c)}
        out["throughput_conditions"] = sm.get("throughput_conditions", "batched eval run")
    return out
```

Three registry methods are new: `per_cluster_table`, `lineage`, `commands_for` (the exact CLI invocations recorded on each run row; add `command TEXT` to `training_runs`, `eval_runs`, and `calibrations` in migration 004 and have every CLI entrypoint store `" ".join(sys.argv)`).

Eval metrics need two additions for the cost block: `prompt_tokens_median` and `completion_tokens_median` on teacher runs (from the API usage), and `throughput_tok_per_s` with `throughput_conditions` on student runs (total generated tokens divided by wall-clock of the batched vLLM generate calls, with the batch size and `max_num_seqs` recorded as the conditions string).

### 2.2 Rendering

`report/html.py` renders `report.html.j2` from `asdict(ReportData)`; `report/svg.py` from v0.3 draws the cost-versus-success curve (analytic points faint, verified points bold) and the reliability diagram (bins as bars against the diagonal). Warnings render at the top in a yellow block; a report with warnings is still a report.

```python
# agentdistill/report/markdown.py
from __future__ import annotations

from agentdistill.report.assemble import ReportData

BEGIN, END = "<!-- agentdistill:results:begin -->", "<!-- agentdistill:results:end -->"


def pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def ci(c) -> str:
    return "" if not c else f" [{c[0] * 100:+.1f}, {c[1] * 100:+.1f}]"


def ci_raw(c) -> str:
    return "" if not c else f" [{c[0]:+.0f}, {c[1]:+.0f}]"


def results_block(r: ReportData) -> str:
    lines = [BEGIN, f"Results generated {r.generated_at} on eval set `{r.eval_set}`.", ""]
    lines += ["| Subject | Success | Schema valid | Divergence | Tokens/task (median) | Run |", "|---|---|---|---|---|---|"]
    for name in ("base", "student", "teacher", "student_unseen", "teacher_unseen"):
        s = r.subjects.get(name)
        if s:
            lines.append(f"| {name} | {pct(s['success'])} | {pct(s['schema_valid'])} | {pct(s['divergence_rate'])} | {s['tokens_median']} | `{s['run_id']}` |")
    p = r.paired.get("student_vs_teacher")
    if p:
        lines += ["", f"Student vs teacher: {p['success']['delta'] * 100:+.1f} pp{ci(p['success']['ci95'])}, McNemar p={p['mcnemar']['p']:.3f}, "
                      f"tokens median {p['tokens']['median_delta']:+.0f}{ci_raw(p['tokens']['ci95'])}."]
    c = r.cost.get("cascade")
    if c:
        lines += ["", f"Cascade at threshold {c['threshold']:.2f}: success {pct(c['success'])}, escalation {pct(c['escalation_rate'])}, "
                      f"cost per task ${c['cost_per_task']:.4f} vs teacher ${r.cost['teacher_cost_per_task']:.4f} ({pct(c['saving_frac'])} saving), "
                      f"break-even {c['breakeven_tasks_per_day']:.0f} tasks/day on the configured GPU."]
    if r.quantization:
        lines += ["", f"Quantization ({r.quantization['method']}): {r.quantization['delta_pp']:+.1f} pp vs bf16 (run `{r.quantization['run_id']}`)."]
    if r.warnings:
        lines += ["", "Warnings:"] + [f"- {w}" for w in r.warnings]
    lines += ["", END]
    return "\n".join(lines)


def inject(readme_text: str, block: str) -> str:
    if BEGIN in readme_text and END in readme_text:
        head = readme_text[: readme_text.index(BEGIN)]
        tail = readme_text[readme_text.index(END) + len(END):]
        return head + block + tail
    return readme_text.rstrip() + "\n\n## Results\n\n" + block + "\n"
```

`agentdistill report --format md --inject README.md` is the only way numbers reach the README. A test asserts that injecting twice is idempotent and that a README without markers gets a Results section appended once.

### 2.3 Tests

- `test_assemble.py`: a seeded fixture registry with base, teacher, student, calibration, quantized rows produces a `ReportData` with no warnings; removing the calibration row produces the "escalate everything" warning; removing the teacher run skips the cost block with a warning.
- `test_markdown.py`: block renders; inject idempotent; no numbers appear without a run id on the same row.
- `test_html.py`: renders from a fixture `ReportData`, contains two `<svg` elements, no external URLs, no `<script`.
- `test_cost.py`: the four functions against hand-computed values, including `breakeven` returning infinity when the cascade costs more than the teacher.

---

## 3. Section 7 remainder: router store, canary, live comparison, retrain

### 3.1 Router store and warm start

```python
# agentdistill/router/store.py
from __future__ import annotations

from agentdistill.router.thompson import ArmState, ThompsonRouter


def load_router(registry, cfg: dict, cost: dict[str, float]) -> ThompsonRouter:
    state = {(row["cluster_id"], row["arm"]): ArmState(row["alpha"], row["beta"]) for row in registry.router_state()}
    r = ThompsonRouter(state, cost=cost, lam=cfg["router"]["lambda_per_usd"], floor=cfg["router"]["floor"],
                       explore_cap=cfg["router"]["explore_cap"], decay=cfg["router"]["decay"])
    if not state:
        counts = registry.per_cluster_counts(prod_adapter=registry.prod_adapter(), eval_set=cfg["eval"]["eval_set"])
        r.warm_start(counts)           # {(cluster, 'student'): (succ, fail), (cluster, 'teacher'): (succ, fail)}
        flush_router(registry, r)
    return r


def flush_router(registry, r: ThompsonRouter) -> None:
    registry.upsert_router_state([(c, arm, s.alpha, s.beta) for (c, arm), s in r.state.items()])
```

Warm start only runs when the table is empty; a retrain that promotes a new prod adapter calls `registry.reset_router_state()` and then `load_router`, so the posteriors restart from the new adapter's eval counts rather than inheriting the old adapter's history.

### 3.2 Canary

`GatewayState` gains `canary_adapter`, `canary_share`. When the router picks `student`, the adapter is the canary if `int(request_id[-2:], 16) % 100 < canary_share * 100`, else prod. `requests.adapter_id` records which. Both adapters must be loaded in vLLM as LoRA modules (`scripts/serve_vllm.sh` emits both when a canary exists).

### 3.3 Live comparison

```python
# agentdistill/router/compare_live.py
from __future__ import annotations

import numpy as np


def compare_live(rows: list[dict], prod: str, canary: str, iters: int = 5000, seed: int = 0) -> dict | None:
    """rows: requests with outcome, adapter_id, cluster_id, arm='student', fallback=False.
    Pairs by cluster: for each cluster with both adapters observed, the difference of success rates; cluster bootstrap over clusters."""
    by: dict[int, dict[str, list[bool]]] = {}
    for r in rows:
        if r["arm"] != "student" or r.get("fallback") or r["outcome"] is None or r["cluster_id"] is None:
            continue
        by.setdefault(r["cluster_id"], {}).setdefault(r["adapter_id"], []).append(bool(r["outcome"]))
    clusters = [c for c, d in by.items() if prod in d and canary in d and len(d[prod]) >= 5 and len(d[canary]) >= 5]
    if len(clusters) < 3:
        return None
    diff = np.array([np.mean(by[c][canary]) - np.mean(by[c][prod]) for c in clusters])
    w = np.array([min(len(by[c][prod]), len(by[c][canary])) for c in clusters], dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(clusters), size=(iters, len(clusters)))
    boots = (diff[idx] * w[idx]).sum(axis=1) / w[idx].sum(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"success": {"delta": float((diff * w).sum() / w.sum()), "ci95": (float(lo), float(hi))}, "n_clusters": len(clusters),
            "n_prod": int(sum(len(by[c][prod]) for c in clusters)), "n_canary": int(sum(len(by[c][canary]) for c in clusters))}
```

`adapter compare-live <prod> <canary> --since 7d` prints it. The lifecycle check `live_not_worse` (v0.3 section 7.3) reads this; `None` means not enough data and the check fails with "insufficient live traffic", which is the right answer.

### 3.4 Retrain stages, wired

The orchestrator from v0.3 section 7.4 exists; this wires real stage functions and gates.

| Stage | Runs | Gate |
|---|---|---|
| `ingest_gateway` | `ingest gateway --since <last retrain>` | at least 50 requests with outcomes |
| `curate` | `curate --config` with the same filters | at least 50 new samples in the new dataset version |
| `train_sft_continue` | one epoch at `lr / 3` from the prod adapter | eval loss finite |
| `onpolicy` | one round from the new adapter | round decision is `promote` |
| `eval` | frozen set, N=5 | `compare(prod, candidate).success.ci95[0] >= -0.01` |
| `calibrate` | fit and verify | holdout ECE at most 0.05 and AUROC at least 0.6 |
| `quantize` | configured method | quantized success within 2 pp of bf16 |
| `promote_canary` | `adapter promote --to canary` | every lifecycle check green |

`retrain --dry-run` prints the plan with counts; `retrain --from eval` resumes. The GitHub Actions workflow with a weekly cron is drafted under `.github/workflows/retrain.yml.draft` and renamed only after `serve_smoke.sh` has passed on real hardware; the runner label is `self-hosted, gpu`.

### 3.5 Tests

- `test_router_store.py`: empty table warm-starts from eval counts and flushes; non-empty table loads without warm start; reset then load restarts.
- `test_canary_split.py`: share 0.1 over 10,000 synthetic request ids lands within 9 to 11 percent, and the same id always maps to the same adapter.
- `test_compare_live.py`: seeded rows where canary is 5 pp better in every cluster produce a positive delta with a CI excluding zero; fewer than three qualifying clusters return `None`; fallback rows are ignored.
- `test_retrain_wired.py`: each gate against a fixture context, plus one full dry run.

---

## 4. Section 6.7: merge, quantize, serve scripts

### 4.1 Merge

```python
# agentdistill/train/merge.py
from __future__ import annotations

import json
import os


def merge_adapter(base_model: str, adapter_path: str, out_dir: str) -> dict:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16, device_map="auto")   # never merge into a 4-bit base
    merged = PeftModel.from_pretrained(model, adapter_path).merge_and_unload()
    os.makedirs(out_dir, exist_ok=True)
    merged.save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)
    with open(os.path.join(out_dir, "agentdistill_merge.json"), "w") as f:
        json.dump({"base_model": base_model, "adapter_path": adapter_path, "dtype": "bfloat16"}, f)
    return {"out_dir": out_dir, "dtype": "bfloat16"}
```

`adapter merge <id>` then runs teacher-forced next-action on 50 held-out turns with the merged weights and with the unmerged adapter and refuses to register the merged artifact if `full_match` differs by more than 2 points; that is the check that catches a wrong `target_modules` list or a merge into the wrong dtype. The merged model is registered as a new adapter row with `merged = true`, `parent_adapter_id` set.

### 4.2 Quantize

```python
# agentdistill/train/quantize.py
from __future__ import annotations

import os


def quantize_awq(merged_dir: str, out_dir: str, calib_prompts: list[str]) -> dict:
    """AWQ via llmcompressor. Calibration on real task prompts, not generic text. Verify API names against the installed version."""
    from llmcompressor import oneshot
    from llmcompressor.modifiers.awq import AWQModifier
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(merged_dir)
    model = AutoModelForCausalLM.from_pretrained(merged_dir, torch_dtype="auto", device_map="auto")
    ds = [{"text": p} for p in calib_prompts]
    oneshot(model=model, tokenizer=tok, dataset=ds, recipe=[AWQModifier(bits=4, symmetric=False, targets="Linear", ignore=["lm_head"])],
            max_seq_length=4096, num_calibration_samples=len(calib_prompts), output_dir=out_dir)
    return {"out_dir": out_dir, "method": "awq", "n_calib": len(calib_prompts)}


def quantize(method: str, merged_dir: str, out_dir: str, calib_prompts: list[str]) -> dict:
    if method == "fp8":
        # vLLM applies FP8 online from bf16 weights; nothing to write except a marker so the registry and serve script agree
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "QUANTIZATION"), "w") as f:
            f.write("fp8-online\n")
        return {"out_dir": merged_dir, "method": "fp8", "online": True}
    if method == "awq":
        return quantize_awq(merged_dir, out_dir, calib_prompts)
    raise ValueError(f"unknown quantization method {method}")
```

Calibration prompts: 128 rendered training prompts (system plus tools plus first user message) from the dataset, so the quantizer sees the tool schemas it will serve. `adapter quantize <id>` registers a new adapter row with `quantization` set and then runs `eval run` on it; the delta goes on the report.

### 4.3 Serve scripts

```bash
#!/usr/bin/env bash
# scripts/serve_vllm.sh   builds the vllm serve command from project.yaml and the registry
set -euo pipefail
CFG="${AGENTDISTILL_CONFIG:-project.yaml}"
BASE="$(agentdistill config get train.base_model)"
PARSER="$(agentdistill config get train.tool_parser.vllm)"
QUANT="$(agentdistill config get serve.quantization)"
MAXLEN="$(agentdistill config get serve.max_model_len 2>/dev/null || echo 16384)"
PROD="$(agentdistill adapter path --status prod 2>/dev/null || true)"
CANARY="$(agentdistill adapter path --status canary 2>/dev/null || true)"
LORA_ARGS=()
if [[ -n "$PROD" ]]; then LORA_ARGS+=(--lora-modules "prod=$PROD"); fi
if [[ -n "$CANARY" ]]; then LORA_ARGS+=(--lora-modules "canary=$CANARY"); fi
ARGS=(serve "$BASE" --enable-prefix-caching --max-model-len "$MAXLEN" --max-num-seqs 64 --enable-auto-tool-choice --tool-call-parser "$PARSER" --port 8000)
if [[ ${#LORA_ARGS[@]} -gt 0 ]]; then ARGS+=(--enable-lora --max-loras 4 --max-lora-rank 64 "${LORA_ARGS[@]}"); fi
if [[ "$QUANT" == "fp8" ]]; then ARGS+=(--quantization fp8); fi
echo "vllm ${ARGS[*]}"
exec vllm "${ARGS[@]}"
```

```bash
#!/usr/bin/env bash
# scripts/serve_smoke.sh   start vLLM and the gateway, run the example agent through both dialects, assert request-log rows, tear down
set -euo pipefail
cleanup() { kill "${VLLM_PID:-}" "${GW_PID:-}" 2>/dev/null || true; }
trap cleanup EXIT
bash scripts/serve_vllm.sh > logs/vllm.log 2>&1 & VLLM_PID=$!
for i in $(seq 1 60); do curl -sf http://127.0.0.1:8000/v1/models >/dev/null && break; sleep 5; done
curl -sf http://127.0.0.1:8000/v1/models >/dev/null || { echo "vllm did not start"; exit 1; }
agentdistill serve --port 8710 > logs/gateway.log 2>&1 & GW_PID=$!
for i in $(seq 1 30); do curl -sf http://127.0.0.1:8710/healthz >/dev/null && break; sleep 2; done
BEFORE="$(agentdistill requests count)"
python -m examples.support_agent.record --model "openai/cascade::auto" --base-url http://127.0.0.1:8710/v1 --n 5 --out /tmp/smoke_openai.jsonl
python -m examples.support_agent.record --model "anthropic/cascade::auto" --base-url http://127.0.0.1:8710 --n 5 --out /tmp/smoke_anthropic.jsonl
AFTER="$(agentdistill requests count)"
[[ $((AFTER - BEFORE)) -ge 10 ]] || { echo "expected at least 10 request-log rows, got $((AFTER - BEFORE))"; exit 1; }
agentdistill requests tail --n 10
echo "serve smoke ok"
```

`record.py` gains `--base-url` and treats an `openai/` or `anthropic/` model prefix as the dialect to speak. `requests count` and `requests tail` are two small CLI additions.

---

## 5. Tiny mode: the CPU rehearsal

This is the de-risking step for the GPU day. Every stage of `gpu_day.sh` runs on a laptop with a tiny model, five tasks, and N=1. The numbers are garbage; the execution path is real.

Changes:

- `AGENTDISTILL_TINY=1` makes `agentdistill config get` return overrides from `project.tiny.yaml`: `train.base_model` = the fixture tiny model already used in tests (or a public ~0.5B instruct model with a tool template), `eval.n_per_task: 1`, eval sets replaced by five-task subsets (`support-holdout-tiny`, `support-calib-tiny`, `support-unseen-tiny`), `train.epochs: 1`, `onpolicy.k_rollouts: 2`, `serve.quantization: fp8` (which is a marker only), `cascade.k_samples: 1`.
- `eval run --backend hf` works on CPU (it already exists for tests) and `gpu_day.sh` picks the backend from `AGENTDISTILL_EVAL_BACKEND` (default `vllm`).
- Stages that need vLLM or a GPU get a `tiny` behavior: `quantize` writes the fp8 marker; `serve_smoke` runs the gateway against the fake vLLM from the test suite (`AGENTDISTILL_FAKE_VLLM=1` starts it on port 8000) so the request-log assertion still runs.
- `stage env` in tiny mode only imports `torch`, `trl`, `peft`.

Patch to the script header:

```bash
if [[ "${AGENTDISTILL_TINY:-0}" == "1" ]]; then
  export AGENTDISTILL_CONFIG="examples/support_agent/project.tiny.yaml"
  export AGENTDISTILL_EVAL_BACKEND="hf"
  export AGENTDISTILL_FAKE_VLLM=1
  echo "== tiny mode: CPU rehearsal, numbers are not meaningful"
fi
BACKEND="${AGENTDISTILL_EVAL_BACKEND:-vllm}"
```

and every `--backend vllm` in the script becomes `--backend "$BACKEND"`.

Expected wall-clock on a laptop: 20 to 40 minutes, dominated by the two training stages. The output is `artifacts/gpu_day/report.html` with the warning block saying "tiny mode" (add that warning to `assemble` when the config name ends in `.tiny`).

The rehearsal will find things. Typical: a registry query that assumes a field only vLLM runs populate, a path that only exists after quantization, a report template key that is `None` in the DPO-discarded case. Fix each in the code, not in the script, and rerun with the markers cleared.

---

## 6. Docs

Written against built code only, in this order, each under 600 words:

- `docs/evaluation.md`: harness, replay policies, graders, paired statistics, what `eval compare` prints and how to read the CI; the judge calibration caveats from v0.3 section 1 with the n=400 numbers.
- `docs/cascade.md`: features (with the NaN convention), labels and their `how` mix, calibration on a disjoint split, threshold selection and harness verification, escalation and fallback semantics, the health alert.
- `docs/router.md`: clusters, Thompson sampling with the floor, warm start and reset on promotion, canary share, `compare-live`, what "insufficient live traffic" means.
- `docs/serving.md`: the serve scripts, quantization methods, the gateway's model-name policies, both dialects, streaming behavior including the buffered header, `/v1/feedback`.
- `docs/retrain.md`: stages and gates, `--dry-run` and `--from`, the workflow draft and when it gets renamed.
- `docs/results.md`: table headers with no numbers, and the sentence "Numbers appear here only via `agentdistill report --format md --inject`."

The README gets the results markers and nothing between them until the GPU day.

---

## 7. GPU day pre-flight

Do these the day before renting:

1. Tiny rehearsal green from a clean checkout: `rm -rf artifacts/gpu_day && AGENTDISTILL_TINY=1 bash scripts/gpu_day.sh`.
2. `agentdistill base-check <real base model>` on the laptop (tokenizer and template only; no weights download beyond the tokenizer) reports `ok` for the round trip.
3. Pin versions: `pip freeze` for `torch`, `transformers`, `trl`, `peft`, `vllm`, `llmcompressor` into `requirements-gpu.txt`; the GPU box installs from that file, not from ranges.
4. Teacher API key and budget: the eval and rollout stages call the teacher; set a hard spend cap on the key.
5. Object storage or a big disk for adapters and the merged model (an 8B merged bf16 model is about 16 GB; AWQ adds another 5 GB).
6. `git tag pre-gpu-day` so the logs branch has a clean parent.

On the day, the only commands you type are `pip install -r requirements-gpu.txt -e ".[train,serve]"` and `bash scripts/gpu_day.sh`. Anything else is a bug report for the next rehearsal.

---

## 8. Sequence

**Day 1**: canonical dedupe (section 1); `decide` tightening with the tokens CI; pair `diff_kind` and the text-only difference check; fallback counting and the health alert; migration 004 (`command` columns, `requests.fallback`); tests.
**Day 2**: `report/assemble.py`, registry additions (`per_cluster_table`, `lineage`, `commands_for`, throughput and token medians on eval runs), `report/markdown.py` with injection, tests.
**Day 3**: `report/html.py` and template with both SVGs; `report` CLI with `--format html|md --inject`; `router/store.py`, canary split, `compare_live`, tests.
**Day 4**: retrain stages wired with gates; workflow draft; `train/merge.py` with the equivalence check; `train/quantize.py`; `adapter merge` and `adapter quantize` CLI; tests with the tiny model for merge.
**Day 5**: `scripts/serve_vllm.sh`, `scripts/serve_smoke.sh`, `record.py --base-url`, `requests count|tail`; tiny mode overrides and `gpu_day.sh` patch; first tiny rehearsal, fix what breaks.
**Day 6**: second tiny rehearsal from a clean checkout; docs; pre-flight items 2 to 6; rent the GPU.

---

## 9. Definition of done

- One canonical implementation in the repo, 19 vectors green.
- `decide` requires a CI-backed cost improvement for the equal-success path.
- Fallbacks counted, excluded from router updates, surfaced on `/healthz` with an error-level log above 20 percent.
- `report` renders HTML and markdown from the registry; README injection is idempotent; no number without a run id.
- Router store, canary split, `compare-live`, and wired retrain stages tested.
- `adapter merge` refuses a merge that changes next-action accuracy by more than 2 points; `adapter quantize` registers and evaluates.
- Tiny rehearsal of `gpu_day.sh` completes from a clean checkout and produces a report.
- Docs written for every built feature; `docs/results.md` has headers and no numbers.
- Pre-flight list complete; GPU booked.
