# agentdistill: Phase 3 Plan

Milestones 4 through 8 built without a GPU, plus the GPU day that produces every number
Version 0.3, September 16, 2026
Builds on: v0.1 (full plan), v0.2 (next phase), docs/progress.md at Milestone 3 (555 tests, harness and bridge built, real traces recorded)

---

## 0. Where you are and what this phase produces

Built and tested: ingest, curate, dataset build, SFT trainer, template round trip, mask invariants, TRL compat, teacher-forced eval, 40 scenario shapes, 600 tasks, 408 training traces, 296 after curation, teacher success 84.3%, replay harness, replay-aware grading, paired statistics, judge calibration, and the three bridge functions for on-policy training.

Blocked on a GPU: the real SFT run (v0.2 section 2.6) and the base-versus-adapter-versus-teacher comparison (v0.2 section 4.9).

This phase has two halves that run in parallel:

**Half A, no GPU needed.** Every remaining component from v0.1 Milestones 4 to 8 is buildable and testable against stub backends and fixture responses: the DPO trainer and round loop, confidence features and calibration, the cascade, the gateway with both dialects, the router, the retrain orchestration, the cost model, and the report. Each ships with tests that run in CI on CPU. When a GPU arrives, nothing is written under time pressure; the GPU day only runs commands that already exist.

**Half B, one rented GPU session.** A single resumable script that trains, evaluates, collects rollouts, runs one on-policy round, collects logprobs, calibrates, quantizes, serves, and writes the report. Budget four to five hours of compute on an L40S or A100; rent eight.

The phase ends when the report from Half B exists, with run ids, and when every command in the CLI does something real.

---

## 1. Corrections and flags from the last round

Three things to fix before building on them, and one to document.

**1. The Rogan-Gladen exactness test is a tautology.** If sensitivity and specificity are estimated on the same labelled set the correction is applied to, the corrected rate equals the true rate by construction; error 0.00000 across seeds is arithmetic, not evidence. Keep the test as a sanity check but add the one that matters: estimate sensitivity and specificity on a labelled split, apply the correction to a disjoint split, and report the error with a bootstrap CI. Ship that number in `calibrate-judge` output as `holdout_error` alongside the in-sample one, and print a warning when the labelled set is under 100 because the CI will be wide.

**2. Judge calibration is a secondary path for this project.** The example uses predicates, so the judge is only exercised by users without state checks. Say so in `docs/evaluation.md`, and make sure `eval run` on the example never silently switches to the judge when a predicate is present.

**3. "Do nothing" scenarios need side-effect assertions.** A task whose right answer is to refuse must assert zero writes (no refund row, no ticket, no address change) in addition to the final message content. Audit the predicates for the 12 refusal-type shapes; a predicate that only checks the message will pass a student that refunds and then apologizes.

**Document the lesson from the curation pass**: fixed-message scenarios collapse under decontamination, varied natural-language openers and closers are part of scenario design, not decoration. Put this in `docs/curation.md` with the 48-to-19 number.

Naming: the recorder calls the teacher agent the "solver". Pick one word and use it everywhere in code, docs, and registry columns. This plan uses "teacher".

---

## 2. The GPU day, first

Everything in Half A exists to make this script boring. Write the script now, with stages that call commands that do not yet all exist, and fill it in as Half A lands. On the day, you run one command and read the log.

### 2.1 Runbook

Machine: one L40S (48 GB) or A100 (40 or 80 GB). An L4 or A10G (24 GB) works with QLoRA but every stage takes about twice as long. Ubuntu 22.04 or 24.04, CUDA 12.x, Python 3.11 or 3.12.

```bash
#!/usr/bin/env bash
# scripts/gpu_day.sh   resumable: each stage writes artifacts/gpu_day/<stage>.done and is skipped on rerun
set -euo pipefail
cd "$(dirname "$0")/.."
export AGENTDISTILL_CONFIG="${AGENTDISTILL_CONFIG:-examples/support_agent/project.yaml}"
mkdir -p artifacts/gpu_day logs
stage() { local name="$1"; shift; if [[ -f "artifacts/gpu_day/$name.done" ]]; then echo "== skip $name"; return; fi
  echo "== $name  $(date -u +%H:%M:%S)"; "$@" 2>&1 | tee "logs/gpu_day.$name.log"; touch "artifacts/gpu_day/$name.done"; }

stage env         bash -c 'pip install -e ".[train,serve,postgres]" && python -c "import torch,vllm,trl,peft;print(torch.cuda.get_device_name(0))"'
stage base_check  agentdistill base-check "$(agentdistill config get train.base_model)" --require-roundtrip
stage sft         agentdistill train sft "$(agentdistill dataset latest --kind sft --name support)" --tag gpu-day
stage merge       agentdistill adapter merge "$(agentdistill adapter latest --tag gpu-day)"
stage eval_base   agentdistill eval run base    --eval-set support-holdout-v1 --n 5 --policy strict --backend vllm
stage eval_sft    agentdistill eval run "$(agentdistill adapter latest --tag gpu-day)" --eval-set support-holdout-v1 --n 5 --policy strict --backend vllm
stage eval_teach  agentdistill eval run teacher --eval-set support-holdout-v1 --n 5 --policy strict
stage cmp_sft     agentdistill eval compare "$(agentdistill eval latest --subject base)" "$(agentdistill eval latest --subject-tag gpu-day)" --out artifacts/gpu_day/cmp_sft.json
stage onpolicy    agentdistill train onpolicy "$(agentdistill adapter latest --tag gpu-day)" --rounds 1 --tag gpu-day-r1
stage eval_r1     agentdistill eval run "$(agentdistill adapter latest --tag gpu-day-r1)" --eval-set support-holdout-v1 --n 5 --policy strict --backend vllm
stage unseen      agentdistill eval run "$(agentdistill adapter best --tag gpu-day*)" --eval-set support-unseen-v1 --n 5 --policy strict --backend vllm
stage logprobs    agentdistill eval run "$(agentdistill adapter best --tag gpu-day*)" --eval-set support-calib-v1 --n 3 --policy fuzzy --backend vllm --logprobs --samples 3
stage calibrate   agentdistill calibrate "$(agentdistill adapter best --tag gpu-day*)" --from-eval "$(agentdistill eval latest --eval-set support-calib-v1)"
stage cascade_ver agentdistill eval run "cascade:$(agentdistill adapter best --tag gpu-day*):auto" --eval-set support-holdout-v1 --n 3 --policy fuzzy --backend vllm --verify-threshold
stage quantize    agentdistill adapter quantize "$(agentdistill adapter best --tag gpu-day*)" --method "$(agentdistill config get serve.quantization)"
stage eval_quant  agentdistill eval run "$(agentdistill adapter latest --quantized)" --eval-set support-holdout-v1 --n 3 --policy strict --backend vllm
stage serve_smoke bash scripts/serve_smoke.sh
stage report      agentdistill report --out artifacts/gpu_day/report.html --include-run-ids
echo "== done  $(date -u +%H:%M:%S)"; ls -la artifacts/gpu_day
```

`support-calib-v1` is a third frozen eval set: 100 training-disjoint tasks used only for fitting the confidence gate (v0.1 section 8.4 forbids fitting on training tasks, and fitting on the holdout would leak into the reported cascade number). Create it now from the remaining scenario instances; if there are not enough, record 100 more tasks before the GPU day.

### 2.2 Expected durations on an L40S

| Stage | Estimate | Abort if |
|---|---|---|
| env | 10 min | vLLM and TRL import fails after pinning |
| base_check | 2 min | round trip not `ok` |
| sft (8B QLoRA, 296 samples, 2 epochs) | 25 to 40 min | eval loss rising from step 1, or next-action accuracy below base at the end |
| merge | 5 min | merged next-action differs from adapter by more than noise |
| eval_base / eval_sft (80 tasks x 5, vLLM offline, batched) | 8 to 12 min each | divergence rate above 40% in strict mode for the adapter |
| eval_teach (API) | 10 to 20 min | teacher success far from the recorded 84% (the harness or the eval set is broken) |
| onpolicy (300 tasks x 8 rollouts, RFT + DPO) | 40 to 60 min | fuzzy replay share above 50% |
| eval_r1, unseen | 10 min each | |
| logprobs (100 tasks x 3 repeats x 3 samples) | 10 to 15 min | |
| calibrate | 2 min | AUROC below 0.6 (the gate is not informative; escalate everything and say so) |
| cascade_ver (three thresholds) | 20 min | |
| quantize (AWQ) | 20 to 40 min; FP8 online: skip | quantized success drops more than 2 pp |
| serve_smoke | 10 min | |
| report | 1 min | |

Four to five hours of compute. Rent eight so a failed stage and a rerun fit. At current cloud rates that is on the order of ten to twenty dollars, plus the teacher's API spend for eval and any teacher-side escalation.

### 2.3 Serverless alternative

If renting a box is inconvenient, `deploy/modal/gpu_day.py` wraps each stage as a Modal function on a shared volume with the same `.done` markers. The script above becomes a driver that calls them in order. Build this only if the rented-box path fails twice; it is a distraction otherwise.

### 2.4 What the day must leave behind

- Registry rows for every training run, adapter, eval run, calibration, and the on-policy round, all tagged.
- `artifacts/gpu_day/report.html` and every `cmp_*.json`.
- `logs/gpu_day.*.log` committed to a `gpu-day-YYYYMMDD` branch (logs only, no weights).
- A one-paragraph `docs/results.md` with the headline numbers and the run ids, written the same day while the numbers are in front of you.

---

## 3. Registry additions (migration 002)

```sql
-- rollouts are traces with provenance
ALTER TABLE traces ADD COLUMN parent_adapter_id TEXT REFERENCES adapters(id);
ALTER TABLE traces ADD COLUMN repeat_idx INTEGER;
ALTER TABLE traces ADD COLUMN replay_policy TEXT;             -- 'strict' | 'fuzzy' for rollouts
ALTER TABLE traces ADD COLUMN fuzzy_hits INTEGER DEFAULT 0;
ALTER TABLE traces ADD COLUMN tag TEXT;
-- source check gains 'rollout'

CREATE TABLE onpolicy_rounds (
  id                 TEXT PRIMARY KEY,
  tag                TEXT,
  round_idx          INTEGER NOT NULL,
  start_adapter_id   TEXT NOT NULL REFERENCES adapters(id),
  rollout_eval_run   TEXT REFERENCES eval_runs(id),
  n_rollouts         INTEGER,
  fuzzy_share        REAL,
  rft_dataset_id     TEXT REFERENCES datasets(id),
  dpo_dataset_id     TEXT REFERENCES datasets(id),
  sft_run_id         TEXT REFERENCES training_runs(id),
  dpo_run_id         TEXT REFERENCES training_runs(id),
  candidate_adapter  TEXT REFERENCES adapters(id),
  eval_run_id        TEXT REFERENCES eval_runs(id),
  compare            JSONB,
  decision           TEXT CHECK (decision IN ('promote','discard','error')),
  reason             TEXT,
  started_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
  ended_at           TIMESTAMPTZ
);

ALTER TABLE eval_results ADD COLUMN logprobs_ref TEXT;         -- blob with per-turn raw choices when --logprobs
ALTER TABLE eval_results ADD COLUMN escalations INTEGER;       -- cascade subjects
ALTER TABLE eval_results ADD COLUMN wasted_student_tokens INTEGER;

ALTER TABLE calibrations ADD COLUMN holdout_metrics JSONB;     -- auroc, brier, ece on the disjoint split
ALTER TABLE calibrations ADD COLUMN reliability_bins JSONB;
ALTER TABLE calibrations ADD COLUMN verified JSONB;            -- harness-measured points at tau, tau-0.05, tau+0.05

ALTER TABLE adapters ADD COLUMN tag TEXT;
ALTER TABLE adapters ADD COLUMN parent_adapter_id TEXT REFERENCES adapters(id);
CREATE TABLE adapter_events (
  id          TEXT PRIMARY KEY,
  adapter_id  TEXT NOT NULL REFERENCES adapters(id),
  from_status TEXT,
  to_status   TEXT NOT NULL,
  checks      JSONB,                                          -- the promotion checks and their results
  actor       TEXT,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE cluster_models (
  id            TEXT PRIMARY KEY,
  dataset_id    TEXT NOT NULL REFERENCES datasets(id),
  embedder      TEXT NOT NULL,
  k             INTEGER NOT NULL,
  centroids_ref TEXT NOT NULL,                                -- blob: float32 [k, dim]
  labels        JSONB,                                        -- cluster id -> short label
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

Every `agentdistill ... latest --tag` and `adapter best --tag` selector in the runbook is a registry query; add them to `registry/select.py` with tests before anything else, because the GPU script depends on them.

---

## 4. Milestone 4: DPO and the on-policy round loop

### 4.1 DPO compat and pair rendering

TRL's conversational DPO format may or may not thread `tools` into the chat template depending on the release. Do not depend on it. Pre-render to the standard string format, which every TRL version accepts, and verify BOS handling once.

```python
# agentdistill/train/dpo_data.py
from __future__ import annotations

import json


def _render(tok, messages, tools, gen: bool) -> str:
    return tok.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=gen)


def render_pair(tok, pair: dict) -> dict:
    """pair: {prompt: [messages], chosen: [assistant msg], rejected: [assistant msg], tools: [...]}.
    Returns {prompt, chosen, rejected} as strings with the template applied, chosen/rejected excluding the prompt."""
    tools = pair["tools"]
    prompt = _render(tok, pair["prompt"], tools, gen=True)
    full_c = _render(tok, pair["prompt"] + pair["chosen"], tools, gen=False)
    full_r = _render(tok, pair["prompt"] + pair["rejected"], tools, gen=False)
    if not (full_c.startswith(prompt) and full_r.startswith(prompt)):
        raise ValueError("template not prefix-stable; cannot render DPO pair")
    return {"prompt": prompt, "chosen": full_c[len(prompt):], "rejected": full_r[len(prompt):],
            "task_id": pair.get("task_id"), "pair_kind": pair.get("pair_kind", "rollout")}


def pair_is_valid(pair: dict) -> tuple[bool, str]:
    if pair["chosen"] == pair["rejected"]:
        return False, "chosen equals rejected"
    if not pair["prompt"] or pair["prompt"][-1]["role"] == "assistant":
        return False, "prompt must end on a user or tool message"
    for side in ("chosen", "rejected"):
        m = pair[side][0]
        if m["role"] != "assistant":
            return False, f"{side} is not an assistant turn"
        for c in m.get("tool_calls") or []:
            try:
                json.loads(c["function"]["arguments"])
            except (json.JSONDecodeError, KeyError, TypeError):
                return False, f"{side} has malformed tool call arguments"
    return True, ""
```

```python
# agentdistill/train/compat.py  (additions)
def dpo_config_kwargs(cfg: dict, out_dir: str) -> dict:
    import dataclasses
    from trl import DPOConfig
    fields = {f.name for f in dataclasses.fields(DPOConfig)}
    want = {
        "output_dir": out_dir, "beta": cfg.get("dpo_beta", 0.1), "num_train_epochs": 1, "learning_rate": cfg.get("dpo_lr", 5e-6),
        "per_device_train_batch_size": 1, "gradient_accumulation_steps": 16, "bf16": True, "gradient_checkpointing": True,
        "logging_steps": 10, "seed": cfg["seed"], "report_to": ["tensorboard"], "loss_type": "sigmoid",
    }
    for cands, value in [(("max_length",), cfg["max_seq_len"]), (("max_prompt_length",), cfg["max_seq_len"] - 512)]:
        for c in cands:
            if c in fields:
                want[c] = value
                break
    unknown = [k for k in want if k not in fields]
    if unknown:
        raise RuntimeError(f"DPOConfig lacks fields {unknown}; update compat.py")
    return want
```

BOS check: after `DPOTrainer` builds its dataset, decode one `prompt_input_ids` and assert the BOS token appears once. If twice, the pre-rendered prompt already contained it; strip the leading BOS text from `render_pair` output for that tokenizer (`template_check` records whether the template emits BOS).

### 4.2 DPO trainer

```python
# agentdistill/train/dpo.py
from __future__ import annotations

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from agentdistill.train.compat import attn_implementation, dpo_config_kwargs
from agentdistill.train.dpo_data import render_pair


def train_dpo(cfg: dict, pairs: list[dict], base_or_merged: str, out_dir: str) -> dict:
    tok = AutoTokenizer.from_pretrained(base_or_merged)
    rendered = [render_pair(tok, p) for p in pairs]
    ds = Dataset.from_list(rendered)
    model = AutoModelForCausalLM.from_pretrained(base_or_merged, torch_dtype=torch.bfloat16, attn_implementation=attn_implementation(cfg))
    lora = LoraConfig(r=cfg.get("dpo_lora_r", 16), lora_alpha=cfg.get("dpo_lora_r", 16) * 2, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    args = DPOConfig(**dpo_config_kwargs(cfg, out_dir))
    trainer = DPOTrainer(model=model, ref_model=None, args=args, train_dataset=ds, processing_class=tok, peft_config=lora)
    trainer.train()
    trainer.save_model(out_dir)
    logs = [x for x in trainer.state.log_history if "rewards/margins" in x]
    return {"steps": trainer.state.global_step, "n_pairs": len(rendered),
            "final_reward_margin": logs[-1]["rewards/margins"] if logs else None,
            "final_reward_accuracy": logs[-1].get("rewards/accuracies") if logs else None}
```

Metrics to watch: `rewards/accuracies` should climb above 0.7 within the first epoch on rollout pairs. If it sits near 0.5, the pairs are not distinguishable (usually a pair builder bug: chosen and rejected differ only in whitespace, or the prompt is not shared) and the run is wasted.

CPU smoke test: `test_dpo_smoke.py` on the tiny fixture model, 4 pairs, 2 steps, asserts the adapter saves and the reward margin key exists.

### 4.3 The round loop

Stages are injected so the loop is testable with stubs; the real wiring lives in `cli.py`.

```python
# agentdistill/train/onpolicy.py
from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from typing import Callable


@dataclass
class RoundCfg:
    k_rollouts: int = 8
    rft_cap_per_task: int = 2
    replay_policy: str = "fuzzy"
    max_fuzzy_share: float = 0.5
    min_pairs: int = 40
    success_tolerance_pp: float = 1.0        # promote if success not worse than this AND cost improved, or success up with CI excluding 0
    schema_floor: float = 0.99
    divergence_slack_pp: float = 5.0


@dataclass
class RoundResult:
    round_idx: int
    start_adapter: str
    decision: str = "error"
    reason: str = ""
    n_rollouts: int = 0
    fuzzy_share: float = 0.0
    n_rft: int = 0
    n_pairs: int = 0
    candidate_adapter: str | None = None
    compare: dict = field(default_factory=dict)
    ids: dict = field(default_factory=dict)


@dataclass
class Stages:
    collect_rollouts: Callable[[str, list[str], int, str], dict]      # (adapter, task_ids, k, policy) -> {rollouts: [...], eval_run_id, fuzzy_share}
    build_rft: Callable[[list[dict], int], tuple[str, int]]           # -> (dataset_id, n)
    build_pairs: Callable[[list[dict], dict[str, dict]], tuple[str, int]]   # (rollouts, teacher_by_task) -> (dataset_id, n)
    train_sft_continue: Callable[[str, str], tuple[str, str]]         # (adapter, dataset_id) -> (training_run_id, adapter_id)
    merge: Callable[[str], str]                                       # adapter -> merged path or adapter id
    train_dpo: Callable[[str, str], tuple[str, str]]                  # (merged, dataset_id) -> (training_run_id, adapter_id)
    run_eval: Callable[[str], str]                                    # adapter -> eval_run_id
    compare: Callable[[str, str], dict]                               # (eval_a, eval_b) -> compare dict (a = current, b = candidate)
    metrics: Callable[[str], dict]                                    # eval_run_id -> aggregate metrics
    record: Callable[[dict], None]                                    # persist RoundResult


def decide(cmp: dict, cur: dict, cand: dict, cfg: RoundCfg) -> tuple[str, str]:
    lo, hi = cmp["success"]["ci95"]
    delta_pp = cmp["success"]["delta"] * 100
    if cand["schema_valid"] < cfg.schema_floor:
        return "discard", f"schema validity {cand['schema_valid']:.3f} below floor"
    if (cand["divergence_rate"] - cur["divergence_rate"]) * 100 > cfg.divergence_slack_pp:
        return "discard", "divergence rate regressed"
    cost_better = cmp["tokens"]["median_delta"] < 0
    if lo > 0:
        return "promote", f"success up {delta_pp:+.1f} pp, CI excludes zero"
    if lo <= 0 <= hi and delta_pp > -cfg.success_tolerance_pp and cost_better:
        return "promote", f"success within tolerance ({delta_pp:+.1f} pp) and tokens down {cmp['tokens']['median_delta']:.0f}"
    return "discard", f"success {delta_pp:+.1f} pp with CI [{lo*100:+.1f}, {hi*100:+.1f}], cost_better={cost_better}"


def run_round(round_idx: int, adapter: str, train_task_ids: list[str], teacher_by_task: dict[str, dict],
              current_eval_run: str, st: Stages, cfg: RoundCfg) -> RoundResult:
    r = RoundResult(round_idx=round_idx, start_adapter=adapter, ids={"round_id": uuid.uuid4().hex})
    try:
        roll = st.collect_rollouts(adapter, train_task_ids, cfg.k_rollouts, cfg.replay_policy)
        r.n_rollouts, r.fuzzy_share, r.ids["rollout_eval_run"] = len(roll["rollouts"]), roll["fuzzy_share"], roll["eval_run_id"]
        if r.fuzzy_share > cfg.max_fuzzy_share:
            r.decision, r.reason = "discard", f"fuzzy replay share {r.fuzzy_share:.2f} above {cfg.max_fuzzy_share}"
            return r
        rft_id, r.n_rft = st.build_rft(roll["rollouts"], cfg.rft_cap_per_task)
        pairs_id, r.n_pairs = st.build_pairs(roll["rollouts"], teacher_by_task)
        r.ids.update(rft_dataset=rft_id, dpo_dataset=pairs_id)
        if r.n_pairs < cfg.min_pairs:
            r.decision, r.reason = "discard", f"only {r.n_pairs} pairs (< {cfg.min_pairs})"
            return r
        sft_run, sft_adapter = st.train_sft_continue(adapter, rft_id)
        merged = st.merge(sft_adapter)
        dpo_run, cand = st.train_dpo(merged, pairs_id)
        r.candidate_adapter = cand
        r.ids.update(sft_run=sft_run, sft_adapter=sft_adapter, dpo_run=dpo_run)
        cand_eval = st.run_eval(cand)
        r.ids["eval_run"] = cand_eval
        r.compare = st.compare(current_eval_run, cand_eval)
        r.decision, r.reason = decide(r.compare, st.metrics(current_eval_run), st.metrics(cand_eval), cfg)
        return r
    except Exception as e:  # the round is recorded as an error, never silently dropped
        r.decision, r.reason = "error", f"{type(e).__name__}: {e}"
        raise
    finally:
        st.record(asdict(r))


def run_rounds(adapter: str, rounds: int, train_task_ids, teacher_by_task, current_eval_run: str, st: Stages, cfg: RoundCfg) -> list[RoundResult]:
    out: list[RoundResult] = []
    cur_adapter, cur_eval = adapter, current_eval_run
    for i in range(rounds):
        r = run_round(i, cur_adapter, train_task_ids, teacher_by_task, cur_eval, st, cfg)
        out.append(r)
        if r.decision == "promote" and r.candidate_adapter:
            cur_adapter, cur_eval = r.candidate_adapter, r.ids["eval_run"]
        else:
            break                                  # a discarded round means the next one would repeat it
    return out
```

Notes:

- `collect_rollouts` uses `VllmOfflineTurnClient` with `temperature=0.8`, `top_p=0.95`; add a `sampling` argument to the client. Rollouts are written as traces with `source="rollout"`, `parent_adapter_id`, `repeat_idx`, `replay_policy`, `fuzzy_hits`.
- `build_pairs` produces two kinds, tagged in `pair_kind`: `rollout` (student success versus student failure on the same task) and `teacher` (teacher success versus student failure). Cap teacher pairs to the same count as rollout pairs; a DPO set dominated by teacher pairs is just SFT with a worse loss.
- `train_sft_continue` trains one epoch at `lr / 3` on the RFT set starting from the current adapter's weights (load the adapter, do not start a new LoRA), then `merge`.
- The comparison in `decide` is candidate versus **current adapter**, not versus teacher. Teacher comparisons are for the report.

Tests (`test_onpolicy_loop.py`): a `Stages` built from stubs that return scripted compare dicts. Cases: promote on clear improvement, promote on tolerance-plus-cost, discard on schema floor, discard on fuzzy share, discard on too few pairs, error propagates and is recorded, two rounds where round 2 uses round 1's candidate, a discarded round stops the loop.

### 4.4 CLI

`train onpolicy <adapter> --rounds N --tag T` replaces the exit-2 stub. It prints one block per round: rollout counts, fuzzy share, pair counts by kind, DPO reward accuracy, the compare table, and the decision with its reason. `--dry-run` prints the stage plan with the task counts and stops.

---

## 5. Milestone 5: confidence features, calibration, cascade

### 5.1 Data collection

`eval run <adapter> --logprobs --samples 3` records, per assistant turn, the raw OpenAI-style choice objects (with `logprobs.content[]` tokens, `top_logprobs`) for the primary sample and the extra samples, to a blob referenced by `eval_results.logprobs_ref`. The vLLM offline client needs `SamplingParams(n=3, logprobs=5)` and a conversion of vLLM's output objects to the OpenAI shape so the feature code has one input format. Write `eval/vllm_shape.py` for that conversion and test it against a fixture from a real vLLM run (capture one on the GPU day and check it in).

### 5.2 Argument token mask

Features need to know which generated tokens belong to tool-call argument JSON.

```python
# agentdistill/cascade/arg_mask.py
from __future__ import annotations

import json


def token_spans(tokens: list[str]) -> list[tuple[int, int]]:
    spans, pos = [], 0
    for t in tokens:
        spans.append((pos, pos + len(t)))
        pos += len(t)
    return spans


def arg_char_spans(text: str, tool_calls: list[dict]) -> list[tuple[int, int]]:
    """Find each tool call's argument JSON inside the generated text. Tries the exact serialized string first,
    then a compact re-serialization, then the argument values one by one."""
    out: list[tuple[int, int]] = []
    cursor = 0
    for c in tool_calls or []:
        raw = c["function"]["arguments"]
        candidates = [raw] if isinstance(raw, str) else []
        try:
            obj = json.loads(raw) if isinstance(raw, str) else raw
            candidates += [json.dumps(obj, separators=(",", ":")), json.dumps(obj)]
        except (json.JSONDecodeError, TypeError):
            obj = None
        found = None
        for cand in candidates:
            i = text.find(cand, cursor)
            if i >= 0:
                found = (i, i + len(cand))
                break
        if found is None and isinstance(obj, dict):
            for v in obj.values():
                s = json.dumps(v) if not isinstance(v, str) else v
                i = text.find(s, cursor)
                if i >= 0:
                    out.append((i, i + len(s)))
            continue
        if found is not None:
            out.append(found)
            cursor = found[1]
    return out


def arg_token_mask(tokens: list[str], text: str, tool_calls: list[dict]) -> list[bool]:
    joined = "".join(tokens)
    if joined != text:
        # some servers return token strings that do not concatenate to the text (byte fallbacks, stripped spaces);
        # fall back to aligning on the joined string, which is what the token spans index anyway
        text = joined
    spans = token_spans(tokens)
    arg_spans = arg_char_spans(text, tool_calls)
    return [any(s < b and e > a for a, b in arg_spans) for s, e in spans]
```

Test: a fixture choice with known tokens; the mask covers exactly the tokens of `{"order_id": "o_1", "amount": 20}` and nothing in the surrounding template.

### 5.3 Turn labels

```python
# agentdistill/cascade/labels.py
from __future__ import annotations

import json

from agentdistill.canonical import args_hash


def _sig(m: dict) -> frozenset:
    out = set()
    for c in m.get("tool_calls") or []:
        a = c["function"]["arguments"]
        out.add(args_hash(c["function"]["name"], json.loads(a) if isinstance(a, str) else a))
    return frozenset(out)


def _prefix_key(messages: list[dict], upto: int) -> str:
    """Prefix identity includes tool names and canonical argument hashes: a call with different arguments leads to a
    different state, so the teacher's next turn is not a valid reference for it."""
    return json.dumps([(m["role"], m.get("content"), sorted(_sig(m))) for m in messages[:upto]], sort_keys=True)


def label_turns(rollout_messages: list[dict], task_success: bool, teacher_messages: list[dict] | None) -> list[tuple[int, bool, str]]:
    """Returns (turn_index, good, how) per assistant turn of the rollout.
    good = task succeeded AND (matches the teacher on the same prefix when the prefix exists in the teacher trace,
           else the turn was not corrected later)."""
    teacher_by_prefix: dict[str, dict] = {}
    if teacher_messages:
        for i, m in enumerate(teacher_messages):
            if m["role"] == "assistant":
                teacher_by_prefix[_prefix_key(teacher_messages, i)] = m
    out = []
    assistant_idx = [i for i, m in enumerate(rollout_messages) if m["role"] == "assistant"]
    for n, i in enumerate(assistant_idx):
        m = rollout_messages[i]
        if not task_success:
            out.append((i, False, "task_failed"))
            continue
        t = teacher_by_prefix.get(_prefix_key(rollout_messages, i))
        if t is not None:
            same = _sig(m) == _sig(t) and bool(m.get("tool_calls")) == bool(t.get("tool_calls"))
            out.append((i, same, "teacher_match" if same else "teacher_mismatch"))
            continue
        # no teacher reference for this prefix: corrected later?
        later = rollout_messages[i + 1:]
        corrected = False
        names = {c["function"]["name"] for c in (m.get("tool_calls") or [])}
        # a tool error right after this turn, or the same tool called again later with different args, counts as corrected
        for j, lm in enumerate(later):
            if lm["role"] == "tool" and j < 2 * max(len(names), 1) and '"error"' in (lm.get("content") or ""):
                corrected = True
                break
            if lm["role"] == "assistant" and names and ({c["function"]["name"] for c in (lm.get("tool_calls") or [])} & names) and _sig(lm) != _sig(m):
                corrected = True
                break
        out.append((i, not corrected, "uncorrected" if not corrected else "corrected_later"))
    return out
```

The `how` field is kept so the calibration report can show how many labels came from each rule. If most labels are `uncorrected`, the label set is weak and the gate's AUROC should be read with that in mind.

### 5.4 Calibration with a disjoint split

Extend v0.1's `calibrate.py`: split turns by task (70/30), fit on the first split, report AUROC, Brier, ECE, and ten reliability bins on the second. Store both in `calibrations.holdout_metrics` and `reliability_bins`. Refit on all data for the artifact that ships, but the report shows the holdout numbers.

Feature vector order is fixed by `cascade.features` in `project.yaml` and stored on the calibration row; the runtime asserts the same order before scoring.

### 5.5 Threshold selection and harness verification

`choose_threshold` from v0.1 section 8.5 runs on the holdout split's turn probabilities and produces `tau`. Then the subject `cascade:<adapter>:auto` resolves `auto` to `tau` and `--verify-threshold` runs the harness at `tau - 0.05`, `tau`, `tau + 0.05` with N=3, writing `calibrations.verified` with measured success, escalation rate, wasted tokens, and cost per task at each. The report uses the verified points; the analytic curve is drawn faintly behind them.

### 5.6 Cascade as a harness client

```python
# agentdistill/cascade/client.py
from __future__ import annotations

import json
from typing import Any, Protocol

import numpy as np

from agentdistill.cascade.arg_mask import arg_token_mask
from agentdistill.cascade.features import as_vector, turn_features


class SamplingBackend(Protocol):
    def chat(self, messages: list[dict], tools: list[dict], n: int, logprobs: bool) -> list[dict]:
        """Return n OpenAI-style choice dicts: {message: {...}, logprobs: {content: [...]}, text: str}."""


class CascadeTurnClient:
    """TurnClient for the harness: student first, teacher when the gate says so. Records what happened per turn."""

    def __init__(self, student: SamplingBackend, teacher: SamplingBackend, calibrator: Any, feature_names: list[str],
                 threshold: float, k_samples: int = 2, cluster_prior: float = 0.5):
        self.student, self.teacher, self.cal, self.names, self.tau, self.k = student, teacher, calibrator, feature_names, threshold, k_samples
        self.cluster_prior = cluster_prior
        self.log: list[dict] = []

    def next_turn(self, messages: list[dict], tools: list[dict]) -> dict:
        choices = self.student.chat(messages, tools, n=1 + self.k, logprobs=True)
        primary, extra = choices[0], choices[1:]
        tokens = [t["token"] for t in primary["logprobs"]["content"]]
        mask = arg_token_mask(tokens, primary.get("text", "".join(tokens)), primary["message"].get("tool_calls") or [])
        turn_idx = sum(1 for m in messages if m["role"] == "assistant")
        prefix_tokens = sum(len(json.dumps(m)) for m in messages) // 4
        f = turn_features(primary, mask, extra, self.cluster_prior, turn_idx, prefix_tokens)
        p = float(self.cal.predict_proba(as_vector(f, self.names)[None, :])[0, 1])
        if p >= self.tau:
            self.log.append({"arm": "student", "p": p, "escalated": False, "student_tokens": len(tokens)})
            return primary["message"]
        t = self.teacher.chat(messages, tools, n=1, logprobs=False)[0]
        self.log.append({"arm": "teacher", "p": p, "escalated": True, "student_tokens": len(tokens)})
        return t["message"]

    def summary(self) -> dict:
        n = len(self.log)
        esc = [x for x in self.log if x["escalated"]]
        return {"turns": n, "escalations": len(esc), "escalation_rate": len(esc) / n if n else 0.0,
                "wasted_student_tokens": sum(x["student_tokens"] for x in esc),
                "mean_p": float(np.mean([x["p"] for x in self.log])) if n else float("nan")}
```

The harness's `run_task` needs one addition: after the task, if the client has `summary()`, copy `escalations` and `wasted_student_tokens` onto the `TaskOutcome`, and reset the log per task.

Tests: stub backends with canned choices and a calibrator stub that returns fixed probabilities; assert escalation routing, the summary counts, and that the harness records escalations on the outcome.

---

## 6. Milestone 6: the gateway

Everything here is testable on CPU against a fake vLLM.

### 6.1 Fake vLLM for tests

```python
# tests/fake_vllm.py
from __future__ import annotations

import json
import uuid
from collections import deque

from fastapi import FastAPI

app = FastAPI()
QUEUE: deque[dict] = deque()      # tests push scripted assistant messages here
CALLS: list[dict] = []


def _tokens(text: str) -> list[dict]:
    out = []
    for w in text.split(" "):
        tok = w + " "
        out.append({"token": tok, "logprob": -0.2, "top_logprobs": [{"token": tok, "logprob": -0.2}, {"token": "x", "logprob": -3.0}]})
    if out:
        out[-1]["token"] = out[-1]["token"].rstrip(" ")
    return out


@app.post("/v1/chat/completions")
async def chat(body: dict):
    CALLS.append(body)
    n = body.get("n", 1)
    choices = []
    for i in range(n):
        msg = QUEUE.popleft() if QUEUE else {"role": "assistant", "content": "ok"}
        text = (msg.get("content") or "") + "".join(
            f'<tool_call>{json.dumps({"name": c["function"]["name"], "arguments": json.loads(c["function"]["arguments"])})}</tool_call>'
            for c in (msg.get("tool_calls") or []))
        choice = {"index": i, "message": msg, "finish_reason": "tool_calls" if msg.get("tool_calls") else "stop"}
        if body.get("logprobs"):
            choice["logprobs"] = {"content": _tokens(text)}
        choices.append(choice)
    return {"id": uuid.uuid4().hex, "object": "chat.completion", "model": body["model"], "choices": choices,
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}}


@app.get("/v1/models")
async def models():
    return {"data": [{"id": "student"}, {"id": "student:support-v1"}]}
```

A pytest fixture starts it with uvicorn on a free port (or mounts it via `httpx.ASGITransport` for in-process speed) and points the gateway's student backend at it.

### 6.2 Backends

```python
# agentdistill/gateway/backends.py
from __future__ import annotations

from typing import Any

import httpx


class StudentBackend:
    """OpenAI-compatible endpoint (vLLM serve). Returns raw choice dicts plus the generated text."""

    def __init__(self, base_url: str, client: httpx.AsyncClient | None = None, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.http = client or httpx.AsyncClient(timeout=timeout)

    async def chat(self, messages: list[dict], tools: list[dict], model: str, n: int = 1, logprobs: bool = False,
                   temperature: float = 0.0, extra: dict[str, Any] | None = None) -> dict:
        body: dict[str, Any] = {"model": model, "messages": messages, "n": n, "temperature": temperature}
        if tools:
            body["tools"] = tools
        if logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = 5
        if extra:
            body.update(extra)
        r = await self.http.post(f"{self.base_url}/v1/chat/completions", json=body)
        r.raise_for_status()
        data = r.json()
        for c in data["choices"]:
            c.setdefault("text", "".join(t["token"] for t in (c.get("logprobs") or {}).get("content", [])))
        return data


class TeacherBackend:
    """Any provider through LiteLLM; returns the same shape as StudentBackend for the cascade's benefit."""

    def __init__(self, model: str, **litellm_kwargs: Any):
        self.model, self.kw = model, litellm_kwargs

    async def chat(self, messages: list[dict], tools: list[dict], n: int = 1, temperature: float = 0.0) -> dict:
        import litellm
        r = await litellm.acompletion(model=self.model, messages=messages, tools=tools or None, n=n, temperature=temperature, **self.kw)
        return r.model_dump()
```

### 6.3 Dialects

Requests arrive in either dialect; internally everything is OpenAI-shaped; responses go back in the dialect they arrived in.

```python
# agentdistill/gateway/dialect.py
from __future__ import annotations

import json
import time
import uuid

from agentdistill.ingest.normalize import anthropic_to_openai


def from_anthropic_request(body: dict) -> dict:
    conv = anthropic_to_openai(body.get("system"), body["messages"], body.get("tools") or [])
    return {"model": body["model"], "messages": conv["messages"], "tools": conv["tools"],
            "max_tokens": body.get("max_tokens", 1024), "temperature": body.get("temperature", 0.0), "stream": bool(body.get("stream"))}


def to_anthropic_response(choice: dict, model: str, usage: dict) -> dict:
    m = choice["message"]
    content: list[dict] = []
    if m.get("content"):
        content.append({"type": "text", "text": m["content"]})
    for c in m.get("tool_calls") or []:
        a = c["function"]["arguments"]
        content.append({"type": "tool_use", "id": c.get("id") or f"toolu_{uuid.uuid4().hex[:12]}", "name": c["function"]["name"],
                        "input": json.loads(a) if isinstance(a, str) else a})
    stop = "tool_use" if m.get("tool_calls") else ("max_tokens" if choice.get("finish_reason") == "length" else "end_turn")
    return {"id": f"msg_{uuid.uuid4().hex[:16]}", "type": "message", "role": "assistant", "model": model, "content": content,
            "stop_reason": stop, "stop_sequence": None,
            "usage": {"input_tokens": usage.get("prompt_tokens", 0), "output_tokens": usage.get("completion_tokens", 0)}}


def from_openai_request(body: dict) -> dict:
    return {"model": body["model"], "messages": body["messages"], "tools": body.get("tools") or [],
            "max_tokens": body.get("max_tokens") or body.get("max_completion_tokens") or 1024,
            "temperature": body.get("temperature", 0.0), "stream": bool(body.get("stream"))}


def to_openai_response(choice: dict, model: str, usage: dict) -> dict:
    return {"id": f"chatcmpl-{uuid.uuid4().hex[:16]}", "object": "chat.completion", "created": int(time.time()), "model": model,
            "choices": [{"index": 0, "message": choice["message"], "finish_reason": choice.get("finish_reason", "stop")}],
            "usage": usage}
```

Round-trip test: an Anthropic-style request with a `tool_result` block becomes an OpenAI `tool` message; an OpenAI choice with `tool_calls` becomes Anthropic `tool_use` blocks with `stop_reason: tool_use`; the official Anthropic and OpenAI Python SDKs parse the gateway's responses without error (contract test, section 6.6).

### 6.4 Model-name policies

```python
# agentdistill/gateway/resolve.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Route:
    mode: str                 # 'teacher' | 'student' | 'cascade' | 'router'
    adapter: str | None = None
    threshold: float | None = None


def resolve(model: str, prod_adapter: str | None, prod_threshold: float | None, teacher_names: set[str]) -> Route:
    if model == "teacher" or model in teacher_names and prod_adapter is None:
        return Route("teacher")
    if model.startswith("student"):
        _, _, adapter = model.partition(":")
        return Route("student", adapter or prod_adapter)
    if model.startswith("cascade:"):
        _, adapter, tau = model.split(":", 2)
        return Route("cascade", adapter or prod_adapter, prod_threshold if tau in ("", "auto") else float(tau))
    if model in teacher_names:
        return Route("router", prod_adapter, prod_threshold)
    raise ValueError(f"unknown model name {model}")
```

When the agent keeps sending its usual teacher model name, the router decides; `student:*` and `cascade:*` are for evals; `teacher` is passthrough.

### 6.5 App

```python
# agentdistill/gateway/app.py
from __future__ import annotations

import time
import uuid

from fastapi import FastAPI, HTTPException, Request

from agentdistill.gateway.dialect import from_anthropic_request, from_openai_request, to_anthropic_response, to_openai_response
from agentdistill.gateway.resolve import resolve
from agentdistill.gateway.state import GatewayState

app = FastAPI()
gw: GatewayState = GatewayState.uninitialized()


async def _handle(req: dict) -> tuple[dict, dict, dict]:
    started = time.time()
    route = resolve(req["model"], gw.prod_adapter, gw.prod_threshold, gw.teacher_names)
    cluster = gw.clusters.assign(req["messages"]) if gw.clusters else None
    meta: dict = {"id": uuid.uuid4().hex, "cluster_id": cluster, "route": route.mode, "adapter": route.adapter}
    if route.mode == "router":
        arm = gw.router.choose(cluster) if cluster is not None else "teacher"
        route = route if arm == "student" else resolve("teacher", None, None, gw.teacher_names)
        meta["arm"] = arm
        if arm == "student":
            route.mode = "cascade"
    if route.mode == "teacher":
        data = await gw.teacher.chat(req["messages"], req["tools"], temperature=req["temperature"])
        choice, usage = data["choices"][0], data["usage"]
        meta.update(arm="teacher", escalated=False, teacher_tokens=usage.get("completion_tokens", 0))
    elif route.mode == "student":
        data = await gw.student.chat(req["messages"], req["tools"], model=f"student:{route.adapter}" if route.adapter else "student", temperature=req["temperature"])
        choice, usage = data["choices"][0], data["usage"]
        meta.update(arm="student", escalated=False, student_tokens=usage.get("completion_tokens", 0))
    else:
        choice, usage, cmeta = await gw.cascade.turn(req["messages"], req["tools"], route.adapter, route.threshold, cluster)
        meta.update(cmeta)
    meta["latency_ms"] = int((time.time() - started) * 1000)
    await gw.log.write(meta, req, choice, usage)
    return choice, usage, meta


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    if body.get("stream"):
        return await gw.stream_openai(from_openai_request(body))
    choice, usage, meta = await _handle(from_openai_request(body))
    out = to_openai_response(choice, body["model"], usage)
    out["agentdistill"] = {"request_id": meta["id"], "arm": meta.get("arm"), "escalated": meta.get("escalated")}
    return out


@app.post("/v1/messages")
async def messages(request: Request):
    body = await request.json()
    if body.get("stream"):
        return await gw.stream_anthropic(from_anthropic_request(body))
    choice, usage, meta = await _handle(from_anthropic_request(body))
    return to_anthropic_response(choice, body["model"], usage)


@app.post("/v1/feedback")
async def feedback(body: dict):
    ok = await gw.log.set_outcome(body["request_id"], bool(body["success"]))
    if not ok:
        raise HTTPException(404, "unknown request_id")
    rec = await gw.log.get(body["request_id"])
    if rec.get("cluster_id") is not None and rec.get("arm") in ("student", "teacher"):
        gw.router.update(rec["cluster_id"], rec["arm"], bool(body["success"]))
        await gw.router_store.flush(gw.router)
    return {"ok": True}


@app.get("/healthz")
async def healthz():
    return {"ok": True, "prod_adapter": gw.prod_adapter, "threshold": gw.prod_threshold}
```

`GatewayState` loads: the prod and canary adapters from the registry, the calibrator artifact for the prod adapter, the cluster model, the router state, the backends, and the request log store. `stream_openai` and `stream_anthropic` implement SSE for teacher passthrough and student-only; for cascade routes they buffer the decided turn and emit it as a short stream, with a header `x-agentdistill-buffered: true` so clients can tell.

### 6.6 Contract tests

`test_gateway_contract.py`: the official `openai` and `anthropic` Python packages, pointed at the gateway (ASGI transport), make a tool-calling request in each dialect against the fake vLLM and a stub teacher; assert the SDKs parse tool calls, `stop_reason`/`finish_reason`, and usage. Run the same test with `stream=True` for the passthrough and student routes.

`test_gateway_cascade.py`: scripted fake-vLLM choices with low logprobs trigger escalation to the stub teacher; the request log row shows `escalated=true` and `student_tokens` counted as wasted; `/v1/feedback` updates the router state.

### 6.7 Serving artifacts

- `scripts/serve_vllm.sh` builds the `vllm serve` command from `project.yaml`: base model, LoRA modules for prod and canary from the registry, quantization, tool parser (the same `train.tool_parser` value the round trip used), prefix caching, max model length.
- `scripts/serve_smoke.sh`: starts vLLM and the gateway in the background, waits for `/healthz`, runs the example agent for five tasks against the gateway in both dialects, asserts five request-log rows, kills both.
- `adapter quantize --method awq` uses `llmcompressor` (or AutoAWQ if that is what installs cleanly) on the merged model, then `eval run` on the result; `--method fp8` records that vLLM's online FP8 will be used and skips the offline step. Either way `adapters.quantization` is set and the eval delta versus bf16 lands in the report.
- `docker-compose.yml`: vllm (GPU), gateway, postgres with pgvector, tensorboard. The compose file is only written when `serve_smoke.sh` passes on a real GPU; a compose file nobody has run is worse than none.

---

## 7. Milestone 7: router, canary, retrain loop

### 7.1 Clusters at serving time

```python
# agentdistill/router/clusters.py
from __future__ import annotations

import hashlib
from functools import lru_cache

import numpy as np


class ClusterAssigner:
    def __init__(self, centroids: np.ndarray, embed, labels: dict[int, str] | None = None):
        self.c = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-9)
        self.embed, self.labels = embed, labels or {}

    @staticmethod
    def task_text(messages: list[dict]) -> str:
        first_user = next((m.get("content") or "" for m in messages if m["role"] == "user"), "")
        system = next((m.get("content") or "" for m in messages if m["role"] == "system"), "")
        return hashlib.sha256(system.encode()).hexdigest()[:8] + " " + first_user[:2000]

    def assign(self, messages: list[dict]) -> int:
        return self._assign_text(self.task_text(messages))

    @lru_cache(maxsize=10_000)
    def _assign_text(self, text: str) -> int:
        v = np.asarray(self.embed([text])[0], dtype=float)
        v = v / (np.linalg.norm(v) + 1e-9)
        return int(np.argmax(self.c @ v))
```

Centroids come from the curation stratification step, persisted via `cluster_models` (migration 002). The router's per-cluster warm start uses the per-cluster success counts from the prod adapter's latest eval run and the teacher's.

### 7.2 Router persistence and canary

`router/store.py` loads `router_state` into the `ThompsonRouter` from v0.1 and flushes on update. Canary: `GatewayState` holds `canary_adapter` and `canary_share`; when the router picks `student`, the cascade uses the canary adapter for `canary_share` of requests (hash of request id modulo 100), logged in `requests.adapter_id`. `agentdistill adapter compare-live <prod> <canary> --since 7d` computes the paired-by-cluster success difference from the request log with the same bootstrap and prints it; that is the promotion evidence for live traffic.

### 7.3 Adapter lifecycle

```python
# agentdistill/registry/lifecycle.py
from __future__ import annotations

TRANSITIONS = {("candidate", "canary"), ("canary", "prod"), ("candidate", "prod"), ("canary", "retired"), ("prod", "retired"), ("candidate", "retired")}


def promotion_checks(registry, adapter_id: str, to: str, cfg: dict) -> dict:
    a = registry.adapter(adapter_id)
    ev = registry.latest_eval_for(adapter_id, eval_set=cfg["eval"]["eval_set"])
    checks: dict[str, dict] = {}
    checks["has_eval"] = {"ok": ev is not None, "detail": ev["id"] if ev else "no eval on the frozen set"}
    if ev:
        checks["schema_valid"] = {"ok": ev["metrics"]["schema_valid"] >= 0.99, "value": ev["metrics"]["schema_valid"]}
        prod = registry.prod_adapter()
        if prod and prod != adapter_id:
            cmp = registry.compare_latest(prod, adapter_id, eval_set=cfg["eval"]["eval_set"])
            lo = cmp["success"]["ci95"][0] * 100 if cmp else None
            checks["not_worse_than_prod"] = {"ok": lo is not None and lo >= -1.0, "ci_low_pp": lo}
            checks["cost_not_worse"] = {"ok": cmp is not None and cmp["tokens"]["median_delta"] <= 0, "median_delta": cmp["tokens"]["median_delta"] if cmp else None}
    if to in ("canary", "prod"):
        cal = registry.latest_calibration(adapter_id)
        checks["calibrated"] = {"ok": cal is not None and (cal["holdout_metrics"] or {}).get("ece", 1.0) <= 0.05, "detail": cal["id"] if cal else None}
    if to == "prod" and a["status"] == "canary":
        live = registry.compare_live(registry.prod_adapter(), adapter_id, since_days=cfg.get("canary_days", 7))
        checks["live_not_worse"] = {"ok": live is not None and live["success"]["ci95"][0] * 100 >= -1.0, "detail": live}
    return checks


def transition(registry, adapter_id: str, to: str, cfg: dict, actor: str, force: bool = False) -> dict:
    a = registry.adapter(adapter_id)
    if (a["status"], to) not in TRANSITIONS:
        raise ValueError(f"illegal transition {a['status']} -> {to}")
    checks = promotion_checks(registry, adapter_id, to, cfg)
    ok = all(c["ok"] for c in checks.values())
    if not ok and not force:
        return {"ok": False, "checks": checks}
    if to == "prod":
        for other in registry.adapters_with_status("prod"):
            if other != adapter_id:
                registry.set_status(other, "retired", checks={"reason": f"superseded by {adapter_id}"}, actor=actor)
    registry.set_status(adapter_id, to, checks=checks | {"forced": not ok}, actor=actor)
    return {"ok": True, "checks": checks}
```

`adapter promote <id> --to canary|prod [--force]` prints the checks table and refuses unless every check is green or `--force` is given, in which case the event records that it was forced.

### 7.4 Retrain orchestration

```python
# agentdistill/retrain.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class Stage:
    name: str
    run: Callable[[dict], dict]        # takes and returns a context dict
    gate: Callable[[dict], tuple[bool, str]] | None = None


def run_pipeline(stages: list[Stage], ctx: dict, marker, log, start_from: str | None = None, dry_run: bool = False) -> dict:
    started = start_from is None
    for s in stages:
        if not started:
            started = s.name == start_from
            if not started:
                log(f"skip {s.name} (before --from)")
                continue
        if marker.done(s.name) and start_from is None:
            log(f"skip {s.name} (done)")
            continue
        if dry_run:
            log(f"would run {s.name}")
            continue
        log(f"run {s.name}")
        ctx = s.run(ctx)
        if s.gate:
            ok, why = s.gate(ctx)
            if not ok:
                log(f"stop at {s.name}: {why}")
                ctx["stopped_at"], ctx["stop_reason"] = s.name, why
                return ctx
        marker.mark(s.name)
    return ctx
```

Stages for `agentdistill retrain`: `ingest_gateway` (requests with outcomes since the last retrain), `curate` (same config, new dataset version; gate: at least 50 new samples), `train_sft_continue` (from prod adapter), `onpolicy` (one round), `eval` (frozen set; gate: not worse than prod within 1 pp), `calibrate` (gate: holdout ECE at most 0.05), `quantize`, `promote_canary`. Markers are per retrain id so a second `retrain` starts fresh. `--dry-run` and `--from <stage>` behave as in the GPU script. A GitHub Actions workflow file with a weekly cron calls it on a self-hosted GPU runner; the workflow is committed only after `serve_smoke.sh` has passed on real hardware.

---

## 8. Milestone 8: cost model and report

### 8.1 Inputs

- Verified cascade points from `calibrations.verified` (three thresholds: success, escalation, wasted tokens).
- Analytic curve over the threshold grid from `choose_threshold`.
- Teacher per-task cost from the eval run's tokens and `model_pricing`.
- Student cost per million tokens from `serve.gpu_usd_per_hour` and the measured throughput of the eval run (tokens per second from `eval_runs` timing), with the utilization assumption from config.
- Per-cluster success for base, student, and teacher from the eval runs.
- Calibration reliability bins and holdout metrics.
- Lineage: dataset ids, filters, training runs, rounds, quantization.

### 8.2 Report

`report/html.py` assembles a dict and renders `report/templates/report.html.j2` with inline SVG (no JavaScript, no external assets, so the file can be attached to an email). Sections in order: headline (cost per task before and after, success with CI, escalation rate), the cost-versus-success curve with the verified points bold and the analytic curve faint, the reliability diagram, per-cluster table, quantization delta, break-even tasks per day for the configured GPU, lineage with run ids, and a "how to reproduce" block listing the exact commands.

```python
# agentdistill/report/svg.py
from __future__ import annotations


def line_chart(points: list[tuple[float, float]], w: int = 520, h: int = 300, pad: int = 40, stroke: str = "#888", marks: list[tuple[float, float]] | None = None) -> str:
    xs, ys = [p[0] for p in points] + [m[0] for m in (marks or [])], [p[1] for p in points] + [m[1] for m in (marks or [])]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    sx = lambda x: pad + (x - x0) / (x1 - x0 + 1e-9) * (w - 2 * pad)
    sy = lambda y: h - pad - (y - y0) / (y1 - y0 + 1e-9) * (h - 2 * pad)
    path = " ".join(f"{'M' if i == 0 else 'L'}{sx(x):.1f},{sy(y):.1f}" for i, (x, y) in enumerate(sorted(points)))
    dots = "".join(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="5" fill="#222"/>' for x, y in (marks or []))
    axes = f'<line x1="{pad}" y1="{h-pad}" x2="{w-pad}" y2="{h-pad}" stroke="#444"/><line x1="{pad}" y1="{pad}" x2="{pad}" y2="{h-pad}" stroke="#444"/>'
    return f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">{axes}<path d="{path}" fill="none" stroke="{stroke}" stroke-width="1.5"/>{dots}</svg>'
```

`agentdistill report --out report.html --include-run-ids` writes the file and prints the headline block to the terminal. The README's numbers section is generated from the same dict by `agentdistill report --format md` so the README and the report can never disagree.

### 8.3 README numbers policy

Numbers appear in the README only with the eval run ids that produced them and the date. The generated markdown block is the only way numbers get in. This is a rule for you, not for users, and it is the reason the report exists.

---

## 9. Tests to add this phase

| Area | Test | Asserts |
|---|---|---|
| Registry | `test_select.py` | `latest --tag`, `best --tag`, `latest --subject` return the right rows on a seeded registry |
| Registry | `test_lifecycle.py` | legal and illegal transitions, promotion checks table, force records `forced: true`, promoting to prod retires the old prod |
| DPO | `test_dpo_data.py` | `render_pair` prefix stability, `pair_is_valid` rejects equal sides and bad JSON, BOS appears once after TRL processing (skipped without TRL) |
| DPO | `test_dpo_smoke.py` | tiny model, 4 pairs, 2 steps, adapter saves |
| On-policy | `test_onpolicy_loop.py` | the eight cases in section 4.3 |
| On-policy | `test_pairs_kinds.py` | rollout and teacher pairs tagged and capped |
| Cascade | `test_arg_mask.py` | mask covers exactly the argument JSON tokens on three fixture choices |
| Cascade | `test_labels.py` | teacher_match, teacher_mismatch, corrected_later, uncorrected, task_failed each produced by a hand-built rollout |
| Cascade | `test_calibrate_holdout.py` | disjoint split, holdout metrics present, reliability bins sum to n |
| Cascade | `test_cascade_client.py` | routing by threshold, summary counts, harness copies escalations onto the outcome |
| Judge | `test_rogan_gladen_holdout.py` | corrected rate on a disjoint split is within the bootstrap CI of truth; warning below 100 labels |
| Gateway | `test_dialect.py` | Anthropic and OpenAI round trips |
| Gateway | `test_resolve.py` | every model-name policy |
| Gateway | `test_gateway_contract.py` | official SDKs parse responses in both dialects, streaming and not |
| Gateway | `test_gateway_cascade.py` | escalation logged, wasted tokens counted, feedback updates router |
| Router | `test_clusters.py` | assignment is stable, cached, and matches the curation-time assignment on the training set |
| Router | `test_router_live_compare.py` | paired-by-cluster comparison from a seeded request log |
| Retrain | `test_retrain_pipeline.py` | markers skip, `--from`, gate stops, dry run runs nothing |
| Report | `test_report.py` | renders from a fixture dict; SVG present; markdown block has run ids |
| Scripts | `test_gpu_day_script.sh` | bash `-n` syntax check and a dry run with every CLI call replaced by `echo` |

---

## 10. Day-by-day (Half A)

**Day 1**: section 1 corrections (holdout Rogan-Gladen test, refusal predicates audit, naming); migration 002; `registry/select.py` with tests; `scripts/gpu_day.sh` with every stage present (commands may not exist yet; the dry-run test replaces them with `echo`).
**Day 2**: `train/dpo_data.py`, `compat.dpo_config_kwargs`, `train/dpo.py`, `test_dpo_data.py`, `test_dpo_smoke.py`.
**Day 3**: `train/onpolicy.py` with `Stages`, `decide`, `run_rounds`; `test_onpolicy_loop.py`; `train onpolicy` CLI with `--dry-run`; pair kinds and caps.
**Day 4**: `eval run --logprobs --samples`, `eval/vllm_shape.py`, `cascade/arg_mask.py`, `cascade/labels.py`, tests.
**Day 5**: calibration holdout split, reliability bins, `calibrate` CLI, `cascade/client.py`, harness escalation fields, `cascade:<adapter>:auto` subject with `--verify-threshold`.
**Day 6**: `tests/fake_vllm.py`, `gateway/backends.py`, `gateway/dialect.py`, `gateway/resolve.py`, tests.
**Day 7**: `gateway/app.py`, `gateway/state.py`, request log store, feedback endpoint, contract tests, cascade tests.
**Day 8**: streaming for passthrough and student routes; `scripts/serve_vllm.sh`, `scripts/serve_smoke.sh`, `adapter quantize` (offline path guarded by an import check).
**Day 9**: `router/clusters.py`, `router/store.py`, canary share, `adapter compare-live`, `registry/lifecycle.py`, `adapter promote`, tests.
**Day 10**: `retrain.py` with stages, gates, markers, `--dry-run`, `--from`; workflow file drafted but not committed; tests.
**Day 11**: `report/cost.py` (from v0.1), `report/svg.py`, `report/html.py`, template, `report --format md`; tests.
**Day 12**: `docs/evaluation.md`, `docs/cascade.md`, `docs/router.md`, `docs/serving.md` (each written against built code only), `docs/results.md` stub with the table headers and no numbers. Final dry run of `gpu_day.sh`. Rent the GPU.

Half B is section 2. If it lands before Day 12, run whatever stages exist and rerun the rest later; the markers make that free.

---

## 11. Things that will go wrong

| Symptom | Likely cause | Fix |
|---|---|---|
| DPO `rewards/accuracies` stuck near 0.5 | Pairs not distinguishable, or prompt not shared | `pair_is_valid` plus a test that chosen and rejected differ in tool name or canonical args, not only whitespace |
| DPO loss NaN in the first steps | Learning rate too high for a merged bf16 model, or double BOS | 5e-6, check BOS once, `max_grad_norm=1.0` |
| Round 1 discards on fuzzy share | Student argument phrasing drifted; canonical rules too strict for this tool set | Inspect the nearest-match scores; add per-tool normalizers; lower the threshold with the predicate as the check |
| Gate AUROC under 0.6 | Labels mostly `uncorrected`, or features carry no signal without `agreement` | Check the label mix; enable `k_samples=2`; if still flat, ship with "escalate everything" and say so in the report |
| Cascade verified success far below the analytic estimate | Analytic model assumes escalated turns are as good as the teacher's; a teacher turn on a student-built prefix is not | Expected to some degree; the verified number is the one reported; lower tau or raise `k_samples` |
| vLLM token strings do not concatenate to the text | Byte-fallback tokens | `arg_token_mask` already aligns on the joined string; add the fixture from the GPU day to the tests |
| Gateway contract test fails on `tool_call.id` | Anthropic SDK requires `toolu_` ids of a certain shape, or OpenAI SDK requires `type: function` | The dialect code sets both; check the SDK versions in the failing test |
| `serve_smoke.sh` hangs | vLLM startup longer than the health timeout | Wait for `/v1/models` on vLLM before `/healthz` on the gateway; 5 minute timeout for model load |
| Quantized adapter drops more than 2 pp | AWQ calibration set unrelated to the task | Calibrate AWQ on 128 training prompts from the dataset, not on generic text |
| Report cost numbers look absurd | Throughput measured with N=1 unbatched | Measure tokens per second from the batched eval run, and print the measurement conditions next to the number |

---

## 12. Definition of done for this phase

- Every CLI command does something real; no `exit 2` stubs remain.
- `pytest` passes on CPU with the fake vLLM, including the contract tests with both official SDKs.
- `scripts/gpu_day.sh` passes its dry run.
- One GPU day has run to completion: report with run ids, `docs/results.md`, logs on a branch.
- The report shows: base, SFT, and on-policy adapters versus the teacher on holdout and unseen sets; the verified cascade point; calibration holdout metrics; the quantization delta; per-cluster table; cost per task before and after; break-even tasks per day.
- An adapter has legitimately reached `canary` through `adapter promote` with every check green. If none can, the report says which check failed and why, and that is still a complete phase.
