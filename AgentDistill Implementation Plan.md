# agentdistill

## Implementation Plan and Build Guide

Version 0.1, September 2026
Owner: Pramod
Companion project: agentreplay (trace format, replay harness, paired statistics)

---

## 0. The promise

> Point it at your production agent's traces. Get back a small model that handles the routine calls, a calibrated escalation gate that sends the rest to the frontier model, and a cost report you can hand to a CFO.

Who it is for: teams running an agent on a frontier model who are paying for the top 5 percent of difficulty on 100 percent of calls. Forward deployed engineers who need to say "we cut inference spend 60 percent and success rate moved by less than one point, here is the confidence interval."

### What the pipeline does, end to end

```
traces  ->  curate  ->  dataset  ->  SFT (LoRA)  ->  DPO / RFT  ->  eval vs teacher
                                                                        |
        gateway (drop-in base_url)  <-  router + cascade  <-  calibrate confidence
                |
        vLLM serving (LoRA, prefix cache, guided JSON, quantized)
                |
        request log  ->  retrain loop  ->  adapter registry  ->  canary  ->  promote
```

### Non-goals for v1

- Training models above ~8B parameters or multi-node training
- Pretraining or continued pretraining
- A web UI beyond a static HTML report
- Reinforcement learning with learned reward models (verifiable rewards only)

### Design principles

1. **The eval harness is the ground truth.** Nothing is promoted on loss curves. Every claim is a paired comparison against the teacher on held-out tasks with a confidence interval.
2. **Drop-in.** The agent code does not change. It points `base_url` at the gateway and keeps its model name. Routing, cascade, and escalation are invisible to it.
3. **Calibrated or silent.** The escalation gate publishes its reliability diagram, ECE, and Brier score. If calibration data is missing, the gate defaults to "escalate everything" and says so.
4. **On-policy before scale.** Exposure bias kills agent distillation. The student is trained on its own rollouts (rejection sampling and DPO) before any claim of parity is made.
5. **Config-driven.** One YAML per project defines sources, filters, base model, training, eval, cascade, and serving. Every run is reproducible from the config plus a dataset hash.

### A warning that belongs at the top

Provider terms of service govern whether you may use a model's outputs to train another model. Several frontier providers restrict using outputs to build competing models. Distilling your own agent's traces for your own internal cost reduction is a different case from building a competing product, but it is not automatically permitted, and the answer differs by provider and by contract. Check the terms for the teacher model before training, put the check in the README, and support open-weight teachers (a large open model as teacher, small open model as student) as a first-class path so the project is useful regardless.

---

## 1. Architecture

```
+------------------+   +------------------+   +------------------+   +------------------+
| Sources          |   | Curation         |   | Datasets         |   | Training         |
| agentreplay db   |-->| normalize        |-->| SFT samples      |-->| TRL + PEFT       |
| OTel / Langfuse  |   | schema-validate  |   | DPO pairs        |   | LoRA / QLoRA     |
| JSONL            |   | dedupe, decontam |   | loss masks       |   | optional Unsloth |
| gateway log      |   | filter, stratify |   | parquet + hash   |   | adapter artifact |
+------------------+   +------------------+   +------------------+   +--------+---------+
                                                                              |
+------------------+   +------------------+   +------------------+            v
| Gateway          |   | Cascade          |   | Calibration      |   +------------------+
| OpenAI-compat    |<--| student first    |<--| logprob feats    |<--| Eval harness     |
| Anthropic-compat |   | escalate if p<t  |   | self-consistency |   | agentreplay or   |
| router (bandit)  |   | teacher fallback |   | isotonic         |   | built-in mocked  |
| request log      |   |                  |   | threshold search |   | paired stats     |
+--------+---------+   +------------------+   +------------------+   +------------------+
         |
         v
+------------------+   +------------------+
| vLLM             |   | Registry         |
| multi-LoRA       |   | datasets         |
| prefix caching   |   | training runs    |
| guided JSON      |   | adapters         |
| FP8 / AWQ        |   | calibrations     |
+------------------+   | router state     |
                       +------------------+
```

### Stack

| Layer | Choice | Why |
|---|---|---|
| Core | Python 3.12 | Training and serving ecosystems are Python |
| Training | transformers, TRL, PEFT, bitsandbytes, optional Unsloth | Standard, well maintained, LoRA and QLoRA out of the box |
| Data | datasets, pyarrow, datasketch (MinHash LSH) | Parquet datasets with content hashes, near-duplicate detection |
| Eval | agentreplay harness when present, built-in mocked-tool harness otherwise | Same paired statistics either way |
| Calibration | scikit-learn (logistic, isotonic), numpy, scipy | No custom math where a library exists |
| Serving | vLLM with LoRA, prefix caching, guided decoding, quantization | Best throughput per dollar on one GPU |
| Gateway | FastAPI, httpx, LiteLLM for teacher calls | OpenAI and Anthropic compatible endpoints |
| Registry | SQLite (local), Postgres (team) via SQLAlchemy core | Same pattern as agentreplay |
| Embeddings | pluggable; default provider API, optional local sentence-transformers | Task clustering for router context |
| Cloud | Docker Compose; examples for one GPU on AWS (g6e / L40S), GCP (L4 / A100), and Modal | One-GPU story first |
| CLI | Typer + Rich | Same as agentreplay |

---

## 2. Data model

### 2.1 Entities

- **trace**: one agent trajectory in normalized OpenAI-style message format with tool schemas, outcome, cost, and source.
- **dataset**: a versioned, hashed set of samples produced from traces by a filter config. Immutable once written.
- **sample**: one training example (SFT trajectory, SFT turn window, or DPO pair) with a reference to its parquet row.
- **training_run**: base model, method, config, metrics, resulting adapter path.
- **adapter**: a versioned artifact with lifecycle status: candidate, canary, prod, retired.
- **eval_run**: an adapter (or the teacher) evaluated on an eval set; metrics and the paired comparison.
- **calibration**: a fitted confidence model for an adapter, its threshold, and its reliability metrics.
- **router_state**: per-cluster Beta posteriors per arm.
- **requests**: gateway log; every request with cluster, arm, confidence, escalation, tokens, cost, and (when known) outcome. This is the source for the next training round.

### 2.2 Postgres DDL

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE traces (
  id             TEXT PRIMARY KEY,
  source         TEXT NOT NULL,            -- 'agentreplay' | 'otel' | 'langfuse' | 'jsonl' | 'gateway'
  source_ref     TEXT,                     -- run id or file path
  task_id        TEXT,
  task_input     JSONB,
  messages       JSONB NOT NULL,           -- normalized OpenAI-style messages incl. tool_calls and role=tool
  tools          JSONB NOT NULL,           -- OpenAI function schemas
  teacher_model  TEXT,
  success        BOOLEAN,
  grader         TEXT,                     -- 'label' | 'exact' | 'llm_judge' | ...
  score          REAL,
  n_turns        INTEGER,
  n_tool_calls   INTEGER,
  prompt_tokens  INTEGER,
  completion_tokens INTEGER,
  cost_usd       NUMERIC(12,6),
  content_hash   TEXT NOT NULL,            -- sha256 of canonicalized messages
  cluster        INTEGER,
  created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (content_hash)
);
CREATE INDEX traces_task ON traces(task_id);
CREATE INDEX traces_success ON traces(success);
CREATE INDEX traces_cluster ON traces(cluster);

CREATE TABLE trace_embeddings (
  trace_id   TEXT PRIMARY KEY REFERENCES traces(id) ON DELETE CASCADE,
  model      TEXT NOT NULL,
  embedding  vector(1024) NOT NULL
);
CREATE INDEX trace_embeddings_hnsw ON trace_embeddings USING hnsw (embedding vector_cosine_ops);

CREATE TABLE datasets (
  id            TEXT PRIMARY KEY,
  name          TEXT NOT NULL,
  version       INTEGER NOT NULL,
  kind          TEXT NOT NULL CHECK (kind IN ('sft','dpo','eval')),
  filter_config JSONB NOT NULL,
  n_samples     INTEGER NOT NULL,
  n_tokens      BIGINT,
  content_hash  TEXT NOT NULL,             -- sha256 over sorted sample hashes
  path          TEXT NOT NULL,             -- parquet directory
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (name, version)
);

CREATE TABLE samples (
  id          TEXT PRIMARY KEY,
  dataset_id  TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
  trace_id    TEXT REFERENCES traces(id),
  kind        TEXT NOT NULL CHECK (kind IN ('trajectory','turn_window','dpo_pair')),
  n_tokens    INTEGER,
  n_target_tokens INTEGER,                 -- unmasked tokens
  row_idx     INTEGER NOT NULL,
  sample_hash TEXT NOT NULL
);
CREATE INDEX samples_dataset ON samples(dataset_id);

CREATE TABLE training_runs (
  id           TEXT PRIMARY KEY,
  dataset_id   TEXT NOT NULL REFERENCES datasets(id),
  base_model   TEXT NOT NULL,
  method       TEXT NOT NULL CHECK (method IN ('sft','dpo','rft','grpo')),
  parent_adapter_id TEXT,
  config       JSONB NOT NULL,
  metrics      JSONB,                      -- train/eval loss curve summary, throughput
  adapter_path TEXT,
  status       TEXT NOT NULL CHECK (status IN ('running','succeeded','failed')),
  started_at   TIMESTAMPTZ NOT NULL,
  ended_at     TIMESTAMPTZ
);

CREATE TABLE adapters (
  id               TEXT PRIMARY KEY,
  training_run_id  TEXT NOT NULL REFERENCES training_runs(id),
  name             TEXT NOT NULL,
  version          INTEGER NOT NULL,
  base_model       TEXT NOT NULL,
  merged           BOOLEAN NOT NULL DEFAULT FALSE,
  quantization     TEXT,                   -- 'fp8' | 'awq' | 'gptq' | NULL
  path             TEXT NOT NULL,
  status           TEXT NOT NULL CHECK (status IN ('candidate','canary','prod','retired')),
  created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (name, version)
);

CREATE TABLE eval_sets (
  id         TEXT PRIMARY KEY,
  name       TEXT NOT NULL,
  trace_ids  TEXT[] NOT NULL,              -- held-out tasks with recorded tool results
  grader     JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE eval_runs (
  id            TEXT PRIMARY KEY,
  eval_set_id   TEXT NOT NULL REFERENCES eval_sets(id),
  subject       TEXT NOT NULL,             -- adapter id, 'teacher', or 'cascade:<adapter>:<threshold>'
  n_per_task    INTEGER NOT NULL,
  metrics       JSONB NOT NULL,            -- success, schema_valid, tool_select_acc, arg_match, tokens, latency
  paired        JSONB,                     -- vs teacher: delta, ci95, mcnemar p
  per_cluster   JSONB,
  started_at    TIMESTAMPTZ,
  ended_at      TIMESTAMPTZ
);

CREATE TABLE calibrations (
  id           TEXT PRIMARY KEY,
  adapter_id   TEXT NOT NULL REFERENCES adapters(id),
  eval_run_id  TEXT NOT NULL REFERENCES eval_runs(id),
  features     TEXT[] NOT NULL,
  model_path   TEXT NOT NULL,              -- pickled sklearn pipeline
  threshold    REAL NOT NULL,
  target       JSONB NOT NULL,             -- {"max_success_drop_pp": 1.0}
  ece          REAL,
  brier        REAL,
  auroc        REAL,
  escalation_rate REAL,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE router_state (
  cluster_id  INTEGER NOT NULL,
  arm         TEXT NOT NULL,               -- 'student' | 'teacher'
  alpha       REAL NOT NULL DEFAULT 1,
  beta        REAL NOT NULL DEFAULT 1,
  updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (cluster_id, arm)
);

CREATE TABLE requests (
  id              TEXT PRIMARY KEY,
  received_at     TIMESTAMPTZ NOT NULL,
  cluster_id      INTEGER,
  arm             TEXT NOT NULL,
  adapter_id      TEXT,
  confidence      REAL,
  escalated       BOOLEAN NOT NULL DEFAULT FALSE,
  student_tokens  INTEGER,
  teacher_tokens  INTEGER,
  cost_usd        NUMERIC(12,6),
  latency_ms      INTEGER,
  outcome         BOOLEAN,                 -- filled later by grader or label
  trace_id        TEXT REFERENCES traces(id)
);
CREATE INDEX requests_time ON requests(received_at);
CREATE INDEX requests_outcome ON requests(outcome) WHERE outcome IS NOT NULL;

CREATE TABLE model_pricing (
  provider              TEXT NOT NULL,
  model                 TEXT NOT NULL,
  input_per_mtok        NUMERIC(10,4) NOT NULL,
  output_per_mtok       NUMERIC(10,4) NOT NULL,
  cache_read_per_mtok   NUMERIC(10,4),
  effective_from        DATE NOT NULL,
  PRIMARY KEY (provider, model, effective_from)
);
```

SQLite variant follows the same substitutions as agentreplay: JSONB to TEXT, arrays to JSON text, vectors computed in Python for local mode.

### 2.3 Normalized message format

Every source is converted to one shape so curation, tokenization, and eval never branch on provider:

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "I will look that up.",
     "tool_calls": [{"id": "call_1", "type": "function",
                     "function": {"name": "search_orders", "arguments": "{\"customer_id\": \"c_9\"}"}}]},
    {"role": "tool", "tool_call_id": "call_1", "content": "[{\"order_id\": \"o_1\", \"status\": \"shipped\"}]"},
    {"role": "assistant", "content": "Your order o_1 shipped yesterday."}
  ],
  "tools": [{"type": "function", "function": {"name": "search_orders", "description": "...", "parameters": {...}}}]
}
```

Anthropic content blocks (`tool_use`, `tool_result`, `text`) are converted in `normalize.py`; the converter is bijective enough that the gateway can answer in either dialect.

```python
# agentdistill/ingest/normalize.py
from __future__ import annotations

import json
from typing import Any


def anthropic_to_openai(system: str | list | None, messages: list[dict], tools: list[dict]) -> dict:
    out: list[dict] = []
    if system:
        text = system if isinstance(system, str) else "".join(b.get("text", "") for b in system)
        out.append({"role": "system", "content": text})
    for m in messages:
        content = m["content"]
        if isinstance(content, str):
            out.append({"role": m["role"], "content": content})
            continue
        if m["role"] == "assistant":
            text = "".join(b["text"] for b in content if b["type"] == "text")
            calls = [
                {"id": b["id"], "type": "function",
                 "function": {"name": b["name"], "arguments": json.dumps(b["input"], ensure_ascii=False)}}
                for b in content if b["type"] == "tool_use"
            ]
            msg: dict[str, Any] = {"role": "assistant", "content": text or None}
            if calls:
                msg["tool_calls"] = calls
            out.append(msg)
        else:  # user turn: may hold tool_result blocks and/or text
            for b in content:
                if b["type"] == "tool_result":
                    c = b["content"]
                    if isinstance(c, list):
                        c = "".join(x.get("text", "") for x in c)
                    out.append({"role": "tool", "tool_call_id": b["tool_use_id"], "content": c})
                elif b["type"] == "text":
                    out.append({"role": "user", "content": b["text"]})
    fn_tools = [
        {"type": "function", "function": {"name": t["name"], "description": t.get("description", ""),
                                          "parameters": t["input_schema"]}}
        for t in tools
    ]
    return {"messages": out, "tools": fn_tools}
```

### 2.4 Ingest sources

| Source | Command | Notes |
|---|---|---|
| agentreplay store | `agentdistill ingest agentreplay --db .agentreplay/agentreplay.db --since 30d` | Reads runs, llm_calls, tool_calls; reconstructs the trajectory from the last llm_call's request plus response; pulls success from eval_results or labels |
| JSONL | `agentdistill ingest jsonl traces.jsonl` | One normalized trace per line; validated against `schemas/trace.schema.json` |
| OTel / Langfuse export | `agentdistill ingest otel export.json` | Requires content on spans; warns loudly on truncation |
| Gateway log | `agentdistill ingest gateway --since 7d` | The retrain loop's source; only requests with a known outcome |

Every ingest computes `content_hash`, rejects exact duplicates, and stores `teacher_model` so a dataset can be restricted to one teacher.

---

## 3. Curation

Garbage traces produce a confident, wrong student. Curation is where most of the quality comes from, and it runs as a pipeline of named filters so the dataset's `filter_config` explains exactly what was kept.

### 3.1 Filters, in order

| Filter | Rule | Default |
|---|---|---|
| `outcome` | keep `success = true` for SFT; keep both for DPO | on |
| `schema_valid` | every tool call's arguments parse as JSON and validate against the tool's `parameters` schema | on |
| `no_error_loops` | drop traces with 3 or more consecutive tool errors or 2 or more identical repeated tool calls | on |
| `length` | 2 to 40 assistant turns; total tokens within `max_seq_len` for trajectory samples | on |
| `teacher` | restrict to one teacher model | off |
| `exact_dedupe` | drop duplicate `content_hash` | on |
| `near_dedupe` | MinHash LSH proposes candidates; each candidate pair is verified with the MinHash Jaccard estimate against the threshold before a drop. LSH banding is approximate and returns pairs below threshold. | on |
| `decontaminate` | drop any trace whose task_input shares an exact match or >= 50 percent 8-gram overlap with any eval set task | on |
| `pii` | run the redaction hook; drop traces where redaction changed a tool argument (the model must not learn placeholder tokens as valid args) | on |
| `quality_judge` | optional: judge rates trajectory efficiency and correctness 1 to 5; drop < 3 | off |
| `stratify` | cluster task inputs (k-means over embeddings, K from config); cap samples per cluster at `cap_per_cluster`; report coverage | on |

### 3.2 Near-duplicate detection

> **As built.** LSH proposes; the threshold disposes. `MinHashLSH.query` is deliberately loose and returns
> candidate pairs well below the threshold it was constructed with, so every candidate is verified with the
> MinHash Jaccard estimate before a drop. Dropping candidates unverified would discard traces the configured rule
> says to keep.
>
> `near_dedupe_normalize_literals` (mask numbers, ids, and timestamps before shingling) is **off by default**. It
> collapses short corpora to roughly one sample per trajectory shape, because once ids are masked every trace of
> a given shape is identical. Turn it on only for corpora with long assistant text.


```python
# agentdistill/curate/dedupe.py
from __future__ import annotations

from datasketch import MinHash, MinHashLSH


def _shingles(text: str, n: int = 5) -> set[str]:
    toks = text.split()
    return {" ".join(toks[i:i + n]) for i in range(max(len(toks) - n + 1, 1))}


def assistant_text(trace: dict) -> str:
    parts = []
    for m in trace["messages"]:
        if m["role"] == "assistant":
            if m.get("content"):
                parts.append(m["content"])
            for c in m.get("tool_calls", []) or []:
                parts.append(c["function"]["name"] + " " + c["function"]["arguments"])
    return "\n".join(parts)


def near_duplicates(traces: list[dict], threshold: float = 0.85, num_perm: int = 128) -> set[str]:
    """Return ids of traces to drop, keeping the first occurrence of each near-duplicate group."""
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    drop: set[str] = set()
    for t in traces:
        mh = MinHash(num_perm=num_perm)
        for s in _shingles(assistant_text(t)):
            mh.update(s.encode("utf-8"))
        if lsh.query(mh):
            drop.add(t["id"])
            continue
        lsh.insert(t["id"], mh)
    return drop
```

### 3.3 Schema validation

```python
# agentdistill/curate/schema.py
from __future__ import annotations

import json

from jsonschema import Draft202012Validator


def tool_calls_valid(trace: dict) -> tuple[bool, list[str]]:
    schemas = {t["function"]["name"]: t["function"].get("parameters", {"type": "object"}) for t in trace["tools"]}
    problems: list[str] = []
    for i, m in enumerate(trace["messages"]):
        for c in m.get("tool_calls", []) or []:
            name = c["function"]["name"]
            if name not in schemas:
                problems.append(f"turn {i}: unknown tool {name}")
                continue
            try:
                args = json.loads(c["function"]["arguments"])
            except json.JSONDecodeError:
                problems.append(f"turn {i}: {name} arguments are not JSON")
                continue
            for err in Draft202012Validator(schemas[name]).iter_errors(args):
                problems.append(f"turn {i}: {name}: {err.message}")
    return (not problems), problems
```

### 3.4 Stratification and coverage report

Embed `task_input` text, fit k-means (K default 32, or from config), assign `traces.cluster`, then cap per cluster. The curation report prints cluster sizes before and after capping, the share of clusters with fewer than 10 samples, and the three most common tool sequences per cluster. Sparse clusters are where the student will fail first; the report says so, and the router floor (section 8) protects them at serving time.

### 3.5 Curation report

`agentdistill curate --config project.yaml` writes `reports/curation-<dataset>.md` with: counts dropped per filter, duplicate rate, decontamination hits, token histogram, turns histogram, tool-call frequency, cluster coverage. Every dataset carries this report; it is the first thing a reviewer reads.

---

## 4. Dataset construction

### 4.1 Sample kinds

- **trajectory**: the full conversation as one sequence; loss only on assistant tokens. Default. Efficient, keeps in-context tool results as-is.
- **turn_window**: for trajectories longer than `max_seq_len`, emit one sample per assistant turn with the system prompt, the tools, and the last `window_turns` messages. Flagged in `samples.kind` so the eval can measure long-context degradation separately.
- **dpo_pair**: prompt prefix plus chosen and rejected assistant turns (section 6).

### 4.2 Rendering with the base model's chat template

Use the tokenizer's own `apply_chat_template(messages, tools=...)`. Modern instruct models ship templates that render tool schemas into the system region and tool calls into their native format. Do not hand-write a tool format; that is how you get a student that emits a syntax the serving stack cannot parse.

Requirements checked at dataset build time:
- the template accepts `tools`
- a round trip `render -> vLLM tool parser` on a sample tool call recovers the same name and arguments
- the template's end-of-turn token is defined

If any check fails, the CLI names the template and stops. Supporting a template without tool calling is a v2 feature, not a silent fallback.

### 4.3 Loss masking

Offsets-based masking is robust to templates that change tokenization at message boundaries.

> **As built.** A token is a target if its **start offset** lies inside an assistant span. Tokens straddling the
> *end* boundary (an end-of-turn marker merged with a following newline) are targets; tokens straddling the
> *start* boundary (a header merged with the first content token) are not. Requiring full containment would drop
> the token most likely to straddle -- the end-of-turn marker -- producing a student that never learns to stop.
>
> Guarded by `tests/test_mask_invariants.py`, which derives the assistant header and end-of-turn string from each
> template and asserts: exactly one end-of-turn marker per assistant turn in the targets, no header leakage, the
> last target ends the turn, and every tool call sits wholly inside the targets. It runs on three templates
> including real hermes and llama3 shapes.

```python
# agentdistill/data/build.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Sample:
    input_ids: list[int]
    labels: list[int]
    n_target_tokens: int


def _render(tok, messages, tools, add_generation_prompt: bool) -> str:
    return tok.apply_chat_template(
        messages, tools=tools, tokenize=False, add_generation_prompt=add_generation_prompt,
    )


def build_trajectory_sample(tok, messages: list[dict], tools: list[dict], max_seq_len: int) -> Sample | None:
    """Loss on assistant turns only (text and tool calls), masked everywhere else."""
    full = _render(tok, messages, tools, add_generation_prompt=False)
    spans: list[tuple[int, int]] = []
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        prefix = _render(tok, messages[:i], tools, add_generation_prompt=True)
        if not full.startswith(prefix):
            raise ValueError(
                "chat template is not prefix-stable at an assistant turn; "
                "this template cannot be used for offsets-based masking"
            )
        end = len(_render(tok, messages[: i + 1], tools, add_generation_prompt=False))
        spans.append((len(prefix), end))

    enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    if len(ids) > max_seq_len:
        return None
    labels = [-100] * len(ids)
    for t, (s, e) in enumerate(enc["offset_mapping"]):
        if e <= s:
            continue
        if any(s >= a and e <= b for a, b in spans):
            labels[t] = ids[t]
    n_target = sum(1 for x in labels if x != -100)
    if n_target == 0:
        return None
    return Sample(input_ids=ids, labels=labels, n_target_tokens=n_target)


def build_turn_windows(tok, messages: list[dict], tools: list[dict], max_seq_len: int, window_turns: int = 12):
    """One sample per assistant turn, keeping system + last window_turns messages before it."""
    system = [m for m in messages[:1] if m["role"] == "system"]
    body = messages[len(system):]
    for i, m in enumerate(body):
        if m["role"] != "assistant":
            continue
        start = max(0, i - window_turns)
        # never start a window on a tool result without its calling assistant turn
        while start > 0 and body[start]["role"] == "tool":
            start -= 1
        window = system + body[start: i + 1]
        s = build_trajectory_sample(tok, window, tools, max_seq_len)
        if s is not None:
            # earlier assistant turns inside the window also receive loss; see the note below
            yield s
```

Note on the window builder: `build_trajectory_sample` masks every non-assistant token, so earlier assistant turns inside the window also receive loss. If you want only the last turn as target, render with the earlier assistant turns included in the prefix computation; both variants are worth an ablation in milestone 2.

### 4.4 Packing

Use TRL's `padding_free=True` with a flash-attention backend so packed sequences do not attend across sample boundaries. Without flash attention, do not pack; cross-contamination in packed batches is a silent quality bug.

### 4.5 Rationale distillation (optional flag)

`--with-rationale` asks the teacher to write a two-sentence justification before each tool call, stored as assistant text preceding the `tool_calls`. Students trained this way tend to select tools better but emit more tokens at inference. The flag is off by default; the eval report shows both if both were trained.

### 4.6 Dataset artifact

Parquet with columns `input_ids`, `labels`, `n_tokens`, `n_target_tokens`, `trace_id`, `kind`, `sample_hash`; a `manifest.json` with the filter config, base model tokenizer id and revision, `max_seq_len`, and the dataset `content_hash`. Datasets are immutable. A change in filters produces a new version.

---

## 5. Supervised fine-tuning

### 5.1 Base model choice

Pick a current open-weight instruct model in the 3B to 8B range whose chat template supports tools and whose license permits your use. Run `agentdistill base-check <model>` which loads the tokenizer, checks the template requirements from 4.2, prints parameter count, license, and the VRAM estimate for LoRA and QLoRA. Ship results for three or four candidates in the docs and let users pick.

### 5.2 Hardware envelope

| Student | Method | VRAM | Typical GPU |
|---|---|---|---|
| 3B | LoRA bf16 | ~16 GB | L4, A10G |
| 8B | QLoRA 4-bit | ~20 GB | L4, A10G |
| 8B | LoRA bf16 | ~40 GB | L40S, A100 40 |
| 3B | full fine-tune | ~48 GB | L40S, A100 80 |

### 5.3 Training config

```yaml
# project.yaml (training section)
train:
  base_model: <org>/<model-8b-instruct>
  method: sft
  lora:
    r: 32
    alpha: 64
    dropout: 0.05
    target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
  quantization: 4bit          # null for bf16 LoRA
  max_seq_len: 8192
  epochs: 2
  lr: 1.0e-4
  scheduler: cosine
  warmup_ratio: 0.03
  per_device_batch: 2
  grad_accum: 8
  packing: true               # requires flash-attn; disabled automatically otherwise
  eval_every_steps: 100
  early_stop_patience: 3
  seed: 17
```

### 5.4 Trainer

```python
# agentdistill/train/sft.py
from __future__ import annotations

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def train_sft(cfg: dict, dataset_path: str, out_dir: str) -> dict:
    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    quant = None
    if cfg.get("quantization") == "4bit":
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], quantization_config=quant, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if cfg.get("packing") else "sdpa",
    )
    lora = LoraConfig(r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"], lora_dropout=cfg["lora"]["dropout"],
                      target_modules=cfg["lora"]["target_modules"], task_type="CAUSAL_LM")

    ds = load_dataset("parquet", data_dir=dataset_path)
    ds = ds["train"].train_test_split(test_size=0.05, seed=cfg["seed"])

    args = SFTConfig(
        output_dir=out_dir,
        num_train_epochs=cfg["epochs"],
        learning_rate=cfg["lr"],
        lr_scheduler_type=cfg["scheduler"],
        warmup_ratio=cfg["warmup_ratio"],
        per_device_train_batch_size=cfg["per_device_batch"],
        gradient_accumulation_steps=cfg["grad_accum"],
        max_length=cfg["max_seq_len"],
        packing=cfg.get("packing", False),
        padding_free=cfg.get("packing", False),
        bf16=True,
        gradient_checkpointing=True,
        eval_strategy="steps",
        eval_steps=cfg["eval_every_steps"],
        save_strategy="steps",
        save_steps=cfg["eval_every_steps"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=10,
        seed=cfg["seed"],
        report_to=["tensorboard"],
        dataset_kwargs={"skip_prepare_dataset": True},   # we pass pre-tokenized input_ids and labels
    )
    trainer = SFTTrainer(model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["test"],
                         processing_class=tok, peft_config=lora)
    trainer.train()
    trainer.save_model(out_dir)
    return {"eval_loss": trainer.evaluate()["eval_loss"], "steps": trainer.state.global_step}
```

Verify `SFTConfig` field names against the installed TRL version; the pre-tokenized path and the `padding_free` flag have moved between releases. Wrap Unsloth as an optional fast path behind `train.backend: unsloth`.

### 5.5 What to log

Train and eval loss, target-token throughput, learning rate, gradient norm, and, every eval step, a **teacher-forced next-action accuracy** on 200 held-out assistant turns (section 7.2). Loss is a proxy; next-action accuracy is what the agent needs.

### 5.6 Merge and quantize

`agentdistill adapter merge <id>` merges the LoRA into the base for serving without multi-LoRA overhead. `agentdistill adapter quantize <id> --method fp8|awq` produces the serving artifact. Both write new `adapters` rows; the unmerged adapter stays available for further DPO rounds.

---

## 6. Preference optimization and on-policy training

SFT on teacher trajectories teaches the student to imitate the teacher when the prefix is the teacher's. At inference the prefix is the student's own, including its own mistakes. That gap is exposure bias, and it is the main reason naive agent distillation disappoints. Two rounds of on-policy training close most of it.

### 6.1 Offline DPO pairs from teacher traces

For tasks that have both a successful and a failed teacher trajectory, pair them at the first divergent assistant turn.

```python
# agentdistill/data/pairs.py
from __future__ import annotations

import hashlib
import json


def _key(m: dict) -> str:
    calls = [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in m.get("tool_calls", []) or []]
    return hashlib.sha256(json.dumps([m.get("content") or "", calls], sort_keys=True).encode()).hexdigest()


def first_divergent_pair(success: dict, failure: dict) -> dict | None:
    """Return a DPO record {prompt, chosen, rejected} at the first assistant turn where the two trajectories differ,
    provided all messages before that turn are identical."""
    s, f = success["messages"], failure["messages"]
    for i in range(min(len(s), len(f))):
        if s[i]["role"] != f[i]["role"]:
            return None
        if s[i]["role"] != "assistant":
            if _key(s[i]) != _key(f[i]):
                return None          # tool results or user turns differ before any assistant divergence
            continue
        if _key(s[i]) == _key(f[i]):
            continue
        return {"prompt": s[:i], "chosen": [s[i]], "rejected": [f[i]], "tools": success["tools"],
                "task_id": success.get("task_id")}
    return None
```

### 6.2 On-policy rejection sampling and DPO (the important part)

Round r:

1. Take the current student (SFT adapter, or previous round).
2. For each training task, run the student k=8 times through the eval harness with mocked tools (section 7). Grade each rollout.
3. **RFT set**: successful rollouts become new SFT samples (rejection-sampling fine-tuning). Cap to 2 per task to avoid over-representing easy tasks.
4. **DPO set**: for tasks with both successful and failed rollouts, build pairs with `first_divergent_pair`. Also pair against the teacher's trajectory when the student failed and the teacher succeeded on the same prefix.
5. Train: SFT on the RFT set (one epoch, low LR), then DPO on the pair set on top of the merged SFT model, `beta=0.1`, LoRA r=16, one epoch.
6. Evaluate against the teacher on the held-out eval set. Keep the round only if the paired delta improves or the CI on the delta includes zero and cost improved.

Two rounds are the default. A third rarely pays for itself.

### 6.3 DPO trainer

```python
# agentdistill/train/dpo.py
from __future__ import annotations

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


def train_dpo(cfg: dict, pairs_path: str, base_or_merged: str, out_dir: str) -> dict:
    tok = AutoTokenizer.from_pretrained(base_or_merged)
    model = AutoModelForCausalLM.from_pretrained(base_or_merged, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    ds = load_dataset("json", data_files=pairs_path)["train"]   # conversational format: prompt, chosen, rejected
    args = DPOConfig(
        output_dir=out_dir, beta=cfg.get("beta", 0.1), num_train_epochs=1, learning_rate=5e-6,
        per_device_train_batch_size=1, gradient_accumulation_steps=16, max_length=cfg["max_seq_len"],
        bf16=True, gradient_checkpointing=True, logging_steps=10, seed=cfg["seed"], report_to=["tensorboard"],
    )
    trainer = DPOTrainer(model=model, ref_model=None, args=args, train_dataset=ds, processing_class=tok, peft_config=lora)
    trainer.train()
    trainer.save_model(out_dir)
    return {"steps": trainer.state.global_step}
```

`ref_model=None` with a PEFT adapter makes TRL use the frozen base as the reference, which is the memory-efficient path. Tools must be passed through the chat template during DPO tokenization; confirm the installed TRL version threads `tools` from the record, otherwise pre-render prompts.

### 6.4 GRPO with verifiable rewards (v2)

Reward per rollout = `1.0 * success + 0.2 * all_tool_calls_schema_valid - 0.001 * completion_tokens`. TRL's `GRPOTrainer` with the harness as the reward function. Requires the mocked-tool harness to run inside the training loop at 8 to 16 rollouts per task, so it is expensive. Leave for v2 once RFT plus DPO has a baseline.

---

## 7. Evaluation

### 7.1 Harness

If agentreplay is installed and the traces came from it, the eval is `agentreplay eval run --model student@gateway --policy strict` and the paired comparison is `agentreplay eval compare`. The gateway exposes each adapter as a model name, so the replay engine swaps models without knowing anything about distillation.

Otherwise the built-in harness does the minimum: for each held-out trace, run the student from the task input with tool results served from the recorded trace by canonical argument hash (the same hash rules as agentreplay), stop on divergence in strict mode or fuzzy-match in fuzzy mode, grade with the eval set's grader. Same paired statistics module (cluster bootstrap over tasks, McNemar, Wilcoxon, Holm), vendored from agentreplay.

### 7.2 Metrics

| Metric | How | Why |
|---|---|---|
| Task success | grader on final outcome, N repeats per task, paired vs teacher | the number that matters |
| Tool-call schema validity | share of tool calls whose arguments validate | catches format collapse early |
| Teacher-forced next-action accuracy | feed the teacher's prefix at each assistant turn; does the student choose the same tool (name) and the same canonical args | cheap, per-turn, diagnostic; runs during training |
| Argument exact match | conditional on same tool name | separates "knows what to do" from "knows how to fill it in" |
| Trajectory length ratio | student steps divided by teacher steps | detects loops and over-calling |
| Tokens and latency per task | measured through the gateway | feeds the cost model |
| Per-cluster success | all of the above by task cluster | tells the router where the student is weak |
| Divergence rate | strict-mode stops per task | second signal of drift |

### 7.3 Report

`agentdistill eval compare <eval_run_student> <eval_run_teacher>`:

```
Task success        teacher 71.0%   student 66.5%   delta -4.5 pp  [95% CI -8.1, -1.0]  McNemar p=0.006
Schema validity     teacher 99.6%   student 98.9%
Next-action acc     student 84.2%  (tool name)   71.8% (name + args)
Length ratio        median 1.08   p90 1.6
Tokens / task       median -38%
Weakest clusters    #7 refunds-multi-item (student 41% vs 78%, n=23)   #12 address-change (52% vs 80%, n=19)
```

The clusters line is the input to both the next curation round (more data there) and the router floor.

---

## 8. Confidence calibration and the escalation cascade

### 8.1 Where the gate sits

Per assistant turn. The student generates the turn; if the calibrated probability that this turn is good falls below the threshold, the turn is discarded and the teacher generates it from the same prefix; the student resumes on the next turn. Per-turn gating is cheaper than per-task routing (the teacher only pays for hard turns) and safer (one bad tool call is caught before it executes). The router in section 9 handles per-task priors on top.

### 8.2 Features

Collected from the vLLM response (request `logprobs=True`, `top_logprobs=5`) and from the trace:

```python
# agentdistill/cascade/features.py
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class TurnFeatures:
    mean_logprob: float          # over all generated tokens
    min_logprob: float
    p10_logprob: float
    arg_mean_logprob: float      # over tokens inside tool-call argument JSON, nan if no tool call
    arg_min_logprob: float
    first_tool_token_entropy: float   # entropy of top_logprobs at the first tool-name token
    n_tokens: int
    n_tool_calls: int
    has_tool_call: int
    agreement: float             # self-consistency: share of k extra samples with same tool name + canonical args
    cluster_prior: float         # posterior mean student success in this task's cluster
    turn_idx: int
    prefix_tokens: int


def _entropy(top: dict[str, float]) -> float:
    ps = np.exp(np.array(list(top.values())))
    ps = ps / ps.sum()
    return float(-(ps * np.log(ps + 1e-12)).sum())


def turn_features(choice: dict, arg_token_mask: list[bool], k_samples: list[dict], cluster_prior: float,
                  turn_idx: int, prefix_tokens: int) -> TurnFeatures:
    lps = np.array([t["logprob"] for t in choice["logprobs"]["content"]], dtype=float)
    arg_lps = lps[np.array(arg_token_mask, dtype=bool)] if any(arg_token_mask) else np.array([np.nan])
    calls = choice["message"].get("tool_calls") or []
    first_tool_tok = next((t for t, m in zip(choice["logprobs"]["content"], arg_token_mask) if m), None)
    ent = _entropy({x["token"]: x["logprob"] for x in first_tool_tok["top_logprobs"]}) if first_tool_tok else 0.0

    def sig(c: dict) -> str:
        cs = c["message"].get("tool_calls") or []
        return json.dumps([(x["function"]["name"], json.loads(x["function"]["arguments"])) for x in cs], sort_keys=True)

    base_sig = sig(choice)
    agree = float(np.mean([sig(s) == base_sig for s in k_samples])) if k_samples else float("nan")

    return TurnFeatures(
        mean_logprob=float(lps.mean()), min_logprob=float(lps.min()), p10_logprob=float(np.percentile(lps, 10)),
        arg_mean_logprob=float(np.nanmean(arg_lps)), arg_min_logprob=float(np.nanmin(arg_lps)),
        first_tool_token_entropy=ent, n_tokens=int(len(lps)), n_tool_calls=len(calls), has_tool_call=int(bool(calls)),
        agreement=agree, cluster_prior=cluster_prior, turn_idx=turn_idx, prefix_tokens=prefix_tokens,
    )


def as_vector(f: TurnFeatures, names: list[str]) -> np.ndarray:
    d = asdict(f)
    return np.array([0.0 if (isinstance(d[n], float) and math.isnan(d[n])) else d[n] for n in names], dtype=float)
```

Self-consistency (`agreement`) costs k extra samples. Default k=2 with `n=3` in one vLLM request, which shares the prefix cache so the marginal cost is small. The ablation in milestone 5 reports gate quality with and without it.

### 8.3 Labels

A turn is "good" if the task eventually succeeded **and** the turn's tool calls match the teacher's on the same prefix in the teacher-forced eval, or, when no teacher reference exists for that prefix, if the task succeeded and no later turn corrected this one. Task-level success alone is too coarse; a task can succeed despite a bad turn that got repaired.

### 8.4 Calibrator

```python
# agentdistill/cascade/calibrate.py
from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score


def fit_calibrator(X: np.ndarray, y: np.ndarray, seed: int = 0):
    base = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300, random_state=seed)
    clf = CalibratedClassifierCV(base, method="isotonic", cv=5)
    clf.fit(X, y)
    return clf


def expected_calibration_error(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi)
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece)


def calibration_report(clf, X: np.ndarray, y: np.ndarray) -> dict:
    p = clf.predict_proba(X)[:, 1]
    return {"auroc": float(roc_auc_score(y, p)), "brier": float(brier_score_loss(y, p)),
            "ece": expected_calibration_error(p, y), "n": int(len(y))}
```

Fit on the eval set's student rollouts (held out from training), report on a second held-out split. Never fit on training tasks; the student's confidence on memorized prefixes is not informative.

### 8.5 Threshold selection

Analytic pass, then verify with the harness.

```python
# agentdistill/cascade/threshold.py
from __future__ import annotations

import numpy as np


def choose_threshold(p_turn: np.ndarray, good_turn: np.ndarray, task_of_turn: np.ndarray,
                     teacher_task_success: dict, cost_student_turn: float, cost_teacher_turn: float,
                     max_success_drop_pp: float = 1.0) -> dict:
    """Grid-search a per-turn threshold. Task succeeds under the cascade if every kept student turn was good
    (escalated turns assumed as good as the teacher's). Returns the cheapest threshold meeting the drop budget."""
    tasks = np.unique(task_of_turn)
    teacher_rate = float(np.mean([teacher_task_success[t] for t in tasks]))
    best = None
    for tau in np.linspace(0.05, 0.99, 95):
        keep = p_turn >= tau
        ok = []
        for t in tasks:
            m = task_of_turn == t
            kept_good = bool(np.all(good_turn[m][keep[m]])) if keep[m].any() else True
            any_escalated = not bool(keep[m].all())
            ok.append(kept_good and (bool(teacher_task_success[t]) if any_escalated else True))
        rate = float(np.mean(ok))
        esc = float(1 - keep.mean())
        cost = cost_student_turn + esc * cost_teacher_turn
        if teacher_rate - rate <= max_success_drop_pp / 100 and (best is None or cost < best["cost_per_turn"]):
            best = {"threshold": float(tau), "cascade_success": rate, "teacher_success": teacher_rate,
                    "escalation_rate": esc, "cost_per_turn": cost}
    return best or {"threshold": 1.0, "note": "no threshold meets the budget; escalate everything"}
```

Then run the harness with `cascade:<adapter>:<tau>` as the subject at `tau`, `tau - 0.05`, and `tau + 0.05`, and pick from measured task success and measured cost. The analytic estimate is for speed; the harness number is what goes in the report.

### 8.6 Cascade runtime

```python
# agentdistill/cascade/runtime.py
from __future__ import annotations


class Cascade:
    def __init__(self, student, teacher, calibrator, feature_names, threshold, k_samples=2):
        self.student, self.teacher = student, teacher
        self.cal, self.names, self.tau, self.k = calibrator, feature_names, threshold, k_samples

    async def turn(self, messages, tools, cluster_prior, turn_idx, prefix_tokens):
        from agentdistill.cascade.features import as_vector, turn_features
        resp = await self.student.chat(messages, tools, n=1 + self.k, logprobs=True, top_logprobs=5)
        choice, extra = resp.choices[0], resp.choices[1:]
        feats = turn_features(choice.raw, choice.arg_token_mask, [e.raw for e in extra],
                              cluster_prior, turn_idx, prefix_tokens)
        p = float(self.cal.predict_proba(as_vector(feats, self.names)[None, :])[0, 1])
        if p >= self.tau:
            return choice, {"arm": "student", "confidence": p, "escalated": False}
        t = await self.teacher.chat(messages, tools)
        return t.choices[0], {"arm": "teacher", "confidence": p, "escalated": True, "student_tokens": choice.n_tokens}
```

Wasted student tokens on escalated turns are counted in the cost model. If the escalation rate rises above the calibrated value by more than 10 points over a rolling hour, the gateway raises an alert: the traffic has drifted from the calibration set.

---

## 9. Router

The cascade decides per turn. The router decides per task, before the first turn, whether this task class should go to the student cascade at all. It is a contextual bandit with clusters as context.

### 9.1 Context

Embed the task input (first user message plus system prompt hash), assign to the nearest of the K cluster centroids fitted during curation. Cache centroid assignment by embedding hash.

### 9.2 Arms and reward

Arms: `student` (cascade) and `teacher`. Reward for a completed task: `success - lambda * cost_usd`, with `lambda` set so that a 1 pp success change is worth the configured dollar amount. Success comes from the grader when one runs online, from labels when they arrive, and from a delayed judge otherwise; the router tolerates delayed feedback.

### 9.3 Thompson sampling with a floor

```python
# agentdistill/router/thompson.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ArmState:
    alpha: float = 1.0
    beta: float = 1.0

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)


class ThompsonRouter:
    def __init__(self, state: dict[tuple[int, str], ArmState], cost: dict[str, float], lam: float,
                 floor: float = 0.5, explore_cap: float = 0.10, decay: float = 0.995, seed: int = 0):
        self.state, self.cost, self.lam = state, cost, lam
        self.floor, self.explore_cap, self.decay = floor, explore_cap, decay
        self.rng = np.random.default_rng(seed)

    def choose(self, cluster_id: int) -> str:
        s = self.state.setdefault((cluster_id, "student"), ArmState())
        t = self.state.setdefault((cluster_id, "teacher"), ArmState())
        if s.mean < self.floor and (s.alpha + s.beta) > 10:
            return "teacher"                                   # hard floor once we have evidence
        if self.rng.random() > self.explore_cap and (s.alpha + s.beta) > 30 and (t.alpha + t.beta) > 30:
            # exploit: compare posterior means, no sampling noise
            return "student" if s.mean - self.lam * self.cost["student"] >= t.mean - self.lam * self.cost["teacher"] else "teacher"
        ts = self.rng.beta(s.alpha, s.beta) - self.lam * self.cost["student"]
        tt = self.rng.beta(t.alpha, t.beta) - self.lam * self.cost["teacher"]
        return "student" if ts >= tt else "teacher"

    def update(self, cluster_id: int, arm: str, success: bool) -> None:
        st = self.state.setdefault((cluster_id, arm), ArmState())
        st.alpha = st.alpha * self.decay + (1.0 if success else 0.0)
        st.beta = st.beta * self.decay + (0.0 if success else 1.0)

    def warm_start(self, eval_counts: dict[tuple[int, str], tuple[int, int]]) -> None:
        for key, (succ, fail) in eval_counts.items():
            self.state[key] = ArmState(alpha=1.0 + succ, beta=1.0 + fail)
```

Warm start from the per-cluster eval counts so the router never starts blind. Decay keeps the posteriors responsive after a retrain. State persists in `router_state`; the gateway loads it at boot and flushes on every update.

### 9.4 Simulation test

Before serving, simulate 20 clusters with known student and teacher success rates and costs for 10,000 tasks; assert regret is within 5 percent of the oracle after 2,000 tasks and that no cluster below the floor ever receives more than the exploration cap of student traffic after 200 observations.

---

## 10. Serving

### 10.1 vLLM

One GPU, one base model, several LoRA adapters, prefix caching on, guided JSON for tool calls, quantized weights.

```
vllm serve <org>/<model-8b-instruct> \
  --enable-lora --max-loras 4 --max-lora-rank 64 \
  --lora-modules support-v3=/adapters/support-v3 support-v4=/adapters/support-v4 \
  --enable-prefix-caching \
  --quantization fp8 \
  --max-model-len 16384 --max-num-seqs 64 \
  --enable-auto-tool-choice --tool-call-parser <parser-for-your-template> \
  --guided-decoding-backend xgrammar \
  --port 8000
```

Verify every flag against the installed vLLM version; names move between releases. Notes:

- **Quantization**: FP8 on Ada and Hopper class GPUs is nearly free in quality. Use AWQ on older cards. Always run the eval on the quantized artifact; report the delta versus bf16.
- **Guided decoding**: constrain tool-call arguments to the tool's JSON schema. This removes almost all schema-validity failures and is worth the small latency cost. Constrain arguments only, not the choice of tool.
- **Prefix caching**: agent turns share long prefixes (system prompt, tools, prior turns). Hit rates above 70 percent are normal and are the main reason per-turn self-consistency sampling is cheap.
- **Speculative decoding**: n-gram speculation helps on tool-heavy outputs that repeat argument keys. Test it; it is not always a win.
- **Multi-LoRA** is for canary and A/B. Merged weights for the prod adapter when only one adapter is live.

### 10.2 Gateway

FastAPI service exposing `POST /v1/chat/completions` (OpenAI dialect) and `POST /v1/messages` (Anthropic dialect). The agent points `base_url` at it and keeps its model name; the gateway maps model names to policies:

| Requested model | Behavior |
|---|---|
| any teacher model name (e.g. the frontier model the agent already uses) | router decides: cascade or teacher passthrough |
| `student` or `student:<adapter>` | student only, no escalation (for evals) |
| `cascade:<adapter>:<tau>` | fixed cascade, no router (for evals) |
| `teacher` | passthrough (for paired evals) |

```python
# agentdistill/gateway/app.py  (skeleton)
from __future__ import annotations

import time
import uuid

from fastapi import FastAPI, Request

from agentdistill.gateway.dialect import from_openai, to_openai, from_anthropic, to_anthropic
from agentdistill.gateway.state import gw   # loaded cascade, router, clients, logger

app = FastAPI()


async def _handle(req: dict, dialect: str) -> dict:
    started = time.time()
    messages, tools, model = req["messages"], req.get("tools") or [], req["model"]
    cluster = gw.cluster_for(messages)
    if model.startswith("student"):
        arm = "student"
    elif model.startswith("cascade") or model == "teacher":
        arm = "cascade" if model.startswith("cascade") else "teacher"
    else:
        arm = gw.router.choose(cluster)
    if arm == "teacher":
        out = await gw.teacher.chat(messages, tools)
        meta = {"arm": "teacher", "escalated": False}
    elif arm == "student":
        out = await gw.student.chat(messages, tools, adapter=gw.adapter_from(model))
        meta = {"arm": "student", "escalated": False}
    else:
        choice, meta = await gw.cascade.turn(messages, tools, gw.router.state_mean(cluster, "student"),
                                             turn_idx=gw.turn_index(messages), prefix_tokens=gw.prefix_tokens(messages))
        out = choice
    rec = {"id": uuid.uuid4().hex, "cluster_id": cluster, "latency_ms": int((time.time() - started) * 1000), **meta}
    await gw.log(rec, req, out)
    return out


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    out = await _handle(from_openai(body), "openai")
    return to_openai(out)


@app.post("/v1/messages")
async def messages(request: Request):
    body = await request.json()
    out = await _handle(from_anthropic(body), "anthropic")
    return to_anthropic(out)
```

Streaming: supported for teacher passthrough and student-only from day one; for the cascade, the student turn must complete before the gate decides, so stream the chosen turn after the decision. Document the added time-to-first-token.

Outcome feedback: `POST /v1/feedback {request_id, success}` lets the agent or a downstream grader report outcomes, which updates `requests.outcome` and the router.

### 10.3 Deployment

- `docker-compose.yml`: vllm, gateway, postgres (pgvector), optional tensorboard. GPU passthrough for vllm.
- `deploy/aws/`: one g6e (L40S) instance, Terraform, S3 for adapters.
- `deploy/gcp/`: one L4 or A100 VM, GCS for adapters.
- `deploy/modal/`: serverless GPU with a cold-start note; good for the training jobs too.
- Adapters live in object storage; the gateway pulls the `prod` and `canary` adapters on boot and on a registry change.

---

## 11. Cost model and report

```python
# agentdistill/report/cost.py
from __future__ import annotations


def student_cost_per_mtok(gpu_usd_per_hour: float, tokens_per_second: float, utilization: float = 0.6) -> float:
    return gpu_usd_per_hour / (tokens_per_second * utilization * 3600) * 1e6


def teacher_cost_per_task(prompt_tokens: float, completion_tokens: float, in_per_mtok: float, out_per_mtok: float,
                          cache_hit_frac: float = 0.0, cache_read_per_mtok: float | None = None) -> float:
    cached = prompt_tokens * cache_hit_frac
    uncached = prompt_tokens - cached
    read_rate = cache_read_per_mtok if cache_read_per_mtok is not None else in_per_mtok
    return (uncached * in_per_mtok + cached * read_rate + completion_tokens * out_per_mtok) / 1e6


def cascade_cost_per_task(student_tokens: float, student_per_mtok: float, escalation_rate: float,
                          teacher_task_cost: float, wasted_student_tokens: float) -> float:
    return (student_tokens + wasted_student_tokens) * student_per_mtok / 1e6 + escalation_rate * teacher_task_cost


def breakeven_tasks_per_day(gpu_usd_per_hour: float, teacher_task_cost: float, cascade_variable_cost: float) -> float:
    """Tasks per day at which the fixed GPU spend is covered by per-task savings."""
    saving = teacher_task_cost - cascade_variable_cost
    if saving <= 0:
        return float("inf")
    return gpu_usd_per_hour * 24 / saving
```

`agentdistill report` produces a single static HTML page:

- the headline: cost per task before and after, success rate before and after with CI, escalation rate
- the cost versus success curve across thresholds (the CFO chart)
- reliability diagram, ECE, Brier, AUROC of the gate
- per-cluster table: traffic share, student success, teacher success, routing decision
- break-even tasks per day for the chosen GPU
- dataset lineage: which traces, which filters, which base model, which rounds

---

## 12. Retrain loop and adapter lifecycle

```
weekly:
  ingest gateway requests with outcomes  ->  curate (same config)  ->  new dataset version
  -> SFT from current prod adapter's base (or continue from adapter)  ->  RFT + DPO round
  -> eval vs teacher on the frozen eval set  ->  calibrate  ->  candidate adapter
  -> canary at 10% of student traffic via multi-LoRA  ->  compare canary vs prod on live outcomes (paired by cluster)
  -> promote or retire
```

Rules:
- The eval set is frozen for a quarter. Rotating it every week makes trends unreadable.
- Promotion requires: paired delta versus prod adapter with CI not worse than -1 pp, schema validity >= 99 percent, calibration ECE <= 0.05, cost not worse.
- Every adapter records its dataset hash and training config; `agentdistill adapter lineage <id>` prints the chain.
- `agentdistill retrain --config project.yaml` runs the whole loop; a cron or GitHub Actions workflow calls it.

---

## 13. Interfaces

### 13.1 Project config

One file drives everything.

```yaml
# project.yaml
name: support-agent
registry: sqlite:///.agentdistill/registry.db      # or postgresql://...
artifacts: ./artifacts                              # or s3://bucket/prefix

sources:
  - type: agentreplay
    db: .agentreplay/agentreplay.db
    since: 60d
  - type: gateway
    since: 7d

teacher:
  model: <frontier-model-name>
  provider: anthropic
  pricing_from_registry: true

curate:
  filters: [outcome, schema_valid, no_error_loops, length, exact_dedupe, near_dedupe, decontaminate, pii, stratify]
  near_dedupe_threshold: 0.85
  max_turns: 40
  clusters: 32
  cap_per_cluster: 400

dataset:
  max_seq_len: 8192
  window_turns: 12
  with_rationale: false

train:
  base_model: <org>/<model-8b-instruct>
  method: sft
  quantization: 4bit
  lora: {r: 32, alpha: 64, dropout: 0.05, target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]}
  epochs: 2
  lr: 1.0e-4
  packing: true
  seed: 17

onpolicy:
  rounds: 2
  k_rollouts: 8
  rft_cap_per_task: 2
  dpo_beta: 0.1

eval:
  eval_set: support-holdout-q3
  n_per_task: 5
  policy: strict
  grader: {type: llm_judge, rubric: rubrics/support.md, judge_model: <mid-tier-model>}

cascade:
  k_samples: 2
  max_success_drop_pp: 1.0
  features: [mean_logprob, min_logprob, p10_logprob, arg_mean_logprob, arg_min_logprob,
             first_tool_token_entropy, n_tokens, n_tool_calls, has_tool_call, agreement, cluster_prior, turn_idx]

router:
  lambda_per_usd: 20.0
  floor: 0.55
  explore_cap: 0.10
  decay: 0.995

serve:
  vllm_url: http://vllm:8000
  quantization: fp8
  gpu_usd_per_hour: 1.20
  guided_json: true
```

### 13.2 CLI

```python
# agentdistill/cli.py  (skeleton)
import typer

app = typer.Typer(no_args_is_help=True, add_completion=False)
ingest_app, adapter_app, eval_app, train_app = typer.Typer(), typer.Typer(), typer.Typer(), typer.Typer()
app.add_typer(ingest_app, name="ingest")
app.add_typer(adapter_app, name="adapter")
app.add_typer(eval_app, name="eval")
app.add_typer(train_app, name="train")


@app.command()
def init(path: str = "project.yaml"):
    """Write a starter project.yaml and create the registry."""


@ingest_app.command("agentreplay")
def ingest_agentreplay(db: str, since: str = "30d", config: str = "project.yaml"):
    """Import runs from an agentreplay store."""


@ingest_app.command("jsonl")
def ingest_jsonl(path: str, config: str = "project.yaml"):
    """Import normalized traces from JSONL."""


@ingest_app.command("gateway")
def ingest_gateway(since: str = "7d", config: str = "project.yaml"):
    """Import gateway requests that have outcomes."""


@app.command()
def curate(config: str = "project.yaml", name: str | None = None):
    """Run the filter pipeline and write a new dataset version plus a curation report."""


@app.command()
def base_check(model: str):
    """Check a base model's chat template for tool support, prefix stability, license, and VRAM needs."""


@train_app.command("sft")
def train_sft_cmd(dataset: str, config: str = "project.yaml"):
    """LoRA / QLoRA supervised fine-tuning."""


@train_app.command("onpolicy")
def train_onpolicy(adapter: str, config: str = "project.yaml", rounds: int | None = None):
    """Rejection sampling + DPO rounds using the eval harness for rollouts."""


@eval_app.command("run")
def eval_run(subject: str, eval_set: str | None = None, config: str = "project.yaml"):
    """Evaluate an adapter, 'teacher', or 'cascade:<adapter>:<tau>' on the eval set."""


@eval_app.command("compare")
def eval_compare(a: str, b: str):
    """Paired statistical comparison of two eval runs."""


@app.command()
def calibrate(adapter: str, config: str = "project.yaml"):
    """Fit the confidence gate, choose the threshold, verify with the harness."""


@adapter_app.command("merge")
def adapter_merge(adapter: str):
    """Merge LoRA into the base for serving."""


@adapter_app.command("quantize")
def adapter_quantize(adapter: str, method: str = "fp8"):
    """Produce a quantized serving artifact and evaluate it."""


@adapter_app.command("promote")
def adapter_promote(adapter: str, to: str = "canary"):
    """Move an adapter to canary or prod after checks."""


@adapter_app.command("lineage")
def adapter_lineage(adapter: str):
    """Print dataset, config, and parent chain."""


@app.command()
def serve(config: str = "project.yaml", host: str = "0.0.0.0", port: int = 8710):
    """Start the gateway."""


@app.command()
def report(config: str = "project.yaml", out: str = "reports/latest.html"):
    """Build the cost and quality report."""


@app.command()
def retrain(config: str = "project.yaml"):
    """Run the full weekly loop: ingest, curate, train, eval, calibrate, canary."""


if __name__ == "__main__":
    app()
```

### 13.3 Integration with agentreplay

- `ingest agentreplay` reads its SQLite or Postgres store directly.
- The gateway registers `student:<adapter>` and `cascade:<adapter>:<tau>` as model names; `agentreplay eval run --model cascade:support-v3:0.62` evaluates the cascade with agentreplay's replay engine and statistics.
- The gateway writes every request as an agentreplay run when `AGENTREPLAY_AUTO=1` is set, so the retrain loop's source is the same store.

---

## 14. Repository skeleton

```
agentdistill/
  pyproject.toml
  README.md
  LICENSE                          (Apache-2.0)
  docker-compose.yml               (vllm + gateway + postgres)
  project.example.yaml
  schemas/
    trace.schema.json
  agentdistill/
    __init__.py
    config.py                      pydantic models for project.yaml
    registry/
      base.py  sqlite.py  postgres.py
      migrations/sqlite/001_init.sql
      migrations/postgres/001_init.sql
    ingest/
      normalize.py                 anthropic <-> openai message formats
      agentreplay_source.py  jsonl_source.py  otel_source.py  gateway_source.py
    curate/
      pipeline.py  filters.py  dedupe.py  schema.py  decontaminate.py  pii.py  stratify.py  report.py
    data/
      build.py                     samples and loss masks
      pairs.py                     DPO pair construction
      template_check.py
      artifact.py                  parquet + manifest + hashes
    train/
      sft.py  dpo.py  onpolicy.py  merge.py  quantize.py  unsloth_backend.py
    eval/
      harness.py                   built-in mocked-tool harness
      agentreplay_bridge.py
      metrics.py                   success, schema validity, next-action accuracy, arg match, length ratio
      stats.py                     vendored from agentreplay
      report.py
    cascade/
      features.py  calibrate.py  threshold.py  runtime.py
    router/
      thompson.py  clusters.py  simulate.py
    gateway/
      app.py  dialect.py  state.py  clients.py  log.py  feedback.py
    report/
      cost.py  html.py  templates/
    cli.py
  deploy/
    aws/  gcp/  modal/
  examples/
    support_agent/                 an agent, 500 recorded traces, an eval set, a rubric
  tests/
    fixtures/traces/               normalized traces incl. anthropic and openai originals
    test_normalize.py
    test_dedupe.py
    test_schema.py
    test_build_masks.py            token-level assertions on a small tokenizer
    test_pairs.py
    test_sft_smoke.py              2 steps on a tiny model, CPU
    test_harness.py
    test_stats.py
    test_features.py
    test_calibrate.py
    test_threshold.py
    test_router_sim.py
    test_gateway_contract.py       OpenAI and Anthropic SDK clients against the gateway
    test_cost.py
  docs/
    quickstart.md  curation.md  training.md  cascade.md  router.md  serving.md  tos.md  faq.md
```

### 14.1 pyproject.toml

```toml
[project]
name = "agentdistill"
version = "0.1.0"
description = "Distill your production agent into a small model with a calibrated escalation gate and a cost report."
readme = "README.md"
requires-python = ">=3.11"
license = { text = "Apache-2.0" }
dependencies = [
  "typer>=0.12",
  "rich>=13",
  "pydantic>=2",
  "pyyaml>=6",
  "sqlalchemy>=2",
  "numpy>=1.26",
  "scipy>=1.12",
  "scikit-learn>=1.4",
  "pyarrow>=15",
  "datasets>=2.19",
  "datasketch>=1.6",
  "jsonschema>=4.21",
  "fastapi>=0.115",
  "uvicorn>=0.30",
  "httpx>=0.27",
  "litellm>=1.50",
  "jinja2>=3.1",
]

[project.optional-dependencies]
train = ["torch>=2.3", "transformers>=4.44", "trl>=0.12", "peft>=0.12", "bitsandbytes>=0.43", "accelerate>=0.33", "tensorboard"]
unsloth = ["unsloth"]
serve = ["vllm>=0.6"]
postgres = ["psycopg[binary]>=3.1", "pgvector>=0.3"]
agentreplay = ["agentreplay>=0.1"]
local-embeddings = ["sentence-transformers>=3"]
dev = ["pytest>=8", "pytest-asyncio", "ruff", "mypy", "hypothesis", "respx"]

[project.scripts]
agentdistill = "agentdistill.cli:app"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.ruff]
line-length = 120
```

---

## 15. Build plan

Sixteen weeks. Each milestone has a definition of done. GPU time starts in milestone 2; everything before it runs on a laptop.

### Milestone 1: ingest, curate, build (weeks 1 to 2)

Build: config models, registry (SQLite), normalize, agentreplay and JSONL sources, all default filters, dedupe, schema validation, decontamination, stratification, curation report, template check, mask builder, parquet artifact with hashes.

Done when:
- 500 example traces go from JSONL to a hashed SFT dataset with a curation report.
- `test_build_masks.py` asserts token-level masks on three chat templates (one that fails prefix stability must raise the named error).
- Re-running curate with the same config produces the same `content_hash`.

### Milestone 2: SFT (weeks 3 to 4)

Build: SFT trainer, QLoRA path, packing with flash attention, teacher-forced next-action accuracy during training, merge, base-check.

Done when:
- An 8B QLoRA run on one 24 GB GPU finishes on the example dataset.
- Next-action accuracy of the student exceeds the zero-shot base model by a margin the report states with a CI.
- `test_sft_smoke.py` runs 2 steps on a tiny model on CPU in CI.

### Milestone 3: evaluation (weeks 5 to 7)

Build: built-in mocked-tool harness, agentreplay bridge, all metrics, vendored stats, per-cluster breakdown, eval report, eval set freezing.

Done when:
- `agentdistill eval compare` prints the section 7.3 report for student versus teacher on the example eval set with N=5.
- The same eval through agentreplay's harness gives the same success counts.
- Weakest clusters are identified and match a hand review.

### Milestone 4: on-policy training (weeks 8 to 9)

Build: offline pair builder, rollout collection through the harness, RFT dataset, DPO trainer, round loop with promotion rule.

Done when:
- Two rounds run unattended from one command.
- The paired delta versus teacher after round 2 is better than after SFT, with the CI reported; if it is not, the report says so and the round is discarded automatically.

### Milestone 5: calibration and cascade (weeks 10 to 11)

Build: feature extraction from vLLM logprobs, turn labels, calibrator, ECE and reliability diagram, threshold search, harness verification, cascade runtime, ablation with and without self-consistency.

Done when:
- Gate AUROC and ECE are reported on a held-out split.
- The cascade at the chosen threshold, measured through the harness, stays within the configured success-drop budget of the teacher and the escalation rate is reported.
- The wasted-token accounting reconciles with the request log.

### Milestone 6: serving and gateway (weeks 12 to 13)

Build: vLLM launch config, quantize command with eval delta, gateway with both dialects, model-name policies, streaming for passthrough and student-only, feedback endpoint, request log, Docker Compose, one cloud recipe.

Done when:
- The example agent runs unchanged with `base_url` pointed at the gateway, in both the OpenAI and Anthropic SDKs.
- `test_gateway_contract.py` passes with both official SDK clients.
- Quantized artifact evaluated; delta versus bf16 in the report.

### Milestone 7: router and retrain loop (weeks 14 to 15)

Build: clusters at serving time, Thompson router with floor and decay, warm start, persistence, simulation test, canary via multi-LoRA, promotion checks, `retrain` command, cron example.

Done when:
- `test_router_sim.py` meets the regret and floor assertions.
- One full `retrain` run produces a candidate, canaries it, and promotes or retires with a printed reason.

### Milestone 8: report, docs, launch (week 16)

Build: HTML report with the cost versus success curve, docs, `docs/tos.md`, example repo with recorded traces so a reader can run the eval without a GPU or an API key, README GIF.

Done when:
- A stranger reaches a report from the example project in under 30 minutes on a rented GPU following only `quickstart.md`.

---

## 16. Testing strategy

- **Fixtures**: 20 normalized traces plus their Anthropic and OpenAI originals, checked in. Every curation and data test starts from these.
- **Normalization**: round trip Anthropic to OpenAI to Anthropic preserves tool names, arguments, and tool result pairing.
- **Masks**: assert exact label positions on a small open tokenizer; property test that masked token count equals the token count of the rendered assistant turns.
- **Dedupe**: Hypothesis test that shuffling message order within a trace does not create false duplicates, and that a trace and its copy with one changed number are near-duplicates.
- **Decontamination**: an eval task planted in the training traces is always removed.
- **Training smoke**: 2 optimizer steps on a tiny model on CPU; asserts the adapter saves and loads.
- **Harness**: replay of a fixture trace with the recorded assistant turns as a fake student reproduces the recorded outcome exactly.
- **Stats**: same simulation suite as agentreplay (CI coverage, McNemar power, Holm false-positive rate).
- **Calibration**: synthetic features with a known success probability; assert ECE below 0.03 and AUROC within tolerance.
- **Threshold**: synthetic turns with known good rates; the chosen threshold meets the budget and no cheaper threshold does.
- **Router**: simulation with oracle regret bound; floor never violated after 200 observations.
- **Gateway contract**: the official OpenAI and Anthropic Python SDKs make tool-calling requests against the gateway and parse responses without modification, streaming and non-streaming.
- **Nightly GPU job**: full pipeline on the example project on one rented GPU; posts the report as a CI artifact.

---

## 17. Launch plan

1. **README** written before milestone 3. Contents: the one-sentence promise, the cost versus success chart from the example project, the three commands (`ingest`, `retrain`, `serve`), a fair "when not to use this" section (low volume, tasks where the frontier model's success is itself marginal, teachers whose terms forbid it), and a link to `docs/tos.md`.
2. **Benchmark post**: distill a public agent task set from a frontier model into an 8B student, report paired success with CIs at three cascade thresholds, escalation rates, per-cluster weaknesses, and dollars per thousand tasks on a named GPU. Include the negative results (clusters where the student cannot be trusted). This post is the marketing and it only works if it is honest.
3. **Example project** with recorded traces and an eval set, runnable without an API key for the eval and report stages.
4. **Notebook** (Colab or Modal) that runs milestone 2 end to end on a free or cheap GPU.
5. **Distribution**: PyPI, Show HN, r/LocalLLaMA (this audience cares about exactly this), the vLLM and TRL communities, and direct messages to teams that have posted about agent inference costs.
6. **First week**: answer every issue within 24 hours; the first ten issues will be chat-template and vLLM flag mismatches, so have the `base-check` command and a template compatibility table ready.

---

## 18. Risks

| Risk | Mitigation |
|---|---|
| Teacher provider terms forbid training on outputs | `docs/tos.md`, README warning, open-weight teacher path tested in the example project, no silent default teacher |
| Exposure bias: student collapses on its own prefixes | On-policy RFT and DPO are milestone 4, not v2; parity claims require the on-policy rounds |
| Tool-format mismatch between template and serving parser | Template check at dataset build, round-trip test through the vLLM parser, guided JSON for arguments |
| Overfitting to dominant task clusters | Per-cluster caps, per-cluster eval, router floor for weak clusters |
| Eval contamination inflates the student | Decontamination filter with n-gram overlap, frozen eval set, planted-task test in CI |
| Judge noise corrupts turn labels | Teacher-forced next-action agreement as the primary turn label; judge calibration when a judge is used (vendored from agentreplay) |
| Gate miscalibrated after traffic drift | Escalation-rate alert, weekly recalibration in the retrain loop, ECE gate on promotion |
| GPU cost surprises | Cost model with break-even tasks per day printed before serving; QLoRA defaults; one-GPU story only |
| Library API churn (TRL, PEFT, vLLM) | Pinned versions in `train` and `serve` extras, nightly GPU job catches breakage, flags verified in `base-check` |
| Long trajectories exceed context | Window samples flagged and evaluated separately; `max_seq_len` in the report |
| Streaming latency with the cascade | Documented added TTFT; student-only and passthrough stream normally |

---

## 19. Immediate next steps

The first five working days. Each day ends with a commit that passes CI. No GPU needed this week.

**Day 1**
- Repo, `pyproject.toml`, ruff and mypy, GitHub Actions on 3.11 and 3.12.
- `config.py` with pydantic models for `project.yaml`; `project.example.yaml`.
- Registry protocol, SQLite implementation, `migrations/sqlite/001_init.sql`.
- `schemas/trace.schema.json` and `ingest/normalize.py` with round-trip tests on the fixtures.

**Day 2**
- `ingest/jsonl_source.py` and `ingest/agentreplay_source.py` (read runs, llm_calls, tool_calls; reconstruct trajectories; pull success from eval_results and labels).
- `curate/schema.py`, `curate/dedupe.py`, `curate/decontaminate.py` with tests, including the planted eval task.

**Day 3**
- `curate/pipeline.py` with all default filters and the curation report.
- `curate/stratify.py` with pluggable embeddings and k-means; coverage table in the report.

**Day 4**
- `data/template_check.py` and `data/build.py` with token-level mask tests on three templates.
- `data/artifact.py`: parquet, manifest, content hash; determinism test.

**Day 5**
- `base-check` command.
- `train/sft.py` written and smoke-tested on CPU with a tiny model for 2 steps.
- Rent one GPU, run the first real SFT on the example dataset overnight, and record throughput and eval loss in the registry.
- Milestone 1 review against its definition of done.

After day 5, follow the milestone order in section 15. Do not build the gateway before the eval harness exists, and do not publish any cost number that was not measured through the harness on the quantized artifact.


---

## 20. Divergences from this plan

The code has diverged from v0.1 in a few places, each because the plan was internally inconsistent or wrong about
how a library behaves. Every divergence is recorded, with its reasoning, in [`docs/progress.md`](docs/progress.md).
Sections 3.1, 3.2, and 4.3 above have been updated in place to describe what the code does.
