# Quickstart

Ten minutes, no GPU, no API key. At the end you have a hashed training dataset and a curation report that
explains exactly which traces went into it and why the rest were dropped.

> **Scope.** Milestone 1 (ingest → curate → dataset) is complete and SFT training runs. Evaluation, the cascade,
> the router, and serving are not built; those commands exit with a pointer to their milestone rather than
> pretending. Until the eval harness exists, an adapter is a `candidate` and nothing more.

## Install

```bash
pip install -e ".[dev,tokenizers]"
```

`tokenizers` pulls in `transformers` for the chat template and the tokenizer. You do not need `torch` to build a
dataset.

## Run the example

The repo ships a synthetic support agent whose corpus deliberately contains the failure modes curation exists to
remove, in known quantities.

```bash
python -m examples.support_agent.record --scripted --error-rate 0.25 --n 260 --seed 7 \
  --out examples/support_agent/traces.jsonl
python examples/support_agent/split_corpus.py

cd examples/support_agent
agentdistill ingest jsonl traces-train.jsonl --config project.yaml
agentdistill evalset add support-holdout-v1 eval-holdout.jsonl --config project.yaml
agentdistill curate --config project.yaml
```

The last command prints something like:

```
                         curation: 639 in → 394 kept
 filter           in  dropped  out  top reason
 outcome         639       59  580  task failed (50)
 schema_valid    580       42  538  turn N: tool call cN has no result (12)
 no_error_loops  538        -  538
 length          538       12  526  N assistant turns, below min_turns=N (12)
 exact_dedupe    526        -  526
 near_dedupe     526       25  501  near-duplicate at Jaccard >= N (25)
 decontaminate   501       81  420  N% N-gram overlap with an eval task (74)
 pii             420        6  414  redaction altered a tool argument (6)
 stratify        414       20  394  cluster N over cap_per_cluster=N (20)
```

Read `reports/curation-support-agent-v1.md` before anything else. It is the artifact that tells you whether the
dataset is worth training on.

## Check the loss mask

The single most damaging silent bug in this pipeline is a wrong loss mask: the model trains on tool results and
user turns, the loss curve looks healthy, and the student is quietly ruined. Look at a sample:

```bash
agentdistill dataset inspect artifacts/datasets/support-agent-v1 --row 0
```

The green text is what the model is trained to produce. It must contain **only** assistant turns — the prose and
the tool calls — and must include the end-of-turn marker. If you see a tool result or the system prompt in green,
stop and open an issue.

## Point it at your own traces

### 1. Get your traces into the normalized format

One JSON object per line. OpenAI-style messages, or Anthropic content blocks (detected and converted for you):

```json
{"task_id": "t-1", "success": true, "teacher_model": "your-teacher",
 "messages": [{"role": "system", "content": "..."},
              {"role": "user", "content": "..."},
              {"role": "assistant", "content": "Looking that up.",
               "tool_calls": [{"id": "c1", "type": "function",
                               "function": {"name": "search_orders", "arguments": "{\"customer_id\": \"c_9\"}"}}]},
              {"role": "tool", "tool_call_id": "c1", "content": "[...]"},
              {"role": "assistant", "content": "It shipped yesterday."}],
 "tools": [{"type": "function", "function": {"name": "search_orders", "parameters": {"type": "object"}}}]}
```

`success` is required for SFT: the `outcome` filter keeps only successes, and a trace with no recorded outcome is
not evidence of anything. The full contract is `schemas/trace.schema.json`.

Already using **agentreplay**? Skip the export:

```bash
agentdistill ingest agentreplay --db .agentreplay/agentreplay.db --since 60d
```

### 2. Pick a base model

```bash
agentdistill base-check <org>/<model-8b-instruct>
```

This refuses models whose chat template cannot render tools or cannot be masked from character offsets. Both are
hard failures, not warnings — a template that silently ignores `tools` would train your student to call tools it
was never shown. Set the winner as `train.base_model`.

### 3. Register your eval set *before* curating

```bash
agentdistill evalset add my-holdout holdout.jsonl
```

Decontamination can only remove what it can compare against. Curating first produces a dataset that cannot be
decontaminated, and every number you measure later will be inflated. Freeze the eval set for a quarter; rotating
it weekly makes trends unreadable.

### 4. Curate

```bash
agentdistill init --name my-agent     # writes project.yaml
agentdistill ingest jsonl traces.jsonl
agentdistill curate
```

## What you get

```
artifacts/datasets/my-agent-v1/
  data.parquet      input_ids, labels, n_tokens, n_target_tokens, trace_id, kind, sample_hash
  manifest.json     tokenizer, max_seq_len, filter config, content hash
reports/
  curation-my-agent-v1.md
```

Datasets are immutable and content-addressed. Re-running `curate` over an unchanged corpus with unchanged filters
reproduces the same hash and reuses the existing version rather than minting a new one:

```
identical to my-agent v1 (hash 357dcd2f0629) — no new dataset written.
```

That is the reproducibility guarantee working. Change a filter and you get a new version with a new hash, and
every training run and eval result downstream is reported against it.

### 5. Train

```bash
pip install -e ".[train]"
agentdistill train sft my-agent
```

This writes a LoRA adapter and records it in the registry as a **candidate**:

```
done 340 steps, eval_loss 0.8123
  adapter artifacts/adapters/my-agent-v1
  status: candidate. Loss is a proxy; run a paired eval against the teacher before trusting it.
```

`candidate` is the only status an adapter can reach today, and deliberately so. Eval loss going down does not
mean the agent works — the paired comparison against the teacher that would justify promoting it is milestone 3.
Do not ship a cost number derived from anything in this step.

## Next

- [`curation.md`](curation.md) — what each filter does and when to change its default
- [`evaluation.md`](evaluation.md) — the harness, divergence, and how a paired comparison is read
- [`canonical-json.md`](canonical-json.md) — the argument-hashing contract shared with mcpgate and agentreplay
- [`tos.md`](tos.md) — **read before training on a teacher's outputs**
- [`progress.md`](progress.md) — what is built, what is measured, and every divergence from the plan
