# Example: a synthetic support agent

Runs the whole implemented pipeline offline — no GPU, no API key, no network.

```bash
python generate_traces.py --n 500
agentdistill ingest jsonl traces.jsonl --config project.yaml
agentdistill evalset add support-holdout eval_tasks.jsonl --config project.yaml
agentdistill curate --config project.yaml
```

## What the corpus contains

The generator is seeded, so the corpus reproduces byte for byte. It is deliberately messy: each failure mode
curation exists to remove is injected in a known quantity, so the curation report can be checked against
expectations rather than merely admired.

From 500 clean trajectories it produces 679 records:

| injected | quantity | caught by |
|---|---|---|
| exact duplicates | 20 | rejected at ingest on `content_hash` |
| near-duplicates (one argument changed) | 25 | `near_dedupe` |
| error loops (3 consecutive tool errors, then recovery) | 25 | `no_error_loops` |
| invalid tool arguments | 20 | `schema_valid` |
| unknown tool names | 10 | `schema_valid` |
| truncated traces (call with no result) | 12 | `schema_valid` |
| single-turn traces | 12 | `length` |
| PII in a tool argument | 10 | `pii` |
| ungraded traces | 20 | `outcome` |
| genuine failures | 25 | `outcome` (kept under `--kind dpo`) |
| tasks also present in the eval set | 8 | `decontaminate` |

The error-loop traces are marked **successful** on purpose. An agent that flails through three failed calls and
then recovers still passes the grader, so `outcome` lets it through and `no_error_loops` is what has to catch it.
That is also the realistic case: the traces worth removing are rarely the ones already labelled as failures.

Tasks are spread across five types with several phrasings each, so clusters are meaningful and decontamination
fires on real overlap rather than on shared template boilerplate.

Every filter above fires at its injected quantity, which is what makes this corpus a regression test as well as a
demo.

## Things worth looking at

**The report.** `reports/curation-support-agent-v1.md` — drop reasons with examples, token and turn histograms,
tool-call frequencies, and the cluster coverage table.

**The loss mask.**

```bash
agentdistill dataset inspect artifacts/datasets/support-agent-v1 --row 0
```

Green is what the model trains on. It should be assistant turns and nothing else.

**Determinism.** Run `agentdistill curate --config project.yaml` twice. The second run reports
`identical to support-agent v1` and writes no new dataset.

**The DPO path.** `agentdistill curate --config project.yaml --kind dpo` keeps the failed trajectories, which is
what preference pairs need.

## The base model

`project.yaml` points `train.base_model` at `../../tests/fixtures/tokenizer` — a 40 KB BPE tokenizer built for
the test suite, carrying a tool-capable chat template. It exists so the example runs offline.

It is **not** a model you can train. For a real run, pick a 3B–8B instruct model, check it, and swap it in:

```bash
agentdistill base-check <org>/<model-8b-instruct>
```
