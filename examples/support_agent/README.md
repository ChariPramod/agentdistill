# Example: a support agent over a real CRM

Runs the whole implemented pipeline offline — no GPU, no API key, no network.

```bash
# from the repo root
python -m examples.support_agent.record --scripted --error-rate 0.25 --n 260 --seed 7 \
  --out examples/support_agent/traces.jsonl
python examples/support_agent/split_corpus.py

cd examples/support_agent
agentdistill ingest jsonl traces-train.jsonl --config project.yaml
agentdistill evalset add support-holdout-v1 eval-holdout.jsonl --config project.yaml
agentdistill evalset add support-unseen-v1  eval-unseen.jsonl  --config project.yaml
agentdistill curate --config project.yaml

PYTHONPATH=../.. agentdistill eval run recorded --eval-set support-holdout-v1 --n 1 --config project.yaml
```

## What is here

| file | what it is |
|---|---|
| `crm.py` | A SQLite-backed CRM, deterministic from a seed. Six tools that enforce real rules |
| `scenarios.py` | 13 scenario shapes, each instantiable with any seed |
| `graders.py` | End-state predicates: pure functions of the final database plus the final message |
| `agent.py` | A plain LiteLLM tool-calling loop — the agent whose traces get distilled |
| `record.py` | Runs tasks, grades them, writes JSONL that `ingest jsonl` accepts |
| `replay_grader.py` | Rebuilds end state from the student's own calls, for grading under replay |
| `scripted_teacher.py` | A rule-based solver so the chain runs with no API key |
| `split_corpus.py` | Splits a recording into training traces and two frozen eval sets |

## The scripted solver is not a teacher

`--scripted` uses a rule-based solver. It reacts to real tool results from a real stateful database, so
trajectories have the right shape, but it is a **generator** — and the whole reason to prefer real traces is that
a student will happily learn the generator, score wonderfully, and fall over on real traffic.

Use it to exercise the pipeline. **Do not train on these traces or publish a number from them.**

For a real corpus:

```bash
python -m examples.support_agent.record --model openai/gpt-4.1 --n 400 --out traces.jsonl
```

Read [`docs/tos.md`](../../docs/tos.md) first — provider terms govern whether you may train on a model's outputs.

## Why the tools refuse things

The interesting part of a trajectory is what an agent does when a tool says no. So:

- Only shipped or delivered orders can be refunded, and only once
- A refund cannot exceed the order total
- An address cannot be changed once any order has shipped
- Unknown emails and order ids are errors, not empty results

Scenarios are built around those refusals: an order too early to refund, an id the customer got wrong, two orders
where only one qualifies, a billing question with nothing refundable behind it.

## The split

| set | n | what it measures |
|---|---:|---|
| `traces-train.jsonl` | 144 | training |
| `support-holdout-v1` | 36 | new instances of scenarios the student trained on |
| `support-unseen-v1` | 80 | four scenario shapes held out of training entirely |

The unseen set scores lower (66% vs 92% for the solver), and that is the point: it is the number that says
whether anything generalized rather than being memorized. Report both.

Both eval sets are frozen. They do not change until there is a v1.0 tag — rotating an eval set makes trends
unreadable.

## The base model

`project.yaml` points `train.base_model` at `../../tests/fixtures/tokenizer`, a 40 KB tokenizer built for the
test suite, so dataset building runs offline. It has no weights and cannot be trained; `train sft` says so. For a
real run pick a 3B–8B instruct model and check it first:

```bash
agentdistill base-check <org>/<model-8b-instruct> --config project.yaml
```
