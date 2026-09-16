# agentdistill

> Point it at your production agent's traces. Get back a small model that handles the routine calls, a calibrated
> escalation gate that sends the rest to the frontier model, and a cost report you can hand to a CFO.

**Status: pre-alpha.** Milestone 1 (ingest → curate → dataset) is complete, and SFT training runs end to end.
Evaluation, the cascade, the router, and serving are not built yet; those commands exit with a pointer to their
milestone rather than pretending. See [the implementation plan](AgentDistill%20Implementation%20Plan.md) for the
full roadmap.

**No cost or quality claim in this README has been measured yet.** Nothing is promoted on a loss curve, and the
paired eval that would justify such a claim lands in milestone 3.

## What it does

```
traces  ->  curate  ->  dataset  ->  SFT (LoRA)  ->  DPO / RFT  ->  eval vs teacher
                                                                        |
        gateway (drop-in base_url)  <-  router + cascade  <-  calibrate confidence
                |
        vLLM serving (LoRA, prefix cache, guided JSON, quantized)
                |
        request log  ->  retrain loop  ->  adapter registry  ->  canary  ->  promote
```

## Install

```bash
pip install agentdistill                  # core: ingest, curate, dataset, registry, CLI
pip install "agentdistill[train]"         # + torch, transformers, TRL, PEFT
pip install "agentdistill[serve]"         # + vLLM
```

## Quickstart

```bash
agentdistill init                                       # writes project.yaml, creates the registry
agentdistill ingest jsonl traces.jsonl                  # or: ingest agentreplay --db .agentreplay/agentreplay.db
agentdistill evalset add holdout holdout.jsonl           # register BEFORE curating, so decontamination can run
agentdistill curate                                     # filters, dedupes, decontaminates, stratifies
agentdistill base-check <org>/<model-8b-instruct>        # is this template usable at all?
agentdistill train sft <dataset>                         # LoRA / QLoRA -> a candidate adapter
```

Every dataset is immutable and carries `reports/curation-<dataset>.md`, which states exactly what was dropped and why.

## Design principles

1. **The eval harness is the ground truth.** Nothing is promoted on loss curves. Every claim is a paired comparison
   against the teacher on held-out tasks, with a confidence interval.
2. **Drop-in.** The agent code does not change. It points `base_url` at the gateway and keeps its model name.
3. **Calibrated or silent.** If calibration data is missing, the escalation gate defaults to "escalate everything"
   and says so.
4. **On-policy before scale.** Exposure bias kills agent distillation; parity claims require the on-policy rounds.
5. **Config-driven.** One YAML defines sources, filters, base model, training, eval, cascade, and serving. Every run
   is reproducible from the config plus a dataset hash.

## Before you train on a teacher's outputs

Provider terms of service govern whether you may use a model's outputs to train another model. Several frontier
providers restrict using outputs to build competing models. Distilling your own agent's traces for your own internal
cost reduction is a different case from building a competing product, but it is **not automatically permitted**, and
the answer differs by provider and by contract. Read [`docs/tos.md`](docs/tos.md) before you run `train`.
Open-weight teachers are a first-class path.

## When not to use this

- **Low volume.** Below the break-even tasks per day for your GPU, a frontier API is cheaper. `agentdistill report`
  prints your break-even number before you commit.
- **The teacher is itself marginal.** If the frontier model succeeds 60% of the time on your tasks, distilling it
  gives you a cheaper way to fail.
- **Terms forbid it** and no open-weight teacher fits your task.

## License

Apache-2.0.
