# The GPU day

One rented session, end to end. On the day the only commands you type are:

```bash
pip install -r requirements-gpu.txt -e ".[train,serve]"
bash scripts/gpu_day.sh
```

Anything else you have to type is a bug report for the next rehearsal.

The script is resumable. Each stage writes `artifacts/gpu_day/<stage>.done` and is skipped on rerun, so a failed
stage costs one stage rather than the session. `rm artifacts/gpu_day/sft.done` forces one stage to run again.

## Pre-flight, the day before

**1. Tiny rehearsal green from a clean checkout.**

```bash
rm -rf artifacts/gpu_day && AGENTDISTILL_TINY=1 bash scripts/gpu_day.sh
```

Twenty to forty minutes on a laptop, dominated by the two training stages. The numbers are garbage and the
report says so; what is being checked is that every stage executes in sequence. See [tiny mode](#tiny-mode).

**2. Base model check.**

```bash
agentdistill base-check <the real base model>
```

Downloads the tokenizer only, not the weights. It reports whether the model has a chat template, accepts tools,
and round-trips a tool call through its own parser — and it names the tool-call family, which is what
`train.tool_parser` and vLLM's `--tool-call-parser` both have to agree on. A base model whose template silently
drops the `tools` argument produces a student that never calls anything, and finding that out on the box costs
the session.

**3. Pin versions.** `requirements-gpu.txt` holds them. Three are not pinnable from a laptop — `torch` (must
match the box's CUDA), `vllm`, `llmcompressor` — and the file says so and says what to do. **Install vLLM
first**: it pulls its own torch and will replace one installed before it.

**4. Teacher API key, with a hard spend cap.** The eval and rollout stages call the teacher, and the on-policy
rounds call it per rollout. Set the cap on the key itself, not in your head. A loop that retries on a 500 will
spend the cap and then stop, which is the outcome you want from a runaway; without a cap it just keeps going.

**5. Disk.** An 8B merged bf16 model is about 16 GB and AWQ adds roughly 5 GB on top. With adapters, rollouts,
and eval trajectories, budget 80 GB and check it before starting rather than at the merge stage. `df -h`.

**6. Tag the commit.**

```bash
git tag pre-gpu-day && git push --tags
```

So the logs branch has a clean parent and you can diff what the day changed.

## Tiny mode

`AGENTDISTILL_TINY=1` runs every stage on a laptop: the fixture tokenizer with a randomly-initialized
2-layer model, ten-task eval sets, N=1, the `hf` backend, and the test suite's fake vLLM standing in for the
real one. `scripts/tiny_setup.sh` builds all of it and is idempotent.

**The numbers mean nothing.** The model is random; it cannot do the task. The report carries a tiny-mode
warning for exactly this reason, and `report --inject` will write that warning into whatever file it targets.

What the rehearsal cannot cover: vLLM itself — no real tool parser, no LoRA loading, no template handling — and
anything CUDA. That is what `scripts/serve_smoke.sh` on the real box is for, and it is the first thing to run
there.

Every rehearsal so far has found something. In order: the fixture tokenizer has no weights to train; 4-bit
quantization needs CUDA; `max_seq_len: 1024` silently dropped every trajectory in curation; five-task eval sets
are below `eval compare`'s minimum of eight; a uv-created venv has no `pip` in it; `train.base_model` as a
relative path resolved against the working directory instead of the config; and `dataset latest` printed an id
where `train sft` accepted only a name. All seven would have surfaced on rented hardware, mid-session.

## On the day

Run `scripts/serve_smoke.sh` first, before `gpu_day.sh`. It is the cheapest thing that proves the serving path
works, and every stage after `sft` depends on it.

Then start the script and read the log. `logs/gpu_day.<stage>.log` has each stage's output.

When it finishes, `artifacts/gpu_day/report.html` is the result, and

```bash
agentdistill report --format md --inject README.md
```

writes the numbers into the README — only the ones it can support, with a run id against each and a list of
every claim it could not make.
