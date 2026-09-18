# The GPU day

One rented session, end to end. On the day the only commands you type are:

```bash
pip install -r requirements-gpu.txt -e ".[train,serve]"
bash scripts/gpu_day.sh
```

Anything else you have to type is a bug report for the next rehearsal.

The script is resumable. Each stage writes `artifacts/gpu_day/<stage>.done` and is skipped on rerun, so a failed
stage costs one stage rather than the session. `rm artifacts/gpu_day/sft.done` forces one stage to run again.

A stage gets its marker only on exit 0. Every stage that is supposed to write a registry row (`eval run`,
`calibrate`, `train onpolicy`, `adapter quantize`, `report`) exits **3** if it wrote none, which stops the day
without a marker so the rerun retries it. The only way to pass without a row is a skip declared in config
(`eval.skip_teacher: true`), and the two log lines look different on purpose:

```
[stage eval_teach] SKIPPED: eval.skip_teacher set
[stage eval_teach] wrote no row: run ev_… on support-holdout-v1 stored no results for teacher
```

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

Tiny mode covers every section of the report, not two thirds of it. Its teacher is a **replay stub**
(`teacher.backend: replay` in `project.tiny.yaml`): it answers with the recorded turns at fixed token counts,
so the teacher row, the cost block, and the cascade verification all execute without an API key. Its runs are
tagged `teacher_backend: replay`, and the report never shows their cost without saying it is structural. The
calibration set is 20 tasks at N=3 with `cascade.min_turns: 20`, enough to make the fit, the reliability bins,
the threshold search and `--verify-threshold` execute. With a random model the gate's AUROC sits near chance, so
its verdict is `uninformative` and the cascade escalates every turn -- the report says exactly that.

**The clean rehearsal** is `bash scripts/clean_rehearsal.sh`. It deletes every tiny artifact (registry,
datasets, adapters, markers, the generated eval sets), reruns the day, and asserts the report's `report.json`
sidecar: base, student and teacher rows with run ids; calibration, cascade, cost and quantization populated; and
no warning except the tiny-mode disclosures (`tiny_mode`, `replay_teacher`, an `uninformative` gate). Warnings
are matched by stable code, so rewording one cannot silently pass or fail the check.

What the rehearsal cannot cover: vLLM itself — no real tool parser, no LoRA loading, no template handling — and
anything CUDA. That is what `scripts/serve_smoke.sh` on the real box is for, and it is the first thing to run
there.

Expect one piece of noise that is not a defect. The tiny model has a 2048-token context, and agent trajectories
run past it, so generation logs `exceeded the model's predefined maximum length`. On a real base model with a
real context window it does not happen. If you want it quiet, raise `max_position_embeddings` in
`scripts/make_tiny_model.py`, delete `artifacts/tiny/model`, and clear the `sft` and `merge` markers.

Every rehearsal so far has found something, and each would otherwise have surfaced on rented hardware,
mid-session, after the stages before it had already run:

| Stage it died at | What was wrong |
|---|---|
| `env` | A uv-created venv has no `pip` in it |
| `sft` | The fixture tokenizer has no weights to train |
| `sft` | 4-bit quantization needs CUDA |
| `sft` | `dataset latest` prints an id; `train sft` accepted only a name |
| `merge` | `train sft --tag` was accepted and dropped, so `adapter latest --tag` found nothing |
| `merge` | `torch_dtype` is deprecated in transformers 5.x, and `device_map="auto"` segfaults on MPS |
| `merge` | `adapter merge` and `train onpolicy` hardcoded the vLLM backend |
| `base_check` | `train.base_model` as a relative path resolved against the CWD, not the config |
| `eval_teach` | **`eval run teacher` was evaluating the base model** |
| `cmp_sft` | `eval latest --subject-tag` does not exist; the flag is `--tag` |
| curation | `max_seq_len: 1024` silently dropped every trajectory |
| `cmp_sft` | Five-task eval sets are below `eval compare`'s minimum of eight |

The teacher one is the reason to take rehearsals seriously. It was not a crash: `eval run teacher` loaded
`train.base_model`, ran it, and wrote the results under the subject name "teacher". Every student-against-teacher
comparison in every report would have been a student-against-base comparison, and nothing would have said so.

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
