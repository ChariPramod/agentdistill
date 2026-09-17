# The retrain loop

A distilled student decays. Traffic drifts from the corpus it was trained on, a tool gains an argument, a policy
changes, and the adapter measured at parity six weeks ago is no longer at parity. The retrain loop turns that
from an incident into a scheduled job.

```bash
agentdistill retrain --dry-run          # the plan, with counts, changing nothing
agentdistill retrain --since 7d         # run it
agentdistill retrain --retrain-id 20260917 --from eval    # resume
```

## Stages and gates

Eight stages. Each has a gate, and the run stops at the first gate it cannot satisfy.

| stage | runs | gate |
|---|---|---|
| `ingest_gateway` | `ingest gateway --since` | at least 50 graded requests |
| `curate` | `curate` with the same filters | at least 50 new samples |
| `train_sft_continue` | one epoch at lr/3 from the prod adapter | eval loss finite |
| `onpolicy` | one round from the new adapter | the round decided `promote` |
| `eval` | frozen set at the configured N | success CI low end ≥ −1 pp vs prod |
| `calibrate` | fit and verify on a disjoint split | holdout ECE ≤ 0.05 **and** AUROC ≥ 0.6 |
| `quantize` | the configured method | quantized success within 2 pp of bf16 |
| `promote_canary` | `adapter promote --to canary` | every lifecycle check green |

The gates are the point. A loop that runs unattended and promotes whatever it produced is a mechanism for
putting an unmeasured model into production every week.

Two gates deserve their reasoning spelled out:

**The eval gate reads the interval, not the point estimate.** A candidate 0.5 pp below prod with an interval
spanning eight points has not been shown to be at parity; it has been measured badly, and the honest response
is to stop rather than to promote on noise.

**Calibration needs AUROC as well as ECE.** ECE alone passes a calibrator that outputs the base rate for every
turn: perfectly calibrated, and useless, because it never separates a turn the student got right from one it
did not. A gate like that saves nothing.

## Stopping is normal

Most weeks there is not enough new graded traffic to justify a retrain. The first gate says so and the run
stops, exiting **0** — a weekly cron should not page anyone because the week was quiet. A stage whose command
actually failed raises and exits 1. The two are deliberately distinguishable.

## Resuming

Each completed stage writes a marker under `artifacts/retrain/<retrain-id>/`. A rerun with the same id skips
what is done; a new id starts fresh. `--from <stage>` overrides the markers from that stage onward, because
resuming means rerunning the stage that failed and a marker from a previous attempt must not skip it.

## Why stages shell out

Each stage runs the same subcommand a person would run by hand. The registry records the exact invocation on
every row it writes, so a retrain leaves behind a sequence of commands that reproduces it — and the unattended
path is the same code an engineer debugs with. A pipeline calling internal functions would leave rows nobody
could rerun and failures nobody could reproduce.

## Continuation, not a fresh run

`train_sft_continue` continues the prod adapter rather than starting from the base model. The new corpus is
mostly the old corpus plus a few weeks of traffic; restarting would discard everything the serving adapter
knows in order to relearn it. Continuation wants a much lower learning rate — the loop uses a third — because
the weights start near a good solution and a fresh-run rate walks out of it.

## It promotes to canary, never to prod

The last stage promotes to **canary**. Prod requires `adapter promote --to prod`, which requires a live
comparison that only accumulates over days of real traffic. That stays a human decision.

## The scheduled workflow

`.github/workflows/retrain.yml.draft` has a weekly cron on a `[self-hosted, gpu]` runner. It ends in `.draft`
deliberately, and is renamed to `retrain.yml` only after:

1. `scripts/serve_smoke.sh` passes on that runner.
2. `agentdistill retrain --dry-run` prints the eight stages with real counts.
3. `agentdistill retrain --since 7d` has been run by hand once, start to finish.
4. The registry database and artifacts directory are on persistent storage. A retrain on an ephemeral runner
   loses the adapter it just trained.

Scheduling an unverified pipeline to train and promote models unattended, every week, is the wrong order to do
things in.
