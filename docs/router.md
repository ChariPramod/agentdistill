# The router

The cascade decides per turn. The router decides per task, before the first turn, whether this kind of task
should go to the student at all. Some clusters the student handles as well as the teacher; some it does not,
and no per-turn gate rescues a task the student was never going to get right.

## Clusters

Clusters come from curation: trajectories are embedded and grouped, the assignment is stored on the trace, and
the centroids are saved as a `cluster_models` row beside the dataset. At boot the gateway loads the latest one and
assigns each incoming request to its nearest centroid, embedding the same text curation embedded with the same
embedder (the stored spec, not today's config). With the `hash` embedder clusters group
by token overlap rather than meaning — fine for a smoke test, not for a routing decision you intend to defend.
Set `curate.embeddings.provider` to something semantic first.

A request the gateway cannot place -- no cluster model, or unreadable centroids -- is **unassigned**. It is
routed on the pooled posterior (both arms' evidence summed across clusters) *without* the floor, logged with
`cluster_id = NULL` and `routing_reason = 'no_cluster_model'`, and kept out of the per-cluster posteriors. The
floor protects clusters the student is measurably bad at; applying it to "we do not know" would turn a missing file
into a teacher bill.

`/healthz` says so on the first request: `cluster_model` is `loaded` (with id and k) or `missing` (with the
reason), `calibration` is `loaded` (with the threshold) or `missing` (including a refused, non-`usable` verdict),
and `traffic` carries the rolling fallback and unassigned rates. Above 20% fallbacks or 50% unassigned over
five minutes (at least 20 requests), `ok` is false.

## Thompson sampling, with two departures

Each `(cluster, arm)` holds a Beta posterior over success. `choose` samples from both and takes the better,
after subtracting `lambda_per_usd` times each arm's dollar cost. Two changes from the textbook version, both
about not learning the wrong thing in production:

**A hard floor.** Once a cluster has `min_observations` (default 10) and the student's posterior mean is below
`floor` (default 0.55), that cluster goes to the teacher and stops being explored. Pure Thompson sampling keeps
a trickle going to a known-bad arm forever, which is fine in a simulation and not fine when each sample is a
customer. The floor is a state, not a sentence: feedback can lift a cluster back out of it.

**Decay.** Posteriors are discounted by `decay` (default 0.995) on every update. Without it, a newly promoted
adapter inherits hundreds of observations about its predecessor and needs hundreds more to escape them.

Decay and the floor interact, and the interaction is checked at config load. Each update multiplies
`alpha + beta` by `decay` and adds one, so evidence converges to `1/(1-decay)` no matter how much traffic
flows. At `decay: 0.9` that ceiling is 8 observations — below the default floor threshold of 10, so the floor
could never engage. That configuration is refused rather than accepted with a silently disabled safety floor.

## Warm start, and reset on promotion

On an empty `router_state` table the router seeds its posteriors from the prod adapter's per-cluster eval
counts. Starting flat means spending the first hundred production requests rediscovering what the eval set
already measured, and paying for the mistakes.

The warm start runs **only** on an empty table. Re-running it over live posteriors would discard what
production taught the router in favour of an offline eval, which is the less reliable of the two.

When a new adapter reaches prod the table is wiped. The old adapter's record is not evidence about the new one.

Posteriors are flushed after every `/v1/feedback` call, so a gateway restart does not lose what the router
learned.

## Costs

`arm_costs` prices both arms from observed traffic: the student from `serve.gpu_usd_per_hour` and measured
throughput, the teacher from configured prices and measured prompt, completion and cached token counts. When
either side cannot be priced — no teacher prices on file, fewer than 50 requests, no teacher traffic — both
arms return zero and the router maximizes success alone. That is the conservative failure: it may route to the
teacher more often than a cost-aware router would, and it will never route to the student for a saving nobody
verified.

## The canary

`serve.canary_share` sends a deterministic share of student-routed traffic to the canary adapter. Deterministic
on a hash of the request id, so a retry lands on the same adapter — otherwise a retry would compare two
adapters on one task and muddy the comparison the canary exists to produce. Both adapters are loaded in vLLM at
once; `serve_vllm.sh` emits `--lora-modules` for each.

## Comparing them

```bash
agentdistill adapter compare-live <prod> <canary> --since 7d
```

Paired within clusters. Live traffic is not a randomized trial: if the canary happened to draw more of an easy
cluster, an unpaired comparison would credit it for the mix rather than the model. The weight on each cluster
is its thinner arm, because a cluster's difference is only as well measured as its smaller side.

**"Not enough live traffic" is an answer, not a failure.** Below three clusters with five observations each,
`compare_live` returns `None` rather than a very wide interval, and the `live_not_worse` promotion check fails.
A caller who sees a number acts on it; there is no number to act on here. Leave the canary running, or raise
`serve.canary_share`.

Fallback requests are excluded — the teacher served those — as are teacher-arm requests and ungraded ones.
