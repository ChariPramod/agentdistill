# Curation

Garbage traces produce a confident, wrong student. Curation is where most of the quality comes from, and it runs
as a pipeline of named filters so that a dataset's `filter_config` explains exactly what was kept.

Every dataset carries `reports/curation-<name>-v<n>.md`. It is the first thing a reviewer should read.

## The filters, in order

Order matters. Cheap per-trace rejections run before expensive corpus-level work, and stratification runs last so
per-cluster caps are computed over the survivors.

| filter | rule | default |
|---|---|---|
| `outcome` | keep `success = true` for SFT; keep both for DPO | on |
| `schema_valid` | tool arguments parse as JSON and validate against the tool's schema; every call has a result and every result answers a call | on |
| `no_error_loops` | drop traces with ≥3 consecutive tool errors, or the same call issued ≥2 times | on |
| `length` | between `min_turns` and `max_turns` assistant turns | on |
| `teacher` | restrict to named teacher models | off |
| `exact_dedupe` | drop duplicate `content_hash` | on |
| `near_dedupe` | MinHash LSH over assistant text, Jaccard ≥ `near_dedupe_threshold` on 5-gram shingles | on |
| `decontaminate` | drop traces overlapping an eval task | on |
| `pii` | redact; drop the trace if redaction changed a **tool argument** | on |
| `quality_judge` | drop traces a judge scored below `quality_judge_min_score` | off |
| `stratify` | cluster task inputs, cap each cluster at `cap_per_cluster`, report coverage | on |

## Notes on the ones that surprise people

### `outcome` drops ungraded traces

A trace with `success: null` is not evidence of anything, so SFT discards it. It is not wasted: `kind=dpo` keeps
failures as rejected sides, and ungraded traces still count toward cluster statistics.

### `schema_valid` also checks pairing

A tool call with no result means the trace was truncated mid-flight. Training on it teaches the student to call a
tool and then stop, which is a failure mode that is very hard to diagnose later.

### `near_dedupe` compares **assistant text only**

Tool results are environment output, not model behavior. Including them would make two genuinely different
trajectories over the same data look identical.

Two tuning notes:

- **LSH proposes, the threshold disposes.** `MinHashLSH.query` is deliberately loose and returns candidate pairs
  well below the threshold it was built with. Each candidate is verified against the real MinHash estimate before
  anything is dropped.
- **`near_dedupe_normalize_literals` is off by default, and should usually stay off.** It masks numbers and ids
  before shingling, which sounds appealing — "the same trajectory with a different order id" — but on a corpus
  whose agent answers in templates, every trace of a given shape becomes byte-identical once ids are masked, and
  the corpus collapses to roughly one sample per trajectory shape. Raw shingling already catches real
  near-duplicates: changing one token in a realistic trace moves Jaccard by a few points, not thirty. Turn
  normalization on deliberately, and read the `near_dedupe` row of the report afterwards.

### `decontaminate` measures overlap against the *training* trace

A short training task fully contained in a long eval task is contamination. Normalizing by the eval task's length
would make that overlap look negligible and let the trace through.

The system prompt is excluded from matching — every task in a project shares it, so including it would flag the
entire corpus.

**If no eval set is registered, this filter drops nothing and says so in the report.** Register the eval set
first.

### `pii` drops rather than rewrites when an argument changes

Redaction inside assistant prose and tool results is applied and kept. Redaction that touches a *tool argument*
is fatal to the trace: a student that learns `<EMAIL>` is a valid argument will send that placeholder to a real
API. Supply your own redactor if the defaults are too blunt:

```python
from agentdistill.curate import pii
pii.set_redactor(my_callable)   # str -> str
```

### `stratify` tells you where the student will fail

The coverage table lists cluster sizes before and after capping and flags clusters with fewer than ten samples.
Those are where the student fails first. They need more data, a per-cluster eval, and — once the router exists —
a routing floor.

The default `hash` embedder needs no API key and is deterministic, which keeps the pipeline testable, but it
clusters by token overlap rather than meaning. The report says so every time. Set
`curate.embeddings.provider: sentence-transformers` (with the `local-embeddings` extra) before trusting the table.

## Scenario phrasing is part of the design, not decoration

If every instance of a task shape uses the same sentence, decontamination will eat the corpus — and it will be
right to. A training instance and a holdout instance of one shape share almost all of their 8-grams when only an
id differs between them, so the overlap check cannot tell "the same task twice" from "a near-duplicate of the
eval set".

Measured on the example project: 40 scenario shapes with one fixed message each lost **48 traces** to
decontamination. Giving each shape varied openers, asides and closers — the way people actually write to
support — dropped that to **19**, and raised the curated corpus from 234 samples to 296.

This is not cosmetic. Varied phrasing is what makes the clusters meaningful, what keeps decontamination measuring
genuine overlap instead of shared boilerplate, and what stops a student learning a template it will never see in
production. Treat it as part of writing a scenario.

## Reproducibility

A dataset's identity is `sha256(sorted sample hashes + tokenizer + max_seq_len + filter config + kind + target)`.

What is deliberately **not** in it: timestamps, paths, GPU prices, router settings. Editing the cost model must
not invalidate a dataset.

What determines survivor *order*, and therefore which member of a duplicate group is kept: the registry's read
order, which is `(created_at, id)`. That is why re-running curation is deterministic.

## Changing filters

Datasets are immutable. Change a filter and you get a new version:

```bash
# edit curate.near_dedupe_threshold in project.yaml
agentdistill curate            # -> my-agent v2, new hash
agentdistill dataset list
```

Both versions stay on disk and in the registry, so an eval run from last week still refers to exactly the data it
was trained on.
