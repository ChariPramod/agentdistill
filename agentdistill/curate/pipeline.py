"""The curation pipeline.

Filters run in the canonical order of section 3.1: cheap per-trace rejections first, then corpus-level work
(dedupe, decontamination), then stratification last so that per-cluster caps are computed over the survivors.

Every stage records what it dropped and why. `filter_config` on the resulting dataset, plus the trace corpus,
fully determines the output: re-running with the same config over the same traces produces the same survivors in
the same order, which is what makes the dataset content hash meaningful.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from agentdistill.config import ProjectConfig
from agentdistill.curate import filters as F
from agentdistill.curate.decontaminate import contaminated_ids
from agentdistill.curate.dedupe import near_duplicates
from agentdistill.curate.pii import redact_trace
from agentdistill.curate.stratify import assign_clusters, cap_per_cluster, coverage, make_embedder

#: Filters whose purpose is to keep both successes and failures (DPO needs the failures).
_SFT_ONLY = {"outcome"}


@dataclass
class StageStats:
    name: str
    n_in: int
    n_dropped: int
    reasons: Counter = field(default_factory=Counter)
    examples: dict[str, str] = field(default_factory=dict)

    @property
    def n_out(self) -> int:
        return self.n_in - self.n_dropped

    def record(self, trace_id: str, reason: str) -> None:
        self.n_dropped += 1
        # Reasons carry specifics ("42 assistant turns, above max_turns=40"); the report groups them by their
        # leading phrase so the histogram stays readable.
        self.reasons[_reason_key(reason)] += 1
        self.examples.setdefault(_reason_key(reason), f"{trace_id}: {reason}")

    def to_dict(self) -> dict:
        return {
            "filter": self.name,
            "n_in": self.n_in,
            "n_dropped": self.n_dropped,
            "n_out": self.n_out,
            "reasons": dict(self.reasons.most_common()),
            "examples": self.examples,
        }


#: Numbers in a drop reason are instance data ("42 assistant turns", "86% overlap", "cluster 7"). Grouping on the
#: numbers would produce one histogram row per distinct value; grouping on the shape produces one row per cause.
_DIGITS = re.compile(r"\d+(?:\.\d+)?")


def _reason_key(reason: str) -> str:
    """Group reasons by shape, so "86% 8-gram overlap" and "91% 8-gram overlap" land in the same bucket.

    The specific reason survives as the stage's stored example, so the report still shows a real one.
    """
    return _DIGITS.sub("N", reason)


@dataclass
class CurationResult:
    traces: list[dict]
    stages: list[StageStats]
    assignments: dict[str, int]
    coverage: dict
    n_input: int
    filter_config: dict
    notes: list[str] = field(default_factory=list)
    embedder_name: str = ""
    #: The k-means centroids behind `assignments`. Saved by `curate` so the gateway can place live requests in
    #: the same clusters the router's posteriors are keyed on.
    centroids: Any = None

    @property
    def n_output(self) -> int:
        return len(self.traces)

    def to_dict(self) -> dict:
        return {
            "n_input": self.n_input,
            "n_output": self.n_output,
            "stages": [s.to_dict() for s in self.stages],
            "coverage": self.coverage,
            "filter_config": self.filter_config,
            "notes": self.notes,
            "embedder": self.embedder_name,
        }


def curate(
    traces: list[dict],
    cfg: ProjectConfig,
    eval_task_inputs: list[Any] | None = None,
    kind: str = "sft",
    judge_scores: dict[str, float] | None = None,
) -> CurationResult:
    """Run the configured filters over `traces`.

    `kind="dpo"` keeps failures: preference pairs need a rejected trajectory, so the `outcome` filter is skipped
    and the report says so.
    """
    c = cfg.curate
    order = c.ordered_filters()
    stages: list[StageStats] = []
    notes: list[str] = []
    current = list(traces)
    n_input = len(current)

    if kind == "dpo":
        skipped = [f for f in order if f in _SFT_ONLY]
        if skipped:
            notes.append(f"kind=dpo: skipped {', '.join(skipped)} so failed trajectories survive as rejected samples")
        order = [f for f in order if f not in _SFT_ONLY]

    for name in order:
        stage = StageStats(name=name, n_in=len(current), n_dropped=0)
        if name in F.PER_TRACE:
            current = _run_per_trace(name, current, c, stage, judge_scores)
        elif name == "exact_dedupe":
            current = _run_exact_dedupe(current, stage)
        elif name == "near_dedupe":
            current = _run_near_dedupe(current, c, stage)
        elif name == "decontaminate":
            current = _run_decontaminate(current, c, stage, eval_task_inputs, notes)
        elif name == "pii":
            current = _run_pii(current, stage)
        elif name == "stratify":
            # Stratification is handled after the loop: it needs the final survivor set to cluster and cap.
            stages.append(stage)
            continue
        stages.append(stage)

    assignments: dict[str, int] = {}
    cov: dict = {}
    embedder_name = ""
    centroids = None
    if "stratify" in order and current:
        before = list(current)
        embedder = make_embedder(c.embeddings)
        embedder_name = embedder.name
        assignments, centroids = assign_clusters(before, embedder, k=c.clusters, seed=0)
        drop = cap_per_cluster(before, assignments, c.cap_per_cluster)
        stage = next(s for s in stages if s.name == "stratify")
        for t in before:
            if t["id"] in drop:
                stage.record(t["id"], f"cluster {assignments[t['id']]} over cap_per_cluster={c.cap_per_cluster}")
        current = [t for t in before if t["id"] not in drop]
        for t in current:
            t["cluster"] = assignments.get(t["id"])
        cov = coverage(before, current, assignments)
        if cov["n_sparse"]:
            notes.append(
                f"{cov['n_sparse']} of {cov['n_clusters']} clusters have fewer than "
                f"{cov['sparse_threshold']} samples; these are where the student will fail first"
            )
        if c.embeddings.provider == "hash":
            notes.append(
                "clusters were built with the `hash` embedder, which groups by token overlap rather than meaning; "
                "set curate.embeddings.provider for a semantic clustering before trusting this table"
            )

    return CurationResult(
        traces=current,
        stages=stages,
        assignments=assignments,
        coverage=cov,
        n_input=n_input,
        filter_config=cfg.curation_fingerprint(),
        notes=notes,
        embedder_name=embedder_name,
        centroids=centroids,
    )


def _run_per_trace(
    name: str, traces: list[dict], c: Any, stage: StageStats, judge_scores: dict[str, float] | None
) -> list[dict]:
    fn = F.PER_TRACE[name]
    kwargs = _kwargs_for(name, c, judge_scores)
    kept = []
    for t in traces:
        ok, reason = fn(t, **kwargs)
        if ok:
            kept.append(t)
        else:
            stage.record(t["id"], reason)
    return kept


def _kwargs_for(name: str, c: Any, judge_scores: dict[str, float] | None) -> dict:
    if name == "no_error_loops":
        return {"max_consecutive": c.max_consecutive_tool_errors, "max_repeats": c.max_repeated_identical_calls}
    if name == "length":
        return {"min_turns": c.min_turns, "max_turns": c.max_turns}
    if name == "teacher":
        return {"models": c.teacher_models}
    if name == "quality_judge":
        return {"min_score": c.quality_judge_min_score, "scores": judge_scores}
    return {}


def _run_exact_dedupe(traces: list[dict], stage: StageStats) -> list[dict]:
    seen: dict[str, str] = {}
    kept = []
    for t in traces:
        h = t["content_hash"]
        if h in seen:
            stage.record(t["id"], f"exact duplicate of {seen[h]}")
            continue
        seen[h] = t["id"]
        kept.append(t)
    return kept


def _run_near_dedupe(traces: list[dict], c: Any, stage: StageStats) -> list[dict]:
    drop = near_duplicates(
        traces,
        threshold=c.near_dedupe_threshold,
        num_perm=c.near_dedupe_num_perm,
        shingle_n=c.near_dedupe_shingle_n,
        normalize=c.near_dedupe_normalize_literals,
    )
    kept = []
    for t in traces:
        if t["id"] in drop:
            stage.record(t["id"], f"near-duplicate at Jaccard >= {c.near_dedupe_threshold}")
        else:
            kept.append(t)
    return kept


def _run_decontaminate(
    traces: list[dict], c: Any, stage: StageStats, eval_task_inputs: list[Any] | None, notes: list[str]
) -> list[dict]:
    if not eval_task_inputs:
        notes.append(
            "decontaminate ran with no eval set registered, so nothing could be checked; "
            "register an eval set before publishing any number from this dataset"
        )
        return traces
    hits = contaminated_ids(traces, eval_task_inputs, n=c.decontaminate_ngram, overlap=c.decontaminate_overlap)
    kept = []
    for t in traces:
        if t["id"] in hits:
            stage.record(t["id"], hits[t["id"]])
        else:
            kept.append(t)
    return kept


def _run_pii(traces: list[dict], stage: StageStats) -> list[dict]:
    kept = []
    for t in traces:
        redacted, argument_changed, _notes = redact_trace(t)
        if argument_changed:
            stage.record(t["id"], "redaction altered a tool argument")
            continue
        kept.append(redacted)
    return kept
