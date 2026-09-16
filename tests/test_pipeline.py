"""The curation pipeline end to end, plus stratification and the report."""

from __future__ import annotations

import json

from agentdistill.curate import report as curation_report
from agentdistill.curate.pipeline import curate
from agentdistill.curate.stratify import HashEmbedder, assign_clusters, cap_per_cluster, coverage, kmeans
from tests.conftest import make_trace


def distinct_closing(i: int) -> str:
    """A closing long enough that near-dedupe has real signal, and distinct enough that these traces are not
    duplicates of each other. Sharing one long closing across a fixture corpus makes every trace a near-duplicate,
    which is correct behaviour but makes the fixture useless."""
    return " ".join(
        f"On step {j} of case {i} I checked the {j + i} record and confirmed detail number {i * 31 + j * 7}"
        for j in range(8)
    )


def _corpus() -> list[dict]:
    """A corpus containing one of each failure mode, so every filter has something to catch."""
    traces = [
        make_trace(f"ok{i}", task=f"Where is order number {i} for my account please tell me", closing=distinct_closing(i))
        for i in range(12)
    ]
    traces.append(make_trace("failed", success=False))
    traces.append(make_trace("ungraded", success=None))

    short = make_trace("short")
    short["messages"] = [*short["messages"][:2], {"role": "assistant", "content": "Sure."}]
    short["n_turns"] = 1
    traces.append(short)

    bad = make_trace("bad_args")
    bad["messages"][2]["tool_calls"][0]["function"]["arguments"] = "{not json"
    traces.append(bad)

    dupe = make_trace("dupe", task="Where is order number 0 for my account please tell me",
                      closing=distinct_closing(0))
    traces.append(dupe)

    pii = make_trace("pii", args={"customer_id": "bob@example.com", "limit": 5})
    traces.append(pii)
    return traces


def test_every_filter_catches_its_case(project_config):
    result = curate(_corpus(), project_config)
    by_name = {s.name: s for s in result.stages}
    assert by_name["outcome"].n_dropped == 2, "one failure, one ungraded"
    assert by_name["schema_valid"].n_dropped == 1
    assert by_name["length"].n_dropped == 1
    assert by_name["exact_dedupe"].n_dropped + by_name["near_dedupe"].n_dropped >= 1
    assert by_name["pii"].n_dropped == 1
    assert result.n_output < result.n_input


def test_filters_run_in_canonical_order(project_config):
    result = curate(_corpus(), project_config)
    assert [s.name for s in result.stages] == project_config.curate.ordered_filters()


def test_stage_counts_chain(project_config):
    result = curate(_corpus(), project_config)
    for prev, nxt in zip(result.stages, result.stages[1:], strict=False):
        assert prev.n_out == nxt.n_in, "each stage must receive exactly what the previous one passed"
    assert result.stages[-1].n_out == result.n_output


def test_dpo_kind_keeps_failures(project_config):
    sft = curate(_corpus(), project_config, kind="sft")
    dpo = curate(_corpus(), project_config, kind="dpo")
    assert "outcome" not in [s.name for s in dpo.stages]
    assert dpo.n_output > sft.n_output
    assert any("kind=dpo" in n for n in dpo.notes)


def test_curation_is_deterministic(project_config):
    corpus = _corpus()
    a = curate(corpus, project_config)
    b = curate(_corpus(), project_config)
    assert [t["id"] for t in a.traces] == [t["id"] for t in b.traces]
    assert a.filter_config == b.filter_config


def test_decontamination_without_an_eval_set_warns_loudly(project_config):
    result = curate(_corpus(), project_config, eval_task_inputs=[])
    assert any("no eval set registered" in n for n in result.notes)
    assert next(s for s in result.stages if s.name == "decontaminate").n_dropped == 0


def test_decontamination_drops_the_planted_task(project_config):
    corpus = _corpus()
    planted = corpus[0]["task_input"]
    result = curate(corpus, project_config, eval_task_inputs=[planted])
    assert corpus[0]["id"] not in {t["id"] for t in result.traces}


def test_pii_redaction_is_applied_to_survivors(project_config):
    corpus = [make_trace("t", task="my email is bob@example.com and I need help with an order")]
    result = curate(corpus, project_config)
    assert "<EMAIL>" in json.dumps(result.traces[0]["messages"])


def test_cluster_assignment_and_cap(project_config):
    project_config.curate.clusters = 3
    project_config.curate.cap_per_cluster = 2
    corpus = [
        make_trace(f"t{i}", task=f"{topic} for order {i}", closing=distinct_closing(i))
        for i, topic in enumerate(["refund my money back"] * 5 + ["change shipping address"] * 5 + ["track my package"] * 5)
    ]
    result = curate(corpus, project_config)
    stage = next(s for s in result.stages if s.name == "stratify")
    assert stage.n_dropped > 0, "caps must bite when a cluster exceeds them"
    assert result.coverage["n_clusters"] >= 1
    assert all(t.get("cluster") is not None for t in result.traces)


def test_sparse_clusters_are_called_out(project_config):
    project_config.curate.clusters = 8
    result = curate(_corpus(), project_config)
    assert any("fail first" in n for n in result.notes)


def test_hash_embedder_note_is_always_present(project_config):
    result = curate(_corpus(), project_config)
    assert any("hash` embedder" in n for n in result.notes)


def test_report_renders_every_section(project_config, tmp_path):
    result = curate(_corpus(), project_config)
    md = curation_report.render(result, "demo", 1)
    for heading in ["# Curation report: demo v1", "## What each filter dropped", "## What survived",
                    "## Reproducing this dataset"]:
        assert heading in md
    assert "| `outcome` |" in md
    # The filter config must be embedded verbatim so the dataset can be reproduced.
    assert json.dumps(result.filter_config, indent=2, sort_keys=True) in md


def test_report_writes_to_disk(project_config, tmp_path):
    result = curate(_corpus(), project_config)
    p = curation_report.write(result, "demo", 1, tmp_path / "sub" / "report.md")
    assert p.exists() and "Curation report" in p.read_text()


# --------------------------------------------------------------------------------------------------------------
# stratification internals
# --------------------------------------------------------------------------------------------------------------


def test_kmeans_recovers_separated_groups():
    import numpy as np

    # numpy broadcasting, not list concatenation
    X = np.vstack([np.zeros((10, 4)) + [1, 0, 0, 0], np.zeros((10, 4)) + [0, 1, 0, 0]])  # noqa: RUF005
    labels, centroids = kmeans(X, k=2, seed=0)
    assert len(set(labels.tolist())) == 2
    assert len(set(labels[:10].tolist())) == 1 and len(set(labels[10:].tolist())) == 1
    assert centroids.shape == (2, 4)


def test_kmeans_handles_k_larger_than_n():
    import numpy as np

    labels, centroids = kmeans(np.zeros((3, 2)) + np.arange(3).reshape(-1, 1), k=10, seed=0)
    assert len(labels) == 3 and centroids.shape[0] <= 3


def test_hash_embedder_is_deterministic_and_normalized():
    import numpy as np

    emb = HashEmbedder(32)
    a = emb.embed(["refund my order please"])
    b = emb.embed(["refund my order please"])
    assert np.allclose(a, b)
    assert np.isclose(np.linalg.norm(a[0]), 1.0)
    assert emb.name == "hash-32"


def test_hash_embedder_handles_empty_text():
    import numpy as np

    out = HashEmbedder(8).embed([""])
    assert out.shape == (1, 8) and np.all(np.isfinite(out))


def test_cap_keeps_the_first_members_in_order():
    traces = [{"id": f"t{i}"} for i in range(6)]
    assignments = {f"t{i}": 0 for i in range(6)}
    drop = cap_per_cluster(traces, assignments, cap=2)
    assert drop == {"t2", "t3", "t4", "t5"}


def test_coverage_reports_before_and_after():
    traces = [{"id": f"t{i}", "messages": []} for i in range(6)]
    assignments = {f"t{i}": i % 2 for i in range(6)}
    cov = coverage(traces, traces[:3], assignments, sparse_threshold=2)
    assert cov["n_clusters"] == 2
    assert sum(c["before"] for c in cov["clusters"]) == 6
    assert sum(c["after"] for c in cov["clusters"]) == 3


def test_assign_clusters_on_empty_input():
    assignments, _centroids = assign_clusters([], HashEmbedder(8), k=3)
    assert assignments == {}
