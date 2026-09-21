"""Dataset assembly: curated traces -> samples -> artifact -> registry."""

from __future__ import annotations

import pytest

from agentdistill.data.dataset import build_dataset
from agentdistill.data.template_check import TemplateError
from tests.conftest import TOKENIZER_DIR, make_trace

LONG_TOOLS = None


def _traces(n: int = 5) -> list[dict]:
    """Distinct enough that near-dedupe would keep them all, long enough to tokenize meaningfully."""
    return [
        make_trace(
            f"t{i}",
            task=f"Where is order {i} for my account, I have been waiting a while now",
            closing=" ".join(f"Point {j} of case {i} is confirmed as {i * 13 + j}" for j in range(6)),
        )
        for i in range(n)
    ]


def _ingest(registry, traces):
    registry.insert_traces(traces)
    return traces


def test_build_writes_artifact_and_registry_rows(project_config, registry, tokenizer):
    built = build_dataset(_ingest(registry, _traces()), project_config, name="demo", version=1, registry=registry, tokenizer=tokenizer)
    assert built.n_samples == 5
    assert built.artifact.parquet_path.exists()
    assert built.artifact.manifest_path.exists()
    assert not built.reused

    ds = registry.get_dataset("demo")
    assert ds["content_hash"] == built.artifact.content_hash
    assert ds["n_samples"] == 5
    assert ds["filter_config"] == project_config.curation_fingerprint()


def test_rebuilding_identical_content_reuses_the_version(project_config, registry, tokenizer):
    first = build_dataset(_ingest(registry, _traces()), project_config, name="demo", version=1, registry=registry, tokenizer=tokenizer)
    second = build_dataset(_ingest(registry, _traces()), project_config, name="demo", version=2, registry=registry, tokenizer=tokenizer)
    assert second.reused
    assert second.version == first.version == 1
    assert second.artifact.content_hash == first.artifact.content_hash
    assert len(registry.list_datasets()) == 1, "no duplicate dataset row"


def test_changed_corpus_produces_a_new_hash(project_config, registry, tokenizer):
    a = build_dataset(_ingest(registry, _traces(5)), project_config, name="demo", version=1, registry=registry, tokenizer=tokenizer)
    b = build_dataset(_ingest(registry, _traces(6)), project_config, name="demo", version=2, registry=registry, tokenizer=tokenizer)
    assert not b.reused
    assert a.artifact.content_hash != b.artifact.content_hash


def test_changed_filter_config_produces_a_new_hash(project_config, registry, tokenizer):
    a = build_dataset(_ingest(registry, _traces()), project_config, name="demo", version=1, registry=registry, tokenizer=tokenizer)
    b = build_dataset(_ingest(registry, _traces()), project_config, name="demo", version=2, registry=registry,
                      tokenizer=tokenizer, filter_config={"filters": ["outcome", "pii"]})
    assert a.artifact.content_hash != b.artifact.content_hash


def test_long_trajectories_become_windows(project_config, registry, tokenizer):
    project_config.dataset.max_seq_len = 330
    project_config.dataset.window_turns = 2
    built = build_dataset(_ingest(registry, _traces(3)), project_config, name="demo", version=1, registry=registry, tokenizer=tokenizer)
    kinds = built.artifact.manifest["kinds"]
    assert kinds.get("turn_window", 0) > 0
    assert any("turn windows" in n for n in built.notes)


def test_traces_that_produce_nothing_are_accounted_for(project_config, registry, tokenizer):
    project_config.dataset.max_seq_len = 16
    project_config.dataset.windows_for_long_trajectories = False
    with pytest.raises(ValueError, match="no samples were produced"):
        build_dataset(_ingest(registry, _traces()), project_config, name="demo", version=1, registry=registry, tokenizer=tokenizer)


def test_bad_template_stops_the_build(project_config, registry, tokenizer_factory):
    tok = tokenizer_factory("no_tools.jinja")
    with pytest.raises(TemplateError, match="accepts_tools"):
        build_dataset(_traces(), project_config, name="demo", version=1, registry=registry, tokenizer=tok)


def test_prefix_unstable_template_stops_the_build(project_config, registry, tokenizer_factory):
    tok = tokenizer_factory("unstable.jinja")
    with pytest.raises(TemplateError, match="prefix_stable"):
        build_dataset(_traces(), project_config, name="demo", version=1, registry=registry, tokenizer=tok)


def test_missing_base_model_is_reported(project_config, registry):
    with pytest.raises(ValueError, match="no base model"):
        build_dataset(_traces(), project_config, name="demo", version=1, registry=registry)


def test_manifest_records_lineage(project_config, registry, tokenizer):
    built = build_dataset(_ingest(registry, _traces()), project_config, name="demo", version=1, registry=registry,
                          tokenizer=tokenizer, base_model=str(TOKENIZER_DIR))
    m = built.artifact.manifest
    assert m["n_traces_in"] == 5 and m["n_traces_used"] == 5
    assert m["max_seq_len"] == project_config.dataset.max_seq_len
    assert m["agentdistill_version"]
    assert m["tokenizer"] == str(TOKENIZER_DIR)


def test_samples_rows_are_written(project_config, registry, tokenizer):
    from sqlalchemy import text

    built = build_dataset(_ingest(registry, _traces()), project_config, name="demo", version=1, registry=registry, tokenizer=tokenizer)
    with registry.engine.connect() as conn:
        n = conn.execute(text("SELECT COUNT(*) FROM samples WHERE dataset_id = :d"),
                         {"d": built.dataset_id}).scalar_one()
    assert n == built.n_samples


def test_build_without_a_registry_still_writes(project_config, tokenizer):
    built = build_dataset(_traces(), project_config, name="demo", version=1, tokenizer=tokenizer)
    assert built.artifact.parquet_path.exists()


def test_the_manifest_names_the_model_that_wrote_the_corpus():
    """The report sets it beside the serving teacher; without it, "the student imitates a scripted solver" is a
    claim nobody can check from the artifact."""
    from agentdistill.data.dataset import corpus_teacher

    assert corpus_teacher([{"teacher_model": "scripted/rule-based-teacher"}] * 3) == "scripted/rule-based-teacher"
    assert corpus_teacher([{"teacher_model": "b"}, {"teacher_model": "a"}]) == "mixed: a, b"
    assert corpus_teacher([{}, {"teacher_model": None}]) is None, "unrecorded reads as unknown, never as a model"
