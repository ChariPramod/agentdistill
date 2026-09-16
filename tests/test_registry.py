"""Registry: migrations, hash dedupe, JSON round-tripping."""

from __future__ import annotations

from agentdistill.registry.base import split_statements
from tests.conftest import make_trace


def test_migrate_is_idempotent(registry):
    from agentdistill.registry.base import SCHEMA_VERSION

    assert registry.schema_version() == SCHEMA_VERSION
    registry.migrate()
    registry.migrate()
    assert registry.schema_version() == SCHEMA_VERSION


def test_every_migration_file_is_applied():
    """A migration added to the list but not to the dialect directory would fail only at runtime."""
    from agentdistill.registry.base import MIGRATION_FILES, MIGRATIONS

    for dialect in ("sqlite", "postgres"):
        for filename in MIGRATION_FILES:
            assert (MIGRATIONS / dialect / filename).exists(), f"missing {dialect}/{filename}"


def test_insert_and_read_back(registry):
    t = make_trace("t1")
    counts = registry.insert_traces([t])
    assert counts == {"added": 1, "skipped_duplicate": 0, "read": 1}

    got = registry.get_trace("t1")
    assert got["messages"] == t["messages"]
    assert got["tools"] == t["tools"]
    assert got["task_input"] == t["task_input"]
    assert got["success"] is True, "SQLite stores booleans as integers; they must come back as bools"


def test_duplicate_content_hash_is_skipped(registry):
    t = make_trace("t1")
    registry.insert_traces([t])
    again = make_trace("t2")  # same content, different id
    counts = registry.insert_traces([again])
    assert counts["added"] == 0 and counts["skipped_duplicate"] == 1
    assert registry.count_traces() == 1


def test_duplicates_within_one_batch_are_skipped(registry):
    t = make_trace("t1")
    counts = registry.insert_traces([t, t, t])
    assert counts["added"] == 1 and counts["skipped_duplicate"] == 2


def test_list_traces_is_ordered_stably(registry):
    traces = [make_trace(f"t{i}", task=f"question number {i}") for i in range(10)]
    registry.insert_traces(traces)
    once = [t["id"] for t in registry.list_traces()]
    twice = [t["id"] for t in registry.list_traces()]
    assert once == twice, "curation reproducibility depends on a stable read order"


def test_null_success_survives(registry):
    registry.insert_traces([make_trace("t1", success=None)])
    assert registry.get_trace("t1")["success"] is None


def test_set_clusters(registry):
    registry.insert_traces([make_trace("t1"), make_trace("t2", task="another question entirely")])
    registry.set_clusters({"t1": 3, "t2": 7})
    assert {t["id"]: t["cluster"] for t in registry.list_traces()} == {"t1": 3, "t2": 7}


def test_dataset_versioning(registry):
    assert registry.next_dataset_version("d") == 1
    registry.insert_dataset({
        "id": "ds1", "name": "d", "version": 1, "kind": "sft", "filter_config": {"filters": []},
        "n_samples": 5, "n_tokens": 100, "content_hash": "abc", "path": "/tmp/d",
    })
    assert registry.next_dataset_version("d") == 2
    assert registry.get_dataset("d")["content_hash"] == "abc"
    assert registry.get_dataset_by_hash("abc")["id"] == "ds1"
    assert registry.get_dataset_by_hash("nope") is None


def test_samples_insert_in_bulk(registry):
    registry.insert_dataset({
        "id": "ds1", "name": "d", "version": 1, "kind": "sft", "filter_config": {},
        "n_samples": 2, "n_tokens": 10, "content_hash": "h", "path": "/tmp/d",
    })
    registry.insert_samples([
        {"id": f"s{i}", "dataset_id": "ds1", "trace_id": None, "kind": "trajectory",
         "n_tokens": 5, "n_target_tokens": 2, "row_idx": i, "sample_hash": f"h{i}"}
        for i in range(2)
    ])
    registry.insert_samples([])  # must not raise


def test_eval_set_roundtrip(registry):
    registry.insert_traces([make_trace("t1"), make_trace("t2", task="second eval question here")])
    registry.insert_eval_set({"id": "es1", "name": "holdout", "trace_ids": ["t1", "t2"], "grader": {"type": "label"}})
    es = registry.get_eval_set("holdout")
    assert es["trace_ids"] == ["t1", "t2"]
    assert es["grader"]["type"] == "label"
    inputs = registry.eval_set_task_inputs()
    assert len(inputs) == 2
    assert all("user" in i["task_input"] for i in inputs)


def test_eval_set_task_inputs_with_no_sets(registry):
    assert registry.eval_set_task_inputs() == []


def test_embeddings_roundtrip(registry):
    registry.insert_traces([make_trace("t1")])
    registry.upsert_embeddings("hash-8", {"t1": [0.1, 0.2, 0.3]})
    registry.upsert_embeddings("hash-8", {"t1": [0.4, 0.5, 0.6]})  # upsert, not duplicate
    got = registry.get_embeddings()
    assert got["t1"] == [0.4, 0.5, 0.6]


def test_relative_sqlite_path_resolves_against_config_root(tmp_path):
    from agentdistill.registry import open_registry

    sub = tmp_path / "project"
    sub.mkdir()
    reg = open_registry("sqlite:///.agentdistill/registry.db", root=sub)
    assert (sub / ".agentdistill" / "registry.db").exists()
    reg.close()


def test_split_statements_ignores_semicolons_in_comments():
    sql = "CREATE TABLE a (x TEXT); -- a comment; with a semicolon\nCREATE TABLE b (y TEXT);"
    stmts = split_statements(sql)
    assert len(stmts) == 2
    assert stmts[0].startswith("CREATE TABLE a")
    assert stmts[1].startswith("CREATE TABLE b")


def test_split_statements_respects_string_literals():
    sql = "INSERT INTO a VALUES ('semi; colon'); SELECT 1;"
    stmts = split_statements(sql)
    assert len(stmts) == 2
    assert "semi; colon" in stmts[0]
