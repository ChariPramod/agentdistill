"""Registry: migrations, hash dedupe, JSON round-tripping."""

from __future__ import annotations

import pytest

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


# --------------------------------------------------------------------------------------------------------------
# migrations apply once
#
# They used to re-run on every open, which was fine while every migration only added columns and indexes. The
# `rft` dataset kind needed a CHECK constraint changed, SQLite cannot alter one in place, and rebuilding the
# datasets table on every process start is both wasteful and a window in which an interrupted run loses rows.
# --------------------------------------------------------------------------------------------------------------


def test_migrations_are_recorded_and_not_replayed(tmp_path):
    from sqlalchemy import text

    from agentdistill.registry import open_registry
    from agentdistill.registry.base import MIGRATION_FILES

    url = f"sqlite:///{tmp_path}/r.db"
    reg = open_registry(url)
    try:
        with reg.engine.connect() as conn:
            applied = {r[0] for r in conn.execute(text("SELECT filename FROM schema_migrations")).fetchall()}
        assert applied == set(MIGRATION_FILES)
    finally:
        reg.close()

    # A second open must not re-run anything. A row written before it has to survive.
    reg = open_registry(url)
    try:
        reg.insert_dataset({"id": "ds1", "name": "d", "version": 1, "kind": "sft", "filter_config": {},
                            "n_samples": 1, "n_tokens": 1, "content_hash": "h", "path": "/tmp/d"})
    finally:
        reg.close()

    reg = open_registry(url)
    try:
        assert reg.get_dataset("ds1") is not None, "reopening the registry destroyed a row"
        with reg.engine.connect() as conn:
            rows = conn.execute(text("SELECT COUNT(*) FROM schema_migrations")).scalar()
        assert rows == len(MIGRATION_FILES), "a migration was recorded twice"
    finally:
        reg.close()


def test_a_database_predating_the_ledger_migrates_without_losing_rows(tmp_path):
    """The upgrade path for an existing registry: no ledger, every migration already applied."""
    from sqlalchemy import text

    from agentdistill.registry import open_registry

    url = f"sqlite:///{tmp_path}/legacy.db"
    reg = open_registry(url)
    try:
        reg.insert_dataset({"id": "ds_keep", "name": "keep", "version": 1, "kind": "sft", "filter_config": {},
                            "n_samples": 7, "n_tokens": 70, "content_hash": "h", "path": "/tmp/keep"})
        with reg.engine.begin() as conn:
            conn.execute(text("DROP TABLE schema_migrations"))
    finally:
        reg.close()

    reg = open_registry(url)
    try:
        kept = reg.get_dataset("ds_keep")
        assert kept is not None and kept["n_samples"] == 7
    finally:
        reg.close()


def test_rft_is_a_dataset_kind_of_its_own(tmp_path):
    """Not filed as `sft`, because `latest_dataset(kind='sft')` is what the retrain loop trains on -- and a
    round's self-generated rollouts showing up there would have the next retrain train the student on its own
    output instead of on curated traffic."""
    from agentdistill.registry import open_registry
    from agentdistill.registry.select import latest_dataset

    reg = open_registry(f"sqlite:///{tmp_path}/r.db")
    try:
        for kind in ("sft", "dpo", "eval", "rft"):
            reg.insert_dataset({"id": f"ds_{kind}", "name": kind, "version": 1, "kind": kind,
                                "filter_config": {}, "n_samples": 1, "n_tokens": 1,
                                "content_hash": kind, "path": f"/tmp/{kind}"})

        assert latest_dataset(reg, kind="rft")["id"] == "ds_rft"
        assert latest_dataset(reg, kind="sft")["id"] == "ds_sft"
    finally:
        reg.close()


def test_an_unknown_dataset_kind_is_still_refused(tmp_path):
    import sqlalchemy.exc

    from agentdistill.registry import open_registry

    reg = open_registry(f"sqlite:///{tmp_path}/r.db")
    try:
        with pytest.raises(sqlalchemy.exc.IntegrityError):
            reg.insert_dataset({"id": "x", "name": "x", "version": 1, "kind": "whatever", "filter_config": {},
                                "n_samples": 1, "n_tokens": 1, "content_hash": "h", "path": "/tmp/x"})
    finally:
        reg.close()
