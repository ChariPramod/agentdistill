"""The schema a fresh registry ends up with, pinned.

A clean rehearsal creates the registry from nothing, which is the one path a long-lived development database
never exercises: every migration in order against an empty file. A migration that only worked because an earlier
hand-run statement had already created a column shows up here as a diff against the snapshot, not on the GPU box.

The snapshot is `PRAGMA table_info` for every table, normalized to plain JSON. When a migration changes the
schema on purpose, regenerate it and review the diff like any other change:

    python -m tests.test_migrations_snapshot --write
    # or: AGENTDISTILL_WRITE_SCHEMA_SNAPSHOT=1 pytest tests/test_migrations_snapshot.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from sqlalchemy import text

from agentdistill.registry.base import MIGRATION_FILES
from agentdistill.registry.sqlite import open_registry

SNAPSHOT = Path(__file__).resolve().parent / "fixtures" / "schema_snapshot.json"


def schema_of(registry) -> dict:
    """`{table: [{name, type, notnull, default, pk}, ...]}`, tables sorted by name, columns in declared order.

    Column order is kept rather than sorted: it is part of what `SELECT *` returns, and a table rebuild that
    reordered columns is a real change. `sqlite_*` internals are left out; they belong to SQLite, not to us.
    """
    with registry.engine.connect() as conn:
        tables = [
            r[0]
            for r in conn.execute(
                text("SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name")
            )
        ]
        out: dict = {}
        for table in tables:
            cols = conn.execute(text(f'PRAGMA table_info("{table}")')).mappings().all()
            out[table] = [
                {
                    "name": c["name"],
                    "type": (c["type"] or "").upper(),
                    "notnull": int(c["notnull"]),
                    "default": c["dflt_value"],
                    "pk": int(c["pk"]),
                }
                for c in cols
            ]
    return out


def fresh_schema(tmp_dir: Path) -> dict:
    reg = open_registry(f"sqlite:///{tmp_dir / 'fresh.db'}")
    try:
        return schema_of(reg)
    finally:
        reg.close()


def write_snapshot(schema: dict) -> None:
    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n")


def test_fresh_registry_matches_the_snapshot(tmp_path):
    schema = fresh_schema(tmp_path)
    if os.environ.get("AGENTDISTILL_WRITE_SCHEMA_SNAPSHOT") == "1":
        write_snapshot(schema)
    assert SNAPSHOT.exists(), "no schema snapshot; run `python -m tests.test_migrations_snapshot --write`"
    expected = json.loads(SNAPSHOT.read_text())
    assert sorted(schema) == sorted(expected), "the set of tables changed; regenerate the snapshot if intended"
    for table in expected:
        assert schema[table] == expected[table], (
            f"{table} differs from tests/fixtures/schema_snapshot.json; if a migration changed it on purpose, "
            f"regenerate with `python -m tests.test_migrations_snapshot --write` and review the diff"
        )


def test_every_migration_is_recorded_once(tmp_path):
    reg = open_registry(f"sqlite:///{tmp_path / 'fresh.db'}")
    try:
        with reg.engine.connect() as conn:
            applied = [r[0] for r in conn.execute(text("SELECT filename FROM schema_migrations ORDER BY filename"))]
        assert applied == sorted(MIGRATION_FILES)
    finally:
        reg.close()


def test_migrating_twice_is_a_no_op(tmp_path):
    reg = open_registry(f"sqlite:///{tmp_path / 'fresh.db'}")
    try:
        before = schema_of(reg)
        with reg.engine.connect() as conn:
            ledger = conn.execute(text("SELECT COUNT(*) FROM schema_migrations")).scalar()
            versions = conn.execute(text("SELECT COUNT(*) FROM schema_version")).scalar()
        reg.migrate()
        with reg.engine.connect() as conn:
            assert conn.execute(text("SELECT COUNT(*) FROM schema_migrations")).scalar() == ledger
            assert conn.execute(text("SELECT COUNT(*) FROM schema_version")).scalar() == versions
        assert schema_of(reg) == before
    finally:
        reg.close()


if __name__ == "__main__":
    import tempfile

    if "--write" not in sys.argv[1:]:
        print("usage: python -m tests.test_migrations_snapshot --write")
        raise SystemExit(2)
    with tempfile.TemporaryDirectory() as d:
        write_snapshot(fresh_schema(Path(d)))
    print(f"wrote {SNAPSHOT}")
