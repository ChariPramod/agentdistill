"""Verify a GPU-day export on the laptop, before the box is terminated.

Three questions, in the order that matters:

1. Did the bytes arrive intact? The checksum written beside the tarball is recomputed here, not trusted.
2. What is in it? Every file is listed with its size, so a missing registry or an empty log is visible.
3. Do the numbers have a database behind them? Every run id in `report.json` is looked up in the exported
   registry. A report whose run ids are not in the registry it shipped with is a report nobody can audit, and
   that is precisely the failure this project exists to prevent.

Standard library only, on purpose: this runs on whatever laptop the tarball landed on.

    python -m agentdistill.ops.verify_export ~/Downloads/agentdistill-export-20260921T101500Z.tar.gz
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import tarfile
import tempfile
from pathlib import Path
from typing import Any

#: Which table an id belongs to, by prefix. Checked in order, and an unknown prefix is searched everywhere.
ID_TABLES: dict[str, str] = {
    "ev_": "eval_runs",
    "cal_": "calibrations",
    "ad_": "adapters",
    "ds_": "datasets",
    "tr_": "training_runs",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_run_ids(node: Any, found: set[str] | None = None) -> set[str]:
    """Every `run_id` anywhere in the report, at any depth.

    The sidecar carries them per subject, and the calibration and cascade blocks carry their own. Walking for
    the key is what keeps this honest when the report grows a section.
    """
    found = set() if found is None else found
    if isinstance(node, dict):
        for key, value in node.items():
            if key in {"run_id", "run_ids"} and value:
                found.update([value] if isinstance(value, str) else [v for v in value if isinstance(v, str)])
            else:
                collect_run_ids(value, found)
    elif isinstance(node, list):
        for item in node:
            collect_run_ids(item, found)
    return found


def _id_exists(conn: sqlite3.Connection, tables: set[str], run_id: str) -> str | None:
    """The table holding this id, or None. Tries the prefix's table first, then every other table."""
    order = [ID_TABLES.get(run_id.split("_")[0] + "_", ""), *sorted(tables)]
    for table in [t for t in order if t and t in tables]:
        try:
            row = conn.execute(f"SELECT 1 FROM {table} WHERE id = ?", (run_id,)).fetchone()
        except sqlite3.DatabaseError:
            continue
        if row:
            return table
    return None


def verify(tarball: Path, checksum: Path | None = None) -> int:
    problems: list[str] = []
    if not tarball.exists():
        print(f"FAIL export.file: no tarball at {tarball}")
        return 1

    # 1. checksum ------------------------------------------------------------------------------------------
    sums = checksum or Path(str(tarball) + ".sha256")
    actual = sha256_file(tarball)
    if not sums.exists():
        problems.append("checksum")
        print(f"FAIL export.checksum: no checksum file at {sums}; the tarball hashes to {actual} but there is "
              f"nothing to compare it with — copy the .sha256 off the box too")
    else:
        expected = sums.read_text().split()[0].strip()
        if expected == actual:
            print(f"PASS export.checksum: {actual}")
        else:
            problems.append("checksum")
            print(f"FAIL export.checksum: expected {expected}, got {actual} — the copy is corrupt or truncated; "
                  f"copy it again and do not terminate the box")

    # 2. contents ------------------------------------------------------------------------------------------
    with tarfile.open(tarball, "r:gz") as tf:
        # AppleDouble `._<name>` entries are macOS resource forks, not files anyone exported. `._registry.db`
        # ends in `.db`, so without this the verifier opens a 163-byte header as the registry and fails.
        members = [m for m in tf.getmembers() if m.isfile() and not Path(m.name).name.startswith("._")]
        print(f"PASS export.contents: {len(members)} files, "
              f"{sum(m.size for m in members) / 1e6:.1f} MB uncompressed")
        for m in sorted(members, key=lambda m: m.name):
            print(f"       {m.size:>12,}  {m.name}")

        registry_members = [m for m in members if m.name.endswith(".db")]
        report_members = [m for m in members if m.name.endswith("report.json")]
        log_members = [m for m in members if "/logs/" in m.name]
        if not registry_members:
            problems.append("registry")
            print("FAIL export.registry: the tarball holds no registry database, so none of the day's numbers "
                  "can be audited. Do not terminate the box: rerun scripts/export_results.sh on it")
        if not log_members:
            print("SKIP export.logs: no logs in the tarball (nothing to diagnose a failure with)")
        else:
            print(f"PASS export.logs: {len(log_members)} log file(s)")

        if not registry_members:
            return 1
        with tempfile.TemporaryDirectory() as tmp:
            tf.extractall(tmp, members=registry_members + report_members)
            db = Path(tmp) / registry_members[0].name
            conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            try:
                tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            except sqlite3.DatabaseError as e:
                print(f"FAIL export.registry: the exported registry does not open as a database: {e}")
                return 1
            n_runs = conn.execute("SELECT COUNT(*) FROM eval_runs").fetchone()[0] if "eval_runs" in tables else 0
            print(f"PASS export.registry: opens, {len(tables)} tables, {n_runs} eval run(s)")

            # 3. the report's run ids --------------------------------------------------------------------
            if not report_members:
                problems.append("report")
                print("FAIL export.report: no report.json in the tarball, so there is nothing to check the "
                      "registry against — the `report` stage never ran, or it ran and wrote nothing")
            else:
                data = json.loads((Path(tmp) / report_members[0].name).read_text())
                ids = sorted(collect_run_ids(data))
                if not ids:
                    problems.append("report")
                    print("FAIL export.run_ids: report.json carries no run ids; every number in a report must "
                          "carry one")
                else:
                    missing = [i for i in ids if _id_exists(conn, tables, i) is None]
                    if missing:
                        problems.append("run_ids")
                        print(f"FAIL export.run_ids: {len(missing)} of {len(ids)} run id(s) in report.json are "
                              f"not in the exported registry: {', '.join(missing)}")
                    else:
                        print(f"PASS export.run_ids: all {len(ids)} run id(s) in report.json resolve in the "
                              f"exported registry ({', '.join(ids[:4])}{'…' if len(ids) > 4 else ''})")
            conn.close()

    print()
    if problems:
        print(f"export NOT verified ({', '.join(sorted(set(problems)))}). DO NOT TERMINATE THE BOX.")
        return 1
    print("export verified: checksum matches, the registry opens, and every run id in the report is in it.")
    print("The box can be terminated.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tarball", type=Path)
    ap.add_argument("--checksum", type=Path, default=None, help="Defaults to <tarball>.sha256.")
    args = ap.parse_args(argv)
    return verify(args.tarball, args.checksum)


if __name__ == "__main__":
    raise SystemExit(main())
