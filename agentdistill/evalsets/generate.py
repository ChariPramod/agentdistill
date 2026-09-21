"""Deterministic selection of a generated eval set.

A clean rehearsal deletes the registry and rebuilds everything, including the small eval sets it carves out of
the example corpus. If that carving depends on file order, `set` iteration or an unseeded RNG, two clean runs
evaluate on different tasks and their numbers are not comparable -- and nothing says so, because each run's set
is internally consistent. So selection is a pure function of the task identities and a salt:

- each task is ranked by `sha256(salt | scenario | instance)` and the first `n` are kept. No RNG, no set
  iteration, and the result does not depend on the order the corpus lists its tasks in;
- the set's identity is `eval_set_hash`, over the *sorted* task ids and the grader, so two selections of the
  same tasks hash equal whatever order they were written in.

The salt is per set name. Two sets drawn from overlapping pools with the same salt would pick the same
top-ranked tasks, which is exactly the correlation a holdout must not have.

    python -m agentdistill.evalsets.generate --src eval-holdout.jsonl --dst eval-holdout-tiny.jsonl \\
        --n 10 --salt support-holdout-tiny
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def stable_task_ids(scenarios: list[dict], n: int, salt: str) -> list[str]:
    """Deterministic selection: rank by hash of (salt, scenario, instance index), take n. No RNG, no set
    iteration order."""
    ranked = sorted(
        (
            (hashlib.sha256(f"{salt}|{s['scenario']}|{s['instance']}".encode()).hexdigest(), s["task_id"])
            for s in scenarios
        ),
        # The task id breaks a hash tie (only possible for duplicate scenario/instance pairs), so the order is
        # total and never falls back to input order.
        key=lambda t: (t[0], t[1]),
    )
    return [task_id for _, task_id in ranked[:n]]


def eval_set_hash(task_ids: list[str], grader: dict) -> str:
    """The identity of an eval set: which tasks, graded how. Order-free."""
    return hashlib.sha256(
        json.dumps({"tasks": sorted(task_ids), "grader": grader}, sort_keys=True).encode()
    ).hexdigest()[:12]


def row_task_id(row: dict) -> str:
    tid = row.get("task_id") or row.get("id")
    if not tid:
        raise ValueError(f"row has neither task_id nor id: {sorted(row)}")
    return str(tid)


def scenario_of(row: dict) -> dict:
    """The `{"scenario", "instance", "task_id"}` triple for one trace row.

    The support example records the scenario in `task_input.scenario` (and `metadata.scenario`) and the instance
    as `metadata.db_seed`. Rows from elsewhere may have neither; the task id then stands in for both, which is
    still deterministic -- it only loses the property that the rank is readable as "scenario, instance".
    """
    tid = row_task_id(row)
    raw_input, raw_meta = row.get("task_input"), row.get("metadata")
    task_input: dict = raw_input if isinstance(raw_input, dict) else {}
    metadata: dict = raw_meta if isinstance(raw_meta, dict) else {}
    scenario = task_input.get("scenario") or metadata.get("scenario") or tid
    instance = metadata.get("db_seed")
    if instance is None:
        instance = metadata.get("instance", tid)
    return {"scenario": str(scenario), "instance": str(instance), "task_id": tid}


def grader_of(rows: list[dict]) -> dict:
    """The grader the rows carry, for the set hash. Rows disagreeing is recorded rather than guessed at."""
    graders = {json.dumps(r.get("grader"), sort_keys=True) for r in rows}
    if len(graders) == 1:
        return json.loads(next(iter(graders))) or {}
    return {"mixed": sorted(graders)}


def select_rows(rows: list[dict], n: int, salt: str) -> list[dict]:
    """Pick `n` distinct tasks from `rows`, returned in rank order.

    A task that appears more than once (several recorded attempts) keeps the row with the smallest canonical
    JSON, not the first one in the file, so the output does not depend on input order even then.
    """
    by_task: dict[str, dict] = {}
    for r in sorted(rows, key=lambda r: json.dumps(r, sort_keys=True)):
        by_task.setdefault(row_task_id(r), r)
    scenarios = [scenario_of(r) for r in by_task.values()]
    return [by_task[tid] for tid in stable_task_ids(scenarios, n, salt)]


def read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_set(rows: list[dict], dst: str | Path) -> None:
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    # `sort_keys` so the bytes depend only on the rows' content, not on the key order of the source file.
    dst.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Select a deterministic eval set from a JSONL corpus.")
    ap.add_argument("--src", type=Path, action="append", required=True,
                    help="Source JSONL; repeat to draw from several files, earlier files first.")
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--salt", required=True, help="Per-set salt; use the set's name.")
    args = ap.parse_args(argv)

    rows: list[dict] = []
    for src in args.src:
        rows.extend(read_jsonl(src))
    picked = select_rows(rows, args.n, args.salt)
    if len(picked) < args.n:
        print(f"{args.dst}: only {len(picked)} distinct tasks available, asked for {args.n}")
        return 2
    write_set(picked, args.dst)
    print(f"{args.dst}: {len(picked)} tasks, hash {eval_set_hash([row_task_id(r) for r in picked], grader_of(picked))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
