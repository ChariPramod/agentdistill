"""The GPU-day lock file: what the day must reproduce, in one committed artifact.

`examples/support_agent/gpu-day.lock.json` records the identity of every input the day depends on -- the base
model and the revision it is pinned to, the tool parser, the corpus `make_corpus.sh` rebuilds, the SFT dataset's
content hash, and each frozen eval set's hash. A fresh clone that rebuilds the corpus and re-curates must land on
the same hashes, and `check` says which one did not.

Two rules this file exists to enforce:

- **Hashes are never parsed out of prose.** They come from the registry and from the files on disk. A number
  copied into `docs/progress.md` and read back is a number nobody can recompute.
- **A mismatch is loud and readable.** `check` exits 1 and prints the field, the locked value and the current
  one, because "the lock failed" at 9am on a rented box is not an actionable message.

The dataset is locked by `content_hash` alone, not by name and version. Curation is deterministic, so a fresh
clone reproduces the hash -- but it mints version 1 where the laptop that wrote the lock was on version 2, and
locking the version would fail every fresh clone for a difference that means nothing.

    python -m agentdistill.ops.lock write --config examples/support_agent/project.yaml
    python -m agentdistill.ops.lock check --config examples/support_agent/project.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

LOCK_FILENAME = "gpu-day.lock.json"
LOCK_VERSION = 1

#: The files `scripts/make_corpus.sh` writes. The corpus is generated and gitignored, so these hashes are the
#: only way a fresh clone can show it rebuilt the same corpus the lock was written against.
CORPUS_FILES: tuple[str, ...] = (
    "traces.jsonl",
    "traces-train.jsonl",
    "eval-holdout.jsonl",
    "eval-unseen.jsonl",
    "eval-calib.jsonl",
)

MISSING = "<missing>"

#: Fields of a recorded trace that are measurements of the machine that recorded it, not content. They are
#: excluded from the corpus hash, because a lock that fails on a one-millisecond timing difference is a lock
#: everyone learns to ignore.
#:
#: Found by rebuilding the corpus from `scripts/make_corpus.sh` and comparing: 799 of 800 traces were
#: byte-identical, and the 800th differed only in `metadata.latency_ms` (0 ms against 1 ms). The recipe claims
#: byte-identical output and is one wall-clock field away from it; see the request to the lead in this phase's
#: report and the divergence entry in docs/progress.md.
VOLATILE_TRACE_FIELDS: tuple[tuple[str, ...], ...] = (("metadata", "latency_ms"),)


class LockError(RuntimeError):
    """The lock cannot be built at all -- no dataset, no eval sets. Distinct from a mismatch, which is a diff."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _drop_volatile(row: Any) -> Any:
    if not isinstance(row, dict):
        return row
    row = json.loads(json.dumps(row))  # a copy, so the caller's object is untouched
    for path in VOLATILE_TRACE_FIELDS:
        node = row
        for key in path[:-1]:
            node = node.get(key) if isinstance(node, dict) else None
            if node is None:
                break
        if isinstance(node, dict):
            node.pop(path[-1], None)
    return row


def corpus_file_hash(path: Path) -> str:
    """The content hash of one corpus file.

    JSONL is hashed record by record in canonical form with the volatile fields dropped, so the hash is an
    identity of what the corpus *says* rather than of how the bytes happened to be laid out on the machine that
    recorded it. Anything else is hashed as bytes.
    """
    if path.suffix != ".jsonl":
        return sha256_file(path)
    h = hashlib.sha256()
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            h.update(json.dumps(_drop_volatile(json.loads(line)), sort_keys=True, separators=(",", ":")).encode())
            h.update(b"\n")
    return h.hexdigest()


def corpus_state(root: Path) -> dict[str, Any]:
    """Per-file hashes of the generated corpus, plus one hash over all of them.

    Per-file as well as combined so that a diff can name the file that changed. A missing file is recorded as
    missing rather than skipped: "the corpus was never built" and "the corpus changed" are different failures and
    the check must be able to say which.
    """
    files = {name: (corpus_file_hash(root / name) if (root / name).exists() else MISSING) for name in CORPUS_FILES}
    combined = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {"hash": combined, "files": files}


def _task_ids(registry: Any, trace_ids: list[str]) -> list[str]:
    """Task ids for a set of trace ids. The eval-set hash is over tasks, not over trace rows."""
    from sqlalchemy import text

    out: list[str] = []
    with registry.engine.connect() as conn:
        for i in range(0, len(trace_ids), 500):
            chunk = trace_ids[i : i + 500]
            params = {f"i{j}": v for j, v in enumerate(chunk)}
            ph = ", ".join(f":{k}" for k in params)
            rows = conn.execute(text(f"SELECT id, task_id FROM traces WHERE id IN ({ph})"), params).fetchall()
            out.extend(str(r[1] or r[0]) for r in rows)
    return out


def eval_set_hashes(registry: Any) -> dict[str, str]:
    """`{name: eval_set_hash}` for every registered eval set.

    Every set, rather than the two the config names: the day evaluates holdout, unseen and calibration, and the
    unseen set is named by the script rather than by the config. Locking whatever is registered also makes a
    fourth set appearing -- which would change what "the eval set" means -- a visible diff rather than a silence.
    """
    from sqlalchemy import text

    from agentdistill.evalsets.generate import eval_set_hash
    from agentdistill.registry.base import loads

    out: dict[str, str] = {}
    with registry.engine.connect() as conn:
        rows = conn.execute(text("SELECT name, trace_ids, grader FROM eval_sets ORDER BY name")).fetchall()
    for name, trace_ids, grader in rows:
        ids = list(trace_ids) if registry.dialect == "postgres" else loads(trace_ids)
        out[str(name)] = eval_set_hash(_task_ids(registry, list(ids)), loads(grader) or {})
    return out


def sft_dataset_hash(registry: Any) -> str:
    """The content hash of the newest SFT dataset, or MISSING when nothing has been curated."""
    sft = [d for d in registry.list_datasets() if d.get("kind") == "sft"]
    if not sft:
        return MISSING
    newest = max(sft, key=lambda d: (d.get("version") or 0, d.get("created_at") or ""))
    return str(newest.get("content_hash") or MISSING)


def build_lock(cfg: Any, registry: Any) -> dict:
    """The lock as this tree would write it now."""
    if cfg.train is None:
        raise LockError("this project has no `train` section, so there is nothing to lock")
    parser = cfg.train.tool_parser
    return {
        "lock_version": LOCK_VERSION,
        "note": (
            "What the GPU day must reproduce. Written by `agentdistill ops lock write`, checked by "
            "`agentdistill ops lock check`. Never edited by hand: a hand-edited lock locks nothing."
        ),
        # The configured string, not the resolved path: a resolved local path is machine-specific and this file
        # is committed. For the real project it is the Hub id, which is what the day pins.
        "base_model": cfg.base_model,
        "base_model_revision": cfg.train.base_model_revision,
        "tool_parser": {"name": parser.name, "family": parser.family},
        "corpus": corpus_state(cfg.root),
        "sft_dataset": {"content_hash": sft_dataset_hash(registry)},
        "eval_sets": eval_set_hashes(registry),
    }


def lock_path(cfg: Any, path: str | Path | None = None) -> Path:
    return Path(path) if path else cfg.root / LOCK_FILENAME


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    """`{"corpus.files.traces.jsonl": "ab12…"}`, so a diff names a field rather than printing two blobs."""
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            out.update(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
        return out
    return {prefix: value}


def diff_locks(locked: dict, current: dict) -> list[str]:
    """Readable lines describing every field where the two disagree. Empty when they match."""
    ignore = {"note", "lock_version"}
    a, b = _flatten(locked), _flatten(current)
    lines: list[str] = []
    for key in sorted(set(a) | set(b)):
        if key.split(".")[0] in ignore:
            continue
        want, have = a.get(key, "<not in lock>"), b.get(key, "<not in this tree>")
        if want != have:
            lines.append(f"  {key}\n      locked:  {want}\n      current: {have}")
    return lines


def write(config: str, path: str | Path | None = None) -> int:
    """Write the lock from the current registry and config. Returns a process exit code."""
    cfg, registry = _open(config)
    try:
        lock = build_lock(cfg, registry)
    except LockError as e:
        print(f"cannot write the lock: {e}")
        return 1
    out = lock_path(cfg, path)
    out.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out}")
    for key, value in sorted(_flatten(lock).items()):
        if key.split(".")[0] not in {"note", "lock_version"}:
            print(f"  {key} = {value}")
    unbuilt = [k for k, v in _flatten(lock).items() if v == MISSING]
    if unbuilt:
        print("\nsome inputs are not built in this tree, and the lock records them as missing:")
        for key in unbuilt:
            print(f"  {key}")
        print("run `bash scripts/make_corpus.sh` and `agentdistill curate`, then write the lock again.")
    return 0


def check(config: str, path: str | Path | None = None) -> int:
    """Compare the lock on disk with this tree. Exit 1 with a diff on any mismatch."""
    cfg, registry = _open(config)
    target = lock_path(cfg, path)
    if not target.exists():
        print(f"FAIL: no lock file at {target}; write one with `agentdistill ops lock write`")
        return 1
    try:
        locked = json.loads(target.read_text())
    except json.JSONDecodeError as e:
        print(f"FAIL: {target} is not valid JSON: {e}")
        return 1
    try:
        current = build_lock(cfg, registry)
    except LockError as e:
        print(f"FAIL: {e}")
        return 1

    lines = diff_locks(locked, current)
    if not lines:
        print(f"lock ok: {target} matches this tree "
              f"({len(current['eval_sets'])} eval sets, dataset {current['sft_dataset']['content_hash'][:12]})")
        return 0
    print(f"FAIL: {target} does not match this tree:")
    print("\n".join(lines))
    print(
        "\nA corpus mismatch means `bash scripts/make_corpus.sh` produced different files than the lock was\n"
        "written against; a dataset mismatch means curation ran with a different config, tokenizer or corpus.\n"
        "Fix the input, or -- if the change is intended -- rewrite the lock with `agentdistill ops lock write`\n"
        "and commit it, so what the day reproduces is a decision rather than a drift."
    )
    return 1


def _open(config: str) -> tuple[Any, Any]:
    from agentdistill.config import ProjectConfig
    from agentdistill.registry import open_registry

    cfg = ProjectConfig.load(config)
    return cfg, open_registry(cfg.registry, root=cfg.root)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["write", "check"])
    ap.add_argument("--config", default="project.yaml")
    ap.add_argument("--path", default=None, help="Lock file path. Defaults to gpu-day.lock.json beside the config.")
    args = ap.parse_args(argv)
    return write(args.config, args.path) if args.action == "write" else check(args.config, args.path)


if __name__ == "__main__":
    raise SystemExit(main())
