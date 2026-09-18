"""Provenance: enough about how a row was produced to get back to it.

The recorded command alone does not reproduce a number. The same `agentdistill eval run ...` on a different commit,
on a tree with uncommitted edits, or against an edited project.yaml measures something else. So every registry
row that carries a result also carries the commit, a dirty flag, the config path, and a hash of the config bytes:

    {"command": str, "commit": str | None, "dirty": bool | None, "config_path": str | None, "config_hash": str | None}

`None` means "could not tell" (no git, not a repository, no config file), never "clean". The report prints these
beneath each command and flags a dirty tree, because a number you cannot get back to is one you cannot defend.

The CLI records which config it loaded (`set_active_config`) so registry writes can find it without a path being
threaded through every stage.
"""

from __future__ import annotations

import functools
import hashlib
import os
import subprocess
import sys
from pathlib import Path

#: argv[0] basenames that are this CLI under another name. `python -m agentdistill.cli` reports an absolute path
#: to cli.py, a console script reports its install path, and the test runner reports pytest. Anything else is left
#: exactly as it was: a wrapper script may do work of its own, and guessing would make the record less true.
KNOWN_ENTRYPOINTS = {
    "cli.py": "agentdistill",
    "__main__.py": "agentdistill",
    "__main__": "agentdistill",
    "agentdistill": "agentdistill",
    "pytest": "agentdistill",
}

_active_config: str | None = None


def set_active_config(path: str | os.PathLike | None) -> None:
    """Record the config this process loaded. Stored as the user passed it, so the report shows their path."""
    global _active_config
    _active_config = os.fspath(path) if path is not None else None


def active_config() -> str | None:
    return _active_config


def normalized_argv() -> list[str]:
    argv = list(sys.argv) or ["agentdistill"]
    replacement = KNOWN_ENTRYPOINTS.get(os.path.basename(argv[0]))
    if replacement is not None:
        argv[0] = replacement
    return argv


def command() -> str:
    return " ".join(normalized_argv())


@functools.cache
def git_state(cwd: str | None = None) -> dict:
    """Commit SHA and dirty flag of the repository containing `cwd` (the process cwd when None).

    Cached per process and directory: the commit does not change mid-run, and the dirty flag is taken once, at the
    first row written. A file edited after that is not noticed, which is the right trade for a check that shells
    out to git on every registry insert otherwise. Never raises; any failure is `None` for both fields.
    """
    # No index lock: a concurrent git command in the same tree must not fail because a run is recording itself.
    env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0", "GIT_TERMINAL_PROMPT": "0"}

    def run(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *args], cwd=cwd, env=env, capture_output=True, text=True, timeout=5, check=True
            ).stdout.strip()
        except Exception:
            return None

    sha = run("rev-parse", "HEAD")
    if sha is None:
        return {"commit": None, "dirty": None}
    status = run("status", "--porcelain")
    return {"commit": sha, "dirty": bool(status) if status is not None else None}


def config_state(path: str | os.PathLike | None) -> dict:
    """The config path as given, and the first 12 hex digits of the sha256 of its bytes (None if unreadable)."""
    if not path:
        return {"config_path": None, "config_hash": None}
    shown = os.fspath(path)
    try:
        digest = hashlib.sha256(Path(shown).resolve().read_bytes()).hexdigest()[:12]
    except OSError:
        digest = None
    return {"config_path": shown, "config_hash": digest}


def _git_dir_for(config_path: str | None) -> str | None:
    """Ask git about the repository holding the config, which is the project being run, when there is one."""
    if config_path:
        parent = Path(config_path).resolve().parent
        if parent.is_dir():
            return str(parent)
    return None


def provenance(config_path: str | os.PathLike | None = None, cmd: str | None = None) -> dict:
    path = os.fspath(config_path) if config_path is not None else None
    return {"command": cmd or command(), **git_state(_git_dir_for(path)), **config_state(path)}


def for_row(explicit: dict | None = None, cmd: str | None = None) -> dict:
    """What a registry insert stores: the caller's own provenance if it passed one, else this process's.

    `cmd` is the row's command column, so the two can never disagree about what was run.
    """
    if explicit is not None:
        return explicit
    return provenance(active_config(), cmd=cmd)
