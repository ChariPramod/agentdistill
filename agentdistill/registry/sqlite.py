"""SQLite entry point. The shared implementation lives in `base.Registry`; this module exists so that
`from agentdistill.registry.sqlite import open_registry` reads naturally at call sites and so that a future
SQLite-only optimization has somewhere to go."""

from __future__ import annotations

from pathlib import Path

from agentdistill.registry.base import Registry


def open_registry(url: str, root: Path | None = None, migrate: bool = True) -> Registry:
    reg = Registry(url, root=root)
    if migrate:
        reg.migrate()
    return reg
