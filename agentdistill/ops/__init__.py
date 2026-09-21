"""Operations: the things that make a rented GPU session safe to start and safe to end.

Four modules, each runnable on its own so the shell scripts do not depend on a CLI registration:

    python -m agentdistill.ops.lock write|check --config <cfg>
    python -m agentdistill.ops.spend --config <cfg>
    python -m agentdistill.ops.preflight --config <cfg>
    python -m agentdistill.ops.verify_export <tarball>

The Typer commands (`agentdistill ops lock write|check`, `agentdistill ops estimate-spend`) are thin wrappers
the lead registers in `cli.py`; the bodies live here, where this package owns them.
"""
