"""Static text the CLI writes out, kept next to the code so `agentdistill init` works from an installed wheel."""

from __future__ import annotations

from pathlib import Path

_EXAMPLE = Path(__file__).resolve().parents[1] / "project.example.yaml"

#: Falls back to the packaged copy when running from a source checkout is not possible.
EXAMPLE_PROJECT_YAML: str = (
    _EXAMPLE.read_text()
    if _EXAMPLE.exists()
    else """name: my-agent
registry: sqlite:///.agentdistill/registry.db
artifacts: ./artifacts
reports: ./reports

sources:
  - type: jsonl
    path: traces.jsonl

curate:
  filters: [outcome, schema_valid, no_error_loops, length, exact_dedupe, near_dedupe, decontaminate, pii, stratify]
  clusters: 32
  cap_per_cluster: 400

dataset:
  max_seq_len: 8192
"""
)
