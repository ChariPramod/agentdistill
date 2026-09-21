"""Every warning code the report can carry.

The list lives in its own module because two things have to agree about it and neither should have to import the
report assembler to find out: `ReportData.warn`, which refuses anything not listed here, and
`scripts/clean_rehearsal.sh`, which must place every code as allowed or forbidden. A code in neither list of the
script passes the rehearsal silently, which is the exact class of silence this project keeps finding, so
`tests/test_warning_code_audit.py` fails on it.

Codes are lower_snake_case strings. Adding one here without placing it in the script is a bug, not a warning.
"""

from __future__ import annotations

CODES: tuple[str, ...] = (
    "tiny_mode", "no_eval_set", "no_run_found", "teacher_skipped", "no_student", "paired_failed",
    "no_calibration", "gate_not_usable", "gate_degenerate", "cascade_unverified", "quantized_unevaluated",
    "quantization_missing", "no_teacher_run", "no_teacher_config", "no_pricing", "no_prompt_tokens",
    "no_throughput", "cost_unbatched", "replay_teacher", "dirty_tree",
    # Phase 3f. `eval_mode_mismatch` belongs in the FORBID list: a report that pairs a live run against a replay
    # run has compared two different questions. `corpus_teacher_differs` belongs in the ALLOW list: it is a true
    # description of an operational comparison, not a hole.
    "eval_mode_mismatch", "corpus_teacher_differs",
)
