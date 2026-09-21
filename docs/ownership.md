# Who owns what, phase 3f

Four work packages run in parallel. A package edits only its own files; anything else is a written request to
the lead. This file is the authority when a prompt and the code disagree about who owns something.

The lead owns `agentdistill/cli.py`, `agentdistill/config.py`, `agentdistill/retrain.py`, CI, `.gitignore`,
this file, the structure of `docs/progress.md`, `docs/gpu-day-go.md`, and the merge order.

Merge order: **WP1, WP3, WP2, WP4**. WP1 changes what the numbers mean and WP3 changes how text renders, so the
later packages are tested against a tree that already has both.

## WP1 — evaluation validity

    agentdistill/eval/live.py                 (new)
    agentdistill/eval/runner.py
    agentdistill/eval/harness.py
    agentdistill/train/onpolicy.py            (rollout tool mode only)
    agentdistill/report/assemble.py
    agentdistill/report/warnings.py           (new; see "warning codes" below)
    agentdistill/ingest/gateway_source.py
    examples/support_agent/project.yaml
    its own tests

## WP2 — lockstep review

    agentdistill/eval/lockstep.py
    agentdistill/eval/clients.py              (batch reply plumbing only; see the split below)
    tests/test_lockstep*.py, tests/test_batched_runner.py
    tests/lockstep_corpus.py                  (shared synthetic corpus, not collected)
    tests/oracle/                             (new)

## WP3 — render boundary audit

    agentdistill/data/
    agentdistill/train/dpo_data.py
    agentdistill/cascade/arg_mask.py
    agentdistill/eval/clients.py              (render call sites only; see the split below)
    agentdistill/eval/teacher_forced.py
    agentdistill/registry/retire.py           (new)
    agentdistill/registry/select.py           (retired-row filtering only)
    agentdistill/report/registry_views.py     (retired-row filtering only)
    its own tests

## WP4 — operations

    scripts/                                  (all)
    examples/support_agent/project.tiny.yaml
    examples/support_agent/gpu-day.lock.json  (new)
    agentdistill/tools/
    agentdistill/ops/                         (new)
    tests/test_preflight.py, tests/test_gpu_day_script.py, tests/test_estimate_spend.py, tests/test_export.py

## Two files two packages need, split by the lead

**`agentdistill/eval/clients.py`.** WP2 needs the batch client to return each reply with the index of the prompt
it answers; WP3 needs every render call to go through the tojson boundary. These are different functions, so:
WP2 owns `next_turns_batch` and the reply/index plumbing, WP3 owns the `_render` / `apply_chat_template` call
sites. Neither touches the other's lines, and neither runs `ruff --fix` on the file.

**Selectors (`registry/select.py`, `report/registry_views.py`).** WP3 owns them for one purpose only: skipping
retired rows. WP1 reads them from `report/assemble.py` but does not edit them.

## Warning codes

`WARNING_CODES` currently lives in `agentdistill/report/assemble.py`. WP1 moves it to
`agentdistill/report/warnings.py` as `CODES`, re-exports it from `assemble` so existing imports keep working, and
updates `tests/test_warning_code_audit.py` to read the new home. Every new code goes there **and** into the allow
or forbid list in `scripts/clean_rehearsal.sh` (WP4 owns the script; WP1 and WP4 coordinate through the lead).
A code in neither list is a bug the audit test fails on.

## Config keys, added by the lead up front

So no package is blocked on a request: `eval.tools`, `onpolicy.tools` (`live` | `replay`, default `replay`),
`ingest.exclude_teacher_turns` (default true), `gpu_day.timing_scale` (default 1.0). Example configs are set to
`live` by their owners: `project.yaml` by WP1, `project.tiny.yaml` by WP4 after WP1 merges.

## Registrations

A package that adds a command implements its body in a module it owns and hands the lead the Typer registration
to add at merge time. Expected this phase: `registry retire` (WP3), `ops lock write|check` and
`ops estimate-spend` (WP4), and `--tools` on `eval run` and `train onpolicy` (WP1).
