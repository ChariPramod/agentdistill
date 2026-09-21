# GPU day: go or no-go

Evidence for each of the nine go criteria in phase 3f, taken at commit `44a8b63` on branch `phase3d`, on
2026-09-21. Each criterion names the command that proves it and the line of output that does.

**Verdict: NO-GO**, on one criterion only. Criterion 1 needs the repository pushed to the private GitHub repo the
owner chose; nothing else is outstanding. Every other criterion is green.

| # | Criterion | State |
|---|---|---|
| 1 | HEAD committed and pushed; clean rehearsal green from a fresh clone | **RED**: committed, not pushed |
| 2 | `project.yaml`: teacher, Hub base pinned by SHA, verified parser, pricing seeded | green |
| 3 | Real dataset rebuilt with the pinned tokenizer; guard passes; lock reproduced by a fresh clone | green |
| 4 | Live tools for evals and rollouts; mode recorded; `compare` refuses mixed modes | green |
| 5 | Render-boundary test on the pinned Qwen template; no retired row selectable | green |
| 6 | Lockstep equivalence tested against the independent oracle | green |
| 7 | `preflight.sh` passes locally for every non-GPU check | green, see note |
| 8 | `gpu_day.sh` ends with the export, which also runs when a stage fails | green |
| 9 | Teacher spend estimate recorded, so the owner can set the cap | green |

Full suite: **1365 passed, 5 skipped**. `ruff` and `mypy` clean.

## 1. Committed, pushed, and green from a fresh clone

- Committed: `git status --short` is empty; `preflight.sh` → `PASS git.clean: working tree clean at 44a8b63`.
- **Not pushed.** `preflight.sh` → `FAIL git.pushed: this repository has no remote`. The owner chose a private
  GitHub repository. Create it empty, then:
  `git remote add origin <url> && git push -u origin --all && git push --tags`.
- Fresh clone of `44a8b63`, `bash scripts/clean_rehearsal.sh` with no dirty-tree allowance →
  `report ok: subjects ['base', 'student', 'teacher'], 5 warning(s) ['corpus_teacher_differs', 'cost_unbatched',
  'gate_degenerate', 'replay_teacher', 'tiny_mode']` and `clean rehearsal ok`. All five are on the allow list;
  none is forbidden.
- Nightly: `.github/workflows/ci.yml` gains a `rehearsal` job running the clean rehearsal from a fresh checkout.

## 2. The GPU-day configuration

`bash scripts/preflight.sh`:

    PASS config.teacher_model: anthropic/claude-opus-5
    PASS config.base_model: Qwen/Qwen2.5-7B-Instruct is a Hub id
    PASS config.base_model_revision: pinned at a09a35458c702b33eeacc393d103063234e8bc28
    PASS config.tool_parser: name=hermes family=hermes
    PASS registry.pricing: anthropic/claude-opus-5 from 2026-09-20: $5/Mtok in, $25/Mtok out

The parser is the one `agentdistill base-check Qwen/Qwen2.5-7B-Instruct` verified: `tool_call_roundtrip PASS via
fallback:hermes`. It uses the regex fallback because vLLM does not install on this Mac. The box re-verifies against
vLLM's real parser. Pricing is seeded with `agentdistill pricing set`, which `bootstrap_box.sh` runs on the box.

## 3. The dataset, the guard, and the lock

- `preflight.sh` → `PASS tokenizer.guard: dataset ds_7a4bd4ffe877451c was tokenized for Qwen/Qwen2.5-7B-Instruct
  at a09a35458c702b33eeacc393d103063234e8bc28`.
- `preflight.sh` → `PASS lock: gpu-day.lock.json matches this tree`.
- **Reproduced by a fresh clone.** In a new clone of `44a8b63`: `make_corpus.sh`, then ingest, eval-set
  registration and `curate` with `project.yaml`, then `agentdistill ops lock check` →
  `lock ok: ... matches this tree (3 eval sets, dataset 7a4bd4ffe877)`.

The first lock failed this check, and was right to. It had been written from the laptop's registry, whose eval
sets were frozen from an older corpus, and it pinned a 342-sample dataset that no fresh clone could build (a fresh
clone builds 334). The registry was rebuilt from the committed recipe and the lock rewritten. See
`docs/progress.md`.

## 4. Live tools and evaluation mode

- Configs: `preflight.sh` → `PASS config.eval_tools: live`, `PASS config.onpolicy_tools: live`. Tiny mode is
  live too, so the rehearsal exercises the GPU day's path.
- Every run records its mode, and a mixed-mode comparison is refused: `tests/test_live_tools.py`
  (`test_every_run_records_the_mode_it_was_graded_under`,
  `test_comparing_a_live_run_against_a_replay_run_is_refused`,
  `test_no_renderer_prints_statistics_for_an_incompatible_comparison`).
- The fresh-clone rehearsal's `report.json` has an evaluation-mode section, `onpolicy.tools_mode: live`, and
  `corpus_teacher: scripted/rule-based-teacher` beside `serving_teacher: replay-stub`.
- Live grading reproduces the recording, and does not penalise a valid alternative call:
  `test_live_replays_scripted_calls_to_the_same_outcome`,
  `test_live_mode_does_not_penalize_valid_alternative_calls`.

## 5. The render boundary, and retirement

- `tests/test_render_boundary.py` and `tests/test_real_template.py` run against the pinned Qwen revision from
  the local HF cache and pass. No rendered path contains `"arguments": "{`.
- `tests/test_retire.py`: every selector (`dataset latest`, `adapter best|latest`, prod, canary, the report's
  quantized selector) skips a retired row.
- On the rebuilt local registry nothing predates the fix, so nothing needs retiring. On the old registry,
  `registry retire --built-before 2026-09-20T07:00:00Z` retired `ds_99a6c87b9a49b745`, the fixture-tokenizer
  dataset.

## 6. Lockstep against the oracle

`tests/test_lockstep_equivalence.py` covers:

- 40 synthetic tasks, including diverging, malformed and max-turns tasks, run through the batched runner and
  the Appendix A oracle, which imports nothing from `agentdistill`.
- Two hand-written transcripts, and the order, timing, refill, repeat and isolation checks.

WP2 found no bug. Its change is that a reply now carries the index of the prompt it answers.

## 7. Pre-flight, locally

`bash scripts/preflight.sh` → `preflight: 12 PASS, 2 FAIL, 3 SKIP`. The three SKIPs are the hardware checks,
which are skipped on a machine with no `nvidia-smi`. The two FAILs are owner actions, not defects:

- `git.pushed`: see criterion 1.
- `env.api_key`: the key is exported by the owner in the box's shell, by design. It never exists on the laptop,
  in the repo, or in a prompt.

`tests/test_preflight.py::test_preflight_never_prints_any_part_of_the_key` checks that no 8-character substring of
a sentinel key ever reaches the output.

## 8. The export, including on failure

- `gpu_day.sh` ends with `stage export` and sets `trap on_exit EXIT`. The trap exports whenever the export
  stage did not complete: a failed stage, an exit 3, or an interrupt.
- The tiny day's tarball, checked by `bash scripts/verify_export.sh <tarball>`:
  `export verified: checksum matches, the registry opens, and every run id in the report is in it.`
- The first verification failed. The verifier had opened a macOS `._registry.tiny.db` resource fork as the
  registry. The export now disables those forks and the verifier skips them.

## 9. Teacher spend

`agentdistill ops estimate-spend --config examples/support_agent/project.yaml`:

    TOTAL                                                 11,847,403      567,299   73.42
    Total: $73.42.  RECOMMENDED CAP: $146.84 (twice the estimate)

This is an upper bound. Every escalating stage is costed as if every turn escalated, with a 1.3 wordiness factor,
and prompt caching (which can only lower the figure) is not modelled. **Set the teacher workspace's spend limit
to $147.**

## Left for the owner

1. Push to the private GitHub repo (criterion 1), then `git push --tags` so `pre-gpu-day` reaches it.
2. Create the Console workspace and key for this project, and set its spend limit to $147.
3. Rent the box (L40S or A100; Ubuntu 22.04/24.04; CUDA 12.x; 100 GB disk), export the key yourself, and run
   `bash scripts/bootstrap_box.sh`. It ends with the pre-flight. Any FAIL: stop.
