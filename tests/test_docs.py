"""The docs describe built code, not intended code.

Documentation that names a flag which does not exist is worse than no documentation: a reader trusts it, runs
it, and gets a parse error with no idea whether the feature is missing or they typed it wrong. These tests keep
the docs honest by checking them against the real CLI and the real constants.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = sorted((ROOT / "docs").glob("*.md"))
PLANNED = {"docs/progress.md"}  # a status document; it is allowed to describe what is not built


def cli_surface() -> tuple[dict[str, set[str]], set[str]]:
    from typer.main import get_command

    from agentdistill.cli import app

    root = get_command(app)
    groups = {n: set(getattr(c, "commands", {})) for n, c in root.commands.items()}  # type: ignore[attr-defined]
    return groups, set(groups) | {f"{g} {s}" for g, subs in groups.items() for s in subs}


def documented_invocations(text: str) -> list[list[str]]:
    out = []
    for m in re.finditer(r"^\s*agentdistill ([^\n]*)$", text, re.M):
        parts = [p for p in m.group(1).split() if p != "\\"]
        if parts:
            out.append(parts)
    return out


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_every_documented_command_exists(doc):
    if str(doc.relative_to(ROOT)) in PLANNED:
        pytest.skip("a status document, which describes what is not built yet")

    groups, available = cli_surface()
    problems = []
    for parts in documented_invocations(doc.read_text()):
        head = parts[0]
        cmd = parts[:2] if head in groups and len(parts) > 1 and parts[1] in groups[head] else parts[:1]
        name = " ".join(cmd)
        if name not in available:
            problems.append(f"no such command `{name}`")
            continue
        help_text = re.sub(r"\s+", " ", subprocess.run(
            [sys.executable, "-m", "agentdistill.cli", *cmd, "--help"],
            capture_output=True, text=True, check=False,
        ).stdout)
        problems += [f"`{name}` has no {f}" for f in sorted({p for p in parts if p.startswith("--")})
                     if f not in help_text]

    assert not problems, f"{doc.name}:\n  " + "\n  ".join(sorted(set(problems)))


def test_the_documented_defaults_match_the_code():
    """Numbers in prose drift from numbers in code. These are the ones a reader would act on."""
    from agentdistill.cascade.calibrate import MAX_ACCEPTABLE_ECE, MIN_TURNS, MIN_USEFUL_AUROC
    from agentdistill.config import RouterConfig
    from agentdistill.gateway.app import FALLBACK_ALERT_RATE
    from agentdistill.retrain import (
        MAX_HOLDOUT_ECE,
        MAX_SUCCESS_REGRESSION_PP,
        MIN_HOLDOUT_AUROC,
        MIN_NEW_REQUESTS,
        MIN_NEW_SAMPLES,
        STAGE_ORDER,
    )
    from agentdistill.train.merge import MAX_FULL_MATCH_DRIFT_PP
    from agentdistill.train.quantize import MAX_QUANTIZATION_DROP_PP, MIN_CALIB_SAMPLES

    router = RouterConfig()
    claims = [
        ("router.md", f"default {router.floor}", router.floor == 0.55),
        ("router.md", f"default {router.decay}", router.decay == 0.995),
        ("router.md", f"default {router.min_observations}", router.min_observations == 10),
        ("router.md", "decay 0.9 ceiling is 8", round(1 / (1 - 0.9) - 2) == 8),
        ("cascade.md", "100 labelled turns", MIN_TURNS == 100),
        ("cascade.md", "20%", FALLBACK_ALERT_RATE == 0.20),
        ("retrain.md", "50 graded requests", MIN_NEW_REQUESTS == 50),
        ("retrain.md", "50 new samples", MIN_NEW_SAMPLES == 50),
        ("retrain.md", "-1 pp", MAX_SUCCESS_REGRESSION_PP == 1.0),
        ("retrain.md", "ECE 0.05", MAX_HOLDOUT_ECE == MAX_ACCEPTABLE_ECE == 0.05),
        ("retrain.md", "AUROC 0.6", MIN_HOLDOUT_AUROC == MIN_USEFUL_AUROC == 0.6),
        ("retrain.md", "eight stages", len(STAGE_ORDER) == 8),
        ("serving.md", "2 pp merge drift", MAX_FULL_MATCH_DRIFT_PP == 2.0),
        ("serving.md", "32 calibration prompts", MIN_CALIB_SAMPLES == 32),
        ("serving.md", "2 pp quantization", MAX_QUANTIZATION_DROP_PP == 2.0),
    ]
    stale = [f"{doc}: {claim}" for doc, claim, ok in claims if not ok]
    assert not stale, "docs state values the code no longer uses:\n  " + "\n  ".join(stale)


def test_the_documented_feature_names_are_the_real_ones():
    from agentdistill.cascade.features import DEFAULT_FEATURES

    text = (ROOT / "docs" / "cascade.md").read_text()
    missing = [f for f in DEFAULT_FEATURES if f"`{f}`" not in text]
    assert not missing, f"docs/cascade.md does not document: {missing}"


def test_the_documented_router_thresholds_come_from_compare_live():
    import inspect

    from agentdistill.router.compare_live import compare_live

    params = inspect.signature(compare_live).parameters
    text = (ROOT / "docs" / "router.md").read_text()
    assert params["min_clusters"].default == 3
    assert params["min_per_arm"].default == 5
    assert "three clusters with five observations" in text


def test_results_and_readme_carry_injectable_markers():
    """`report --inject` writes between these. Without them the command has nowhere to put anything."""
    from agentdistill.report.markdown import BEGIN, END

    for path in (ROOT / "README.md", ROOT / "docs" / "results.md"):
        text = path.read_text()
        assert text.count(BEGIN) == 1, f"{path.name} needs exactly one {BEGIN}"
        assert text.count(END) == 1, f"{path.name} needs exactly one {END}"
        assert text.index(BEGIN) < text.index(END)


def test_the_results_markers_are_still_empty():
    """No numbers until a GPU run. A README with plausible numbers is worse than one with none."""
    from agentdistill.report.markdown import BEGIN, END

    for path in (ROOT / "README.md", ROOT / "docs" / "results.md"):
        text = path.read_text()
        between = text[text.index(BEGIN) + len(BEGIN):text.index(END)].strip()
        assert not between, (
            f"{path.name} has numbers between its results markers. If a GPU run produced them, delete this "
            f"test; if something injected a rehearsal, revert it -- tiny-mode numbers measure nothing."
        )


def test_every_doc_the_plan_asked_for_exists():
    expected = {"cascade", "curation", "canonical-json", "evaluation", "quickstart",
                "results", "retrain", "router", "serving", "tos"}
    assert expected <= {p.stem for p in DOCS}


def test_nothing_reads_the_raw_base_model_path():
    """`train.base_model` may be a path relative to the config, so every consumer must resolve it.

    This has bitten three times in one rehearsal -- `base-check`, the rollout tokenizer, and loading an adapter's
    stored base model -- because each site read `cfg.train.base_model` directly. `cfg.base_model` resolves it;
    `cfg.resolve_model(...)` resolves a value read from a registry row. Recording the configured spelling is
    fine, so the check is narrow: no *unresolved read* outside config.py itself.
    """
    offenders = []
    for path in sorted((ROOT / "agentdistill").rglob("*.py")):
        if path.name == "config.py":
            continue
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            if ".train.base_model" in line and "resolve_model" not in line and "getattr(cfg" not in line:
                offenders.append(f"{path.relative_to(ROOT)}:{i}: {line.strip()}")

    assert not offenders, (
        "these read train.base_model without resolving it, so they work from beside the config and fail from "
        "anywhere else:\n  " + "\n  ".join(offenders)
    )
