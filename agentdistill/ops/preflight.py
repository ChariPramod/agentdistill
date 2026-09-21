"""The checks of `scripts/preflight.sh` that need the project loaded: config, registry, lock, tokenizer,
versions, hardware.

The shell script owns the two cheapest groups (git and the environment) and calls this for the rest, so the
ordering the runbook promises -- cheapest first, hardware last -- is preserved across the two halves.

One line per check, `PASS`/`FAIL`/`SKIP` first so the output greps. Exit 1 if any line is FAIL. A SKIP is a check
that cannot run here and says why; it never stands in for a pass.

    python -m agentdistill.ops.preflight --config examples/support_agent/project.yaml
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

HUB_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
SHA40 = re.compile(r"^[0-9a-f]{40}$")

#: Minimum free VRAM in GB. bf16 LoRA on an 8B-class model needs roughly 40; 4-bit QLoRA fits in 22.
VRAM_BF16_GB = 40
VRAM_QLORA_GB = 22
#: Merged bf16 weights, a quantized copy, adapters, rollouts and eval trajectories. docs/gpu-day.md budgets 80.
DISK_FREE_GB = 60


class Report:
    """Collects one line per check and remembers whether anything failed."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.failed = False

    def _add(self, status: str, name: str, detail: str) -> None:
        self.lines.append(f"{status} {name}: {detail}")
        print(f"{status} {name}: {detail}", flush=True)

    def ok(self, name: str, detail: str) -> None:
        self._add("PASS", name, detail)

    def fail(self, name: str, detail: str) -> None:
        self.failed = True
        self._add("FAIL", name, detail)

    def skip(self, name: str, detail: str) -> None:
        self._add("SKIP", name, detail)

    def check(self, name: str, condition: bool, ok_detail: str, fail_detail: str) -> bool:
        (self.ok if condition else self.fail)(name, ok_detail if condition else fail_detail)
        return condition


def check_config(rep: Report, cfg: Any) -> None:
    """The settings that decide what the day trains, prices and measures."""
    teacher = cfg.teacher
    rep.check(
        "config.teacher_model", bool(teacher and teacher.model),
        f"{teacher.provider}/{teacher.model}" if teacher else "",
        "no teacher.model in the config; the cost block and the cascade have nothing to price against",
    )

    if cfg.train is None:
        rep.fail("config.base_model", "the config has no `train` section")
        return

    base = cfg.base_model
    rep.check(
        "config.base_model", bool(HUB_ID.match(base or "")) and not Path(base).exists(),
        f"{base} is a Hub id",
        f"train.base_model is {base!r}, which is a local path rather than a Hub id; the GPU day must train "
        f"the model the report names",
    )
    rev = cfg.train.base_model_revision or ""
    rep.check(
        "config.base_model_revision", bool(SHA40.match(rev)),
        f"pinned at {rev}",
        f"train.base_model_revision is {rev or 'unset'!r}, not a 40-character commit sha; a tag moves and then "
        f"a rerun trains on different weights than the report says",
    )
    parser = cfg.train.tool_parser
    rep.check(
        "config.tool_parser", bool(parser.name and parser.family),
        f"name={parser.name} family={parser.family}",
        f"train.tool_parser needs both name (vLLM's parser) and family (the fallback regex); have "
        f"name={parser.name!r} family={parser.family!r}. `agentdistill base-check` names the right one",
    )
    for key, value in (("config.eval_tools", cfg.eval.tools), ("config.onpolicy_tools", cfg.onpolicy.tools)):
        rep.check(
            key, value == "live", f"{value}",
            f"{key.split('.')[1].replace('_', '.')} is {value!r}; replay grades a subject on how closely it "
            f"imitates the recorded solver, which is not whether it solved the task",
        )


def check_pricing(rep: Report, cfg: Any, registry: Any) -> None:
    if not (cfg.teacher and cfg.teacher.model):
        rep.skip("registry.pricing", "no teacher configured, so there is no price to look for")
        return
    rows = [
        r for r in registry.list_pricing()
        if r["model"] == cfg.teacher.model and r["provider"] == cfg.teacher.provider
    ]
    if not rows:
        rep.fail(
            "registry.pricing",
            f"no pricing row for {cfg.teacher.provider}/{cfg.teacher.model}. Seed it: `agentdistill pricing set "
            f"{cfg.teacher.model} --provider {cfg.teacher.provider} --input <usd> --output <usd> "
            f"--cache-read <usd>` (docs/gpu-day.md, pre-flight step 3b)",
        )
        return
    row = max(rows, key=lambda r: r["effective_from"])
    rep.ok(
        "registry.pricing",
        f"{row['provider']}/{row['model']} from {row['effective_from']}: ${row['input_per_mtok']}/Mtok in, "
        f"${row['output_per_mtok']}/Mtok out",
    )


def check_lock(rep: Report, cfg: Any, registry: Any) -> None:
    """The same comparison `agentdistill ops lock check` makes, in process."""
    import io
    from contextlib import redirect_stdout

    from agentdistill.ops import lock as lockmod

    target = lockmod.lock_path(cfg)
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = lockmod.check(str(cfg.source_path), target)
    if code == 0:
        rep.ok("lock", f"{target.name} matches this tree (same check as `agentdistill ops lock check`)")
        return
    detail = buf.getvalue().strip().replace("\n", "\n    ")
    rep.fail("lock", f"`agentdistill ops lock check` fails:\n    {detail}")


def check_tokenizer(rep: Report, cfg: Any, registry: Any) -> None:
    """The guard `train sft` applies, run before the box is rented rather than after `sft` has started."""
    from agentdistill.train.sft import TokenizerMismatch, check_dataset_tokenizer

    sft = [d for d in registry.list_datasets() if d.get("kind") == "sft"]
    if not sft:
        rep.fail("tokenizer.guard", "no SFT dataset in the registry; run `agentdistill curate` first")
        return
    newest = max(sft, key=lambda d: (d.get("version") or 0, d.get("created_at") or ""))
    path = cfg.resolve(newest["path"])
    try:
        warnings = check_dataset_tokenizer(cfg.train_config(), path)
    except TokenizerMismatch as e:
        rep.fail("tokenizer.guard", str(e))
        return
    except FileNotFoundError as e:
        rep.fail("tokenizer.guard", f"dataset {newest['id']} is in the registry but not on disk: {e}")
        return
    if warnings:
        rep.fail("tokenizer.guard", "; ".join(warnings))
        return
    rep.ok(
        "tokenizer.guard",
        f"dataset {newest['id']} was tokenized for {cfg.base_model} at "
        f"{cfg.train.base_model_revision}",
    )


def requirement_pins(path: Path) -> dict[str, str]:
    """`pkg==version` lines from requirements-gpu.txt. Comments are the file's way of saying "not pinnable here"
    and are left alone."""
    pins: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if "==" in line:
            name, version = line.split("==", 1)
            pins[name.strip()] = version.strip()
    return pins


def check_versions(rep: Report, root: Path) -> None:
    import importlib.metadata as md

    req = root / "requirements-gpu.txt"
    if not req.exists():
        rep.fail("python.versions", f"no {req}; the box would install whatever released this morning")
        return
    pins = requirement_pins(req)
    wrong, missing = [], []
    for name, want in sorted(pins.items()):
        try:
            have = md.version(name)
        except md.PackageNotFoundError:
            missing.append(f"{name} (want {want})")
            continue
        if have != want:
            wrong.append(f"{name} {have} != {want}")
    if wrong or missing:
        parts = []
        if wrong:
            parts.append("mismatched: " + ", ".join(wrong))
        if missing:
            parts.append("not installed: " + ", ".join(missing))
        rep.fail("python.versions", f"{'; '.join(parts)}. `pip install -r requirements-gpu.txt -e \".[train,serve]\"`")
        return
    rep.ok("python.versions", f"all {len(pins)} pinned packages match requirements-gpu.txt")


def _nvidia_smi(query: str) -> list[str]:
    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False, timeout=60,
    )
    return [line.strip() for line in out.stdout.splitlines() if line.strip()]


def _free_disk_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / 1e9


def check_hardware(rep: Report, cfg: Any, root: Path) -> None:
    """GPU, VRAM and disk. Skipped as a group where there is no nvidia-smi, because these are the box's numbers
    and a laptop's answers would be noise -- but the free-disk figure is still reported, since it is the one a
    person may want to know before they rent anything."""
    free_gb = _free_disk_gb(root)
    if shutil.which("nvidia-smi") is None:
        rep.skip("hw.gpu", "no nvidia-smi: this is not the GPU box")
        rep.skip("hw.vram", "no nvidia-smi: this is not the GPU box")
        rep.skip("hw.disk", f"not the GPU box; {free_gb:.0f} GB free here, the box needs {DISK_FREE_GB}")
        return

    names = _nvidia_smi("name")
    if not rep.check(
        "hw.gpu", bool(names), ", ".join(names),
        "nvidia-smi reports no GPU; the day cannot train here",
    ):
        rep.skip("hw.vram", "no GPU to measure")
    else:
        quantized = bool(cfg.train and cfg.train.quantization)
        need = VRAM_QLORA_GB if quantized else VRAM_BF16_GB
        method = f"{'QLoRA (' + str(cfg.train.quantization) + ')' if quantized else 'bf16 LoRA'}"
        try:
            vram = max(float(v) for v in _nvidia_smi("memory.total")) / 1024
        except ValueError:
            vram = 0.0
        rep.check(
            "hw.vram", vram >= need,
            f"{vram:.0f} GB, enough for {method} (needs {need} GB)",
            f"{vram:.0f} GB is below the {need} GB {method} needs; either rent a bigger card or set "
            f"train.quantization to 4bit and rehearse that",
        )
    rep.check(
        "hw.disk", free_gb >= DISK_FREE_GB,
        f"{free_gb:.0f} GB free",
        f"{free_gb:.0f} GB free, below the {DISK_FREE_GB} GB the merge and quantize stages need",
    )


def run(config: str) -> int:
    from agentdistill.config import ProjectConfig
    from agentdistill.registry import open_registry

    rep = Report()
    try:
        cfg = ProjectConfig.load(config)
    except Exception as e:  # a config that will not load fails every check after it
        rep.fail("config.load", f"{config} did not load: {e}")
        return 1
    repo_root = Path(__file__).resolve().parents[2]

    check_config(rep, cfg)
    try:
        registry = open_registry(cfg.registry, root=cfg.root)
    except Exception as e:
        rep.fail("registry.open", f"cannot open {cfg.registry}: {e}")
        check_versions(rep, repo_root)
        check_hardware(rep, cfg, repo_root)
        return 1
    check_pricing(rep, cfg, registry)
    check_lock(rep, cfg, registry)
    check_tokenizer(rep, cfg, registry)
    check_versions(rep, repo_root)
    check_hardware(rep, cfg, repo_root)
    return 1 if rep.failed else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="project.yaml")
    args = ap.parse_args(argv)
    return run(args.config)


if __name__ == "__main__":
    raise SystemExit(main())
