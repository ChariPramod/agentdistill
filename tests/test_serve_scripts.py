"""The serving scripts.

Neither can run here -- one starts vLLM and the other needs a GPU behind it. What is testable is everything
that would fail on rented hardware for a reason a laptop could have caught: invalid bash, a subcommand that
does not exist, a flag the CLI does not accept, and a serve command assembled from the wrong settings.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SERVE = ROOT / "scripts" / "serve_vllm.sh"
SMOKE = ROOT / "scripts" / "serve_smoke.sh"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")


def fake_cli(tmp_path: Path, values: dict[str, str], adapters: dict[str, str]) -> Path:
    """A stand-in `agentdistill` that answers `config get` and `adapter path` from a fixture.

    The script is tested against a fake rather than the real CLI so a test can express "the registry has a
    canary" without building a registry, and so a missing config key can be made to fail the way the real CLI
    fails it.
    """
    lines = ["#!/usr/bin/env bash", 'if [[ "$1 $2" == "config get" ]]; then', '  case "$3" in']
    lines += [f'    {key}) echo "{value}" ;;' for key, value in values.items()]
    lines += ['    *) exit 1 ;;', '  esac', '  exit 0', 'fi',
              'if [[ "$1 $2" == "adapter path" ]]; then', '  case "$4" in']
    lines += [f'    {status}) echo "{path}" ;;' for status, path in adapters.items()]
    lines += ['    *) exit 1 ;;', '  esac', '  exit 0', 'fi', 'exit 1']

    script = tmp_path / "agentdistill"
    script.write_text("\n".join(lines) + "\n")
    script.chmod(0o755)
    return script


def serve_command(tmp_path, values=None, adapters=None) -> str:
    settings = {
        "train.base_model": "Qwen/Qwen3-8B",
        "train.tool_parser.name": "hermes",
        "serve.quantization": "fp8",
        "serve.max_model_len": "16384",
        "serve.vllm_port": "8000",
    }
    settings.update(values or {})
    cli = fake_cli(tmp_path, settings, adapters or {})
    proc = subprocess.run(
        ["bash", str(SERVE), "--dry-run"], capture_output=True, text=True, cwd=ROOT, check=False,
        env={"PATH": f"{cli.parent}:/usr/bin:/bin", "AGENTDISTILL": str(cli), "HOME": str(tmp_path)},
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


def test_both_scripts_are_valid_bash():
    for script in (SERVE, SMOKE):
        proc = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True, check=False)
        assert proc.returncode == 0, f"{script.name}: {proc.stderr}"


def test_the_serve_command_uses_the_configured_model_and_parser(tmp_path):
    cmd = serve_command(tmp_path)
    assert "vllm serve Qwen/Qwen3-8B" in cmd
    assert "--tool-call-parser hermes" in cmd
    assert "--max-model-len 16384" in cmd


def test_prefix_caching_is_always_on(tmp_path):
    """Agent prompts repeat a system prompt and tool schemas on every turn. Without prefix caching the
    throughput number in the cost model is wrong by a large factor."""
    assert "--enable-prefix-caching" in serve_command(tmp_path)


def test_a_prod_adapter_is_loaded_as_a_lora_module(tmp_path):
    cmd = serve_command(tmp_path, adapters={"prod": "/artifacts/prod-v3"})
    assert "--enable-lora" in cmd
    assert "--lora-modules prod=/artifacts/prod-v3" in cmd


def test_prod_and_canary_are_both_loaded(tmp_path):
    """The canary is served alongside prod, not instead of it; a split across a restart is not a comparison."""
    cmd = serve_command(tmp_path, adapters={"prod": "/a/prod", "canary": "/a/canary"})
    assert "--lora-modules prod=/a/prod" in cmd
    assert "--lora-modules canary=/a/canary" in cmd


def test_no_adapters_means_no_lora_flags(tmp_path):
    cmd = serve_command(tmp_path, adapters={})
    assert "--enable-lora" not in cmd
    assert "--lora-modules" not in cmd


def test_fp8_is_flagged_and_awq_is_not(tmp_path):
    """vLLM applies fp8 online. AWQ weights carry their own config and must not be double-flagged."""
    assert "--quantization fp8" in serve_command(tmp_path, {"serve.quantization": "fp8"})
    assert "--quantization" not in serve_command(tmp_path, {"serve.quantization": "awq"})


def test_a_missing_base_model_fails_loudly(tmp_path):
    cli = fake_cli(tmp_path, {"serve.quantization": "fp8"}, {})
    proc = subprocess.run(
        ["bash", str(SERVE), "--dry-run"], capture_output=True, text=True, cwd=ROOT, check=False,
        env={"PATH": f"{cli.parent}:/usr/bin:/bin", "AGENTDISTILL": str(cli), "HOME": str(tmp_path)},
    )
    assert proc.returncode != 0
    assert "no train.base_model" in proc.stderr


def test_the_smoke_script_only_calls_commands_that_exist():
    """A typo in a subcommand would surface on rented hardware."""
    from typer.main import get_command

    from agentdistill.cli import app

    root = get_command(app)
    available = set()
    for name, cmd in root.commands.items():  # type: ignore[attr-defined]
        available.add(name)
        available.update(f"{name} {sub}" for sub in getattr(cmd, "commands", {}))

    text = SMOKE.read_text() + SERVE.read_text()
    invoked = set()
    for line in text.splitlines():
        stripped = line.strip()
        if '"$AD"' not in stripped:
            continue
        after = stripped.split('"$AD"', 1)[1].split()
        words = [w for w in after if not w.startswith(("-", "$", '"', "|", ">"))]
        if not words:
            continue
        head = words[0]
        candidate = f"{head} {words[1]}" if len(words) > 1 and f"{head} {words[1]}" in available else head
        invoked.add(candidate)

    assert invoked, "the smoke script invokes no agentdistill commands"
    assert not (invoked - available), f"unknown commands: {sorted(invoked - available)}"


def test_the_smoke_script_fails_when_everything_fell_back():
    """A run where vLLM never served anything would otherwise pass: the gateway falls back to the teacher,
    every request succeeds, and the request log fills up."""
    text = SMOKE.read_text()
    assert "fell back to the teacher" in text
    assert "FALLBACKS" in text


def test_the_smoke_script_drives_both_dialects():
    text = SMOKE.read_text()
    assert 'openai/${SMOKE_MODEL}' in text
    assert 'anthropic/${SMOKE_MODEL}' in text


def test_the_real_run_drives_the_cascade_and_tiny_mode_drives_the_student():
    """The cascade is the thing worth smoke-testing. Tiny mode has no teacher to escalate to, so it drives the
    student -- still exercising the gateway, both dialects and the request log, and honest about the gap."""
    text = SMOKE.read_text()
    assert 'SMOKE_MODEL="${SMOKE_MODEL:-cascade::auto}"' in text
    assert 'SMOKE_MODEL="${SMOKE_MODEL:-student}"' in text
    assert "no teacher to escalate to" in text


def test_record_accepts_the_flags_the_smoke_script_passes():
    from examples.support_agent.record import main

    with pytest.raises(SystemExit):
        main(["--help"])

    # The dialect prefix is required with --base-url, because it is what picks which translation is exercised.
    with pytest.raises(SystemExit):
        main(["--model", "gpt-4.1", "--base-url", "http://127.0.0.1:8710/v1", "--n", "1"])
