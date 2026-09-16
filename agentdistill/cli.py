"""agentdistill command line.

Commands that are not implemented yet exit with a clear "not built yet, see milestone N" rather than a stack
trace or, worse, a silent no-op. The help text is the roadmap.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from agentdistill import __version__

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__.split("\n")[0])
ingest_app = typer.Typer(no_args_is_help=True, help="Import traces from a source into the registry.")
dataset_app = typer.Typer(no_args_is_help=True, help="Build and inspect dataset artifacts.")
adapter_app = typer.Typer(no_args_is_help=True, help="Adapter lifecycle: merge, quantize, promote, lineage.")
eval_app = typer.Typer(no_args_is_help=True, help="Run and compare evaluations.")
train_app = typer.Typer(no_args_is_help=True, help="Supervised and on-policy training.")
evalset_app = typer.Typer(no_args_is_help=True, help="Register and freeze held-out eval sets.")
app.add_typer(ingest_app, name="ingest")
app.add_typer(dataset_app, name="dataset")
app.add_typer(adapter_app, name="adapter")
app.add_typer(eval_app, name="eval")
app.add_typer(train_app, name="train")
app.add_typer(evalset_app, name="evalset")

console = Console()
err = Console(stderr=True)


def _not_built(what: str, milestone: str) -> None:
    err.print(f"[yellow]{what} is not built yet[/yellow] — {milestone}.")
    err.print("The implementation plan's section 15 has the milestone order.")
    raise typer.Exit(code=2)


def _load(config: str) -> Any:
    from agentdistill.config import ProjectConfig

    try:
        return ProjectConfig.load(config)
    except FileNotFoundError as e:
        err.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1) from e
    except Exception as e:
        err.print(f"[red]{config} is not valid:[/red] {e}")
        raise typer.Exit(code=1) from e


def _registry(cfg: Any) -> Any:
    from agentdistill.registry import open_registry

    return open_registry(cfg.registry, root=cfg.root)


def _report_ingest(label: str, counts: dict, problems: list[str]) -> None:
    console.print(
        f"[green]{label}[/green]: read {counts['read']}, added [bold]{counts['added']}[/bold], "
        f"skipped {counts['skipped_duplicate']} duplicate"
    )
    if problems:
        err.print(f"[yellow]{len(problems)} record(s) could not be ingested:[/yellow]")
        for p in problems[:10]:
            err.print(f"  - {p}")
        if len(problems) > 10:
            err.print(f"  …and {len(problems) - 10} more")


@app.callback(invoke_without_command=True)
def main(version: bool = typer.Option(False, "--version", help="Print the version and exit.")) -> None:
    if version:
        console.print(__version__)
        raise typer.Exit()


@app.command()
def init(
    path: str = typer.Option("project.yaml", help="Where to write the starter config."),
    name: str = typer.Option("my-agent", help="Project name."),
    force: bool = typer.Option(False, help="Overwrite an existing config."),
) -> None:
    """Write a starter project.yaml and create the registry."""
    from agentdistill.templates import EXAMPLE_PROJECT_YAML

    p = Path(path)
    if p.exists() and not force:
        err.print(f"[red]{p} already exists[/red]; pass --force to overwrite.")
        raise typer.Exit(code=1)
    p.write_text(EXAMPLE_PROJECT_YAML.replace("name: support-agent", f"name: {name}"))
    console.print(f"[green]wrote[/green] {p}")

    cfg = _load(str(p))
    reg = _registry(cfg)
    console.print(f"[green]registry ready[/green] at {reg.url} (schema v{reg.schema_version()})")
    console.print(
        "\nNext: [bold]agentdistill ingest jsonl <traces.jsonl>[/bold], "
        "then [bold]agentdistill curate[/bold]"
    )


@ingest_app.command("jsonl")
def ingest_jsonl(
    path: str = typer.Argument(..., help="JSONL file, one trace per line."),
    config: str = typer.Option("project.yaml", help="Project config."),
    lenient: bool = typer.Option(False, help="Skip invalid records instead of stopping."),
) -> None:
    """Import normalized traces from JSONL. Anthropic-shaped records are converted automatically."""
    from agentdistill.ingest.jsonl_source import IngestError, load_traces

    cfg = _load(config)
    try:
        traces, problems = load_traces(path, strict=not lenient)
    except IngestError as e:
        err.print(f"[red]{e}[/red]")
        err.print("Pass --lenient to skip bad records.")
        raise typer.Exit(code=1) from e
    counts = _registry(cfg).insert_traces(traces)
    _report_ingest("jsonl", counts, problems)


@ingest_app.command("agentreplay")
def ingest_agentreplay(
    db: str = typer.Option(..., help="Path or URL of the agentreplay store."),
    since: str = typer.Option("30d", help="Only runs started within this window."),
    config: str = typer.Option("project.yaml", help="Project config."),
    limit: int | None = typer.Option(None, help="Cap the number of runs read."),
) -> None:
    """Import runs from an agentreplay store."""
    from agentdistill.ingest.agentreplay_source import AgentReplayError, load_traces

    cfg = _load(config)
    try:
        traces, problems = load_traces(db, since=since, limit=limit)
    except AgentReplayError as e:
        err.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1) from e
    counts = _registry(cfg).insert_traces(traces)
    _report_ingest("agentreplay", counts, problems)


@ingest_app.command("otel")
def ingest_otel(path: str, config: str = "project.yaml") -> None:
    """Import an OTel or Langfuse export."""
    _not_built("otel ingest", "milestone 1 stretch; jsonl and agentreplay are the supported sources today")


@ingest_app.command("gateway")
def ingest_gateway(since: str = "7d", config: str = "project.yaml") -> None:
    """Import gateway requests that have outcomes."""
    _not_built("gateway ingest", "needs the gateway, which lands in milestone 6")


@app.command()
def curate(
    config: str = typer.Option("project.yaml", help="Project config."),
    name: str | None = typer.Option(None, help="Dataset name; defaults to the project name."),
    kind: str = typer.Option("sft", help="sft keeps successes only; dpo keeps failures too."),
    build: bool = typer.Option(True, help="Tokenize survivors into a dataset artifact."),
    base_model: str | None = typer.Option(None, help="Override train.base_model for tokenization."),
    dry_run: bool = typer.Option(False, help="Report what would be kept without writing anything."),
) -> None:
    """Run the filter pipeline, write a curation report, and build a dataset version."""
    from agentdistill.curate import report as curation_report
    from agentdistill.curate.pipeline import curate as run_curate

    cfg = _load(config)
    reg = _registry(cfg)
    traces = reg.list_traces()
    if not traces:
        err.print("[red]no traces in the registry[/red] — run `agentdistill ingest …` first.")
        raise typer.Exit(code=1)

    eval_inputs = [e["task_input"] for e in reg.eval_set_task_inputs()]
    result = run_curate(traces, cfg, eval_task_inputs=eval_inputs, kind=kind)

    ds_name = name or cfg.name
    version = reg.next_dataset_version(ds_name)

    table = Table(title=f"curation: {result.n_input} in → {result.n_output} kept", box=None)
    table.add_column("filter")
    table.add_column("in", justify="right")
    table.add_column("dropped", justify="right", style="red")
    table.add_column("out", justify="right")
    table.add_column("top reason", overflow="fold")
    for s in result.stages:
        top = s.reasons.most_common(1)
        table.add_row(s.name, str(s.n_in), str(s.n_dropped) if s.n_dropped else "-", str(s.n_out),
                      f"{top[0][0]} ({top[0][1]})" if top else "")
    console.print(table)
    for note in result.notes:
        console.print(f"[yellow]note:[/yellow] {note}")

    if result.n_output == 0:
        err.print("[red]curation kept nothing[/red]; loosen the filters or ingest more traces.")
        raise typer.Exit(code=1)

    if dry_run:
        console.print("[yellow]dry run: nothing written[/yellow]")
        return

    if result.assignments:
        reg.set_clusters(result.assignments)

    # The dataset is built first because it resolves the version: an unchanged corpus reproduces the previous
    # hash and reuses that version rather than minting a new one, and the report must be named accordingly.
    built = _build_dataset(cfg, reg, result, ds_name, version, kind, base_model)
    resolved_version = built.version if built else version
    resolved_name = built.name if built else ds_name

    report_path = cfg.reports_dir / f"curation-{resolved_name}-v{resolved_version}.md"
    curation_report.write(result, resolved_name, resolved_version, report_path)
    console.print(f"[green]curation report[/green] {report_path}")
    if built is not None:
        reg.set_dataset_report_path(built.dataset_id, str(report_path))


def _build_dataset(cfg, reg, result, ds_name, version, kind, base_model):
    """Tokenize the survivors. Returns the BuildResult, or None when no base model is configured."""
    from agentdistill.data.dataset import build_dataset
    from agentdistill.data.template_check import TemplateError

    model = base_model or (cfg.train.base_model if cfg.train else None)
    if not model:
        err.print("[yellow]no train.base_model set, so no dataset was tokenized.[/yellow]")
        err.print("Set train.base_model in project.yaml or pass --base-model, then re-run.")
        return None
    try:
        built = build_dataset(
            result.traces, cfg, name=ds_name, version=version, registry=reg, base_model=model,
            filter_config=result.filter_config, kind=kind,
        )
    except TemplateError as e:
        err.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1) from e
    except (ValueError, ImportError) as e:
        err.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1) from e

    a = built.artifact
    if built.reused:
        console.print(
            f"[green]identical to {built.name} v{built.version}[/green] "
            f"(hash [bold]{a.content_hash[:12]}[/bold]) — no new dataset written."
        )
        console.print("  [dim]Curation is deterministic: same traces plus same filters means the same dataset.[/dim]")
        console.print(f"  {a.path}")
        return built
    console.print(
        f"[green]dataset[/green] {built.name} v{built.version} — {a.n_samples} samples, "
        f"{a.n_tokens:,} tokens ({a.n_target_tokens:,} trained), hash [bold]{a.content_hash[:12]}[/bold]"
    )
    console.print(f"  {a.path}")
    for note in built.notes[:5]:
        console.print(f"  [dim]{note}[/dim]")
    return built


@dataset_app.command("list")
def dataset_list(config: str = typer.Option("project.yaml", help="Project config.")) -> None:
    """List dataset versions in the registry."""
    cfg = _load(config)
    rows = _registry(cfg).list_datasets()
    if not rows:
        console.print("no datasets yet")
        return
    table = Table(box=None)
    for col in ("name", "v", "kind", "samples", "tokens", "hash", "path"):
        table.add_column(col, justify="right" if col in {"v", "samples", "tokens"} else "left")
    for r in rows:
        table.add_row(r["name"], str(r["version"]), r["kind"], f"{r['n_samples']:,}",
                      f"{r['n_tokens']:,}" if r["n_tokens"] else "-", r["content_hash"][:12], r["path"])
    console.print(table)


@dataset_app.command("verify")
def dataset_verify(path: str = typer.Argument(..., help="Dataset directory.")) -> None:
    """Recompute a dataset's content hash and compare it to the manifest."""
    from agentdistill.data.artifact import verify

    try:
        ok, detail = verify(path)
    except FileNotFoundError as e:
        err.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1) from e
    if ok:
        console.print(f"[green]ok[/green] {detail}")
    else:
        err.print(f"[red]corrupt[/red] {detail}")
        raise typer.Exit(code=1)


@dataset_app.command("inspect")
def dataset_inspect(
    path: str = typer.Argument(..., help="Dataset directory."),
    row: int = typer.Option(0, help="Which sample to show."),
    base_model: str | None = typer.Option(None, help="Tokenizer to decode with; defaults to the manifest's."),
) -> None:
    """Show what a sample actually trains on.

    If the highlighted text does not read like assistant turns, the loss mask is wrong and nothing downstream
    will tell you: the loss curve looks fine while the model learns to predict tool results.
    """
    from agentdistill.data.artifact import read_manifest, read_samples
    from agentdistill.data.dataset import load_tokenizer

    try:
        manifest = read_manifest(path)
        table = read_samples(path)
    except FileNotFoundError as e:
        err.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1) from e
    if row >= table.num_rows:
        err.print(f"[red]row {row} out of range[/red]; the dataset has {table.num_rows} samples.")
        raise typer.Exit(code=1)
    tok = load_tokenizer(base_model or manifest["tokenizer"])
    ids = table.column("input_ids")[row].as_py()
    labels = table.column("labels")[row].as_py()

    console.print(
        f"[bold]sample {row}[/bold] of {table.num_rows}  "
        f"kind={table.column('kind')[row].as_py()}  trace={table.column('trace_id')[row].as_py()}"
    )
    console.print(f"{len(ids)} tokens, {table.column('n_target_tokens')[row].as_py()} trained\n")
    console.print("[dim]--- full rendering ---[/dim]")
    console.print(tok.decode(ids))
    console.print("\n[dim]--- trained on (loss applies here) ---[/dim]")
    buf: list[int] = []
    for tid, lab in zip(ids, labels, strict=True):
        if lab == -100:
            if buf:
                console.print(f"[green]{tok.decode(buf)}[/green]")
                buf = []
        else:
            buf.append(tid)
    if buf:
        console.print(f"[green]{tok.decode(buf)}[/green]")


@evalset_app.command("add")
def evalset_add(
    name: str = typer.Argument(..., help="Eval set name."),
    path: str = typer.Argument(..., help="JSONL of held-out tasks, same format as training traces."),
    config: str = typer.Option("project.yaml", help="Project config."),
    freeze: bool = typer.Option(
        True, help="Mark the set frozen. Freeze for a quarter; rotating it weekly makes trends unreadable."
    ),
) -> None:
    """Register held-out tasks as an eval set.

    The tasks are stored as traces marked `eval`, and decontamination drops any training trace that overlaps them.
    Register the eval set *before* curating, or the dataset cannot be decontaminated against it.
    """
    from agentdistill.ingest.jsonl_source import IngestError, load_traces
    from agentdistill.registry import utcnow

    cfg = _load(config)
    reg = _registry(cfg)
    if reg.get_eval_set(name):
        err.print(f"[red]eval set {name!r} already exists[/red]; eval sets are frozen on purpose. "
                  f"Use a new name for a new quarter.")
        raise typer.Exit(code=1)
    try:
        traces, problems = load_traces(path, strict=True)
    except IngestError as e:
        err.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1) from e
    counts = reg.insert_traces(traces)
    ids = [t["id"] for t in traces]
    reg.insert_eval_set({
        "id": f"es_{name}",
        "name": name,
        "trace_ids": ids,
        "grader": cfg.eval.grader.model_dump(),
        "frozen_at": utcnow() if freeze else None,
    })
    console.print(f"[green]eval set[/green] {name}: {len(ids)} tasks "
                  f"({counts['added']} new traces, {counts['skipped_duplicate']} already present)"
                  + (" [dim]frozen[/dim]" if freeze else ""))
    if problems:
        err.print(f"[yellow]{len(problems)} task(s) skipped[/yellow]")


@evalset_app.command("list")
def evalset_list(config: str = typer.Option("project.yaml")) -> None:
    """List registered eval sets."""
    from sqlalchemy import text

    cfg = _load(config)
    reg = _registry(cfg)
    with reg.engine.connect() as conn:
        rows = conn.execute(text("SELECT name, frozen_at FROM eval_sets ORDER BY name")).fetchall()
    if not rows:
        console.print("no eval sets registered")
        return
    for name, frozen in rows:
        es = reg.get_eval_set(name)
        status = f"  [dim]frozen {frozen}[/dim]" if frozen else "  [yellow]not frozen[/yellow]"
        console.print(f"{name}: {len(es['trace_ids'])} tasks{status}")


@app.command("base-check")
def base_check(
    model: str = typer.Argument(..., help="Base model id or local path."),
    json_out: bool = typer.Option(False, "--json", help="Emit the report as JSON."),
) -> None:
    """Check a base model's chat template for tool support, prefix stability, and masking compatibility."""
    from agentdistill.data.dataset import load_tokenizer
    from agentdistill.data.template_check import check_template, roundtrip_tool_call

    try:
        tok = load_tokenizer(model)
    except Exception as e:
        err.print(f"[red]could not load a tokenizer for {model!r}:[/red] {e}")
        raise typer.Exit(code=1) from e

    report = check_template(tok, model)
    if json_out:
        console.print_json(json.dumps(report.to_dict()))
        raise typer.Exit(code=0 if report.ok else 1)

    table = Table(title=f"chat template: {model}", box=None)
    table.add_column("check")
    table.add_column("", justify="center")
    table.add_column("detail", overflow="fold")
    for c in report.checks:
        table.add_row(c.name, "[green]PASS[/green]" if c.ok else "[red]FAIL[/red]", c.detail)
    if report.ok:
        ok, detail = roundtrip_tool_call(tok)
        table.add_row("tool_call_roundtrip", "[green]PASS[/green]" if ok else "[red]FAIL[/red]", detail)
    console.print(table)

    if report.ok:
        console.print("\n[green]usable as a base model.[/green] "
                      "Verify the vLLM tool-call parser for this template before serving.")
    else:
        console.print("\n[red]not usable as a base model for dataset building.[/red]")
        raise typer.Exit(code=1)


@train_app.command("sft")
def train_sft_cmd(dataset: str, config: str = "project.yaml") -> None:
    """LoRA / QLoRA supervised fine-tuning."""
    _not_built("SFT training", "milestone 2")


@train_app.command("onpolicy")
def train_onpolicy(adapter: str, config: str = "project.yaml", rounds: int | None = None) -> None:
    """Rejection sampling and DPO rounds using the eval harness for rollouts."""
    _not_built("on-policy training", "milestone 4; it depends on the eval harness from milestone 3")


@eval_app.command("run")
def eval_run(subject: str, eval_set: str | None = None, config: str = "project.yaml") -> None:
    """Evaluate an adapter, 'teacher', or 'cascade:<adapter>:<tau>' on the eval set."""
    _not_built("the eval harness", "milestone 3")


@eval_app.command("compare")
def eval_compare(a: str, b: str) -> None:
    """Paired statistical comparison of two eval runs."""
    _not_built("eval compare", "milestone 3")


@app.command()
def calibrate(adapter: str, config: str = "project.yaml") -> None:
    """Fit the confidence gate, choose the threshold, verify with the harness."""
    _not_built("calibration", "milestone 5")


@adapter_app.command("merge")
def adapter_merge(adapter: str) -> None:
    """Merge LoRA into the base for serving."""
    _not_built("adapter merge", "milestone 2")


@adapter_app.command("quantize")
def adapter_quantize(adapter: str, method: str = "fp8") -> None:
    """Produce a quantized serving artifact and evaluate it."""
    _not_built("adapter quantize", "milestone 6")


@adapter_app.command("promote")
def adapter_promote(adapter: str, to: str = "canary") -> None:
    """Move an adapter to canary or prod after checks."""
    _not_built("adapter promote", "milestone 7")


@adapter_app.command("lineage")
def adapter_lineage(adapter: str) -> None:
    """Print dataset, config, and parent chain."""
    _not_built("adapter lineage", "milestone 7")


@app.command()
def serve(config: str = "project.yaml", host: str = "0.0.0.0", port: int = 8710) -> None:
    """Start the gateway."""
    _not_built("the gateway", "milestone 6; do not build it before the eval harness exists")


@app.command()
def report(config: str = "project.yaml", out: str = "reports/latest.html") -> None:
    """Build the cost and quality report."""
    _not_built("the cost report", "milestone 8")


@app.command()
def retrain(config: str = "project.yaml") -> None:
    """Run the full weekly loop: ingest, curate, train, eval, calibrate, canary."""
    _not_built("the retrain loop", "milestone 7")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(app())
