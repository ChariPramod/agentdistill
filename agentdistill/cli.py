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

    # Eval traces share the traces table. They are never training candidates, so they are removed before the
    # filters run rather than left for decontamination to catch as duplicates of themselves.
    eval_ids = reg.eval_set_trace_ids()
    if eval_ids:
        before = len(traces)
        traces = [t for t in traces if t["id"] not in eval_ids]
        held = before - len(traces)
        if held:
            console.print(f"[dim]excluded {held} traces belonging to registered eval sets[/dim]")
    if not traces:
        err.print("[red]every trace belongs to an eval set[/red]; ingest training traces before curating.")
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
    config: str | None = typer.Option(None, help="Project config, for train.tool_parser."),
    parser_name: str | None = typer.Option(None, help="vLLM tool parser name, e.g. hermes."),
    family: str | None = typer.Option(None, help="Fallback parser family: hermes, llama3_json."),
    allow_unparsed: bool = typer.Option(
        False, help="Accept a template whose tool calls no parser can recover. You will not be able to serve it."
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit the report as JSON."),
) -> None:
    """Check a base model's chat template: tool support, prefix stability, masking, and the tool-call round trip.

    The round trip is the check that prevents a wasted GPU day. vLLM recovers tool calls from generated *text*
    with a template-specific parser; if training data renders them in a shape that parser cannot read, the student
    is useless however good its loss looks.
    """
    from agentdistill.data.dataset import load_tokenizer
    from agentdistill.data.template_check import check_template, detect_family, roundtrip_tool_call

    if config and (parser_name is None or family is None):
        cfg = _load(config)
        if cfg.train is not None:
            parser_name = parser_name or cfg.train.tool_parser.name
            family = family or cfg.train.tool_parser.family

    try:
        tok = load_tokenizer(model)
    except Exception as e:
        err.print(f"[red]could not load a tokenizer for {model!r}:[/red] {e}")
        raise typer.Exit(code=1) from e

    report = check_template(tok, model)
    trip = roundtrip_tool_call(tok, parser_name=parser_name, family=family) if report.ok else None

    if json_out:
        payload = report.to_dict()
        payload["roundtrip"] = trip
        payload["detected_family"] = detect_family(tok)
        console.print_json(json.dumps(payload, default=str))
        raise typer.Exit(code=0 if report.ok and (trip is None or trip["ok"] or allow_unparsed) else 1)

    table = Table(title=f"chat template: {model}", box=None)
    table.add_column("check")
    table.add_column("", justify="center")
    table.add_column("detail", overflow="fold")
    for c in report.checks:
        table.add_row(c.name, "[green]PASS[/green]" if c.ok else "[red]FAIL[/red]", c.detail)
    if trip is not None:
        table.add_row(
            "tool_call_roundtrip",
            "[green]PASS[/green]" if trip["ok"] else "[red]FAIL[/red]",
            f"via {trip['parser']}. {trip['detail']}".strip(),
        )
    console.print(table)

    if not report.ok:
        console.print("\n[red]not usable as a base model for dataset building.[/red]")
        raise typer.Exit(code=1)

    detected = detect_family(tok)
    if trip is not None and not trip["ok"]:
        console.print("\n[red]the tool-call round trip failed.[/red] The template rendered:")
        console.print(f"  [dim]{trip['model_output'][:400]!r}[/dim]")
        console.print(f"  expected the parser to recover {trip['expected']}, got {trip['parsed']}")
        if not allow_unparsed:
            console.print(
                "\nTraining on this would produce a student whose tool calls the serving stack silently drops. "
                "Set train.tool_parser, or pass --allow-unparsed if you know what you are doing."
            )
            raise typer.Exit(code=1)
        console.print("\n[yellow]--allow-unparsed: continuing anyway. You will not be able to serve this.[/yellow]")

    console.print("\n[green]usable as a base model.[/green]")
    if detected and not parser_name:
        console.print(
            f"  Detected tool-call family [bold]{detected}[/bold]. Set train.tool_parser in project.yaml "
            f"(name = vLLM's parser name, family = {detected}) so serving uses the same answer."
        )
    if trip is not None and trip["parser"].startswith("fallback"):
        console.print("  [dim]Verified with the fallback regex; vLLM is not installed here.[/dim]")


@train_app.command("sft")
def train_sft_cmd(
    dataset: str = typer.Argument(..., help="Dataset name (latest version) or a dataset directory."),
    config: str = typer.Option("project.yaml", help="Project config."),
    name: str | None = typer.Option(None, help="Adapter name; defaults to the project name."),
    out: str | None = typer.Option(None, help="Where to write the adapter; defaults under artifacts/adapters."),
    max_steps: int | None = typer.Option(None, help="Cap training steps. Useful for a smoke run."),
) -> None:
    """LoRA / QLoRA supervised fine-tuning on a built dataset.

    The resulting adapter enters the registry as a `candidate`. Nothing is promoted on loss curves: promotion
    requires a paired evaluation against the teacher, which lands in milestone 3.
    """
    import uuid

    from agentdistill.registry import utcnow

    cfg = _load(config)
    if cfg.train is None:
        err.print("[red]project.yaml has no `train` section[/red]; add one with a base_model and re-run.")
        raise typer.Exit(code=1)
    reg = _registry(cfg)

    ds = reg.get_dataset(dataset)
    if ds is not None:
        dataset_path, dataset_id = ds["path"], ds["id"]
    else:
        path = Path(dataset)
        if not path.exists():
            err.print(f"[red]no dataset named {dataset!r} in the registry and no directory at {path}[/red]")
            err.print("Run `agentdistill dataset list` to see what is available.")
            raise typer.Exit(code=1)
        dataset_path, dataset_id = str(path), None

    try:
        from agentdistill.train.sft import NoSuchBaseModel, TrainingUnavailable, train_sft
    except ImportError as e:  # pragma: no cover
        err.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1) from e

    train_cfg = cfg.train.model_dump()
    if max_steps is not None:
        train_cfg["max_steps"] = max_steps

    adapter_name = name or cfg.name
    version = reg.next_adapter_version(adapter_name)
    out_dir = Path(out) if out else cfg.artifacts_dir / "adapters" / f"{adapter_name}-v{version}"

    run_id = f"tr_{uuid.uuid4().hex[:16]}"
    if dataset_id is not None:
        reg.insert_training_run({
            "id": run_id, "dataset_id": dataset_id, "base_model": train_cfg["base_model"], "method": "sft",
            "config": train_cfg, "status": "running", "started_at": utcnow(),
        })

    console.print(f"[bold]training[/bold] {adapter_name} v{version} from {dataset_path}")
    console.print(f"  base model  {train_cfg['base_model']}")
    console.print(f"  quantization {train_cfg.get('quantization') or 'none'}, LoRA r={train_cfg['lora']['r']}")

    try:
        result = train_sft(train_cfg, dataset_path, out_dir)
    except (TrainingUnavailable, NoSuchBaseModel) as e:
        if dataset_id is not None:
            reg.finish_training_run(run_id, "failed", {"error": str(e)}, None)
        err.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1) from e
    except Exception as e:
        if dataset_id is not None:
            reg.finish_training_run(run_id, "failed", {"error": f"{type(e).__name__}: {e}"}, None)
        raise

    if dataset_id is not None:
        reg.finish_training_run(run_id, "succeeded", result.to_dict(), str(out_dir))
        reg.insert_adapter({
            "id": f"ad_{uuid.uuid4().hex[:16]}", "training_run_id": run_id, "name": adapter_name,
            "version": version, "base_model": train_cfg["base_model"], "path": str(out_dir),
        })

    loss = f"{result.eval_loss:.4f}" if result.eval_loss is not None else "n/a"
    console.print(f"[green]done[/green] {result.steps} steps, eval_loss {loss}")
    console.print(f"  adapter {out_dir}")
    console.print(
        "  [dim]status: candidate. Loss is a proxy; run a paired eval against the teacher before trusting it.[/dim]"
    )


@adapter_app.command("list")
def adapter_list(config: str = typer.Option("project.yaml")) -> None:
    """List adapters and their lifecycle status."""
    cfg = _load(config)
    rows = _registry(cfg).list_adapters()
    if not rows:
        console.print("no adapters yet")
        return
    table = Table(box=None)
    for col in ("name", "v", "status", "base model", "path"):
        table.add_column(col, justify="right" if col == "v" else "left")
    for r in rows:
        table.add_row(r["name"], str(r["version"]), r["status"], r["base_model"], r["path"])
    console.print(table)


@train_app.command("onpolicy")
def train_onpolicy(adapter: str, config: str = "project.yaml", rounds: int | None = None) -> None:
    """Rejection sampling and DPO rounds using the eval harness for rollouts."""
    _not_built("on-policy training", "milestone 4; it depends on the eval harness from milestone 3")


def _resolve_grader(cfg: Any):
    """Turn `eval.grader` into a callable `(trace, outcome) -> (success, detail)`."""
    from agentdistill.eval.runner import label_grader

    kind = cfg.eval.grader.type
    if kind in ("predicate", "replay_predicate"):
        source = cfg.eval.grader.predicate_source or "examples.support_agent.replay_grader:grade_outcome"
        module_name, _, attr = source.partition(":")
        import importlib

        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            err.print(
                f"[red]could not import the predicate source {source!r}:[/red] {e}\n"
                f"Set eval.grader.predicate_source to a `module:callable` that grades a (trace, outcome) pair."
            )
            raise typer.Exit(code=1) from e
        return getattr(module, attr)
    if kind == "llm_judge":
        from agentdistill.eval.calibration import make_llm_judge

        grader_cfg = cfg.eval.grader
        rubric_path = cfg.resolve(grader_cfg.rubric) if grader_cfg.rubric else None
        if rubric_path is None or not rubric_path.exists():
            err.print(f"[red]eval.grader.rubric is required for llm_judge and was not found[/red]: {rubric_path}")
            raise typer.Exit(code=1)
        judge_client = _resolve_client(f"http:{grader_cfg.judge_model}@{cfg.eval.judge_base_url}", cfg, "http")
        console.print(
            "[yellow]judge grading:[/yellow] results will be reported with the judge's agreement and error "
            "rates, never as a bare number. Calibrate with `agentdistill judge calibrate`."
        )
        return make_llm_judge(judge_client, rubric_path.read_text(), grader_cfg.judge_model or "")
    return label_grader


def _resolve_client(subject: str, cfg: Any, backend: str, tok: Any = None):
    """Turn a subject string into a TurnClient.

    Subjects: an adapter name, `base`, `teacher`, `recorded`, or `http:<model>@<url>`.
    """
    from agentdistill.eval.clients import HfTurnClient, HttpTurnClient

    parser = cfg.train.tool_parser if cfg.train else None
    parser_name = parser.name if parser else None
    family = parser.family if parser else None

    if subject.startswith("http:"):
        rest = subject[len("http:") :]
        remote_model, _, url = rest.partition("@")
        if not url:
            err.print("[red]http subjects look like http:<model>@<base-url>[/red]")
            raise typer.Exit(code=1)
        return HttpTurnClient(base_url=url, model=remote_model)

    if subject == "recorded":
        # The control: replays the recorded turns. Used to prove the harness reproduces a trace.
        return "recorded"

    if cfg.train is None:
        err.print("[red]project.yaml has no `train` section, so a model subject cannot be resolved[/red]")
        raise typer.Exit(code=1)

    base_model = cfg.train.base_model
    adapter_path = None
    if subject not in ("base", "teacher"):
        reg = _registry(cfg)
        match = next((a for a in reg.list_adapters() if a["name"] == subject or a["id"] == subject), None)
        if match is None:
            err.print(f"[red]no adapter named {subject!r}[/red]; try `agentdistill adapter list`, "
                      f"or use `base`, `recorded`, or `http:<model>@<url>`.")
            raise typer.Exit(code=1)
        adapter_path, base_model = match["path"], match["base_model"]

    if backend == "vllm":
        from agentdistill.eval.clients import VllmOfflineTurnClient

        return VllmOfflineTurnClient(base_model, tok, parser_name, family, lora_path=adapter_path)

    from agentdistill.data.dataset import load_tokenizer

    tok = tok or load_tokenizer(base_model)
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as e:
        err.print("[red]the hf backend needs transformers; install `agentdistill[train]`[/red]")
        raise typer.Exit(code=1) from e
    loaded: Any = AutoModelForCausalLM.from_pretrained(base_model)
    if adapter_path:
        from peft import PeftModel

        loaded = PeftModel.from_pretrained(loaded, adapter_path)
    return HfTurnClient(loaded, tok, parser_name, family)


@eval_app.command("run")
def eval_run(
    subject: str = typer.Argument(..., help="Adapter name, 'base', 'recorded', or 'http:<model>@<url>'."),
    eval_set: str | None = typer.Option(None, help="Eval set name; defaults to eval.eval_set."),
    n: int | None = typer.Option(None, "--n", help="Repeats per task."),
    policy: str = typer.Option("strict", help="strict or fuzzy replay."),
    backend: str = typer.Option("hf", help="hf, vllm, or http."),
    max_turns: int = typer.Option(12),
    fuzzy_threshold: float = typer.Option(0.92),
    store_messages: bool = typer.Option(True, help="Keep full trajectories for hand review."),
    config: str = typer.Option("project.yaml"),
) -> None:
    """Evaluate a subject on an eval set, replaying tool results from the recorded traces."""
    from agentdistill.eval.report import render_run
    from agentdistill.eval.runner import RunSpec, run_eval

    cfg = _load(config)
    reg = _registry(cfg)
    name = eval_set or cfg.eval.eval_set
    if not name:
        err.print("[red]no eval set[/red]; pass --eval-set or set eval.eval_set in project.yaml.")
        raise typer.Exit(code=1)
    es = reg.get_eval_set(name)
    if es is None:
        err.print(f"[red]no eval set named {name!r}[/red]; see `agentdistill evalset list`.")
        raise typer.Exit(code=1)

    traces_by_task = {}
    for trace_id in es["trace_ids"]:
        trace = reg.get_trace(trace_id)
        if trace is not None:
            traces_by_task[trace_id] = trace
    if not traces_by_task:
        err.print(f"[red]eval set {name!r} references no traces that are still in the registry[/red]")
        raise typer.Exit(code=1)

    grader = _resolve_grader(cfg)
    client = _resolve_client(subject, cfg, backend)

    spec = RunSpec(
        subject=subject,
        eval_set=name,
        n_per_task=n or cfg.eval.n_per_task,
        policy=policy,
        max_turns=max_turns,
        fuzzy_threshold=fuzzy_threshold,
    )
    console.print(f"[bold]eval[/bold] {subject} on {name}: {len(traces_by_task)} tasks x {spec.n_per_task}")

    from rich.progress import Progress

    with Progress(transient=True) as bar:
        task_bar = bar.add_task("running", total=len(traces_by_task) * spec.n_per_task)

        def tick(done: int, total: int) -> None:
            bar.update(task_bar, completed=done)

        if client == "recorded":
            run_id = _run_recorded(reg, es, traces_by_task, grader, spec, store_messages, tick)
        else:
            run_id = run_eval(reg, es, traces_by_task, client, grader, spec,
                              store_messages=store_messages, progress=tick)

    run = reg.get_eval_run(run_id)
    console.print()
    console.print(render_run(run))
    console.print()
    console.print(f"[dim]compare with: agentdistill eval compare {run_id[:10]} <other-run>[/dim]")


def _run_recorded(reg, es, traces_by_task, grader, spec, store_messages, tick):
    """`recorded` needs a fresh client per task, since each replays that task's own turns."""
    import uuid

    from agentdistill.eval.clients import RecordedTurnClient
    from agentdistill.eval.harness import run_task
    from agentdistill.eval.replay import ReplayToolProvider
    from agentdistill.eval.runner import aggregate

    run_id = f"ev_{uuid.uuid4().hex[:16]}"
    reg.start_eval_run(run_id, es["id"], spec.subject, spec.n_per_task)
    done = 0
    for trace_id in es["trace_ids"]:
        trace = traces_by_task.get(trace_id)
        if trace is None:
            continue
        for k in range(spec.n_per_task):
            provider = ReplayToolProvider(trace, policy=spec.policy, fuzzy_threshold=spec.fuzzy_threshold)
            outcome = run_task(trace, RecordedTurnClient(trace), provider, repeat_idx=k,
                               max_turns=spec.max_turns)
            success, detail = grader(trace, outcome)
            outcome.success = success
            outcome.grader_detail = str(detail.get("detail", ""))[:500]
            reg.write_eval_result(run_id, outcome, cluster=trace.get("cluster"),
                                  store_messages=store_messages)
            done += 1
            tick(done, 0)
    metrics, per_cluster = aggregate(reg.eval_results(run_id), traces_by_task)
    reg.finish_eval_run(run_id, metrics, per_cluster)
    return run_id


@eval_app.command("compare")
def eval_compare(
    run_a: str = typer.Argument(..., help="Run id, id prefix, or latest:<subject>."),
    run_b: str = typer.Argument(..., help="The baseline. Positive deltas favour run_a."),
    alpha: float = typer.Option(0.05),
    out: str | None = typer.Option(None, help="Also write the report as Markdown."),
    config: str = typer.Option("project.yaml"),
) -> None:
    """Paired statistical comparison of two eval runs."""
    from agentdistill.eval.report import render_comparison, render_run_markdown
    from agentdistill.eval.runner import compare
    from agentdistill.eval.stats import TooFewTasks

    cfg = _load(config)
    reg = _registry(cfg)
    a, b = reg.find_eval_run(run_a), reg.find_eval_run(run_b)
    if a is None or b is None:
        err.print(f"[red]could not resolve {'run_a' if a is None else 'run_b'}[/red]; "
                  f"try `agentdistill eval list`.")
        raise typer.Exit(code=1)
    try:
        result = compare(reg, a["id"], b["id"], alpha=alpha)
    except (TooFewTasks, ValueError) as e:
        err.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1) from e

    console.print(render_comparison(result))
    if out:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_run_markdown(a, result))
        console.print(f"\n[green]wrote[/green] {path}")


@eval_app.command("calibrate-judge")
def eval_calibrate_judge(
    judge_run: str = typer.Argument(..., help="An eval run graded by the judge."),
    truth_run: str = typer.Argument(..., help="The same tasks graded by a predicate or human labels."),
    out: str | None = typer.Option(None, help="Where to write the calibration JSON."),
    config: str = typer.Option("project.yaml"),
) -> None:
    """Measure a judge against trusted labels on the same tasks.

    A judge is a measuring instrument with its own error rate, and it is usually asymmetric: most judges call a
    mediocre trajectory a success far more readily than they call a good one a failure. Without this, that bias
    sits inside every number the judge produces.
    """
    from agentdistill.eval.calibration import MIN_CALIBRATION_ITEMS, calibrate_judge, report_line

    cfg = _load(config)
    reg = _registry(cfg)
    a, b = reg.find_eval_run(judge_run), reg.find_eval_run(truth_run)
    if a is None or b is None:
        err.print("[red]could not resolve both runs[/red]; try `agentdistill eval list`.")
        raise typer.Exit(code=1)

    judged = {(r["task_id"], r["repeat_idx"]): bool(r["success"]) for r in reg.eval_results(a["id"])}
    truth = {(r["task_id"], r["repeat_idx"]): bool(r["success"]) for r in reg.eval_results(b["id"])}
    shared = sorted(set(judged) & set(truth))
    if not shared:
        err.print("[red]the two runs share no task/repeat pairs[/red]; they must cover the same items.")
        raise typer.Exit(code=1)

    cal = calibrate_judge(
        [judged[k] for k in shared], [truth[k] for k in shared],
        judge_model=cfg.eval.grader.judge_model or "", rubric=cfg.eval.grader.rubric or "",
    )
    console.print(report_line(cal.judge_positive_rate, cal))
    if not cal.usable:
        console.print(
            f"[yellow]only {cal.n} labelled items; {MIN_CALIBRATION_ITEMS} are needed before a correction is "
            f"anything but noise.[/yellow]"
        )
    path = Path(out) if out else cfg.artifacts_dir / "calibration" / f"judge-{a['id'][:10]}.json"
    cal.save(path)
    console.print(f"[green]wrote[/green] {path}")


@eval_app.command("list")
def eval_list(config: str = typer.Option("project.yaml")) -> None:
    """List eval runs."""
    cfg = _load(config)
    runs = _registry(cfg).list_eval_runs()
    if not runs:
        console.print("no eval runs yet")
        return
    table = Table(box=None)
    for col in ("run", "subject", "eval set", "success", "divergence", "started"):
        table.add_column(col)
    for r in runs:
        m = r["metrics"] or {}
        table.add_row(
            r["id"][:12], r["subject"], r["eval_set_id"],
            f"{m.get('success', float('nan')) * 100:.1f}%" if m.get("success") is not None else "-",
            f"{m.get('divergence_rate', 0) * 100:.1f}%",
            (r["started_at"] or "")[:19],
        )
    console.print(table)


@eval_app.command("show")
def eval_show(
    run: str = typer.Argument(..., help="Run id or prefix."),
    failures_only: bool = typer.Option(False, help="Only tasks that failed at least once."),
    divergences: bool = typer.Option(False, help="Show the divergence summary instead of per-task rows."),
    config: str = typer.Option("project.yaml"),
) -> None:
    """Per-task outcomes, for hand review. Reading failures is the only way to learn why a number moved."""
    from agentdistill.eval.report import per_task_table, render_run, summarize_divergences

    cfg = _load(config)
    reg = _registry(cfg)
    found = reg.find_eval_run(run)
    if found is None:
        err.print(f"[red]no eval run matching {run!r}[/red]")
        raise typer.Exit(code=1)
    rows = reg.eval_results(found["id"])
    console.print(render_run(found))
    console.print()

    if divergences:
        summary = summarize_divergences(rows)
        if not summary:
            console.print("no divergences")
            return
        table = Table(box=None, title="divergences by tool")
        for col in ("tool", "n", "max nearest score", "example args"):
            table.add_column(col, overflow="fold")
        for entry in summary:
            table.add_row(entry["tool"], str(entry["n"]), f"{entry['max_score']:.2f}",
                          json.dumps(entry["example"].get("args"))[:80])
        console.print(table)
        console.print(
            "\n[dim]A high nearest score is an argument-phrasing gap: add a per-tool rule in canonical.py. "
            "A low one means the student genuinely went elsewhere.[/dim]"
        )
        return

    table = Table(box=None)
    for col in ("task", "success", "diverged", "first failure detail"):
        table.add_column(col, overflow="fold")
    for task, success, diverged, detail in per_task_table(rows):
        if failures_only and success.split("/")[0] == success.split("/")[1]:
            continue
        table.add_row(task, success, diverged, detail)
    console.print(table)


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
