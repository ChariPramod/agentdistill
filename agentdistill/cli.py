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
def ingest_gateway(
    since: str = typer.Option("7d", help="Only requests received within this window."),
    require_outcome: bool = typer.Option(True, help="Skip requests nobody graded."),
    limit: int | None = typer.Option(None),
    config: str = typer.Option("project.yaml"),
) -> None:
    """Import served requests that have outcomes. This is the retrain loop's source."""
    from agentdistill.ingest.gateway_source import load_traces, summarize

    cfg = _load(config)
    reg = _registry(cfg)

    stats = summarize(reg, since=since)
    console.print(
        f"request log ({since}): {stats['requests']} requests, {stats['graded']} graded "
        f"({stats['successful']} successful), {stats['escalated']} escalated"
    )
    if stats["requests"] and not stats["graded"]:
        console.print(
            "[yellow]nothing is graded[/yellow]. Outcomes arrive through POST /v1/feedback; without them the "
            "log records what was served but not whether it worked, and nothing here can be trained on."
        )

    traces, problems = load_traces(reg, since=since, require_outcome=require_outcome, limit=limit)
    counts = reg.insert_traces(traces)
    _report_ingest("gateway", counts, problems)


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


# --------------------------------------------------------------------------------------------------------------
# Selectors. The GPU-day script substitutes these straight into the next command, so they print one bare id and
# exit non-zero with a message rather than printing nothing.
# --------------------------------------------------------------------------------------------------------------


def _select(fn, **kwargs) -> dict:
    from agentdistill.registry.select import Ambiguous, NoMatch

    try:
        return fn(**kwargs)
    except (NoMatch, Ambiguous) as e:
        err.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1) from e


@dataset_app.command("latest")
def dataset_latest(
    name: str | None = typer.Option(None, help="Dataset name."),
    kind: str | None = typer.Option(None, help="sft, dpo, or eval."),
    config: str = typer.Option("project.yaml"),
) -> None:
    """Print the newest dataset's id. Used by scripts/gpu_day.sh."""
    from agentdistill.registry.select import latest_dataset

    row = _select(latest_dataset, registry=_registry(_load(config)), name=name, kind=kind)
    print(row["id"])


@adapter_app.command("latest")
def adapter_latest(
    tag: str | None = typer.Option(None, help="Exact tag, or a glob like 'gpu-day*'."),
    status: str | None = typer.Option(None),
    quantized: bool = typer.Option(False, "--quantized", help="Only quantized artifacts."),
    config: str = typer.Option("project.yaml"),
) -> None:
    """Print the newest adapter's id."""
    from agentdistill.registry.select import latest_adapter

    row = _select(latest_adapter, registry=_registry(_load(config)), tag=tag, status=status,
                  quantized=True if quantized else None)
    print(row["id"])


@adapter_app.command("best")
def adapter_best(
    tag: str | None = typer.Option(None, help="Exact tag, or a glob like 'gpu-day*'."),
    eval_set: str | None = typer.Option(None, help="Rank by success on this eval set."),
    config: str = typer.Option("project.yaml"),
) -> None:
    """Print the id of the adapter with the highest measured success.

    Ranks by measured success, so an adapter with no eval run is not a candidate however new it is: picking one
    would put an unmeasured model into the report.
    """
    from agentdistill.registry.select import best_adapter

    cfg = _load(config)
    row = _select(best_adapter, registry=_registry(cfg), tag=tag, eval_set=eval_set or cfg.eval.eval_set)
    print(row["id"])


@eval_app.command("latest")
def eval_latest(
    subject: str | None = typer.Option(None),
    tag: str | None = typer.Option(None, help="Exact tag, or a glob."),
    eval_set: str | None = typer.Option(None),
    config: str = typer.Option("project.yaml"),
) -> None:
    """Print the newest matching eval run's id."""
    from agentdistill.registry.select import latest_eval

    row = _select(latest_eval, registry=_registry(_load(config)), subject=subject, tag=tag, eval_set=eval_set)
    print(row["id"])


@app.command("config")
def config_cmd(
    action: str = typer.Argument(..., help="Only 'get' is supported."),
    key: str = typer.Argument(..., help="Dotted path, e.g. train.base_model."),
    config: str = typer.Option("project.yaml"),
) -> None:
    """Print one config value. Used by scripts/gpu_day.sh so the script has no duplicated settings."""
    if action != "get":
        err.print("[red]only `config get <key>` is supported[/red]")
        raise typer.Exit(code=1)
    cfg = _load(config)
    node: Any = cfg
    for part in key.split("."):
        node = getattr(node, part, None) if not isinstance(node, dict) else node.get(part)
        if node is None:
            err.print(f"[red]no config value at {key!r}[/red]")
            raise typer.Exit(code=1)
    print(node)


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
def train_onpolicy(
    adapter: str = typer.Argument(..., help="The adapter to improve."),
    rounds: int = typer.Option(1, help="How many rounds to attempt. A discarded round stops the loop."),
    tag: str | None = typer.Option(None, help="Groups this session's artifacts for the selectors."),
    k: int | None = typer.Option(None, "--k", help="Rollouts per task."),
    dry_run: bool = typer.Option(False, help="Print the stage plan and stop."),
    config: str = typer.Option("project.yaml"),
) -> None:
    """Rejection sampling and DPO rounds, using the eval harness for rollouts.

    Each round rolls the current student out on training tasks, keeps what worked as new SFT data, pairs what did
    not against what did, trains, evaluates, and compares the candidate against the *current adapter*. The
    teacher comparison belongs in the report; here the only question is whether this round improved on the last.
    """
    from agentdistill.train.onpolicy import RoundCfg, plan, run_rounds

    cfg = _load(config)
    reg = _registry(cfg)

    op = cfg.onpolicy
    round_cfg = RoundCfg(
        k_rollouts=k or op.k_rollouts,
        rft_cap_per_task=op.rft_cap_per_task,
        min_pairs=op.min_pairs,
        max_fuzzy_share=op.max_fuzzy_share,
    )

    train_traces = [t for t in reg.list_traces() if t["id"] not in reg.eval_set_trace_ids()]
    task_ids = [t["id"] for t in train_traces]
    if not task_ids:
        err.print("[red]no training traces[/red]; ingest and curate before running on-policy rounds.")
        raise typer.Exit(code=1)

    if dry_run:
        for line in plan(adapter, rounds, task_ids, round_cfg):
            console.print(f"  {line}")
        console.print("\n[yellow]dry run: nothing executed[/yellow]")
        return

    current_eval = _select(
        __import__("agentdistill.registry.select", fromlist=["latest_eval"]).latest_eval,
        registry=reg, subject=adapter, eval_set=cfg.eval.eval_set,
    )
    stages = _onpolicy_stages(cfg, reg, tag, round_cfg)
    teacher_by_task = {t.get("task_id") or t["id"]: t for t in train_traces}

    results = run_rounds(adapter, rounds, task_ids, teacher_by_task, current_eval["id"], stages, round_cfg)
    for r in results:
        _print_round(r)
    promoted = [r for r in results if r.promoted]
    if promoted:
        console.print(f"\n[green]kept[/green] {promoted[-1].candidate_adapter} after {len(promoted)} round(s)")
    else:
        console.print("\n[yellow]no round was kept[/yellow]; the starting adapter is still the best you have")


def _print_round(r) -> None:
    style = "green" if r.promoted else ("red" if r.decision == "error" else "yellow")
    console.print(f"\n[bold]round {r.round_idx}[/bold] from {r.start_adapter}")
    console.print(f"  rollouts    {r.n_rollouts}  (fuzzy share {r.fuzzy_share:.0%})")
    console.print(f"  RFT samples {r.n_rft}")
    console.print(f"  pairs       {r.n_pairs}  {r.pair_kinds or ''}")
    if r.compare:
        s = r.compare.get("success") or {}
        lo, hi = s.get("ci95", (0, 0))
        console.print(
            f"  success     {s.get('delta', 0) * 100:+.1f} pp  [CI {lo * 100:+.1f}, {hi * 100:+.1f}]"
        )
        console.print(f"  tokens      {(r.compare.get('tokens') or {}).get('median_delta', 0):+.0f} median")
    console.print(f"  [{style}]{r.decision}[/{style}]: {r.reason}")


def _onpolicy_stages(cfg, reg, tag, round_cfg):
    """Wire the loop's stages to the real trainers, harness, and registry."""
    import uuid

    from agentdistill.eval.rollouts import build_pairs as build_pairs_fn
    from agentdistill.eval.rollouts import build_rft as build_rft_fn
    from agentdistill.eval.rollouts import collect_rollouts
    from agentdistill.eval.runner import RunSpec, run_eval
    from agentdistill.eval.runner import compare as compare_runs
    from agentdistill.train.onpolicy import Stages

    grader = _resolve_grader(cfg)
    state: dict[str, Any] = {}

    def _collect(adapter, task_ids, k, policy):
        client = _resolve_client(adapter, cfg, "hf")
        traces = [reg.get_trace(t) for t in task_ids]
        rollouts = collect_rollouts(
            [t for t in traces if t], client, grader, k=k, policy=policy,
            fuzzy_threshold=round_cfg.max_fuzzy_share, adapter_id=adapter,
        )
        state["rollouts"] = rollouts
        return {"rollouts": rollouts.rollouts, "fuzzy_share": rollouts.replay.get("fuzzy_share", 0.0),
                "eval_run_id": None}

    def _build_rft(rollouts, cap):
        from agentdistill.eval.rollouts import RolloutSet

        rs = state.get("rollouts") or RolloutSet(rollouts=rollouts)
        picked = build_rft_fn(rs, cap_per_task=cap)
        state["rft"] = picked
        return f"ds_rft_{uuid.uuid4().hex[:8]}", len(picked)

    def _build_pairs(rollouts, teacher_by_task):
        from agentdistill.eval.rollouts import RolloutSet
        from agentdistill.train.dpo_data import balance_kinds, filter_pairs

        rs = state.get("rollouts") or RolloutSet(rollouts=rollouts)
        pairs = build_pairs_fn(rs, list(teacher_by_task.values()))
        usable, _ = filter_pairs(pairs)
        balanced, kinds = balance_kinds(usable, max_teacher_ratio=round_cfg.max_teacher_ratio)
        state["pairs"] = balanced
        return f"ds_pairs_{uuid.uuid4().hex[:8]}", len(balanced), kinds

    def _train_sft_continue(adapter, dataset_id):
        _not_built("continuing SFT from an existing adapter inside a round", "needs a GPU; see scripts/gpu_day.sh")
        raise AssertionError("unreachable")

    def _merge(adapter):
        _not_built("adapter merge", "milestone 2 on a GPU; see scripts/gpu_day.sh")
        raise AssertionError("unreachable")

    def _train_dpo(merged, dataset_id):
        _not_built("DPO inside a round", "needs a GPU; the trainer itself is tested on CPU")
        raise AssertionError("unreachable")

    def _run_eval(adapter):
        es = reg.get_eval_set(cfg.eval.eval_set)
        traces = {t: reg.get_trace(t) for t in es["trace_ids"]}
        return run_eval(reg, es, {k: v for k, v in traces.items() if v},
                        _resolve_client(adapter, cfg, "hf"), grader,
                        RunSpec(subject=adapter, eval_set=cfg.eval.eval_set,
                                n_per_task=cfg.eval.n_per_task, tag=tag))

    def _compare(a, b):
        return compare_runs(reg, a, b)

    def _metrics(eval_run):
        run = reg.get_eval_run(eval_run)
        return (run or {}).get("metrics") or {}

    def _record(row):
        reg.record_round({**row, "tag": tag})

    return Stages(
        collect_rollouts=_collect, build_rft=_build_rft, build_pairs=_build_pairs,
        train_sft_continue=_train_sft_continue, merge=_merge, train_dpo=_train_dpo,
        run_eval=_run_eval, compare=_compare, metrics=_metrics, record=_record,
    )


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
    reg.start_eval_run(run_id, es["id"], spec.subject, spec.n_per_task, tag=spec.tag)
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
    from agentdistill.eval.calibration import (
        MIN_CALIBRATION_ITEMS,
        calibrate_judge,
        holdout_error,
        report_line,
    )

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

    judge_labels = [judged[k] for k in shared]
    truth_labels = [truth[k] for k in shared]
    cal = calibrate_judge(
        judge_labels, truth_labels,
        judge_model=cfg.eval.grader.judge_model or "", rubric=cfg.eval.grader.rubric or "",
    )
    holdout = holdout_error(judge_labels, truth_labels)
    console.print(report_line(cal.judge_positive_rate, cal, holdout))
    if not cal.usable:
        console.print(
            f"[yellow]only {cal.n} labelled items; {MIN_CALIBRATION_ITEMS} are needed before a correction is "
            f"anything but noise.[/yellow]"
        )
    path = Path(out) if out else cfg.artifacts_dir / "calibration" / f"judge-{a['id'][:10]}.json"
    cal.save(path)
    # The holdout check is what says whether the correction generalizes, so it ships beside the calibration.
    Path(str(path).replace(".json", "-holdout.json")).write_text(json.dumps(holdout, indent=2, sort_keys=True))
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
def calibrate(
    adapter: str = typer.Argument(..., help="The adapter whose gate is being fitted."),
    from_eval: str | None = typer.Option(None, "--from-eval", help="Eval run supplying the labelled turns."),
    out: str | None = typer.Option(None, help="Where to write the calibration artifact."),
    max_drop_pp: float | None = typer.Option(None, help="Success-drop budget for the threshold search."),
    config: str = typer.Option("project.yaml"),
) -> None:
    """Fit the confidence gate, choose a threshold, and report the gate's own reliability.

    Fitted on a task-disjoint split and reported on the half it never saw: in-sample calibration error is
    optimistic by construction. The artifact that ships is refit on everything, because holdout numbers are about
    honesty rather than shipping a weaker model.

    If the gate is not good enough to threshold on, this says so and the cascade escalates everything. That is the
    documented default: a cascade with a meaningless gate is worse than no cascade.
    """
    import numpy as np

    from agentdistill.cascade.calibrate import fit_calibrator, save
    from agentdistill.cascade.features import matrix
    from agentdistill.cascade.labels import label_rollouts
    from agentdistill.cascade.threshold import choose_threshold, verification_points

    cfg = _load(config)
    reg = _registry(cfg)

    if not from_eval:
        err.print(
            "[red]--from-eval is required[/red]: the gate is fitted on turns from an eval run recorded with "
            "`eval run <adapter> --logprobs`, on a task set disjoint from training."
        )
        raise typer.Exit(code=1)
    run = reg.find_eval_run(from_eval)
    if run is None:
        err.print(f"[red]no eval run matching {from_eval!r}[/red]")
        raise typer.Exit(code=1)

    rows = reg.eval_results(run["id"])
    rollouts = [
        {"id": f"{r['task_id']}#{r['repeat_idx']}", "task_id": r["task_id"], "success": r["success"],
         "messages": r["messages"] or []}
        for r in rows if r.get("messages")
    ]
    if not rollouts:
        err.print(
            "[red]that eval run stored no trajectories[/red], so there are no turns to label. "
            "Re-run it without --no-store-messages."
        )
        raise typer.Exit(code=1)

    teacher_by_task = {}
    for trace_id in reg.get_eval_set(run["eval_set_id"].removeprefix("es_"))["trace_ids"] \
            if reg.get_eval_set(run["eval_set_id"].removeprefix("es_")) else []:
        trace = reg.get_trace(trace_id)
        if trace:
            teacher_by_task[trace.get("task_id") or trace["id"]] = trace

    records, label_stats = label_rollouts(rollouts, teacher_by_task)
    console.print(f"labelled {label_stats['n']} turns  mix={label_stats['mix']}")
    if label_stats["weak_share"] > 0.5:
        console.print(
            f"[yellow]{label_stats['weak_share']:.0%} of labels are `uncorrected`[/yellow], which is an "
            f"assumption rather than evidence. Read the AUROC with that in mind."
        )

    # Without stored logprobs there are no confidence features to fit on.
    if not any(r.get("features") for r in records):
        err.print(
            "[red]no per-turn logprobs in that eval run[/red]. The gate's features come from "
            "`eval run <adapter> --logprobs --samples 3`, which needs the vLLM backend on a GPU. "
            "See scripts/gpu_day.sh, stage `logprobs`."
        )
        raise typer.Exit(code=1)

    features = [r["features"] for r in records]
    y = np.array([int(r["good"]) for r in records])
    task_ids = [r["task_id"] for r in records]
    names = list(cfg.cascade.features)
    result = fit_calibrator(matrix(features, names), y, task_ids, names, label_mix=label_stats["mix"])

    for note in result.notes:
        console.print(f"[yellow]note:[/yellow] {note}")
    if result.holdout:
        h = result.holdout
        console.print(
            f"holdout  AUROC {h.get('auroc', float('nan')):.3f}  Brier {h.get('brier', float('nan')):.3f}  "
            f"ECE {h.get('ece', float('nan')):.3f}  (n={h.get('n')})"
        )
        console.print(f"in-sample AUROC {result.in_sample.get('auroc', float('nan')):.3f} [dim](optimistic)[/dim]")

    path = Path(out) if out else cfg.artifacts_dir / "calibration" / f"{adapter}"
    save(result, path)
    console.print(f"[green]wrote[/green] {path}")

    if not result.usable:
        console.print("[yellow]gate not usable; the cascade will escalate everything and the report says so.[/yellow]")
        return

    p = result.model.predict_proba(matrix(features, names))[:, 1]
    choice = choose_threshold(
        p, y.astype(bool), np.array(task_ids), dict.fromkeys(set(task_ids), True),
        max_success_drop_pp=max_drop_pp if max_drop_pp is not None else cfg.cascade.max_success_drop_pp,
    )
    if choice.chosen is None:
        console.print(f"[yellow]{choice.note}[/yellow]")
        return
    console.print(
        f"threshold {choice.chosen.threshold:.2f}  escalation {choice.chosen.escalation_rate:.0%}  "
        f"estimated success {choice.chosen.cascade_success:.1%}"
    )
    console.print(
        "  [dim]analytic estimate; it assumes an escalated turn is as good as the teacher's, which is optimistic "
        "because the teacher answers on a prefix the student built. Verify with "
        f"`eval run cascade:{adapter}:auto --verify-threshold` at {verification_points(choice.chosen.threshold)}."
        "[/dim]"
    )


@adapter_app.command("merge")
def adapter_merge(adapter: str) -> None:
    """Merge LoRA into the base for serving."""
    _not_built("adapter merge", "milestone 2")


@adapter_app.command("quantize")
def adapter_quantize(adapter: str, method: str = "fp8") -> None:
    """Produce a quantized serving artifact and evaluate it."""
    _not_built("adapter quantize", "milestone 6")


@adapter_app.command("promote")
def adapter_promote(
    adapter: str = typer.Argument(..., help="Adapter id or name."),
    to: str = typer.Option("canary", help="canary, prod, or retired."),
    force: bool = typer.Option(False, help="Promote despite failing checks. The event records that you did."),
    actor: str = typer.Option("cli", help="Who is promoting, recorded on the event."),
    config: str = typer.Option("project.yaml"),
) -> None:
    """Move an adapter through its lifecycle, gated on measured evidence.

    Prints the checks and refuses unless every one is green. Months from now, `adapter lineage` answers "why is
    this adapter in prod" with the comparison that justified it.
    """
    from agentdistill.registry.lifecycle import IllegalTransition, transition

    cfg = _load(config)
    reg = _registry(cfg)
    try:
        result = transition(reg, adapter, to, cfg, actor=actor, force=force)
    except (LookupError, IllegalTransition) as e:
        err.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1) from e

    table = Table(box=None, title=f"{adapter}: {result.from_status} -> {to}")
    table.add_column("check")
    table.add_column("", justify="center")
    table.add_column("detail", overflow="fold")
    for name, check in result.checks.items():
        value = f"  [dim]({check.value})[/dim]" if check.value is not None else ""
        table.add_row(name, "[green]PASS[/green]" if check.ok else "[red]FAIL[/red]", check.detail + value)
    console.print(table)

    if not result.ok:
        console.print(f"\n[red]not promoted[/red]: {', '.join(result.failed)} failed.")
        console.print("Fix the failing checks, or pass --force to override and have that recorded.")
        raise typer.Exit(code=1)
    if result.forced:
        console.print(f"\n[yellow]forced[/yellow] to {to} despite {', '.join(result.failed)}.")
    else:
        console.print(f"\n[green]promoted[/green] to {to}.")


@adapter_app.command("lineage")
def adapter_lineage(
    adapter: str = typer.Argument(..., help="Adapter id or name."),
    config: str = typer.Option("project.yaml"),
) -> None:
    """Print an adapter's provenance: dataset, training run, parents, and every status change."""
    from agentdistill.registry.lifecycle import adapter as get_adapter
    from agentdistill.registry.lifecycle import events

    cfg = _load(config)
    reg = _registry(cfg)
    try:
        row = get_adapter(reg, adapter)
    except LookupError as e:
        err.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1) from e

    console.print(f"[bold]{row['name']} v{row['version']}[/bold]  ({row['id']})")
    console.print(f"  status       {row['status']}" + (f"   tag {row['tag']}" if row.get("tag") else ""))
    console.print(f"  base model   {row['base_model']}")
    console.print(f"  path         {row['path']}")
    if row.get("quantization"):
        console.print(f"  quantization {row['quantization']}")

    run = reg.get_training_run(row["training_run_id"])
    if run:
        console.print(f"  trained by   {run['id']}  ({run['method']}, {run['status']})")
        dataset = next((d for d in reg.list_datasets() if d["id"] == run["dataset_id"]), None)
        if dataset:
            console.print(
                f"  dataset      {dataset['name']} v{dataset['version']}  "
                f"{dataset['n_samples']} samples  hash {dataset['content_hash'][:12]}"
            )
            if dataset.get("report_path"):
                console.print(f"  curation     {dataset['report_path']}")

    parent = row.get("parent_adapter_id")
    while parent:
        try:
            prow = get_adapter(reg, parent)
        except LookupError:
            break
        console.print(f"  parent       {prow['name']} v{prow['version']} ({prow['id']})")
        parent = prow.get("parent_adapter_id")

    history = events(reg, row["id"])
    if history:
        console.print("\n  history")
        for event in history:
            checks = event["checks"] or {}
            failed = [k for k, v in (checks.get("checks") or {}).items() if not v.get("ok")]
            suffix = f"  [yellow]forced past {failed}[/yellow]" if checks.get("forced") else ""
            console.print(f"    {event['created_at'][:19]}  {event['from_status']} -> {event['to_status']}  "
                          f"by {event['actor']}{suffix}")
    else:
        console.print("\n  [dim]no status changes recorded[/dim]")


@app.command()
def serve(
    config: str = typer.Option("project.yaml"),
    host: str | None = typer.Option(None, help="Defaults to serve.host."),
    port: int | None = typer.Option(None, help="Defaults to serve.port."),
    dry_run: bool = typer.Option(False, help="Load and report what would be served, then stop."),
) -> None:
    """Start the gateway.

    The agent points `base_url` here and keeps its model name. Routing, escalation, and the cascade are invisible
    to it, and every response says which arm answered.
    """
    from agentdistill.gateway import app as gateway_app
    from agentdistill.gateway.state import load_state

    cfg = _load(config)
    reg = _registry(cfg)
    state = load_state(cfg, reg)
    gateway_app.set_state(state)

    table = Table(box=None, title="gateway")
    table.add_column("")
    table.add_column("", overflow="fold")
    table.add_row("student", cfg.serve.vllm_url)
    table.add_row("teacher", cfg.teacher.model if cfg.teacher else "[red]none configured[/red]")
    table.add_row("prod adapter", state.prod_adapter or "[yellow]none; the agent's model passes through[/yellow]")
    table.add_row("canary", f"{state.canary_adapter} at {state.canary_share:.0%}" if state.canary_adapter else "-")
    table.add_row("gate", f"threshold {state.prod_threshold}" if state.cascade_available
                  else "[yellow]no usable calibration; escalating every turn[/yellow]")
    console.print(table)
    for note in state.notes:
        console.print(f"[yellow]note:[/yellow] {note}")

    if dry_run:
        console.print("\n[yellow]dry run: not serving[/yellow]")
        return

    try:
        import uvicorn
    except ImportError as e:
        err.print("[red]uvicorn is required to serve[/red]; it ships with the core install.")
        raise typer.Exit(code=1) from e

    bind_host = host or cfg.serve.host
    bind_port = port or cfg.serve.port
    console.print(f"\nlistening on http://{bind_host}:{bind_port}  (POST /v1/chat/completions, /v1/messages)")
    uvicorn.run(gateway_app.app, host=bind_host, port=bind_port, log_level="info")


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
