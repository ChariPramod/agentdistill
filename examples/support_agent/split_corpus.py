"""Split a recorded corpus into training traces and two frozen eval sets.

Per the next-phase plan §3.3, the split is by scenario **instance**, not by scenario: every trained shape also
appears in the holdout, so the student is tested on new instances of things it has seen. A second set holds
several shapes out of training entirely, for the generalization number — which is always worse, and is the
honest one.

A third set, `support-calib-v1`, exists only to fit the confidence gate. It has to be disjoint from both: fitting
on training tasks measures the student's confidence on prefixes it memorized, and fitting on the holdout leaks
into the cascade number that gets reported.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from examples.support_agent import scenarios

HERE = Path(__file__).resolve().parent


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traces", type=Path, default=HERE / "traces.jsonl")
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--calib-frac", type=float, default=0.15,
                    help="Share of each trained shape reserved for fitting the confidence gate.")
    args = ap.parse_args(argv)

    traces = [json.loads(line) for line in args.traces.open()]
    held_out = set(scenarios.HELD_OUT_SCENARIOS)

    seen = [t for t in traces if t["metadata"]["scenario"] not in held_out]
    unseen = [t for t in traces if t["metadata"]["scenario"] in held_out]

    by_scenario: dict[str, list[dict]] = {}
    for t in seen:
        by_scenario.setdefault(t["metadata"]["scenario"], []).append(t)

    train: list[dict] = []
    holdout: list[dict] = []
    calib: list[dict] = []
    for _name, group in sorted(by_scenario.items()):
        group.sort(key=lambda t: t["task_id"])
        n = len(group)
        train_cut = int(n * args.train_frac)
        calib_cut = train_cut + int(n * args.calib_frac)
        train.extend(group[:train_cut])
        calib.extend(group[train_cut:calib_cut])
        holdout.extend(group[calib_cut:])

    for filename, rows, label in [
        ("traces-train.jsonl", train, "training"),
        ("eval-holdout.jsonl", holdout, "held-out instances of trained scenarios"),
        ("eval-unseen.jsonl", unseen, "scenarios never seen in training"),
        ("eval-calib.jsonl", calib, "gate fitting only; disjoint from train and holdout"),
    ]:
        (HERE / filename).write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        rate = sum(r["success"] for r in rows) / len(rows) if rows else 0.0
        print(f"{filename:24} {len(rows):4} traces  success {rate:.0%}  ({label})")

    overlap = {t["id"] for t in train} & ({t["id"] for t in holdout} | {t["id"] for t in calib})
    if overlap:
        raise SystemExit(f"splits overlap on {len(overlap)} traces; the eval numbers would be inflated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
