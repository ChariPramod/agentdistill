"""Exact and near-duplicate detection.

Duplicates inflate the apparent size of a dataset and over-weight whatever the agent happens to do often. Exact
duplicates are caught on `content_hash` at ingest; this module catches the near ones: the same trajectory with a
different order id, a different customer name, a different number.
"""

from __future__ import annotations

import re

from datasketch import MinHash, MinHashLSH

from agentdistill.ingest.normalize import canonical_arguments

#: Instance data that varies between otherwise identical trajectories: numbers, and id-like tokens such as
#: `o_1234`, `c_9`, `r_77`, `1Z999AA10123456784`.
_NUMBER = re.compile(r"\b\d+(?:\.\d+)?\b")
_IDENTIFIER = re.compile(r"\b(?=[A-Za-z_]*[\d_])[A-Za-z][A-Za-z0-9_]*\d[A-Za-z0-9_]*\b")


def normalize_literals(text: str) -> str:
    """Replace instance data with type placeholders before shingling.

    This is **off by default**, and the default is the one to keep unless you know otherwise.

    Raw shingling already catches the case the filter exists for. A copy of a realistic trace with one changed
    number keeps Jaccard around 0.95 -- changing one token invalidates only ~`n` shingles out of hundreds -- so it
    is caught comfortably above the 0.85 threshold. The overlap only falls below the threshold on very short
    trajectories, where a handful of shingles is all there is.

    Normalizing is the aggressive option, for a corpus whose agent answers in fixed templates. There, every trace
    of a given shape has *identical* assistant text once ids are masked, so the whole corpus collapses to roughly
    one sample per trajectory shape. That is occasionally what you want (50k traces covering three shapes), and
    usually not: different customers' orders are genuinely different training examples. Turn it on deliberately,
    and read the near_dedupe row of the curation report afterwards.
    """
    text = _IDENTIFIER.sub("<ID>", text)
    return _NUMBER.sub("<NUM>", text)


def shingles(text: str, n: int = 5) -> set[str]:
    toks = text.split()
    if len(toks) <= n:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def assistant_text(trace: dict) -> str:
    """Only what the assistant produced. Tool results are environment output, not model behavior; including them
    would make two different trajectories over the same data look identical."""
    parts: list[str] = []
    for m in trace["messages"]:
        if m["role"] != "assistant":
            continue
        if m.get("content"):
            parts.append(m["content"])
        for c in m.get("tool_calls") or []:
            parts.append(c["function"]["name"] + " " + canonical_arguments(c["function"]["arguments"]))
    return "\n".join(parts)


def _text_for_hashing(trace: dict, normalize: bool = False) -> str:
    text = assistant_text(trace)
    return normalize_literals(text) if normalize else text


def _minhash(trace: dict, num_perm: int, shingle_n: int, normalize: bool = False) -> MinHash:
    mh = MinHash(num_perm=num_perm)
    for s in shingles(_text_for_hashing(trace, normalize), n=shingle_n):
        mh.update(s.encode("utf-8"))
    return mh


def jaccard(a: dict, b: dict, shingle_n: int = 5, normalize: bool = False) -> float:
    """Exact Jaccard between two traces' shingle sets. Used by tests to check the MinHash approximation."""
    sa = shingles(_text_for_hashing(a, normalize), n=shingle_n)
    sb = shingles(_text_for_hashing(b, normalize), n=shingle_n)
    if not sa and not sb:
        return 1.0
    union = sa | sb
    return len(sa & sb) / len(union) if union else 0.0


def near_duplicates(
    traces: list[dict], threshold: float = 0.85, num_perm: int = 128, shingle_n: int = 5, normalize: bool = False
) -> set[str]:
    """Return ids of traces to drop, keeping the first occurrence of each near-duplicate group.

    Input order therefore determines which member survives. Callers pass traces in a stable order (the registry
    sorts by created_at, id) so that re-running curation drops the same members and the dataset hash is stable.

    LSH is used only to *propose* candidates; each candidate is then verified against the threshold with the
    MinHash Jaccard estimate. Banding is deliberately loose -- `MinHashLSH.query` returns pairs well below the
    threshold it was constructed with -- so dropping its candidates unverified would silently discard traces the
    configured rule says to keep.
    """
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    kept: dict[str, MinHash] = {}
    drop: set[str] = set()
    for t in traces:
        if not assistant_text(t).strip():
            continue
        mh = _minhash(t, num_perm, shingle_n, normalize)
        if any(mh.jaccard(kept[cand]) >= threshold for cand in lsh.query(mh) if cand in kept):
            drop.add(t["id"])
            continue
        lsh.insert(t["id"], mh)
        kept[t["id"]] = mh
    return drop


def duplicate_groups(
    traces: list[dict], threshold: float = 0.85, num_perm: int = 128, shingle_n: int = 5, normalize: bool = False
) -> dict[str, list[str]]:
    """Map kept trace id -> ids dropped as near-duplicates of it. Used by the curation report."""
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    kept: dict[str, MinHash] = {}
    groups: dict[str, list[str]] = {}
    for t in traces:
        if not assistant_text(t).strip():
            continue
        mh = _minhash(t, num_perm, shingle_n, normalize)
        match = next(
            (cand for cand in lsh.query(mh) if cand in kept and mh.jaccard(kept[cand]) >= threshold), None
        )
        if match is not None:
            groups.setdefault(match, []).append(t["id"])
            continue
        lsh.insert(t["id"], mh)
        kept[t["id"]] = mh
        groups.setdefault(t["id"], [])
    return {k: v for k, v in groups.items() if v}
