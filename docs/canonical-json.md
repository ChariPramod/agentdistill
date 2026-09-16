# Canonical JSON for argument hashing, v1

This spec is shared across **agentdistill**, **mcpgate**, and **agentreplay**. All three hash tool-call arguments,
and if their hashes disagree, replay lookups fail and audit records never correlate. Treat this document as a
contract, not a description.

- Reference implementation: [`agentdistill/canonical.py`](../agentdistill/canonical.py) (Python)
- Companion implementation: `mcpgate packages/gateway/src/audit/canonical.ts` (TypeScript)
- Shared vectors: [`schemas/canonical-vectors.json`](../schemas/canonical-vectors.json) — every implementation
  must pass them

## Why it exists

The replay harness serves a recorded tool result by hashing the call's arguments and looking the hash up. That
makes the hash function part of the measurement:

- **Too strict** and a student that wrote `{"limit": 5, "customer_id": "c_9"}` instead of
  `{"customer_id": "c_9", "limit": 5}` is scored as a divergence when it behaved identically. Divergences
  truncate the trajectory, so the task fails, so the student's measured success rate is wrong.
- **Too loose** and a student that asked for a *different* order gets the recorded result for the right one. That
  inflates the score, which is worse, because nothing downstream will catch it.

Version this spec. Changing the rules changes every stored hash.

## The rules

Applied in order:

1. **Per-tool normalizers first**, if registered (`register_rules`). Use these for genuine tool quirks — a tool
   that accepts `email` or `customer_email` for the same field — not to paper over a student's mistakes.
2. **Objects**: sort keys byte-wise. Drop keys in the drop list.
   Default drop list: `request_id`, `trace_id`, `timestamp`, `ts`, `cursor`, `page_token`, `nonce`.
3. **Arrays**: preserve order, normalize each element. Order is meaning: `["b","a"]` is not `["a","b"]`.
4. **Strings**: trim whitespace. ISO-8601 timestamps become `<ts>`. RFC 4122 UUIDs become `<uuid>`.
   Both patterns are **anchored** — an id embedded in a sentence is left alone, because there the text is content,
   not an identifier.
5. **Numbers**: floats round to 6 decimals; `-0.0` becomes `0.0`. Integers and booleans pass through unchanged.
   `null` passes through. A bool is never treated as an integer, despite Python's type hierarchy.
6. **Serialize** with sorted keys, separators `,` and `:`, UTF-8, no ASCII escaping.
7. **Hash**: `args_hash = sha256(canonical({"args": <normalized>, "tool": <name>}))`, lowercase hex.

The tool name is inside the hashed object, so the same arguments to different tools never collide.

## Deliberate non-rules

- **`5` and `5.0` hash differently.** An integer argument and a float argument are different arguments; a tool
  with `{"type": "integer"}` in its schema would reject one of them.
- **Array order is never sorted.** For a list of line items or a sequence of steps, order is the argument.
- **Case is preserved.** `"SHIPPED"` and `"shipped"` are different strings. If a specific tool is
  case-insensitive, register a rule for that tool rather than weakening the default.
- **Only the listed volatile keys are dropped.** Add to the drop list per tool; do not extend the default without
  changing the version, because it changes every stored hash.

## Drift is a signal, not a nuisance

When the harness reports a high divergence rate, the fix is usually **not** to loosen these rules. Read
`Divergence.nearest` and `score` first:

- Different *values* — the student asked for a different order. That is a real behavioral difference. Leave it.
- Different *shape* for the same meaning — an extra defaulted key, a tool-specific alias. That is a normalization
  gap. Register a per-tool rule and re-run.

Loosening the default to make a number look better is how a cost claim becomes fiction.

## Adding an implementation

Load `schemas/canonical-vectors.json` and, for each vector, assert your implementation reproduces `normalized`,
`canonical`, and `args_hash` byte for byte. The vectors cover key ordering, nesting, unicode, trimming, float
rounding, integer-versus-float, negative zero, booleans and null, dropped volatile keys, three timestamp shapes,
a UUID inside an array, array ordering, and empty arguments.

If you need a rule the vectors do not cover, add a vector in the same commit as the rule.
