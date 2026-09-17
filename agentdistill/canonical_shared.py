"""The MCPGate-compatible argument hash (`args_hash_v=1` on their side).

This is a thin selection of a ruleset, not a second implementation. `canonical.py` holds the only
canonicalizer in this repo; the two versions differ on exactly one axis -- how numbers render -- and that is a
field on `Rules`:

- **`canonical.args_hash`** uses JSON semantics, where `1` and `1.0` are different arguments. A tool whose schema
  says `{"type": "integer"}` accepts one and rejects the other, so the replay index must not collide them.
- **`canonical_shared.args_hash`** uses JavaScript semantics, where an integral float renders as an integer,
  because that is what MCPGate's TypeScript implementation produces and what its stored rows assume.

Both hash sets stay valid, and there is one code path to keep correct. Two canonicalizers that agree today are
two that disagree after the next edit.

Do not use this for approval or integrity hashes: the number normalization is lossy by design.
"""

from __future__ import annotations

from typing import Any

from agentdistill.canonical import SHARED_RULES, canonical_json
from agentdistill.canonical import args_hash as _args_hash
from agentdistill.canonical import normalize as _normalize

ARGS_HASH_VERSION = 1

#: Re-exported so callers can see which rules they are getting.
RULES = SHARED_RULES
DROP = set(SHARED_RULES.drop_keys)


def normalize(value: Any) -> Any:
    """Normalize under the shared ruleset."""
    return _normalize(value, SHARED_RULES)


def canonical(value: Any) -> str:
    """Serialize an already-normalized value the way JSON.stringify would."""
    return canonical_json(value, SHARED_RULES.number_format)


def args_hash(tool: str, args: Any) -> str:
    """The shared replay hash. Lowercase hex sha256."""
    return _args_hash(tool, args, SHARED_RULES)
