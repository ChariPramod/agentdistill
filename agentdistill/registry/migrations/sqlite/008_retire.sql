-- Phase 3f: retirement. A row can be wrong for a reason that has nothing to do with how it was trained.
--
-- Tool-call arguments reached the chat template as the OpenAI wire format's JSON *string* until the render
-- boundary fix, so every dataset built before it rendered `"arguments": "{\"a\": 1}"` and every adapter trained
-- on one learned to emit a call the serving stack's parser drops. Deleting those rows would destroy the record
-- of what was run; leaving them selectable would let `dataset latest` hand the GPU day an invalid dataset. So
-- they are marked, and the selectors skip them.
--
-- `retired_at` is the marker, not `adapters.status`: status is the promotion lifecycle (candidate -> canary ->
-- prod -> retired) and a *superseded* prod adapter is retired in that sense while still being perfectly valid.
-- Two different facts need two different columns. `registry retire` sets both for an adapter, because an
-- invalid adapter must also leave the promotion path, and records the reason as an adapter_events row.
--
-- `retired_reason` is NOT NULL-able but never written empty: a retirement with no reason is one nobody can
-- audit. `registry retire` refuses an empty reason rather than letting the column carry the policy.
ALTER TABLE datasets ADD COLUMN retired_at TEXT;
ALTER TABLE datasets ADD COLUMN retired_reason TEXT;

ALTER TABLE adapters ADD COLUMN retired_at TEXT;
ALTER TABLE adapters ADD COLUMN retired_reason TEXT;

-- The selectors filter on this in Python (they already load whole rows), but a report that scans every adapter
-- on a long-lived registry should not read retired ones off disk to discard them.
CREATE INDEX IF NOT EXISTS datasets_retired ON datasets(retired_at);
CREATE INDEX IF NOT EXISTS adapters_retired ON adapters(retired_at);
