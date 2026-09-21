-- Phase 3f: retirement. See the sqlite copy for why `retired_at` is separate from `adapters.status`.
ALTER TABLE datasets ADD COLUMN IF NOT EXISTS retired_at TEXT;
ALTER TABLE datasets ADD COLUMN IF NOT EXISTS retired_reason TEXT;

ALTER TABLE adapters ADD COLUMN IF NOT EXISTS retired_at TEXT;
ALTER TABLE adapters ADD COLUMN IF NOT EXISTS retired_reason TEXT;

CREATE INDEX IF NOT EXISTS datasets_retired ON datasets(retired_at);
CREATE INDEX IF NOT EXISTS adapters_retired ON adapters(retired_at);
