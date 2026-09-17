-- Phase 3c, Postgres variant.
ALTER TABLE requests ADD COLUMN IF NOT EXISTS fallback BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE requests ADD COLUMN IF NOT EXISTS fallback_reason TEXT;
CREATE INDEX IF NOT EXISTS requests_fallback ON requests(fallback) WHERE fallback;

ALTER TABLE training_runs ADD COLUMN IF NOT EXISTS command TEXT;
ALTER TABLE eval_runs ADD COLUMN IF NOT EXISTS command TEXT;
ALTER TABLE calibrations ADD COLUMN IF NOT EXISTS command TEXT;

ALTER TABLE eval_runs ADD COLUMN IF NOT EXISTS throughput_tok_per_s REAL;
ALTER TABLE eval_runs ADD COLUMN IF NOT EXISTS throughput_conditions TEXT;
