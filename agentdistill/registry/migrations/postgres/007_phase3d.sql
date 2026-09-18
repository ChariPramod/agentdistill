-- Phase 3d: gate verdicts, routing reasons, pair statistics, provenance. See the sqlite copy for why.
ALTER TABLE calibrations ADD COLUMN IF NOT EXISTS verdict TEXT;
ALTER TABLE calibrations ADD COLUMN IF NOT EXISTS report JSONB;
ALTER TABLE requests ADD COLUMN IF NOT EXISTS routing_reason TEXT;
ALTER TABLE onpolicy_rounds ADD COLUMN IF NOT EXISTS pair_stats JSONB;
ALTER TABLE training_runs ADD COLUMN IF NOT EXISTS provenance JSONB;
ALTER TABLE eval_runs ADD COLUMN IF NOT EXISTS provenance JSONB;
ALTER TABLE calibrations ADD COLUMN IF NOT EXISTS provenance JSONB;
ALTER TABLE adapters ADD COLUMN IF NOT EXISTS provenance JSONB;

-- A marker table from before the migration ledger existed. Nothing reads it; 002 no longer creates it.
DROP TABLE IF EXISTS schema_version_002;
