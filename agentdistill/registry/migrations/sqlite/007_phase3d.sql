-- Phase 3d: stages that produce nothing must be visible, and every row must say how to get back to it.
--
-- `calibrations.verdict` is the gate's own assessment (usable, uninformative, ...). The gateway refuses to load
-- anything but `usable`, so a gate at chance can never quietly decide which turns the teacher sees.
-- `calibrations.report` carries what the threshold search concluded, including the predicted escalation rate
-- that `--verify-threshold` measures against.
ALTER TABLE calibrations ADD COLUMN verdict TEXT;
ALTER TABLE calibrations ADD COLUMN report TEXT;

-- Why a request was routed where it was. `no_cluster_model` marks traffic the router could not place; it is
-- counted on /healthz rather than silently billed to the teacher.
ALTER TABLE requests ADD COLUMN routing_reason TEXT;

-- The pair builder's per-task statistics, so a round that trained on a cross product says so on its row.
ALTER TABLE onpolicy_rounds ADD COLUMN pair_stats TEXT;

-- Commit, dirty flag, config path and hash beside the command. A number recorded from a dirty tree is a number
-- nobody can get back to.
ALTER TABLE training_runs ADD COLUMN provenance TEXT;
ALTER TABLE eval_runs ADD COLUMN provenance TEXT;
ALTER TABLE calibrations ADD COLUMN provenance TEXT;
ALTER TABLE adapters ADD COLUMN provenance TEXT;

-- A marker table from before the migration ledger existed. Nothing reads it; 002 no longer creates it.
DROP TABLE IF EXISTS schema_version_002;
