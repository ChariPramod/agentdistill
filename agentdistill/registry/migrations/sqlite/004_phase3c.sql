-- Phase 3c: fallback accounting and the exact command that produced each row.

-- A silent fallback is how a broken vLLM becomes a quiet 100% teacher bill. Counting it is the whole point.
ALTER TABLE requests ADD COLUMN fallback INTEGER NOT NULL DEFAULT 0;
ALTER TABLE requests ADD COLUMN fallback_reason TEXT;
CREATE INDEX IF NOT EXISTS requests_fallback ON requests(fallback) WHERE fallback = 1;

-- The invocation that produced a row, so a report can print how to reproduce it.
ALTER TABLE training_runs ADD COLUMN command TEXT;
ALTER TABLE eval_runs ADD COLUMN command TEXT;
ALTER TABLE calibrations ADD COLUMN command TEXT;

-- Cost-block inputs that only the run itself can measure.
ALTER TABLE eval_runs ADD COLUMN throughput_tok_per_s REAL;
ALTER TABLE eval_runs ADD COLUMN throughput_conditions TEXT;
