-- Phase 3: rollout provenance, on-policy rounds, cascade eval columns, calibration holdout, adapter lifecycle.

-- Rollouts are traces with provenance: which adapter produced them, which repeat, under which replay policy.
ALTER TABLE traces ADD COLUMN parent_adapter_id TEXT REFERENCES adapters(id);
ALTER TABLE traces ADD COLUMN repeat_idx INTEGER;
ALTER TABLE traces ADD COLUMN replay_policy TEXT;
ALTER TABLE traces ADD COLUMN fuzzy_hits INTEGER DEFAULT 0;
ALTER TABLE traces ADD COLUMN tag TEXT;
CREATE INDEX IF NOT EXISTS traces_tag ON traces(tag);
CREATE INDEX IF NOT EXISTS traces_parent_adapter ON traces(parent_adapter_id);

CREATE TABLE IF NOT EXISTS onpolicy_rounds (
  id                TEXT PRIMARY KEY,
  tag               TEXT,
  round_idx         INTEGER NOT NULL,
  start_adapter_id  TEXT NOT NULL REFERENCES adapters(id),
  rollout_eval_run  TEXT REFERENCES eval_runs(id),
  n_rollouts        INTEGER,
  fuzzy_share       REAL,
  rft_dataset_id    TEXT REFERENCES datasets(id),
  dpo_dataset_id    TEXT REFERENCES datasets(id),
  sft_run_id        TEXT REFERENCES training_runs(id),
  dpo_run_id        TEXT REFERENCES training_runs(id),
  candidate_adapter TEXT REFERENCES adapters(id),
  eval_run_id       TEXT REFERENCES eval_runs(id),
  compare           TEXT,
  decision          TEXT CHECK (decision IN ('promote','discard','error')),
  reason            TEXT,
  started_at        TEXT NOT NULL,
  ended_at          TEXT
);
CREATE INDEX IF NOT EXISTS onpolicy_rounds_tag ON onpolicy_rounds(tag);

-- Cascade subjects record what the gate did, so wasted tokens can be reconciled against the request log.
ALTER TABLE eval_results ADD COLUMN logprobs_ref TEXT;
ALTER TABLE eval_results ADD COLUMN escalations INTEGER;
ALTER TABLE eval_results ADD COLUMN wasted_student_tokens INTEGER;

-- The holdout numbers are what the report shows; the in-sample ones are kept for comparison.
ALTER TABLE calibrations ADD COLUMN holdout_metrics TEXT;
ALTER TABLE calibrations ADD COLUMN reliability_bins TEXT;
ALTER TABLE calibrations ADD COLUMN verified TEXT;
ALTER TABLE calibrations ADD COLUMN feature_order TEXT;

ALTER TABLE adapters ADD COLUMN tag TEXT;
ALTER TABLE adapters ADD COLUMN parent_adapter_id TEXT REFERENCES adapters(id);
CREATE INDEX IF NOT EXISTS adapters_tag ON adapters(tag);
CREATE INDEX IF NOT EXISTS adapters_status ON adapters(status);

-- Every status change is an event with the checks that justified it, so a promotion can be audited later.
CREATE TABLE IF NOT EXISTS adapter_events (
  id          TEXT PRIMARY KEY,
  adapter_id  TEXT NOT NULL REFERENCES adapters(id),
  from_status TEXT,
  to_status   TEXT NOT NULL,
  checks      TEXT,
  actor       TEXT,
  created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS adapter_events_adapter ON adapter_events(adapter_id);

CREATE TABLE IF NOT EXISTS cluster_models (
  id            TEXT PRIMARY KEY,
  dataset_id    TEXT NOT NULL REFERENCES datasets(id),
  embedder      TEXT NOT NULL,
  k             INTEGER NOT NULL,
  centroids_ref TEXT NOT NULL,
  labels        TEXT,
  created_at    TEXT NOT NULL
);

ALTER TABLE eval_runs ADD COLUMN tag TEXT;
CREATE INDEX IF NOT EXISTS eval_runs_subject ON eval_runs(subject);
CREATE INDEX IF NOT EXISTS eval_runs_tag ON eval_runs(tag);
