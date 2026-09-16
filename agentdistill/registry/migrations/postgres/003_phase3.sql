-- Phase 3, Postgres variant. Same shape; JSONB where SQLite uses TEXT.
ALTER TABLE traces ADD COLUMN IF NOT EXISTS parent_adapter_id TEXT REFERENCES adapters(id);
ALTER TABLE traces ADD COLUMN IF NOT EXISTS repeat_idx INTEGER;
ALTER TABLE traces ADD COLUMN IF NOT EXISTS replay_policy TEXT;
ALTER TABLE traces ADD COLUMN IF NOT EXISTS fuzzy_hits INTEGER DEFAULT 0;
ALTER TABLE traces ADD COLUMN IF NOT EXISTS tag TEXT;
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
  compare           JSONB,
  decision          TEXT CHECK (decision IN ('promote','discard','error')),
  reason            TEXT,
  started_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
  ended_at          TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS onpolicy_rounds_tag ON onpolicy_rounds(tag);

ALTER TABLE eval_results ADD COLUMN IF NOT EXISTS logprobs_ref TEXT;
ALTER TABLE eval_results ADD COLUMN IF NOT EXISTS escalations INTEGER;
ALTER TABLE eval_results ADD COLUMN IF NOT EXISTS wasted_student_tokens INTEGER;

ALTER TABLE calibrations ADD COLUMN IF NOT EXISTS holdout_metrics JSONB;
ALTER TABLE calibrations ADD COLUMN IF NOT EXISTS reliability_bins JSONB;
ALTER TABLE calibrations ADD COLUMN IF NOT EXISTS verified JSONB;
ALTER TABLE calibrations ADD COLUMN IF NOT EXISTS feature_order JSONB;

ALTER TABLE adapters ADD COLUMN IF NOT EXISTS tag TEXT;
ALTER TABLE adapters ADD COLUMN IF NOT EXISTS parent_adapter_id TEXT REFERENCES adapters(id);
CREATE INDEX IF NOT EXISTS adapters_tag ON adapters(tag);
CREATE INDEX IF NOT EXISTS adapters_status ON adapters(status);

CREATE TABLE IF NOT EXISTS adapter_events (
  id          TEXT PRIMARY KEY,
  adapter_id  TEXT NOT NULL REFERENCES adapters(id),
  from_status TEXT,
  to_status   TEXT NOT NULL,
  checks      JSONB,
  actor       TEXT,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS adapter_events_adapter ON adapter_events(adapter_id);

CREATE TABLE IF NOT EXISTS cluster_models (
  id            TEXT PRIMARY KEY,
  dataset_id    TEXT NOT NULL REFERENCES datasets(id),
  embedder      TEXT NOT NULL,
  k             INTEGER NOT NULL,
  centroids_ref TEXT NOT NULL,
  labels        JSONB,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE eval_runs ADD COLUMN IF NOT EXISTS tag TEXT;
CREATE INDEX IF NOT EXISTS eval_runs_subject ON eval_runs(subject);
CREATE INDEX IF NOT EXISTS eval_runs_tag ON eval_runs(tag);
