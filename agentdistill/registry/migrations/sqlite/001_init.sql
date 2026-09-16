-- agentdistill registry, SQLite variant.
-- Same shape as the Postgres DDL with the agentreplay substitutions: JSONB -> TEXT holding JSON,
-- arrays -> JSON text, TIMESTAMPTZ -> TEXT holding ISO-8601 UTC, vectors computed in Python.

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS traces (
  id                TEXT PRIMARY KEY,
  source            TEXT NOT NULL,
  source_ref        TEXT,
  task_id           TEXT,
  task_input        TEXT,
  messages          TEXT NOT NULL,
  tools             TEXT NOT NULL,
  teacher_model     TEXT,
  success           INTEGER,
  grader            TEXT,
  score             REAL,
  n_turns           INTEGER,
  n_tool_calls      INTEGER,
  prompt_tokens     INTEGER,
  completion_tokens INTEGER,
  cost_usd          NUMERIC,
  content_hash      TEXT NOT NULL UNIQUE,
  cluster           INTEGER,
  metadata          TEXT,
  created_at        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS traces_task ON traces(task_id);
CREATE INDEX IF NOT EXISTS traces_success ON traces(success);
CREATE INDEX IF NOT EXISTS traces_cluster ON traces(cluster);
CREATE INDEX IF NOT EXISTS traces_source ON traces(source);

CREATE TABLE IF NOT EXISTS trace_embeddings (
  trace_id  TEXT PRIMARY KEY REFERENCES traces(id) ON DELETE CASCADE,
  model     TEXT NOT NULL,
  dim       INTEGER NOT NULL,
  embedding TEXT NOT NULL            -- JSON array of floats; cosine search happens in Python in local mode
);

CREATE TABLE IF NOT EXISTS datasets (
  id            TEXT PRIMARY KEY,
  name          TEXT NOT NULL,
  version       INTEGER NOT NULL,
  kind          TEXT NOT NULL CHECK (kind IN ('sft','dpo','eval')),
  filter_config TEXT NOT NULL,
  n_samples     INTEGER NOT NULL,
  n_tokens      INTEGER,
  content_hash  TEXT NOT NULL,
  path          TEXT NOT NULL,
  report_path   TEXT,
  created_at    TEXT NOT NULL,
  UNIQUE (name, version)
);

CREATE TABLE IF NOT EXISTS samples (
  id              TEXT PRIMARY KEY,
  dataset_id      TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
  trace_id        TEXT REFERENCES traces(id),
  kind            TEXT NOT NULL CHECK (kind IN ('trajectory','turn_window','dpo_pair')),
  n_tokens        INTEGER,
  n_target_tokens INTEGER,
  row_idx         INTEGER NOT NULL,
  sample_hash     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS samples_dataset ON samples(dataset_id);

CREATE TABLE IF NOT EXISTS training_runs (
  id                TEXT PRIMARY KEY,
  dataset_id        TEXT NOT NULL REFERENCES datasets(id),
  base_model        TEXT NOT NULL,
  method            TEXT NOT NULL CHECK (method IN ('sft','dpo','rft','grpo')),
  parent_adapter_id TEXT,
  config            TEXT NOT NULL,
  metrics           TEXT,
  adapter_path      TEXT,
  status            TEXT NOT NULL CHECK (status IN ('running','succeeded','failed')),
  started_at        TEXT NOT NULL,
  ended_at          TEXT
);

CREATE TABLE IF NOT EXISTS adapters (
  id              TEXT PRIMARY KEY,
  training_run_id TEXT NOT NULL REFERENCES training_runs(id),
  name            TEXT NOT NULL,
  version         INTEGER NOT NULL,
  base_model      TEXT NOT NULL,
  merged          INTEGER NOT NULL DEFAULT 0,
  quantization    TEXT,
  path            TEXT NOT NULL,
  status          TEXT NOT NULL CHECK (status IN ('candidate','canary','prod','retired')),
  created_at      TEXT NOT NULL,
  UNIQUE (name, version)
);

CREATE TABLE IF NOT EXISTS eval_sets (
  id         TEXT PRIMARY KEY,
  name       TEXT NOT NULL UNIQUE,
  trace_ids  TEXT NOT NULL,           -- JSON array
  grader     TEXT NOT NULL,
  frozen_at  TEXT,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS eval_runs (
  id          TEXT PRIMARY KEY,
  eval_set_id TEXT NOT NULL REFERENCES eval_sets(id),
  subject     TEXT NOT NULL,
  n_per_task  INTEGER NOT NULL,
  metrics     TEXT NOT NULL,
  paired      TEXT,
  per_cluster TEXT,
  started_at  TEXT,
  ended_at    TEXT
);

CREATE TABLE IF NOT EXISTS calibrations (
  id              TEXT PRIMARY KEY,
  adapter_id      TEXT NOT NULL REFERENCES adapters(id),
  eval_run_id     TEXT NOT NULL REFERENCES eval_runs(id),
  features        TEXT NOT NULL,       -- JSON array
  model_path      TEXT NOT NULL,
  threshold       REAL NOT NULL,
  target          TEXT NOT NULL,
  ece             REAL,
  brier           REAL,
  auroc           REAL,
  escalation_rate REAL,
  created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS router_state (
  cluster_id INTEGER NOT NULL,
  arm        TEXT NOT NULL,
  alpha      REAL NOT NULL DEFAULT 1,
  beta       REAL NOT NULL DEFAULT 1,
  updated_at TEXT NOT NULL,
  PRIMARY KEY (cluster_id, arm)
);

CREATE TABLE IF NOT EXISTS requests (
  id             TEXT PRIMARY KEY,
  received_at    TEXT NOT NULL,
  cluster_id     INTEGER,
  arm            TEXT NOT NULL,
  adapter_id     TEXT,
  confidence     REAL,
  escalated      INTEGER NOT NULL DEFAULT 0,
  student_tokens INTEGER,
  teacher_tokens INTEGER,
  cost_usd       NUMERIC,
  latency_ms     INTEGER,
  outcome        INTEGER,
  trace_id       TEXT REFERENCES traces(id),
  payload        TEXT
);
CREATE INDEX IF NOT EXISTS requests_time ON requests(received_at);
CREATE INDEX IF NOT EXISTS requests_outcome ON requests(outcome) WHERE outcome IS NOT NULL;

CREATE TABLE IF NOT EXISTS model_pricing (
  provider            TEXT NOT NULL,
  model               TEXT NOT NULL,
  input_per_mtok      NUMERIC NOT NULL,
  output_per_mtok     NUMERIC NOT NULL,
  cache_read_per_mtok NUMERIC,
  effective_from      TEXT NOT NULL,
  PRIMARY KEY (provider, model, effective_from)
);

CREATE TABLE IF NOT EXISTS schema_version (
  version    INTEGER PRIMARY KEY,
  applied_at TEXT NOT NULL
);
