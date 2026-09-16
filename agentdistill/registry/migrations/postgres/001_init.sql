-- agentdistill registry, Postgres variant. Requires pgvector for trace_embeddings.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS traces (
  id             TEXT PRIMARY KEY,
  source         TEXT NOT NULL,
  source_ref     TEXT,
  task_id        TEXT,
  task_input     JSONB,
  messages       JSONB NOT NULL,
  tools          JSONB NOT NULL,
  teacher_model  TEXT,
  success        BOOLEAN,
  grader         TEXT,
  score          REAL,
  n_turns        INTEGER,
  n_tool_calls   INTEGER,
  prompt_tokens  INTEGER,
  completion_tokens INTEGER,
  cost_usd       NUMERIC(12,6),
  content_hash   TEXT NOT NULL,
  cluster        INTEGER,
  metadata       JSONB,
  created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (content_hash)
);
CREATE INDEX IF NOT EXISTS traces_task ON traces(task_id);
CREATE INDEX IF NOT EXISTS traces_success ON traces(success);
CREATE INDEX IF NOT EXISTS traces_cluster ON traces(cluster);

CREATE TABLE IF NOT EXISTS trace_embeddings (
  trace_id   TEXT PRIMARY KEY REFERENCES traces(id) ON DELETE CASCADE,
  model      TEXT NOT NULL,
  embedding  vector(1024) NOT NULL
);
CREATE INDEX IF NOT EXISTS trace_embeddings_hnsw ON trace_embeddings USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS datasets (
  id            TEXT PRIMARY KEY,
  name          TEXT NOT NULL,
  version       INTEGER NOT NULL,
  kind          TEXT NOT NULL CHECK (kind IN ('sft','dpo','eval')),
  filter_config JSONB NOT NULL,
  n_samples     INTEGER NOT NULL,
  n_tokens      BIGINT,
  content_hash  TEXT NOT NULL,
  path          TEXT NOT NULL,
  report_path   TEXT,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (name, version)
);

CREATE TABLE IF NOT EXISTS samples (
  id          TEXT PRIMARY KEY,
  dataset_id  TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
  trace_id    TEXT REFERENCES traces(id),
  kind        TEXT NOT NULL CHECK (kind IN ('trajectory','turn_window','dpo_pair')),
  n_tokens    INTEGER,
  n_target_tokens INTEGER,
  row_idx     INTEGER NOT NULL,
  sample_hash TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS samples_dataset ON samples(dataset_id);

CREATE TABLE IF NOT EXISTS training_runs (
  id           TEXT PRIMARY KEY,
  dataset_id   TEXT NOT NULL REFERENCES datasets(id),
  base_model   TEXT NOT NULL,
  method       TEXT NOT NULL CHECK (method IN ('sft','dpo','rft','grpo')),
  parent_adapter_id TEXT,
  config       JSONB NOT NULL,
  metrics      JSONB,
  adapter_path TEXT,
  status       TEXT NOT NULL CHECK (status IN ('running','succeeded','failed')),
  started_at   TIMESTAMPTZ NOT NULL,
  ended_at     TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS adapters (
  id               TEXT PRIMARY KEY,
  training_run_id  TEXT NOT NULL REFERENCES training_runs(id),
  name             TEXT NOT NULL,
  version          INTEGER NOT NULL,
  base_model       TEXT NOT NULL,
  merged           BOOLEAN NOT NULL DEFAULT FALSE,
  quantization     TEXT,
  path             TEXT NOT NULL,
  status           TEXT NOT NULL CHECK (status IN ('candidate','canary','prod','retired')),
  created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (name, version)
);

CREATE TABLE IF NOT EXISTS eval_sets (
  id         TEXT PRIMARY KEY,
  name       TEXT NOT NULL UNIQUE,
  trace_ids  TEXT[] NOT NULL,
  grader     JSONB NOT NULL,
  frozen_at  TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS eval_runs (
  id            TEXT PRIMARY KEY,
  eval_set_id   TEXT NOT NULL REFERENCES eval_sets(id),
  subject       TEXT NOT NULL,
  n_per_task    INTEGER NOT NULL,
  metrics       JSONB NOT NULL,
  paired        JSONB,
  per_cluster   JSONB,
  started_at    TIMESTAMPTZ,
  ended_at      TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS calibrations (
  id           TEXT PRIMARY KEY,
  adapter_id   TEXT NOT NULL REFERENCES adapters(id),
  eval_run_id  TEXT NOT NULL REFERENCES eval_runs(id),
  features     TEXT[] NOT NULL,
  model_path   TEXT NOT NULL,
  threshold    REAL NOT NULL,
  target       JSONB NOT NULL,
  ece          REAL,
  brier        REAL,
  auroc        REAL,
  escalation_rate REAL,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS router_state (
  cluster_id  INTEGER NOT NULL,
  arm         TEXT NOT NULL,
  alpha       REAL NOT NULL DEFAULT 1,
  beta        REAL NOT NULL DEFAULT 1,
  updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (cluster_id, arm)
);

CREATE TABLE IF NOT EXISTS requests (
  id              TEXT PRIMARY KEY,
  received_at     TIMESTAMPTZ NOT NULL,
  cluster_id      INTEGER,
  arm             TEXT NOT NULL,
  adapter_id      TEXT,
  confidence      REAL,
  escalated       BOOLEAN NOT NULL DEFAULT FALSE,
  student_tokens  INTEGER,
  teacher_tokens  INTEGER,
  cost_usd        NUMERIC(12,6),
  latency_ms      INTEGER,
  outcome         BOOLEAN,
  trace_id        TEXT REFERENCES traces(id),
  payload         JSONB
);
CREATE INDEX IF NOT EXISTS requests_time ON requests(received_at);
CREATE INDEX IF NOT EXISTS requests_outcome ON requests(outcome) WHERE outcome IS NOT NULL;

CREATE TABLE IF NOT EXISTS model_pricing (
  provider              TEXT NOT NULL,
  model                 TEXT NOT NULL,
  input_per_mtok        NUMERIC(10,4) NOT NULL,
  output_per_mtok       NUMERIC(10,4) NOT NULL,
  cache_read_per_mtok   NUMERIC(10,4),
  effective_from        DATE NOT NULL,
  PRIMARY KEY (provider, model, effective_from)
);

CREATE TABLE IF NOT EXISTS schema_version (
  version    INTEGER PRIMARY KEY,
  applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
