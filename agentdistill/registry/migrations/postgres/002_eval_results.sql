CREATE TABLE IF NOT EXISTS eval_results (
  id                    TEXT PRIMARY KEY,
  eval_run_id           TEXT NOT NULL REFERENCES eval_runs(id) ON DELETE CASCADE,
  task_id               TEXT NOT NULL,
  repeat_idx            INTEGER NOT NULL,
  success               BOOLEAN,
  schema_valid          BOOLEAN,
  diverged              BOOLEAN NOT NULL DEFAULT FALSE,
  divergence            JSONB,
  n_turns               INTEGER,
  n_tool_calls          INTEGER,
  completion_tokens_est INTEGER,
  latency_ms            INTEGER,
  stop_reason           TEXT,
  grader_detail         TEXT,
  replay_stats          JSONB,
  final_text            TEXT,
  messages              JSONB,
  cluster               INTEGER
);
CREATE INDEX IF NOT EXISTS eval_results_run ON eval_results(eval_run_id);
CREATE INDEX IF NOT EXISTS eval_results_task ON eval_results(eval_run_id, task_id);

CREATE TABLE IF NOT EXISTS schema_version_002 (marker INTEGER);
