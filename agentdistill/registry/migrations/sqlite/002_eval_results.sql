-- Per-repeat eval outcomes. eval_runs holds the aggregate; this holds the rows the aggregate was computed from,
-- so a comparison can be re-run, a per-task review is possible, and a claim can be audited after the fact.
CREATE TABLE IF NOT EXISTS eval_results (
  id                    TEXT PRIMARY KEY,
  eval_run_id           TEXT NOT NULL REFERENCES eval_runs(id) ON DELETE CASCADE,
  task_id               TEXT NOT NULL,
  repeat_idx            INTEGER NOT NULL,
  success               INTEGER,
  schema_valid          INTEGER,
  diverged              INTEGER NOT NULL DEFAULT 0,
  divergence            TEXT,
  n_turns               INTEGER,
  n_tool_calls          INTEGER,
  completion_tokens_est INTEGER,
  latency_ms            INTEGER,
  stop_reason           TEXT,
  grader_detail         TEXT,
  replay_stats          TEXT,
  final_text            TEXT,
  messages              TEXT,
  cluster               INTEGER
);
CREATE INDEX IF NOT EXISTS eval_results_run ON eval_results(eval_run_id);
CREATE INDEX IF NOT EXISTS eval_results_task ON eval_results(eval_run_id, task_id);

