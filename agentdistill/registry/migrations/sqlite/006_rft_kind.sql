-- `rft` as a dataset kind of its own.
--
-- An RFT set is supervised data, so filing it as `sft` would be defensible -- and wrong in one specific way
-- that matters: `latest_dataset(kind='sft')` is what the retrain loop feeds to training, and if a round's
-- self-generated rollouts were the latest `sft` dataset, the next retrain would train on the student's own
-- output instead of on curated production traffic. That is how a model quietly collapses onto itself.
--
-- SQLite cannot alter a CHECK constraint, so the table is rebuilt. Row count is asserted by the migration test.
PRAGMA foreign_keys=OFF;

CREATE TABLE datasets_new (
  id            TEXT PRIMARY KEY,
  name          TEXT NOT NULL,
  version       INTEGER NOT NULL,
  kind          TEXT NOT NULL CHECK (kind IN ('sft','dpo','eval','rft')),
  filter_config TEXT NOT NULL,
  n_samples     INTEGER NOT NULL,
  n_tokens      INTEGER NOT NULL,
  content_hash  TEXT NOT NULL,
  path          TEXT NOT NULL,
  report_path   TEXT,
  created_at    TEXT NOT NULL,
  UNIQUE (name, version)
);

INSERT INTO datasets_new
  SELECT id, name, version, kind, filter_config, n_samples, n_tokens, content_hash, path, report_path,
         created_at
  FROM datasets;

DROP TABLE datasets;
ALTER TABLE datasets_new RENAME TO datasets;

PRAGMA foreign_keys=ON;
