-- `rft` as a dataset kind of its own. See the sqlite copy for why it is not filed as `sft`.
ALTER TABLE datasets DROP CONSTRAINT IF EXISTS datasets_kind_check;
ALTER TABLE datasets ADD CONSTRAINT datasets_kind_check CHECK (kind IN ('sft','dpo','eval','rft'));
