-- Prompt tokens, so the teacher's cost can be measured rather than assumed. See the sqlite copy for why.
ALTER TABLE requests ADD COLUMN IF NOT EXISTS prompt_tokens INTEGER;
ALTER TABLE requests ADD COLUMN IF NOT EXISTS cached_prompt_tokens INTEGER;
