-- Prompt tokens, so the teacher's cost can be measured rather than assumed.
--
-- The log recorded completion tokens only. For agent traffic the prompt is the larger half -- a system prompt
-- and tool schemas repeat on every turn -- so pricing the teacher from completions alone understated it, and an
-- understated teacher is exactly the error that makes a distillation claim look worse than it is.
ALTER TABLE requests ADD COLUMN prompt_tokens INTEGER;
ALTER TABLE requests ADD COLUMN cached_prompt_tokens INTEGER;
