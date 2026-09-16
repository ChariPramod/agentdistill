"""JSONL and agentreplay ingest."""

from __future__ import annotations

import json
import sqlite3

import pytest

from agentdistill.ingest.agentreplay_source import AgentReplayError
from agentdistill.ingest.agentreplay_source import load_traces as load_agentreplay
from agentdistill.ingest.jsonl_source import IngestError, load_traces


def _write(tmp_path, records):
    p = tmp_path / "traces.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return p


def test_openai_records_load(tmp_path):
    p = _write(tmp_path, [{"task_id": "t1", "success": True,
                           "messages": [{"role": "user", "content": "hi"},
                                        {"role": "assistant", "content": "hello"}], "tools": []}])
    traces, problems = load_traces(p)
    assert len(traces) == 1 and problems == []
    assert traces[0]["source"] == "jsonl" and traces[0]["n_turns"] == 1


def test_anthropic_records_are_detected_and_converted(tmp_path, anthropic_conversation):
    p = _write(tmp_path, [{**anthropic_conversation, "task_id": "t1", "success": True}])
    traces, problems = load_traces(p)
    assert problems == []
    roles = [m["role"] for m in traces[0]["messages"]]
    assert roles == ["system", "user", "assistant", "tool", "tool", "assistant"]
    assert traces[0]["tools"][0]["type"] == "function"


def test_blank_lines_are_skipped(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text('{"messages": [{"role": "user", "content": "hi"}], "tools": []}\n\n\n')
    traces, _ = load_traces(p)
    assert len(traces) == 1


def test_malformed_json_names_the_line(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text('{"messages": [{"role": "user", "content": "hi"}], "tools": []}\n{not json\n')
    with pytest.raises(IngestError, match=":2:"):
        load_traces(p)


def test_missing_messages_is_rejected(tmp_path):
    p = _write(tmp_path, [{"task_id": "t"}])
    with pytest.raises(IngestError, match="no `messages`"):
        load_traces(p)


def test_lenient_mode_collects_problems_instead_of_raising(tmp_path):
    p = _write(tmp_path, [{"task_id": "t"},
                          {"messages": [{"role": "user", "content": "hi"}], "tools": []}])
    traces, problems = load_traces(p, strict=False)
    assert len(traces) == 1 and len(problems) == 1


def test_missing_file_is_reported(tmp_path):
    with pytest.raises(IngestError, match="no such file"):
        load_traces(tmp_path / "absent.jsonl")


def test_schema_violation_is_rejected(tmp_path):
    p = _write(tmp_path, [{"messages": [{"role": "wizard", "content": "hi"}], "tools": []}])
    with pytest.raises(IngestError):
        load_traces(p)


# --------------------------------------------------------------------------------------------------------------
# agentreplay
# --------------------------------------------------------------------------------------------------------------


def _agentreplay_db(tmp_path, *, with_response=True):
    db = tmp_path / "agentreplay.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE runs (id TEXT PRIMARY KEY, task_id TEXT, model TEXT, started_at TEXT);
        CREATE TABLE llm_calls (id TEXT PRIMARY KEY, run_id TEXT, model TEXT, request TEXT, response TEXT,
                                prompt_tokens INT, completion_tokens INT, cost_usd REAL, started_at TEXT);
        CREATE TABLE eval_results (run_id TEXT, passed INT, grader TEXT, score REAL);
        """
    )
    request = {
        "messages": [
            {"role": "system", "content": "You are a support agent."},
            {"role": "user", "content": "Where is my order?"},
            {"role": "assistant", "content": "Looking.",
             "tool_calls": [{"id": "c1", "type": "function",
                             "function": {"name": "search_orders", "arguments": '{"customer_id": "c_9"}'}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "[]"},
        ],
        "tools": [{"type": "function", "function": {"name": "search_orders", "parameters": {"type": "object"}}}],
    }
    response = {"choices": [{"message": {"role": "assistant", "content": "It shipped yesterday."}}]}
    conn.execute("INSERT INTO runs VALUES (?,?,?,?)", ("r1", "task-1", "teacher-v1", "2026-09-10T00:00:00+00:00"))
    conn.execute(
        "INSERT INTO llm_calls VALUES (?,?,?,?,?,?,?,?,?)",
        ("l1", "r1", "teacher-v1", json.dumps(request), json.dumps(response) if with_response else None,
         100, 20, 0.01, "2026-09-10T00:00:01+00:00"),
    )
    conn.execute("INSERT INTO eval_results VALUES (?,?,?,?)", ("r1", 1, "label", 1.0))
    conn.commit()
    conn.close()
    return db


def test_agentreplay_reconstructs_the_trajectory(tmp_path):
    traces, problems = load_agentreplay(_agentreplay_db(tmp_path), since="365d")
    assert problems == [], problems
    assert len(traces) == 1
    t = traces[0]
    assert [m["role"] for m in t["messages"]] == ["system", "user", "assistant", "tool", "assistant"]
    assert t["messages"][-1]["content"] == "It shipped yesterday.", "the final response completes the trajectory"
    assert t["success"] is True and t["grader"] == "label"
    assert t["teacher_model"] == "teacher-v1"
    assert t["prompt_tokens"] == 100 and t["cost_usd"] == 0.01
    assert t["metadata"]["agentreplay_run_id"] == "r1"


def test_agentreplay_without_a_final_response(tmp_path):
    traces, _ = load_agentreplay(_agentreplay_db(tmp_path, with_response=False), since="365d")
    assert [m["role"] for m in traces[0]["messages"]] == ["system", "user", "assistant", "tool"]


def test_agentreplay_since_window_excludes_old_runs(tmp_path):
    traces, _ = load_agentreplay(_agentreplay_db(tmp_path), since="1d")
    assert traces == []


def test_non_agentreplay_db_is_reported(tmp_path):
    db = tmp_path / "other.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE unrelated (x TEXT)")
    conn.commit()
    conn.close()
    with pytest.raises(AgentReplayError, match="does not look like an agentreplay store"):
        load_agentreplay(db)


def test_missing_db_is_reported(tmp_path):
    with pytest.raises(AgentReplayError, match="no agentreplay store"):
        load_agentreplay(tmp_path / "absent.db")
