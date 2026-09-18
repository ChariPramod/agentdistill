"""Shared fixtures.

Everything here is offline: the tokenizer is the tiny BPE fixture in `tests/fixtures/tokenizer`, so no test
touches the network or needs a GPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parent / "fixtures"
TOKENIZER_DIR = FIXTURES / "tokenizer"
TEMPLATES = FIXTURES / "templates"


def load_template(name: str) -> str:
    return (TEMPLATES / name).read_text()


@pytest.fixture(scope="session")
def tokenizer():
    """The tool-capable, prefix-stable template. The happy path."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    tok.chat_template = load_template("toolchat.jinja")
    return tok


@pytest.fixture
def tokenizer_factory():
    """Build a tokenizer with any of the fixture templates."""
    from transformers import AutoTokenizer

    def make(template: str):
        tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
        tok.chat_template = load_template(template)
        return tok

    return make


def make_tool(name: str = "search_orders", required: list[str] | None = None) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"The {name} tool.",
            "parameters": {
                "type": "object",
                "properties": {"customer_id": {"type": "string"}, "limit": {"type": "integer"}},
                "required": required if required is not None else ["customer_id"],
            },
        },
    }


def make_call(call_id: str, name: str, args: dict) -> dict:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def make_trace(
    trace_id: str = "t1",
    *,
    success: bool | None = True,
    task: str = "Where is the order for customer c_9?",
    task_id: str | None = "task-1",
    tool_name: str = "search_orders",
    args: dict | None = None,
    closing: str = "Your order shipped yesterday and should arrive on Thursday.",
) -> dict:
    """A well-formed two-turn trace: an assistant turn with a tool call, its result, and a closing turn."""
    from agentdistill.ingest.normalize import normalize_trace

    raw = {
        "id": trace_id,
        "task_id": task_id,
        "success": success,
        "teacher_model": "teacher-v1",
        "messages": [
            {"role": "system", "content": "You are a support agent."},
            {"role": "user", "content": task},
            {
                "role": "assistant",
                "content": "Let me look that up for you right away.",
                "tool_calls": [make_call("c1", tool_name, args or {"customer_id": "c_9", "limit": 5})],
            },
            {"role": "tool", "tool_call_id": "c1", "content": '[{"order_id": "o_1", "status": "shipped"}]'},
            {"role": "assistant", "content": closing},
        ],
        "tools": [make_tool(tool_name)],
    }
    t = normalize_trace(raw, source="jsonl")
    t["id"] = trace_id
    return t


@pytest.fixture
def trace():
    return make_trace()


@pytest.fixture
def anthropic_conversation() -> dict:
    """An Anthropic-dialect conversation exercising text, tool_use, parallel tool_result, and a final answer."""
    return {
        "system": "You are a support agent.",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Where are orders for c_9 and c_10?"}]},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Looking both up."},
                    {"type": "tool_use", "id": "call_1", "name": "search_orders", "input": {"customer_id": "c_9"}},
                    {"type": "tool_use", "id": "call_2", "name": "search_orders", "input": {"customer_id": "c_10"}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "call_1", "content": [{"type": "text", "text": "shipped"}]},
                    {"type": "tool_result", "tool_use_id": "call_2", "content": "pending"},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "One shipped, one is pending."}]},
        ],
        "tools": [
            {
                "name": "search_orders",
                "description": "Find orders.",
                "input_schema": {
                    "type": "object",
                    "properties": {"customer_id": {"type": "string"}},
                    "required": ["customer_id"],
                },
            }
        ],
    }


@pytest.fixture
def project_config(tmp_path):
    """A config rooted in tmp_path, using the fixture tokenizer."""
    from agentdistill.config import ProjectConfig

    cfg = ProjectConfig(
        name="test-project",
        registry=f"sqlite:///{tmp_path}/registry.db",
        artifacts=str(tmp_path / "artifacts"),
        reports=str(tmp_path / "reports"),
    )
    cfg.source_path = tmp_path / "project.yaml"
    return cfg


@pytest.fixture
def registry(project_config):
    from agentdistill.registry import open_registry

    reg = open_registry(project_config.registry, root=project_config.root)
    yield reg
    reg.close()


@pytest.fixture(autouse=True)
def _pin_git_state(request, monkeypatch):
    """Rows record the git state of the tree the suite runs in, and a developer's tree is usually dirty. Pinned
    clean everywhere except the provenance tests, which exercise git itself, so `dirty_tree` appears only where a
    test asks for it."""
    if request.module.__name__.endswith("test_provenance"):
        return
    import agentdistill.provenance as prov

    monkeypatch.setattr(prov, "git_state", lambda cwd=None: {"commit": "abc1234", "dirty": False})
