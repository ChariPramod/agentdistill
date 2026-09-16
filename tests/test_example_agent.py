"""The example agent: CRM semantics, predicates, scenarios, and recording.

The predicates are the example's ground truth -- every success label in the corpus comes from one -- so they are
tested against hand-built final states rather than against whatever the agent happened to do.
"""

from __future__ import annotations

import json

import pytest

from examples.support_agent import graders, scenarios
from examples.support_agent.agent import final_assistant_text, run_agent
from examples.support_agent.crm import CRM, TOOL_NAMES, TOOLS, Customer, ToolError
from examples.support_agent.record import record_one, summarize
from examples.support_agent.scripted_teacher import ScriptedTeacher


@pytest.fixture
def crm():
    c = CRM.empty(1)
    c.add_customer(Customer(id="c_1", email="ada@example.com", name="Ada N", address="1 High St, Boston",
                            tier="standard"))
    c.add_order("o_1", "c_1", "delivered", 100.0, "desk lamp", "2026-08-01", carrier="UPS")
    c.add_order("o_2", "c_1", "processing", 50.0, "wool blanket", "2026-09-01")
    return c


# --------------------------------------------------------------------------------------------------------------
# CRM: the rules that make trajectories interesting
# --------------------------------------------------------------------------------------------------------------


def test_seeded_database_is_deterministic():
    assert CRM.from_seed(11).state_hash() == CRM.from_seed(11).state_hash()
    assert CRM.from_seed(11).state_hash() != CRM.from_seed(12).state_hash()


def test_lookup_is_case_and_whitespace_insensitive(crm):
    assert crm.call("get_customer", {"email": "  ADA@example.com "})["customer_id"] == "c_1"


def test_unknown_customer_is_a_tool_error(crm):
    with pytest.raises(ToolError, match="no customer"):
        crm.call("get_customer", {"email": "nobody@example.com"})


def test_unknown_tool_names_the_available_ones(crm):
    with pytest.raises(ToolError, match="available tools"):
        crm.call("teleport", {})


def test_refund_requires_a_shipped_or_delivered_order(crm):
    with pytest.raises(ToolError, match="only shipped or delivered"):
        crm.call("issue_refund", {"order_id": "o_2", "amount": 50.0, "reason": "damaged"})
    assert crm.refunds() == []


def test_refund_happens_once(crm):
    crm.call("issue_refund", {"order_id": "o_1", "amount": 100.0, "reason": "damaged"})
    with pytest.raises(ToolError, match="already been refunded"):
        crm.call("issue_refund", {"order_id": "o_1", "amount": 10.0, "reason": "damaged"})
    assert len(crm.refunds()) == 1


def test_refund_cannot_exceed_the_total(crm):
    with pytest.raises(ToolError, match="exceeds the order total"):
        crm.call("issue_refund", {"order_id": "o_1", "amount": 500.0, "reason": "damaged"})


def test_refund_reason_is_constrained(crm):
    with pytest.raises(ToolError, match="reason must be one of"):
        crm.call("issue_refund", {"order_id": "o_1", "amount": 10.0, "reason": "because"})


def test_address_change_blocked_once_anything_shipped():
    c = CRM.empty(2)
    c.add_customer(Customer("c_1", "a@b.com", "A", "old address", "standard"))
    c.add_order("o_1", "c_1", "shipped", 10.0, "mug", "2026-09-01", carrier="UPS")
    with pytest.raises(ToolError, match="already shipped"):
        c.call("update_address", {"customer_id": "c_1", "address": "new address"})
    assert c.customer("c_1")["address"] == "old address", "a refused change must not partially apply"


def test_address_change_succeeds_when_nothing_shipped(crm):
    crm.call("update_address", {"customer_id": "c_1", "address": "9 New Road, Denver"})
    assert crm.customer("c_1")["address"] == "9 New Road, Denver"


def test_every_call_is_recorded(crm):
    crm.call("get_customer", {"email": "ada@example.com"})
    with pytest.raises(ToolError):
        crm.call("get_customer", {"email": "no@one.com"})
    assert [name for name, _ in crm.calls] == ["get_customer", "get_customer"], "refusals are recorded too"


def test_tool_schemas_are_valid_json_schema():
    from jsonschema import Draft202012Validator

    for tool in TOOLS:
        Draft202012Validator.check_schema(tool["function"]["parameters"])
        assert tool["function"]["description"], f"{tool['function']['name']} has no description"


def test_tool_names_match_handlers():
    c = CRM.empty(0)
    for name in TOOL_NAMES:
        assert hasattr(c, f"_tool_{name}"), f"no handler for {name}"


# --------------------------------------------------------------------------------------------------------------
# predicates, against hand-built final states
# --------------------------------------------------------------------------------------------------------------


def test_refund_exactly_requires_the_amount_in_the_reply(crm):
    crm.call("issue_refund", {"order_id": "o_1", "amount": 100.0, "reason": "damaged"})
    predicate = graders.refund_exactly("o_1", 100.0)
    assert predicate(crm, "I refunded $100.00 to your card.")[0]
    ok, detail = predicate(crm, "All sorted, thanks for your patience.")
    assert not ok and "never states the amount" in detail


def test_refund_exactly_rejects_the_wrong_order(crm):
    crm.add_order("o_3", "c_1", "delivered", 100.0, "mug", "2026-08-02", carrier="UPS")
    crm.call("issue_refund", {"order_id": "o_3", "amount": 100.0, "reason": "damaged"})
    ok, detail = graders.refund_exactly("o_1", 100.0)(crm, "I refunded 100.00")
    assert not ok and "expected o_1" in detail


def test_refund_exactly_rejects_the_wrong_amount(crm):
    crm.call("issue_refund", {"order_id": "o_1", "amount": 60.0, "reason": "damaged"})
    ok, detail = graders.refund_exactly("o_1", 100.0)(crm, "I refunded 60.00")
    assert not ok and "expected 100" in detail


def test_refund_exactly_rejects_extra_refunds(crm):
    crm.add_order("o_3", "c_1", "delivered", 20.0, "mug", "2026-08-02", carrier="UPS")
    crm.call("issue_refund", {"order_id": "o_1", "amount": 100.0, "reason": "damaged"})
    crm.call("issue_refund", {"order_id": "o_3", "amount": 20.0, "reason": "damaged"})
    ok, detail = graders.refund_exactly("o_1", 100.0)(crm, "refunded 100.00")
    assert not ok and "expected exactly one" in detail


@pytest.mark.parametrize(
    ("reply", "ok"),
    [
        ("I refunded $100.00 today.", True),
        ("I refunded 100.00 today.", True),
        ("I refunded 100 today.", True),            # the same number, stated without cents
        ("I refunded $1,00.00", False),             # malformed; not a way of writing 100.00
        ("I refunded $10.00 today.", False),        # a different number is the failure being checked for
        ("Refunded against order 100.", True),      # a bare 100 anywhere still states the amount
        ("All sorted for you.", False),
    ],
)
def test_amount_must_be_stated_precisely(crm, reply, ok):
    crm.call("issue_refund", {"order_id": "o_1", "amount": 100.0, "reason": "damaged"})
    assert graders.refund_exactly("o_1", 100.0)(crm, reply)[0] is ok


@pytest.mark.parametrize(
    ("reply", "amount", "ok"),
    [
        ("I refunded $1,234.50.", 1234.50, True),
        ("I refunded 42.5", 42.50, True),
        ("I refunded 42", 42.50, False),            # 42 is not 42.50
        ("order o_4250 is done", 42.50, False),     # a digit run inside an id is not an amount
    ],
)
def test_amount_matching_handles_formats(crm, reply, amount, ok):
    crm.add_order("o_9", "c_1", "delivered", amount, "mug", "2026-08-01", carrier="UPS")
    crm.call("issue_refund", {"order_id": "o_9", "amount": amount, "reason": "damaged"})
    assert graders.refund_exactly("o_9", amount)(crm, reply)[0] is ok


def test_no_refund_but_explained(crm):
    predicate = graders.no_refund_but_explained(must_mention=("processing",))
    assert predicate(crm, "That order is still processing, so I cannot refund it yet.")[0]
    assert not predicate(crm, "Sorry about that.")[0], "must explain why"
    assert not predicate(crm, "")[0]
    crm.call("issue_refund", {"order_id": "o_1", "amount": 100.0, "reason": "damaged"})
    ok, detail = predicate(crm, "That order is still processing.")
    assert not ok and "not eligible" in detail


def test_address_updated_rejects_a_ticket_instead_of_the_work(crm):
    crm.call("update_address", {"customer_id": "c_1", "address": "9 New Road"})
    predicate = graders.address_updated("c_1", "9 New Road")
    assert predicate(crm, "Updated.")[0]
    crm.call("create_ticket", {"customer_id": "c_1", "category": "shipping", "summary": "x"})
    ok, detail = predicate(crm, "Updated.")
    assert not ok and "ticket was opened instead" in detail


def test_address_unchanged_and_ticket():
    c = CRM.empty(3)
    c.add_customer(Customer("c_1", "a@b.com", "A", "old", "standard"))
    c.add_order("o_1", "c_1", "shipped", 10.0, "mug", "2026-09-01", carrier="UPS")
    predicate = graders.address_unchanged_and_ticket("c_1", "old")
    assert not predicate(c, "Escalated.")[0], "no ticket yet"
    c.call("create_ticket", {"customer_id": "c_1", "category": "shipping", "summary": "redirect"})
    assert predicate(c, "Escalated.")[0]


def test_tracking_reported_requires_both_number_and_carrier(crm):
    order = crm.order("o_1")
    predicate = graders.tracking_reported("o_1")
    assert predicate(crm, f"It's with {order['carrier']}, tracking {order['tracking']}.")[0]
    assert not predicate(crm, f"Tracking is {order['tracking']}.")[0], "carrier missing"
    assert not predicate(crm, f"It's with {order['carrier']}.")[0], "number missing"


def test_read_only_predicates_reject_state_changes(crm):
    order = crm.order("o_1")
    reply = f"It's with {order['carrier']}, tracking {order['tracking']}."
    assert graders.tracking_reported("o_1")(crm, reply)[0]
    crm.call("create_ticket", {"customer_id": "c_1", "category": "other", "summary": "x"})
    ok, detail = graders.tracking_reported("o_1")(crm, reply)
    assert not ok and "read-only" in detail


def test_refund_one_of_two_rejects_refunding_both(crm):
    crm.add_order("o_3", "c_1", "delivered", 20.0, "mug", "2026-08-02", carrier="UPS")
    predicate = graders.refund_one_of_two("o_1", "o_3", 100.0)
    crm.call("issue_refund", {"order_id": "o_1", "amount": 100.0, "reason": "damaged"})
    assert predicate(crm, "done")[0]
    crm.call("issue_refund", {"order_id": "o_3", "amount": 20.0, "reason": "damaged"})
    ok, detail = predicate(crm, "done")
    assert not ok and "not eligible" in detail


def test_ticket_opened_checks_the_category(crm):
    crm.call("create_ticket", {"customer_id": "c_1", "category": "billing", "summary": "dupe"})
    assert graders.ticket_opened("c_1", "billing")(crm, "raised")[0]
    assert not graders.ticket_opened("c_1", "shipping")(crm, "raised")[0]


def test_nothing_changed_requires_a_reply(crm):
    assert graders.nothing_changed()(crm, "I could not find that account.")[0]
    assert not graders.nothing_changed()(crm, "")[0]


def test_all_of_reports_the_first_failure(crm):
    predicate = graders.all_of(graders.nothing_changed(), graders.ticket_opened("c_1", "billing"))
    ok, detail = predicate(crm, "hello")
    assert not ok and "no ticket" in detail


# --------------------------------------------------------------------------------------------------------------
# scenarios
# --------------------------------------------------------------------------------------------------------------


def test_every_scenario_builds_and_is_deterministic():
    for name in scenarios.SCENARIOS:
        a, b = scenarios.build_task(name, 5), scenarios.build_task(name, 5)
        assert a.user_message == b.user_message
        assert a.fresh_crm().state_hash() == b.fresh_crm().state_hash()
        assert a.task_id == b.task_id


def test_scenarios_vary_with_the_seed():
    messages = {scenarios.build_task("refund_delivered", s).user_message for s in range(8)}
    assert len(messages) > 1, "instances must differ or the corpus is one task repeated"


def test_held_out_scenarios_are_real_and_excluded():
    for name in scenarios.HELD_OUT_SCENARIOS:
        assert name in scenarios.SCENARIOS
        assert name not in scenarios.training_scenarios()


def test_sample_spreads_across_scenarios():
    tasks = scenarios.sample(2 * len(scenarios.SCENARIOS), seed=1)
    counts: dict[str, int] = {}
    for t in tasks:
        counts[t.scenario] = counts.get(t.scenario, 0) + 1
    assert len(counts) == len(scenarios.SCENARIOS)
    assert max(counts.values()) - min(counts.values()) <= 1


def test_sample_can_restrict_scenarios():
    tasks = scenarios.sample(6, seed=1, scenarios=["refund_delivered"])
    assert {t.scenario for t in tasks} == {"refund_delivered"}
    assert len({t.user_message for t in tasks}) > 1


def test_unknown_scenario_is_rejected():
    with pytest.raises(KeyError, match="unknown scenario"):
        scenarios.build_task("no_such_scenario", 0)


# --------------------------------------------------------------------------------------------------------------
# recording
# --------------------------------------------------------------------------------------------------------------


def test_recorded_trace_has_everything_ingest_needs():
    from agentdistill.ingest.normalize import normalize_trace, validate_trace

    task = scenarios.build_task("refund_delivered", 21)
    trace = record_one(task, "scripted", completion=ScriptedTeacher(error_rate=0.0))
    assert validate_trace(normalize_trace(trace, source="jsonl")) == []
    assert trace["grader"] == "predicate"
    assert isinstance(trace["success"], bool)
    assert trace["metadata"]["scenario"] == "refund_delivered"
    assert trace["metadata"]["system_prompt_version"]
    assert trace["task_input"]["text"] == task.user_message


def test_recording_is_reproducible():
    task = scenarios.build_task("refund_delivered", 22)
    a = record_one(task, "scripted", completion=ScriptedTeacher(error_rate=0.0))
    b = record_one(task, "scripted", completion=ScriptedTeacher(error_rate=0.0))
    assert a["id"] == b["id"]
    assert a["metadata"]["final_state_hash"] == b["metadata"]["final_state_hash"]


def test_tool_errors_stay_in_the_trajectory():
    """Recovering from a refusal is the behaviour worth learning; it must not be swallowed."""
    task = scenarios.build_task("address_after_ship", 23)
    trace = record_one(task, "scripted", completion=ScriptedTeacher(error_rate=0.0))
    tool_contents = [m["content"] for m in trace["messages"] if m["role"] == "tool"]
    assert any("error" in c for c in tool_contents), "the blocked address change must appear as a tool error"


def test_agent_loop_stops_when_the_model_answers():
    task = scenarios.build_task("order_status", 24)
    crm = task.fresh_crm()
    run = run_agent(task, crm, TOOLS, "scripted", completion=ScriptedTeacher(error_rate=0.0))
    assert run["stop_reason"] == "answered"
    assert final_assistant_text(run["messages"])


def test_agent_loop_respects_max_turns():
    class NeverStops:
        def __call__(self, model, messages, tools, temperature=0.0, **_):
            from examples.support_agent.scripted_teacher import _call, _respond

            return _respond(None, [_call(0, "list_orders", {"customer_id": "c_x"})], 1)

    task = scenarios.build_task("refund_delivered", 25)
    run = run_agent(task, task.fresh_crm(), TOOLS, "loop", max_turns=3, completion=NeverStops())
    assert run["stop_reason"] == "max_turns"
    assert sum(1 for m in run["messages"] if m["role"] == "assistant") == 3


def test_malformed_tool_arguments_become_a_tool_error():
    class BadArgs:
        def __call__(self, model, messages, tools, temperature=0.0, **_):
            from examples.support_agent.scripted_teacher import (
                _Choice,
                _Function,
                _Message,
                _Response,
                _ToolCall,
                _Usage,
            )

            if any(m["role"] == "tool" for m in messages):
                return _Response([_Choice(_Message("Sorry, something went wrong.", None))], _Usage(1, 1))
            call = _ToolCall("c0", "function", _Function("get_customer", "{not json"))
            return _Response([_Choice(_Message(None, [call]))], _Usage(1, 1))

    task = scenarios.build_task("refund_delivered", 26)
    run = run_agent(task, task.fresh_crm(), TOOLS, "bad", completion=BadArgs())
    tool_msg = next(m for m in run["messages"] if m["role"] == "tool")
    assert json.loads(tool_msg["content"])["error"] == "arguments were not valid JSON"


def test_summarize_reports_per_scenario():
    tasks = scenarios.sample(6, seed=31)
    traces = [record_one(t, "scripted", completion=ScriptedTeacher(error_rate=0.0)) for t in tasks]
    s = summarize(traces)
    assert s["n"] == 6
    assert 0.0 <= s["success_rate"] <= 1.0
    assert set(s["by_scenario"]) <= set(scenarios.SCENARIOS)


def test_summarize_empty():
    assert summarize([])["n"] == 0


def test_scripted_solver_produces_both_outcomes():
    """Without failures there are no DPO pairs and the outcome filter has nothing to drop."""
    tasks = scenarios.sample(40, seed=33)
    teacher = ScriptedTeacher(error_rate=0.3, seed=1)
    outcomes = {record_one(t, "scripted", completion=teacher)["success"] for t in tasks}
    assert outcomes == {True, False}
