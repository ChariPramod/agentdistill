"""Generate the support-agent example corpus.

Deterministic (seeded), so the example reproduces byte for byte and the milestone-1 "same config, same hash"
check is meaningful. The corpus is synthetic but deliberately messy: it contains the failure modes curation
exists to remove, in known quantities, so the curation report has something real to show and so the example
doubles as an end-to-end test.

    python examples/support_agent/generate_traces.py --n 500

Writes `traces.jsonl` (training corpus) and `eval_tasks.jsonl` (held-out tasks, used to prove decontamination).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent

SYSTEM = (
    "You are a support agent for an online store. Use the tools provided to look up, modify, and refund orders. "
    "Never promise a refund you have not issued."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_orders",
            "description": "Find orders for a customer.",
            "parameters": {
                "type": "object",
                "properties": {"customer_id": {"type": "string"}, "limit": {"type": "integer"}},
                "required": ["customer_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "track_package",
            "description": "Get carrier tracking for an order.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "refund_order",
            "description": "Issue a refund.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "amount": {"type": "number"},
                    "reason": {"type": "string", "enum": ["damaged", "late", "wrong_item", "unwanted"]},
                },
                "required": ["order_id", "amount", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_address",
            "description": "Change the shipping address on an unshipped order.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "street": {"type": "string"},
                    "city": {"type": "string"},
                },
                "required": ["order_id", "street", "city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_order",
            "description": "Cancel an unshipped order.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    },
]

# Real support tickets vary in wording; a corpus where every task of a kind shares the same sentence would make
# decontamination fire on the shared boilerplate rather than on genuine overlap, and would make the clusters
# trivial. Each task type therefore draws from several phrasings and several framing details.
TRACK_PHRASINGS = [
    "Where is my order? My customer id is {cust}.",
    "I placed an order last week and it still has not turned up. Customer {cust}. Can you tell me what happened?",
    "Hi, could you check the delivery status on my recent purchase? My account is {cust}.",
    "Any update on shipping? I have been waiting eleven days now. Customer number {cust}.",
    "My tracking page has not moved since Tuesday. Account {cust}. Is the parcel actually moving?",
    "Good morning. I need an estimated arrival date for whatever is currently on its way to me. Customer {cust}.",
    "The confirmation email said three days and it has been nine. Customer id {cust}. What is going on?",
    "Can someone tell me which courier has my package and when they plan to deliver it? I am customer {cust}.",
]
REFUND_PHRASINGS = [
    "My order arrived {reason}. I want my money back. Customer {cust}.",
    "The item I received is not acceptable, it came {reason}. Please return my payment. Account {cust}.",
    "I would like a refund. The product was {reason} when I opened the box. Customer number {cust}.",
    "This purchase was a complete waste, it showed up {reason}. Refund me please. Customer {cust}.",
    "Hello, I need to return something that arrived {reason} and get reimbursed. My id is {cust}.",
    "Please process a refund for my last purchase. Reason: it was {reason}. Customer {cust}.",
    "I am very disappointed. The goods were {reason} on arrival and I want the charge reversed. Account {cust}.",
    "Can you reverse the payment on my recent order? It came through {reason}. Customer id {cust}.",
]
ADDRESS_PHRASINGS = [
    "I moved. Please send order to {street}, {city}. Customer {cust}.",
    "Change of plans, I need delivery redirected to {street} in {city}. My account is {cust}.",
    "Could you update where my parcel is going? The new place is {street}, {city}. Customer {cust}.",
    "I gave the wrong address at checkout. It should go to {street}, {city} instead. Customer id {cust}.",
    "We relocated last weekend. New address is {street}, {city}. Please amend my pending order. Account {cust}.",
    "Urgent: redirect my shipment to {street}, {city} before it leaves the warehouse. Customer {cust}.",
    "Hi there, can the delivery destination be switched to {street} in {city}? I am customer {cust}.",
    "My order is still processing so please change the ship-to address to {street}, {city}. Account {cust}.",
]
CANCEL_PHRASINGS = [
    "Cancel my most recent order please. Customer {cust}.",
    "I changed my mind and no longer want what I ordered. Please call it off. Account {cust}.",
    "Could you stop my latest purchase from shipping? I do not need it anymore. Customer {cust}.",
    "Please void the order I placed this morning. Customer number {cust}.",
    "I found the same thing cheaper elsewhere, so please cancel mine. Customer id {cust}.",
    "Kindly halt the order that is still being prepared. I want to back out. Account {cust}.",
    "Scrap my pending order, circumstances have changed. Customer {cust}.",
    "I need to withdraw the purchase I made yesterday before it goes out. Customer {cust}.",
]
MULTI_PHRASINGS = [
    "Two items in my order were wrong. Refund both. Customer {cust}.",
    "I received a pair of incorrect products in the same shipment and need both charges reversed. Account {cust}.",
    "Neither of the two things that arrived matches what I ordered. Please refund each of them. Customer {cust}.",
    "There were two mistakes in one delivery. I want money back for both lines. Customer id {cust}.",
]

CITIES = ["Boston", "Austin", "Denver", "Seattle", "Miami", "Chicago", "Portland", "Atlanta"]
STREETS = ["12 Main Street", "40 Oak Avenue", "7 Birch Lane", "221 Harbor Road", "98 Cedar Court"]
REASONS = ["damaged", "late", "wrong_item", "unwanted"]


def call(cid: str, name: str, args: dict) -> dict:
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def asst(text: str | None, calls: list[dict] | None = None) -> dict:
    m: dict = {"role": "assistant", "content": text}
    if calls:
        m["tool_calls"] = calls
    return m


def result(cid: str, payload) -> dict:
    body = payload if isinstance(payload, str) else json.dumps(payload)
    return {"role": "tool", "tool_call_id": cid, "content": body}


def track_task(rng: random.Random, i: int) -> dict:
    cust, order = f"c_{rng.randint(100, 999)}", f"o_{rng.randint(1000, 9999)}"
    eta = f"2026-09-{rng.randint(10, 28):02d}"
    return {
        "task_id": f"track-{i}",
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": rng.choice(TRACK_PHRASINGS).format(cust=cust)},
            asst("Let me look up your orders.",
                 [call("c1", "search_orders", {"customer_id": cust, "limit": 5})]),
            result("c1", [{"order_id": order, "status": "shipped"}]),
            asst("Checking the carrier now.", [call("c2", "track_package", {"order_id": order})]),
            result("c2", {"carrier": "UPS", "eta": eta, "status": "in_transit"}),
            asst(f"Thanks for waiting. I checked your account and then queried the carrier directly. "
                 f"Your order {order} is in transit with UPS and is currently scheduled to arrive on {eta}. "
                 f"The parcel has cleared the regional sorting facility, so it is moving normally now. "
                 f"You will get an email from the carrier on the morning of delivery with a time window. "
                 f"If it has not arrived by the end of that day, reply here and I will open a trace with UPS."),
        ],
    }


def refund_task(rng: random.Random, i: int) -> dict:
    cust, order = f"c_{rng.randint(100, 999)}", f"o_{rng.randint(1000, 9999)}"
    amount, reason = round(rng.uniform(10, 200), 2), rng.choice(REASONS)
    return {
        "task_id": f"refund-{i}",
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user",
             "content": rng.choice(REFUND_PHRASINGS).format(cust=cust, reason=reason.replace("_", " "))},
            asst("I am sorry about that. Let me find the order.",
                 [call("c1", "search_orders", {"customer_id": cust, "limit": 5})]),
            result("c1", [{"order_id": order, "status": "delivered", "total": amount}]),
            asst("Issuing the refund now.",
                 [call("c2", "refund_order", {"order_id": order, "amount": amount, "reason": reason})]),
            result("c2", {"ok": True, "refund_id": f"r_{rng.randint(10, 99)}", "amount": amount}),
            asst(f"I am sorry again that this happened. I have gone ahead and issued a full refund of "
                 f"${amount:.2f} against order {order}, recorded under the reason {reason}. "
                 f"The money goes back to the original payment method, which normally takes three to five "
                 f"business days to appear depending on your bank. You do not need to return anything. "
                 f"I have also added a note to your account in case you contact us about this again."),
        ],
    }


def address_task(rng: random.Random, i: int) -> dict:
    cust, order = f"c_{rng.randint(100, 999)}", f"o_{rng.randint(1000, 9999)}"
    street, city = rng.choice(STREETS), rng.choice(CITIES)
    return {
        "task_id": f"address-{i}",
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": rng.choice(ADDRESS_PHRASINGS).format(cust=cust, street=street, city=city)},
            asst("Let me pull up that order.",
                 [call("c1", "search_orders", {"customer_id": cust, "limit": 5})]),
            result("c1", [{"order_id": order, "status": "processing"}]),
            asst("It has not shipped, so I can change the address.",
                 [call("c2", "update_address", {"order_id": order, "street": street, "city": city})]),
            result("c2", {"ok": True}),
            asst(f"That is all set now. I confirmed the order had not yet left the warehouse, so I was able "
                 f"to change it. Order {order} will now ship to {street}, {city}. "
                 f"The delivery estimate does not change because the order had not been dispatched. "
                 f"You will see the updated address on the shipping confirmation email. "
                 f"If anything else about the order needs changing, let me know before it ships."),
        ],
    }


def cancel_task(rng: random.Random, i: int) -> dict:
    cust, order = f"c_{rng.randint(100, 999)}", f"o_{rng.randint(1000, 9999)}"
    return {
        "task_id": f"cancel-{i}",
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": rng.choice(CANCEL_PHRASINGS).format(cust=cust)},
            asst("Checking whether it can still be cancelled.",
                 [call("c1", "search_orders", {"customer_id": cust, "limit": 1})]),
            result("c1", [{"order_id": order, "status": "processing"}]),
            asst("It has not shipped yet.", [call("c2", "cancel_order", {"order_id": order})]),
            result("c2", {"ok": True}),
            asst(f"Done. I checked that order {order} was still in processing and cancelled it for you. "
                 f"You will not be charged, and any authorization hold on your card should drop off "
                 f"within a few business days. Nothing will ship. "
                 f"You will receive a cancellation confirmation by email shortly. "
                 f"If you change your mind you are welcome to place a new order at any time."),
        ],
    }


def multi_item_refund_task(rng: random.Random, i: int) -> dict:
    """A deliberately thin cluster: only a handful are generated, so the coverage report flags it."""
    cust = f"c_{rng.randint(100, 999)}"
    o1, o2 = f"o_{rng.randint(1000, 9999)}", f"o_{rng.randint(1000, 9999)}"
    a1, a2 = round(rng.uniform(10, 90), 2), round(rng.uniform(10, 90), 2)
    return {
        "task_id": f"multi-refund-{i}",
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": rng.choice(MULTI_PHRASINGS).format(cust=cust)},
            asst("Let me find both orders.",
                 [call("c1", "search_orders", {"customer_id": cust, "limit": 10})]),
            result("c1", [{"order_id": o1, "total": a1}, {"order_id": o2, "total": a2}]),
            asst("Refunding the first.",
                 [call("c2", "refund_order", {"order_id": o1, "amount": a1, "reason": "wrong_item"})]),
            result("c2", {"ok": True, "amount": a1}),
            asst("And the second.",
                 [call("c3", "refund_order", {"order_id": o2, "amount": a2, "reason": "wrong_item"})]),
            result("c3", {"ok": True, "amount": a2}),
            asst(f"Both refunds are now issued. I processed ${a1:.2f} for the first item and ${a2:.2f} "
                 f"for the second, each recorded as a wrong-item return. "
                 f"They were submitted separately so they may appear as two lines on your statement. "
                 f"Expect them within three to five business days on the original payment method. "
                 f"Please keep the incorrect items; you do not need to send them back to us."),
        ],
    }


GENERATORS = [
    (track_task, 0.30),
    (refund_task, 0.28),
    (address_task, 0.20),
    (cancel_task, 0.17),
    (multi_item_refund_task, 0.05),
]


def corrupt(trace: dict, rng: random.Random, mode: str) -> dict:
    """Inject a known failure mode so curation has something to catch."""
    t = json.loads(json.dumps(trace))
    if mode == "error_loop":
        cid = "e"
        errs = []
        for k in range(3):
            errs.append(asst("Trying again.", [call(f"{cid}{k}", "track_package", {"order_id": f"o_{k}"})]))
            errs.append(result(f"{cid}{k}", {"error": "carrier timeout"}))
        t["messages"] = [*t["messages"][:2], *errs, asst("The carrier finally responded; your parcel is moving.")]
        # Marked successful on purpose: an agent that flails through three failed calls and then recovers still
        # counts as a success to the grader, so `outcome` lets it through and `no_error_loops` is what must catch
        # it. Training on it teaches the student to keep hammering a tool that is not working.
        t["success"] = True
    elif mode == "bad_args":
        for m in t["messages"]:
            for c in m.get("tool_calls") or []:
                if c["function"]["name"] == "refund_order":
                    c["function"]["arguments"] = '{"order_id": "o_1", "amount": "a lot", "reason": "damaged"}'
                    return t
        t["messages"][2]["tool_calls"][0]["function"]["arguments"] = "{not valid json"
    elif mode == "unknown_tool":
        t["messages"][2]["tool_calls"][0]["function"]["name"] = "lookup_customer_v2"
    elif mode == "pii_arg":
        t["messages"][2]["tool_calls"][0]["function"]["arguments"] = json.dumps(
            {"customer_id": f"bob{rng.randint(1, 99)}@example.com", "limit": 5}
        )
    elif mode == "truncated":
        t["messages"] = t["messages"][:3]  # a tool call with no result
    elif mode == "too_short":
        t["messages"] = [*t["messages"][:2], asst("Sure, I can help with that.")]
    return t


def build(n: int, seed: int = 17) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    traces: list[dict] = []
    gens, weights = zip(*GENERATORS, strict=True)

    for i in range(n):
        gen = rng.choices(gens, weights=weights)[0]
        t = gen(rng, i)
        t["success"] = True
        t["grader"] = "label"
        t["teacher_model"] = "teacher-frontier-v1"
        traces.append(t)

    # Known contamination: 8 tasks that also appear in the eval set. Decontamination must remove exactly these.
    eval_tasks = [json.loads(json.dumps(t)) for t in rng.sample(traces, 8)]
    for e in eval_tasks:
        e["task_id"] = f"eval-{e['task_id']}"

    # Exact duplicates.
    for t in rng.sample(traces, max(1, n // 25)):
        traces.append(json.loads(json.dumps(t)))

    # Near-duplicates: same trajectory, one number changed.
    for t in rng.sample(traces[:n], max(1, n // 20)):
        d = json.loads(json.dumps(t))
        d["task_id"] = f"{d['task_id']}-near"
        for m in d["messages"]:
            if m["role"] == "assistant" and m.get("tool_calls"):
                args = json.loads(m["tool_calls"][0]["function"]["arguments"])
                if "limit" in args:
                    args["limit"] = args["limit"] + 1
                m["tool_calls"][0]["function"]["arguments"] = json.dumps(args)
                break
        traces.append(d)

    # Failure modes, in known quantities.
    for mode, count in [
        ("error_loop", max(1, n // 20)),
        ("bad_args", max(1, n // 25)),
        ("unknown_tool", max(1, n // 50)),
        ("pii_arg", max(1, n // 50)),
        ("truncated", max(1, n // 40)),
        ("too_short", max(1, n // 40)),
    ]:
        for t in rng.sample(traces[:n], count):
            c = corrupt(t, rng, mode)
            c["task_id"] = f"{c['task_id']}-{mode}"
            traces.append(c)

    # Ungraded traces: no outcome recorded at all.
    for t in rng.sample(traces[:n], max(1, n // 25)):
        u = json.loads(json.dumps(t))
        u["task_id"] = f"{u['task_id']}-ungraded"
        u["success"] = None
        traces.append(u)

    # Genuine failures, which SFT drops but DPO keeps as rejected sides.
    for t in rng.sample(traces[:n], max(1, n // 20)):
        f = json.loads(json.dumps(t))
        f["task_id"] = t["task_id"]  # same task as the success, so pairs can be built
        f["success"] = False
        for m in f["messages"]:
            if m["role"] == "assistant" and m.get("tool_calls"):
                m["tool_calls"][0]["function"]["name"] = "cancel_order"
                m["tool_calls"][0]["function"]["arguments"] = json.dumps({"order_id": "o_0"})
                break
        traces.append(f)

    for t in traces:
        t["tools"] = TOOLS
    for e in eval_tasks:
        e["tools"] = TOOLS
    rng.shuffle(traces)
    return traces, eval_tasks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500, help="Number of clean traces before corruption.")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", type=Path, default=HERE)
    args = ap.parse_args()

    traces, eval_tasks = build(args.n, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    tp = args.out / "traces.jsonl"
    ep = args.out / "eval_tasks.jsonl"
    tp.write_text("\n".join(json.dumps(t) for t in traces) + "\n")
    ep.write_text("\n".join(json.dumps(t) for t in eval_tasks) + "\n")
    print(f"wrote {len(traces)} traces -> {tp}")
    print(f"wrote {len(eval_tasks)} eval tasks -> {ep}")


if __name__ == "__main__":
    main()
