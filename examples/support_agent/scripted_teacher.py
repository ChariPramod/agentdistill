"""A deterministic, rule-based solver that drives the real agent loop.

**This is not a teacher and its traces are not training data.** The next-phase plan is explicit about why
generated traces are worthless for distillation: an 8B student will learn the generator, every number computed on
them will look wonderful, and the first person to run the pipeline on real traffic gets a student that falls over.
A rule-based solver is a generator.

What it is for: exercising the whole chain -- CRM state, tool refusals, predicates, ingest, curation, the replay
harness -- without an API key or a network call. It reacts to real tool results from a real stateful database, so
the trajectories have the right *shape*; they simply have no model in them.

For a corpus worth training on, point `record.py` at a real model with `--model`.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from typing import Any

EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.\w{2,}")
ORDER_ID = re.compile(r"\bo_[\w]+\b")

REFUND_WORDS = ("refund", "money back", "reimburse", "return my payment", "charge reversed", "refunded",
                "want that back", "return what i can", "send it back", "damaged", "wrong thing")
TRACK_WORDS = ("where is", "tracking", "hasn't arrived", "has not arrived", "when will", "get here", "update on")
ADDRESS_WORDS = ("address", "moved", "redirect", "send my order to", "ship to", "relocated")
BILLING_WORDS = ("charged twice", "duplicate", "billed", "charge twice")
STATUS_WORDS = ("what's happening", "gone out", "status")


@dataclass
class _Function:
    name: str
    arguments: str


@dataclass
class _ToolCall:
    id: str
    type: str
    function: _Function


@dataclass
class _Message:
    content: str | None
    tool_calls: list[_ToolCall] | None


@dataclass
class _Choice:
    message: _Message


@dataclass
class _Usage:
    prompt_tokens: int
    completion_tokens: int


@dataclass
class _Response:
    choices: list[_Choice]
    usage: _Usage


def _call(idx: int, name: str, args: dict) -> _ToolCall:
    return _ToolCall(id=f"call_{idx}", type="function", function=_Function(name=name, arguments=json.dumps(args)))


def _respond(text: str | None, calls: list[_ToolCall] | None, n_in: int) -> _Response:
    completion = len((text or "").split()) + sum(len(c.function.arguments) // 4 for c in calls or [])
    return _Response(choices=[_Choice(_Message(text, calls))], usage=_Usage(n_in, max(completion, 1)))


class ScriptedTeacher:
    """A `completion`-compatible callable driving the agent loop by rules.

    `error_rate` makes it occasionally take a wrong action, so a recorded corpus contains both successes and
    failures. Without failures there are no DPO pairs, the `outcome` filter has nothing to drop, and the harness
    cannot be tested against a mixed corpus.
    """

    def __init__(self, error_rate: float = 0.0, seed: int = 0) -> None:
        self.error_rate = error_rate
        self.rng = random.Random(seed)

    # ----------------------------------------------------------------------------------------------------------

    def __call__(self, model: str, messages: list[dict], tools: list[dict], temperature: float = 0.0, **_: Any):
        user = next((m["content"] for m in messages if m["role"] == "user"), "")
        results = self._results(messages)
        n_in = sum(len(json.dumps(m)) // 4 for m in messages)
        intent = self._intent(user)
        step = len(results)

        # Step 0: identify the customer.
        if "get_customer" not in results:
            email = EMAIL.search(user)
            if not email:
                return _respond("I could not find an email address in your message. Could you share it?", None, n_in)
            return _respond(None, [_call(step, "get_customer", {"email": email.group(0)})], n_in)

        customer = results["get_customer"]
        if "error" in customer:
            return _respond(
                "I could not find an account with that email address, so I have not made any changes. "
                "Could you confirm the address you used when ordering?",
                None,
                n_in,
            )
        customer_id = customer["customer_id"]

        # Step 1: see their orders.
        if "list_orders" not in results:
            return _respond(None, [_call(step, "list_orders", {"customer_id": customer_id})], n_in)
        orders = results["list_orders"].get("orders", [])

        handler = {
            "refund": self._refund,
            "track": self._track,
            "address": self._address,
            "billing": self._billing,
            "status": self._status,
        }[intent]
        return handler(user, customer_id, orders, results, step, n_in)

    # ----------------------------------------------------------------------------------------------------------

    def _intent(self, user: str) -> str:
        low = user.lower()
        if any(w in low for w in BILLING_WORDS):
            return "billing"
        if any(w in low for w in REFUND_WORDS):
            return "refund"
        if any(w in low for w in ADDRESS_WORDS):
            return "address"
        if any(w in low for w in TRACK_WORDS):
            return "track"
        if any(w in low for w in STATUS_WORDS):
            return "status"
        return "track"

    def _results(self, messages: list[dict]) -> dict[str, Any]:
        """Latest result per tool name, plus every result keyed by call id."""
        by_id: dict[str, str] = {}
        for m in messages:
            for c in m.get("tool_calls") or []:
                by_id[c["id"]] = c["function"]["name"]
        out: dict[str, Any] = {}
        for m in messages:
            if m["role"] != "tool":
                continue
            name = by_id.get(m["tool_call_id"], "?")
            try:
                out[name] = json.loads(m["content"])
            except (json.JSONDecodeError, TypeError):
                out[name] = {"raw": m["content"]}
        return out

    def _eligible(self, orders: list[dict]) -> list[dict]:
        return [o for o in orders if o["status"] in ("delivered", "shipped")]

    def _pick_by_item(self, user: str, orders: list[dict]) -> dict | None:
        """The order whose item the customer named.

        When two items are mentioned ("I got an X that is wrong, the Y coming is fine"), the one named first is
        the subject of the request. A real model reads the sentence; this fixture uses position.
        """
        low = user.lower()
        best, best_pos = None, len(low) + 1
        for o in orders:
            pos = low.find(o["item"].lower())
            if pos != -1 and pos < best_pos:
                best, best_pos = o, pos
        return best

    def _maybe_err(self) -> bool:
        return self.error_rate > 0 and self.rng.random() < self.error_rate

    # ----------------------------------------------------------------------------------------------------------

    def _refund(self, user, customer_id, orders, results, step, n_in):
        eligible = self._eligible(orders)
        if "issue_refund" in results:
            r = results["issue_refund"]
            if "error" in r:
                return _respond(
                    f"I was not able to issue that refund: {r['error']}. I have not changed anything on the "
                    f"account.",
                    None,
                    n_in,
                )
            return _respond(
                f"I have refunded ${r['amount']:.2f} on order {r['order_id']}. "
                f"It goes back to your original payment method within three to five business days.",
                None,
                n_in,
            )
        if not eligible:
            statuses = ", ".join(sorted({o["status"] for o in orders})) or "none"
            return _respond(
                f"I can only refund orders that have shipped or been delivered. Your order is currently "
                f"{statuses}, so there is nothing I can refund yet. Let me know once it arrives.",
                None,
                n_in,
            )
        target = self._pick_by_item(user, eligible) or eligible[0]
        if self._maybe_err() and len(orders) > 1:
            target = next((o for o in orders if o["order_id"] != target["order_id"]), target)
        amount = target["total"]
        if self._maybe_err():
            amount = round(amount / 2, 2)
        return _respond(
            None,
            [_call(step, "issue_refund", {"order_id": target["order_id"], "amount": amount, "reason": "damaged"})],
            n_in,
        )

    def _track(self, user, customer_id, orders, results, step, n_in):
        if "get_order" in results:
            o = results["get_order"]
            if "error" in o:
                return _respond("I could not look up that order. Could you confirm the order number?", None, n_in)
            if o.get("tracking"):
                return _respond(
                    f"Your {o['item']} is on its way with {o['carrier']}, tracking number {o['tracking']}. "
                    f"The order is currently {o['status']}.",
                    None,
                    n_in,
                )
            return _respond(
                f"Your {o['item']} order is still {o['status']}, so there is no tracking number yet. "
                f"You will get one by email as soon as it ships.",
                None,
                n_in,
            )
        if not orders:
            return _respond("I do not see any orders on this account.", None, n_in)
        shipped = [o for o in orders if o["status"] in ("shipped", "delivered")]
        target = self._pick_by_item(user, orders) or (shipped[0] if shipped else orders[0])
        return _respond(None, [_call(step, "get_order", {"order_id": target["order_id"]})], n_in)

    def _address(self, user, customer_id, orders, results, step, n_in):
        new_address = self._extract_address(user)
        if "update_address" in results:
            r = results["update_address"]
            if "error" not in r:
                return _respond(
                    f"Done — your shipping address is now {r['address']}. Anything still in processing will go "
                    f"there.",
                    None,
                    n_in,
                )
            if "create_ticket" in results:
                return _respond(
                    "One of your orders has already shipped, so I cannot redirect it. I have passed this to our "
                    "shipping team, who will contact the carrier and get back to you.",
                    None,
                    n_in,
                )
            return _respond(
                None,
                [
                    _call(
                        step,
                        "create_ticket",
                        {
                            "customer_id": customer_id,
                            "category": "shipping",
                            "summary": f"Customer asked to redirect a shipped order to {new_address}.",
                        },
                    )
                ],
                n_in,
            )
        return _respond(None, [_call(step, "update_address", {"customer_id": customer_id,
                                                              "address": new_address})], n_in)

    def _billing(self, user, customer_id, orders, results, step, n_in):
        if "create_ticket" in results:
            return _respond(
                "I can see only one order on your account, so I have raised this with our billing team. "
                "They will check for a duplicate charge and contact you.",
                None,
                n_in,
            )
        return _respond(
            None,
            [
                _call(
                    step,
                    "create_ticket",
                    {"customer_id": customer_id, "category": "billing",
                     "summary": "Customer reports a possible duplicate charge."},
                )
            ],
            n_in,
        )

    def _status(self, user, customer_id, orders, results, step, n_in):
        if not orders:
            return _respond("I do not see any orders on this account.", None, n_in)
        target = self._pick_by_item(user, orders) or orders[0]
        return _respond(
            f"Your {target['item']} order is currently {target['status']}. It was placed on "
            f"{target['placed_on']} for ${target['total']:.2f}.",
            None,
            n_in,
        )

    @staticmethod
    def _extract_address(user: str) -> str:
        """Pull the address out of the request. Crude on purpose: this is a fixture, not a product."""
        for marker in (" to ", " should be ", " is "):
            if marker in user:
                tail = user.split(marker, 1)[1]
                tail = tail.split(". ")[0].split(" instead")[0].split(" My email")[0].split(" Account")[0]
                cleaned = EMAIL.sub("", tail).strip(" .,")
                if cleaned:
                    return cleaned
        return "unknown address"
