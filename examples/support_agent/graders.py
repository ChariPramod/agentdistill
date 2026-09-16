"""End-state predicates.

A predicate is a pure function of the final database plus the final assistant message. That is what makes the
example's labels trustworthy: no judge, no rubric, no model in the loop. A task succeeded or it did not.

Two rules learned the hard way, both from the plan's "things that will go wrong" table:

- Predicates check **state and the final message** where the task demands something be said to the customer.
  A refund that happened but was never communicated is not a successful support interaction.
- Predicates check that the agent did **not** do extra damage: refunding the wrong order as well as the right
  one is a failure even though the right one was refunded.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

#: (crm, final_assistant_text) -> (success, detail)
Predicate = Callable[[Any, str], tuple[bool, str]]


#: A properly formatted money amount: optional $, optional thousands groups, optional cents.
_AMOUNT = re.compile(r"\$?\d{1,3}(?:,\d{3})+(?:\.\d+)?|\$?\d+(?:\.\d+)?")


def _amounts_in(text: str) -> list[float]:
    """Every number the reply states, parsed.

    Extraction and numeric comparison rather than substring matching: `"42.50" in text` also matches inside an
    order id, and stripping commas first would accept a malformed `$1,00.00` as 100.00.
    """
    out = []
    for match in _AMOUNT.finditer(text):
        token = match.group(0).lstrip("$").replace(",", "")
        try:
            out.append(float(token))
        except ValueError:
            continue
    return out


def _money_mentioned(text: str, amount: float) -> bool:
    """Is this amount stated in the reply?

    `100`, `100.00`, `$100.00` and `1,234.50` all count -- they are the same number. A *different* number does
    not, because telling a customer the wrong figure is the failure being checked for.
    """
    target = round(float(amount), 2)
    return any(round(v, 2) == target for v in _amounts_in(text))


def _mentions(text: str, needle: str) -> bool:
    return needle.lower() in text.lower()


def refund_exactly(order_id: str, amount: float, *, require_amount_in_reply: bool = True) -> Predicate:
    """Exactly one refund, on the right order, for the right amount, and stated to the customer."""

    def predicate(crm: Any, final_text: str) -> tuple[bool, str]:
        refunds = crm.refunds()
        if not refunds:
            return False, "no refund was issued"
        if len(refunds) > 1:
            return False, f"{len(refunds)} refunds issued; expected exactly one"
        r = refunds[0]
        if r["order_id"] != order_id:
            return False, f"refunded {r['order_id']}, expected {order_id}"
        if round(r["amount"], 2) != round(amount, 2):
            return False, f"refunded {r['amount']}, expected {amount}"
        if require_amount_in_reply and not _money_mentioned(final_text, amount):
            return False, f"refund is correct but the reply never states the amount {amount:.2f}"
        return True, "refund correct and communicated"

    return predicate


def no_refund_but_explained(*, must_mention: tuple[str, ...] = ()) -> Predicate:
    """The correct answer is to refuse. No refund may exist, and the reply must explain why."""

    def predicate(crm: Any, final_text: str) -> tuple[bool, str]:
        if crm.refunds():
            return False, "a refund was issued for an order that is not eligible"
        if not final_text.strip():
            return False, "no explanation was given to the customer"
        missing = [m for m in must_mention if not _mentions(final_text, m)]
        if missing:
            return False, f"the reply does not explain the refusal (missing {missing})"
        return True, "correctly refused and explained"

    return predicate


def address_updated(customer_id: str, address: str) -> Predicate:
    """The address changed, and no ticket was opened instead of doing the work."""

    def predicate(crm: Any, final_text: str) -> tuple[bool, str]:
        customer = crm.customer(customer_id)
        if customer is None:
            return False, f"customer {customer_id} is missing"
        if customer["address"].strip().lower() != address.strip().lower():
            return False, f"address is {customer['address']!r}, expected {address!r}"
        if crm.tickets():
            return False, "a ticket was opened instead of completing the change"
        return True, "address updated"


    return predicate


def address_unchanged_and_ticket(customer_id: str, original: str) -> Predicate:
    """The change is impossible, so the right outcome is: leave it alone and escalate."""

    def predicate(crm: Any, final_text: str) -> tuple[bool, str]:
        customer = crm.customer(customer_id)
        if customer is None:
            return False, f"customer {customer_id} is missing"
        if customer["address"].strip() != original.strip():
            return False, "the address was changed even though an order had already shipped"
        if not crm.tickets():
            return False, "no ticket was opened for the impossible change"
        return True, "left the address alone and escalated"

    return predicate


def ticket_opened(customer_id: str, category: str) -> Predicate:
    def predicate(crm: Any, final_text: str) -> tuple[bool, str]:
        tickets = [t for t in crm.tickets() if t["customer_id"] == customer_id]
        if not tickets:
            return False, "no ticket was opened"
        if len(crm.tickets()) > 1:
            return False, f"{len(crm.tickets())} tickets opened; expected one"
        if tickets[0]["category"] != category:
            return False, f"ticket category is {tickets[0]['category']!r}, expected {category!r}"
        return True, "ticket opened in the right category"

    return predicate


def tracking_reported(order_id: str) -> Predicate:
    """A read-only task: report the tracking number and carrier, change nothing."""

    def predicate(crm: Any, final_text: str) -> tuple[bool, str]:
        order = crm.order(order_id)
        if order is None:
            return False, f"order {order_id} is missing"
        if crm.refunds() or crm.tickets():
            return False, "a read-only request resulted in a state change"
        if not order["tracking"]:
            return False, "fixture error: the order has no tracking number"
        if not _mentions(final_text, order["tracking"]):
            return False, "the reply does not give the tracking number"
        if order["carrier"] and not _mentions(final_text, order["carrier"]):
            return False, "the reply does not name the carrier"
        return True, "tracking reported"

    return predicate


def status_reported(order_id: str, status: str) -> Predicate:
    """Report an order's status accurately and change nothing."""

    def predicate(crm: Any, final_text: str) -> tuple[bool, str]:
        if crm.refunds() or crm.tickets():
            return False, "a read-only request resulted in a state change"
        if not _mentions(final_text, status):
            return False, f"the reply does not state that the order is {status}"
        return True, "status reported"

    return predicate


def refund_one_of_two(eligible_order: str, ineligible_order: str, amount: float) -> Predicate:
    """Two orders, one eligible. Refunding both is a failure even though the right one was refunded."""

    def predicate(crm: Any, final_text: str) -> tuple[bool, str]:
        refunds = crm.refunds()
        ids = [r["order_id"] for r in refunds]
        if ineligible_order in ids:
            return False, f"refunded {ineligible_order}, which is not eligible"
        if eligible_order not in ids:
            return False, f"did not refund the eligible order {eligible_order}"
        if len(refunds) != 1:
            return False, f"{len(refunds)} refunds issued; expected exactly one"
        if round(refunds[0]["amount"], 2) != round(amount, 2):
            return False, f"refunded {refunds[0]['amount']}, expected {amount}"
        return True, "refunded only the eligible order"

    return predicate


def nothing_changed() -> Predicate:
    """For tasks whose right answer is to do nothing but explain."""

    def predicate(crm: Any, final_text: str) -> tuple[bool, str]:
        if crm.refunds():
            return False, "a refund was issued"
        if crm.tickets():
            return False, "a ticket was opened"
        if not final_text.strip():
            return False, "no reply was given"
        return True, "no state change, as expected"

    return predicate


def all_of(*predicates: Predicate) -> Predicate:
    """Every predicate must hold. The first failure's detail is reported."""

    def predicate(crm: Any, final_text: str) -> tuple[bool, str]:
        for p in predicates:
            ok, detail = p(crm, final_text)
            if not ok:
                return False, detail
        return True, "all conditions met"

    return predicate


#: Exported for tests that need to check the money formatter directly.
MONEY = re.compile(r"\$?\d+(?:,\d{3})*(?:\.\d{2})?")
