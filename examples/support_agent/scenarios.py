"""Scenario templates.

Each scenario builds a seeded database, a user message, and an end-state predicate. Instantiating one with a
different seed gives a different customer, different orders, different amounts, and different phrasing, but the
same *shape* of problem.

The wrinkles are the point. A corpus of "refund this order" teaches nothing a lookup table could not do, and a
teacher scoring 98% on it cannot be distinguished from a student scoring 96%. These scenarios include orders that
cannot be refunded, ids the customer gets wrong, two orders where only one is eligible, and requests whose
correct answer is to refuse and explain.

Splitting: `sample()` returns instances. Instances of the same scenario go to both train and holdout (new
instances of a known shape), while `HELD_OUT_SCENARIOS` never appear in training at all, for the generalization
number that is always worse and always the honest one.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from examples.support_agent import graders
from examples.support_agent.crm import CRM, Customer

Builder = Callable[[random.Random, int], "Task"]


@dataclass
class Task:
    task_id: str
    scenario: str
    user_message: str
    db_seed: int
    predicate: graders.Predicate
    #: How the database is built. Re-running this with `db_seed` reproduces the starting state exactly, which is
    #: what lets the replay grader reconstruct the end state from the student's calls.
    build: Callable[[], CRM] = field(repr=False, default=None)  # type: ignore[assignment]
    notes: str = ""

    def fresh_crm(self) -> CRM:
        return self.build()


SCENARIOS: dict[str, Builder] = {}
#: Scenarios deliberately excluded from training, for the unseen-shape eval set.
HELD_OUT_SCENARIOS = ("refund_partial_shipment", "wrong_item_two_orders", "address_after_ship", "duplicate_charge")


def scenario(name: str) -> Callable[[Builder], Builder]:
    def register(fn: Builder) -> Builder:
        SCENARIOS[name] = fn
        return fn

    return register


# --------------------------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------------------------

STREETS = ["12 Main Street", "40 Oak Avenue", "7 Birch Lane", "221 Harbor Road", "98 Cedar Court",
           "3 Juniper Way", "154 Quarry Road"]
CITIES = ["Boston", "Austin", "Denver", "Seattle", "Miami", "Chicago", "Portland", "Atlanta"]
ITEMS = ["desk lamp", "wool blanket", "ceramic mug", "hiking boots", "espresso grinder", "wall clock",
         "yoga mat", "bluetooth speaker", "cast iron pan", "linen shirt"]


def _customer(rng: random.Random, seed: int, idx: int = 0) -> Customer:
    first = rng.choice(["Ada", "Bo", "Cleo", "Dev", "Elif", "Farid", "Greta", "Hana", "Ivo", "Jun"])
    last = rng.choice(["Nowak", "Oyelaran", "Park", "Quinn", "Ramos", "Silva", "Tan", "Ueda"])
    return Customer(
        id=f"c_{seed}_{idx}",
        email=f"{first.lower()}.{last.lower()}{rng.randint(10, 99)}@example.com",
        name=f"{first} {last}",
        address=f"{rng.choice(STREETS)}, {rng.choice(CITIES)}",
        tier=rng.choice(["standard", "standard", "gold"]),
    )


def _phrase(rng: random.Random, options: list[str], **kw: Any) -> str:
    return rng.choice(options).format(**kw)


# --------------------------------------------------------------------------------------------------------------
# scenarios
# --------------------------------------------------------------------------------------------------------------

REFUND_PHRASINGS = [
    "My {item} arrived {problem}. I'd like a refund please. My email is {email}.",
    "Hi — the {item} I ordered turned up {problem} and I want my money back. Account: {email}.",
    "I'm not happy. The {item} came {problem}. Please refund me. You can find me at {email}.",
    "Could you refund the {item}? It arrived {problem}. My email address is {email}.",
    "The {item} was {problem} when it arrived. I would like that order refunded. {email}",
]


@scenario("refund_delivered")
def _refund_delivered(rng: random.Random, seed: int) -> Task:
    """The straightforward case: one delivered order, refund it in full."""
    cust = _customer(rng, seed)
    item, total = rng.choice(ITEMS), round(rng.uniform(20, 180), 2)
    order_id = f"o_{seed}_0"
    problem = rng.choice(["damaged", "broken", "cracked", "scratched"])

    def build() -> CRM:
        c = CRM.empty(seed)
        c.add_customer(cust)
        c.add_order(order_id, cust.id, "delivered", total, item, "2026-08-14", carrier="UPS")
        return c

    return Task(
        task_id=f"refund_delivered-{seed}",
        scenario="refund_delivered",
        user_message=_phrase(rng, REFUND_PHRASINGS, item=item, problem=problem, email=cust.email),
        db_seed=seed,
        predicate=graders.refund_exactly(order_id, total),
        build=build,
        notes="single delivered order, full refund, amount must be stated",
    )


@scenario("refund_processing_ineligible")
def _refund_processing(rng: random.Random, seed: int) -> Task:
    """The order has not shipped, so it cannot be refunded. The right answer is to refuse and explain."""
    cust = _customer(rng, seed)
    item, total = rng.choice(ITEMS), round(rng.uniform(20, 180), 2)
    order_id = f"o_{seed}_0"

    def build() -> CRM:
        c = CRM.empty(seed)
        c.add_customer(cust)
        c.add_order(order_id, cust.id, "processing", total, item, "2026-09-02")
        return c

    return Task(
        task_id=f"refund_processing_ineligible-{seed}",
        scenario="refund_processing_ineligible",
        user_message=_phrase(
            rng,
            [
                "I changed my mind about the {item} and want a refund. {email}",
                "Please refund my {item} order — I no longer need it. Account {email}.",
                "Can I get my money back for the {item}? {email}",
            ],
            item=item, email=cust.email,
        ),
        db_seed=seed,
        predicate=graders.no_refund_but_explained(must_mention=("processing",)),
        build=build,
        notes="not yet shipped; must refuse and say why",
    )


@scenario("refund_already_refunded")
def _refund_twice(rng: random.Random, seed: int) -> Task:
    """Already refunded. The tool will refuse; the agent must notice and explain rather than loop."""
    cust = _customer(rng, seed)
    item, total = rng.choice(ITEMS), round(rng.uniform(20, 180), 2)
    order_id = f"o_{seed}_0"

    def build() -> CRM:
        c = CRM.empty(seed)
        c.add_customer(cust)
        c.add_order(order_id, cust.id, "delivered", total, item, "2026-07-20", carrier="DHL")
        c.call("issue_refund", {"order_id": order_id, "amount": total, "reason": "damaged"})
        c.calls.clear()  # the setup refund is not part of the agent's trajectory
        return c

    def predicate(crm: Any, final_text: str) -> tuple[bool, str]:
        if len(crm.refunds()) != 1:
            return False, f"{len(crm.refunds())} refunds; the order was already refunded before the conversation"
        if not final_text.strip():
            return False, "no reply"
        if "refund" not in final_text.lower():
            return False, "the reply does not address the refund"
        return True, "recognized the existing refund"

    return Task(
        task_id=f"refund_already_refunded-{seed}",
        scenario="refund_already_refunded",
        user_message=_phrase(
            rng,
            [
                "I still haven't seen my refund for the {item}. {email}",
                "Where is the refund for my {item} order? Email is {email}.",
            ],
            item=item, email=cust.email,
        ),
        db_seed=seed,
        predicate=predicate,
        build=build,
        notes="refund already exists; must not double-refund",
    )


@scenario("track_order")
def _track(rng: random.Random, seed: int) -> Task:
    """Read-only: report carrier and tracking, change nothing."""
    cust = _customer(rng, seed)
    item = rng.choice(ITEMS)
    order_id = f"o_{seed}_0"
    carrier = rng.choice(["UPS", "DHL", "USPS"])

    def build() -> CRM:
        c = CRM.empty(seed)
        c.add_customer(cust)
        c.add_order(order_id, cust.id, "shipped", round(rng.uniform(20, 180), 2), item, "2026-09-05",
                    carrier=carrier)
        return c

    return Task(
        task_id=f"track_order-{seed}",
        scenario="track_order",
        user_message=_phrase(
            rng,
            [
                "Where is my {item}? It hasn't arrived. {email}",
                "Any update on the {item} I ordered? My email is {email}.",
                "Can you tell me when the {item} will get here? Account {email}.",
            ],
            item=item, email=cust.email,
        ),
        db_seed=seed,
        predicate=graders.tracking_reported(order_id),
        build=build,
        notes="read-only; must give tracking and carrier",
    )


@scenario("update_address_processing")
def _address_ok(rng: random.Random, seed: int) -> Task:
    """Nothing has shipped, so the address change should go through."""
    cust = _customer(rng, seed)
    new_address = f"{rng.choice(STREETS)}, {rng.choice(CITIES)}"
    while new_address == cust.address:
        new_address = f"{rng.choice(STREETS)}, {rng.choice(CITIES)}"

    def build() -> CRM:
        c = CRM.empty(seed)
        c.add_customer(cust)
        c.add_order(f"o_{seed}_0", cust.id, "processing", round(rng.uniform(20, 180), 2), rng.choice(ITEMS),
                    "2026-09-09")
        return c

    return Task(
        task_id=f"update_address_processing-{seed}",
        scenario="update_address_processing",
        user_message=_phrase(
            rng,
            [
                "I've moved. Please send my order to {addr}. My email is {email}.",
                "Change my shipping address to {addr}, please. Account {email}.",
                "Wrong address at checkout — it should go to {addr}. {email}",
            ],
            addr=new_address, email=cust.email,
        ),
        db_seed=seed,
        predicate=graders.address_updated(cust.id, new_address),
        build=build,
        notes="nothing shipped; change should succeed",
    )


@scenario("address_after_ship")
def _address_blocked(rng: random.Random, seed: int) -> Task:
    """An order has shipped, so the tool refuses. The right answer is to escalate, not to keep retrying."""
    cust = _customer(rng, seed)
    new_address = f"{rng.choice(STREETS)}, {rng.choice(CITIES)}"

    def build() -> CRM:
        c = CRM.empty(seed)
        c.add_customer(cust)
        c.add_order(f"o_{seed}_0", cust.id, "shipped", round(rng.uniform(20, 180), 2), rng.choice(ITEMS),
                    "2026-09-01", carrier="UPS")
        return c

    return Task(
        task_id=f"address_after_ship-{seed}",
        scenario="address_after_ship",
        user_message=_phrase(
            rng,
            [
                "Urgent — redirect my package to {addr}. {email}",
                "I need my order sent to {addr} instead. Account {email}.",
            ],
            addr=new_address, email=cust.email,
        ),
        db_seed=seed,
        predicate=graders.address_unchanged_and_ticket(cust.id, cust.address),
        build=build,
        notes="already shipped; tool refuses; must open a ticket",
    )


@scenario("wrong_order_id_first")
def _wrong_id(rng: random.Random, seed: int) -> Task:
    """The customer supplies an id that does not exist. The agent must look it up properly rather than give up."""
    cust = _customer(rng, seed)
    item, total = rng.choice(ITEMS), round(rng.uniform(20, 180), 2)
    real_order = f"o_{seed}_0"
    bogus = f"o_{rng.randint(90000, 99999)}"

    def build() -> CRM:
        c = CRM.empty(seed)
        c.add_customer(cust)
        c.add_order(real_order, cust.id, "delivered", total, item, "2026-08-02", carrier="USPS")
        return c

    return Task(
        task_id=f"wrong_order_id_first-{seed}",
        scenario="wrong_order_id_first",
        user_message=(
            f"Order {bogus} arrived damaged and I want a refund. My email is {cust.email} "
            f"if that id is wrong."
        ),
        db_seed=seed,
        predicate=graders.refund_exactly(real_order, total),
        build=build,
        notes="the id in the message is wrong; must recover via email lookup",
    )


@scenario("refund_one_of_two")
def _one_of_two(rng: random.Random, seed: int) -> Task:
    """Two orders, only one eligible. Refunding both is a failure."""
    cust = _customer(rng, seed)
    eligible, ineligible = f"o_{seed}_0", f"o_{seed}_1"
    amount = round(rng.uniform(20, 180), 2)
    item_a, item_b = rng.sample(ITEMS, 2)

    def build() -> CRM:
        c = CRM.empty(seed)
        c.add_customer(cust)
        c.add_order(eligible, cust.id, "delivered", amount, item_a, "2026-08-11", carrier="UPS")
        c.add_order(ineligible, cust.id, "processing", round(rng.uniform(20, 180), 2), item_b, "2026-09-10")
        return c

    return Task(
        task_id=f"refund_one_of_two-{seed}",
        scenario="refund_one_of_two",
        user_message=(
            f"I have two open orders and I want to return what I can. The {item_a} was damaged. "
            f"My email is {cust.email}."
        ),
        db_seed=seed,
        predicate=graders.refund_one_of_two(eligible, ineligible, amount),
        build=build,
        notes="only the delivered order may be refunded",
    )


@scenario("wrong_item_two_orders")
def _wrong_item(rng: random.Random, seed: int) -> Task:
    """Two delivered orders; the customer names the item, not the id. Refund the one they named."""
    cust = _customer(rng, seed)
    item_a, item_b = rng.sample(ITEMS, 2)
    target, other = f"o_{seed}_0", f"o_{seed}_1"
    amount = round(rng.uniform(20, 180), 2)

    def build() -> CRM:
        c = CRM.empty(seed)
        c.add_customer(cust)
        c.add_order(target, cust.id, "delivered", amount, item_a, "2026-08-03", carrier="UPS")
        c.add_order(other, cust.id, "delivered", round(rng.uniform(20, 180), 2), item_b, "2026-08-20",
                    carrier="DHL")
        return c

    return Task(
        task_id=f"wrong_item_two_orders-{seed}",
        scenario="wrong_item_two_orders",
        user_message=(
            f"You sent me the wrong thing — I got a {item_a} that isn't what I ordered. "
            f"Please refund that one. I also have a {item_b} coming which is fine. {cust.email}"
        ),
        db_seed=seed,
        predicate=graders.refund_one_of_two(target, other, amount),
        build=build,
        notes="must identify the order by item, not refund both",
    )


@scenario("order_status")
def _status(rng: random.Random, seed: int) -> Task:
    cust = _customer(rng, seed)
    status = rng.choice(["processing", "delivered"])
    order_id = f"o_{seed}_0"
    item = rng.choice(ITEMS)

    def build() -> CRM:
        c = CRM.empty(seed)
        c.add_customer(cust)
        c.add_order(order_id, cust.id, status, round(rng.uniform(20, 180), 2), item, "2026-09-03",
                    carrier="UPS" if status == "delivered" else None)
        return c

    return Task(
        task_id=f"order_status-{seed}",
        scenario="order_status",
        user_message=_phrase(
            rng,
            [
                "What's happening with my {item} order? {email}",
                "Has my {item} order gone out yet? Account {email}.",
            ],
            item=item, email=cust.email,
        ),
        db_seed=seed,
        predicate=graders.status_reported(order_id, status),
        build=build,
        notes="read-only status report",
    )


@scenario("refund_partial_shipment")
def _partial(rng: random.Random, seed: int) -> Task:
    """The customer asks for more than the order total. The agent must refund the total, not the asked amount."""
    cust = _customer(rng, seed)
    total = round(rng.uniform(40, 120), 2)
    asked = round(total + rng.uniform(20, 80), 2)
    order_id = f"o_{seed}_0"
    item = rng.choice(ITEMS)

    def build() -> CRM:
        c = CRM.empty(seed)
        c.add_customer(cust)
        c.add_order(order_id, cust.id, "delivered", total, item, "2026-08-07", carrier="UPS")
        return c

    return Task(
        task_id=f"refund_partial_shipment-{seed}",
        scenario="refund_partial_shipment",
        user_message=(
            f"The {item} was damaged. I paid about ${asked:.2f} for it and want that back. {cust.email}"
        ),
        db_seed=seed,
        predicate=graders.refund_exactly(order_id, total),
        build=build,
        notes="customer overstates the amount; refund the real total",
    )


@scenario("duplicate_charge")
def _duplicate(rng: random.Random, seed: int) -> Task:
    """A billing question with no refundable order. The right answer is a ticket, not a refund."""
    cust = _customer(rng, seed)

    def build() -> CRM:
        c = CRM.empty(seed)
        c.add_customer(cust)
        c.add_order(f"o_{seed}_0", cust.id, "processing", round(rng.uniform(20, 180), 2), rng.choice(ITEMS),
                    "2026-09-11")
        return c

    return Task(
        task_id=f"duplicate_charge-{seed}",
        scenario="duplicate_charge",
        user_message=(
            f"I think I've been charged twice this month and I can only see one order. "
            f"Can someone look into it? {cust.email}"
        ),
        db_seed=seed,
        predicate=graders.all_of(graders.ticket_opened(cust.id, "billing")),
        build=build,
        notes="no refundable order; escalate as billing",
    )


@scenario("unknown_email")
def _unknown_email(rng: random.Random, seed: int) -> Task:
    """The email does not exist. The agent must not invent a customer; it should say so."""
    cust = _customer(rng, seed)
    bogus_email = f"nobody{rng.randint(100, 999)}@example.com"

    def build() -> CRM:
        c = CRM.empty(seed)
        c.add_customer(cust)
        c.add_order(f"o_{seed}_0", cust.id, "delivered", 50.0, rng.choice(ITEMS), "2026-08-01", carrier="UPS")
        return c

    return Task(
        task_id=f"unknown_email-{seed}",
        scenario="unknown_email",
        user_message=f"I want a refund on my last order. My email is {bogus_email}.",
        db_seed=seed,
        predicate=graders.nothing_changed(),
        build=build,
        notes="no such customer; must not refund anything",
    )


# --------------------------------------------------------------------------------------------------------------
# sampling
# --------------------------------------------------------------------------------------------------------------


def build_task(scenario_name: str, seed: int) -> Task:
    if scenario_name not in SCENARIOS:
        raise KeyError(f"unknown scenario {scenario_name!r}; known: {sorted(SCENARIOS)}")
    return SCENARIOS[scenario_name](random.Random(seed), seed)


def sample(n: int, seed: int = 0, scenarios: list[str] | None = None) -> list[Task]:
    """`n` tasks spread evenly across the scenarios, deterministic in `seed`."""
    names = scenarios if scenarios is not None else sorted(SCENARIOS)
    if not names:
        return []
    tasks = []
    for i in range(n):
        name = names[i % len(names)]
        tasks.append(build_task(name, seed * 100_000 + i))
    return tasks


def training_scenarios() -> list[str]:
    return [n for n in sorted(SCENARIOS) if n not in HELD_OUT_SCENARIOS]
