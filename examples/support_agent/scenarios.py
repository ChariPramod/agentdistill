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
HELD_OUT_SCENARIOS = (
    "refund_partial_shipment",      # the customer overstates the amount
    "wrong_item_two_orders",        # pick the right order from a description
    "address_after_ship",           # a tool refusal that must become an escalation
    "duplicate_charge",             # a billing question with nothing refundable behind it
    "two_damaged_both_eligible",    # more than one refund is correct
    "refund_all_ineligible",        # every order is ineligible
)


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
# compact scenario builder
# --------------------------------------------------------------------------------------------------------------



#: Real customers open and close a message in a dozen different ways. Without this, every instance of a scenario
#: is near-identical text, which makes decontamination fire on the shared boilerplate between a training instance
#: and a holdout instance of the same shape, and makes clustering trivial.
_OPENERS = [
    "", "Hi, ", "Hello, ", "Hi there — ", "Good morning. ", "Afternoon. ", "Hey, ",
    "Sorry to bother you, but ", "Quick question: ", "I hope you can help. ",
]
_CLOSERS = [
    "", " Thanks.", " Thanks in advance.", " Please let me know.", " Appreciate your help.",
    " Looking forward to hearing back.", " Cheers.", " Many thanks for your time.",
    " Let me know what you can do.", " I would appreciate a quick reply.",
]
_ASIDES = [
    "", " I have been a customer for years.", " This is my first time contacting support.",
    " Apologies if this is the wrong channel.", " I did try the help pages first.",
    " No rush, whenever you get a chance.", " I am happy to provide more detail if needed.",
]


def _vary(rng: random.Random, message: str) -> str:
    """Wrap a request in a varied opener, aside, and closer."""
    opener = rng.choice(_OPENERS)
    body = message[0].lower() + message[1:] if opener and message[:1].isupper() and not message[:1].isdigit() \
        else message
    return f"{opener}{body}{rng.choice(_ASIDES)}{rng.choice(_CLOSERS)}".strip()


@dataclass
class OrderSpec:
    """One order to seed. `key` names it so a predicate can refer to it without knowing the id format."""

    key: str
    status: str
    total: float | None = None
    item: str | None = None
    placed_on: str = "2026-08-15"
    carrier: str | None = None


def simple(
    name: str,
    *,
    orders: Callable[[random.Random], list[OrderSpec]],
    message: Callable[[random.Random, dict], str],
    predicate: Callable[[dict], graders.Predicate],
    notes: str = "",
    setup: Callable[[CRM, dict], None] | None = None,
) -> Builder:
    """Register a scenario from its parts.

    `orders` and `message` receive the rng; `predicate` and `setup` receive a `ctx` dict carrying the customer,
    the resolved order ids by key, and the order specs. This keeps each scenario to its distinctive parts --
    the database state it needs and what counts as success -- instead of repeating the same scaffolding.
    """

    def build_task(rng: random.Random, seed: int) -> Task:
        cust = _customer(rng, seed)
        specs = orders(rng)
        resolved: list[OrderSpec] = []
        for spec in specs:
            resolved.append(
                OrderSpec(
                    key=spec.key,
                    status=spec.status,
                    total=spec.total if spec.total is not None else round(rng.uniform(18, 190), 2),
                    item=spec.item or rng.choice(ITEMS),
                    placed_on=spec.placed_on,
                    carrier=spec.carrier
                    or (rng.choice(["UPS", "DHL", "USPS"]) if spec.status in ("shipped", "delivered") else None),
                )
            )
        ids = {spec.key: f"o_{seed}_{i}" for i, spec in enumerate(resolved)}
        ctx = {
            "customer": cust,
            "ids": ids,
            "orders": {spec.key: spec for spec in resolved},
            "seed": seed,
            # Baseline addresses, so a "do nothing" predicate can assert nothing moved. Without this a student
            # that correctly declines a refund and then rewrites the shipping address still passes.
            "addresses": {cust.id: cust.address},
        }

        def make_crm() -> CRM:
            crm = CRM.empty(seed)
            crm.add_customer(cust)
            for spec in resolved:
                crm.add_order(ids[spec.key], cust.id, spec.status, spec.total, spec.item, spec.placed_on,
                              carrier=spec.carrier)
            if setup is not None:
                setup(crm, ctx)
                crm.calls.clear()  # setup calls are not part of the agent's trajectory
            return crm

        return Task(
            task_id=f"{name}-{seed}",
            scenario=name,
            user_message=_vary(rng, message(rng, ctx)),
            db_seed=seed,
            predicate=predicate(ctx),
            build=make_crm,
            notes=notes,
        )

    SCENARIOS[name] = build_task
    return build_task


def _pick(rng: random.Random, options: list[str], **kw: Any) -> str:
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
        predicate=graders.no_refund_but_explained(must_mention=("processing",),
                                                  addresses={cust.id: cust.address}),
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
        predicate=graders.tracking_reported(order_id, {cust.id: cust.address}),
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
        predicate=graders.status_reported(order_id, status, {cust.id: cust.address}),
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
        predicate=graders.ticket_opened(cust.id, "billing", {cust.id: cust.address}),
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
        predicate=graders.nothing_changed({cust.id: cust.address}),
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


# --------------------------------------------------------------------------------------------------------------
# Refund shapes
#
# Each reaches the refund tool under a different database state, so the interesting behaviour is how the agent
# handles what the tool says back. That is deliberately where the difficulty lives: an agent that only knows the
# happy path scores well on "refund a delivered order" and falls apart on everything below.
# --------------------------------------------------------------------------------------------------------------

_REFUND_ASKS = [
    "The {item} arrived damaged and I would like a refund. {email}",
    "Please refund my {item} — it turned up broken. Account {email}.",
    "I want my money back for the {item}. It was not in usable condition. {email}",
    "Refund request for the {item} please, it came through damaged. My email is {email}.",
]

simple(
    "refund_cancelled_order",
    orders=lambda rng: [OrderSpec("target", "cancelled")],
    message=lambda rng, ctx: _pick(rng, _REFUND_ASKS, item=ctx["orders"]["target"].item,
                                   email=ctx["customer"].email),
    predicate=lambda ctx: graders.no_refund_but_explained(addresses=ctx["addresses"]),
    notes="already cancelled; nothing to refund, must explain",
)

simple(
    "refund_no_orders",
    orders=lambda rng: [],
    message=lambda rng, ctx: f"I want a refund on my last purchase. My email is {ctx['customer'].email}.",
    predicate=lambda ctx: graders.nothing_changed(ctx["addresses"]),
    notes="account exists but has no orders at all",
)

simple(
    "refund_oldest_of_three",
    orders=lambda rng: [
        OrderSpec("old", "delivered", placed_on="2026-06-02"),
        OrderSpec("mid", "delivered", placed_on="2026-07-14"),
        OrderSpec("new", "processing", placed_on="2026-09-12"),
    ],
    message=lambda rng, ctx: (
        f"The {ctx['orders']['old'].item} I ordered back in June was damaged. Can I get a refund on that one? "
        f"{ctx['customer'].email}"
    ),
    predicate=lambda ctx: graders.refund_exactly(ctx["ids"]["old"], ctx["orders"]["old"].total),
    notes="three orders; the named item picks the right one",
)

simple(
    "refund_shipped_not_delivered",
    orders=lambda rng: [OrderSpec("target", "shipped")],
    message=lambda rng, ctx: (
        f"The {ctx['orders']['target'].item} on its way to me arrived damaged at my neighbour's. "
        f"Please refund it. {ctx['customer'].email}"
    ),
    predicate=lambda ctx: graders.refund_exactly(ctx["ids"]["target"], ctx["orders"]["target"].total),
    notes="shipped is refundable; an over-cautious agent refuses this wrongly",
)

simple(
    "refund_one_already_done",
    orders=lambda rng: [OrderSpec("done", "delivered"), OrderSpec("pending", "delivered")],
    setup=lambda crm, ctx: crm.call(
        "issue_refund",
        {"order_id": ctx["ids"]["done"], "amount": ctx["orders"]["done"].total, "reason": "damaged"},
    ),
    message=lambda rng, ctx: (
        f"You refunded my {ctx['orders']['done'].item} already, thank you. The {ctx['orders']['pending'].item} "
        f"was damaged too — can you do that one? {ctx['customer'].email}"
    ),
    # The setup refund is part of the starting state, so the correct end state has *two* refunds: the one that
    # was already there and the new one. A predicate expecting a single refund could never pass.
    predicate=lambda ctx: graders.refunds_exactly(
        {ctx["ids"]["done"]: ctx["orders"]["done"].total,
         ctx["ids"]["pending"]: ctx["orders"]["pending"].total}
    ),
    notes="one refund already exists; must add the other without retrying the first",
)

simple(
    "refund_all_ineligible",
    orders=lambda rng: [OrderSpec("a", "processing"), OrderSpec("b", "cancelled")],
    message=lambda rng, ctx: f"I would like refunds on everything I have ordered. {ctx['customer'].email}",
    predicate=lambda ctx: graders.no_refund_but_explained(addresses=ctx["addresses"]),
    notes="nothing is refundable; must refuse rather than force one through",
)

simple(
    "refund_amount_under_total",
    orders=lambda rng: [OrderSpec("target", "delivered", total=120.00)],
    message=lambda rng, ctx: (
        f"One of the two {ctx['orders']['target'].item}s in my order was damaged. I paid $120.00 for the pair, "
        f"so please refund half. {ctx['customer'].email}"
    ),
    predicate=lambda ctx: graders.refund_at_most(ctx["ids"]["target"], 120.00),
    notes="a partial refund is legitimate; any amount up to the total counts",
)


# --------------------------------------------------------------------------------------------------------------
# Tracking and status shapes
# --------------------------------------------------------------------------------------------------------------

simple(
    "track_not_yet_shipped",
    orders=lambda rng: [OrderSpec("target", "processing")],
    message=lambda rng, ctx: _pick(
        rng,
        [
            "Where is my {item}? It has been days. {email}",
            "Any tracking for the {item} yet? Account {email}.",
        ],
        item=ctx["orders"]["target"].item, email=ctx["customer"].email,
    ),
    predicate=lambda ctx: graders.status_reported(ctx["ids"]["target"], "processing", ctx["addresses"]),
    notes="no tracking exists yet; must say so rather than invent one",
)

simple(
    "track_already_delivered",
    orders=lambda rng: [OrderSpec("target", "delivered", placed_on="2026-07-30")],
    message=lambda rng, ctx: (
        f"My {ctx['orders']['target'].item} still has not shown up. Where is it? {ctx['customer'].email}"
    ),
    predicate=lambda ctx: graders.status_reported(ctx["ids"]["target"], "delivered", ctx["addresses"]),
    notes="the record says delivered but the customer disagrees",
)

simple(
    "track_multiple_shipped",
    orders=lambda rng: [
        OrderSpec("first", "shipped", placed_on="2026-09-01"),
        OrderSpec("second", "shipped", placed_on="2026-09-08"),
    ],
    message=lambda rng, ctx: (
        f"Can you tell me where the {ctx['orders']['first'].item} is? {ctx['customer'].email}"
    ),
    predicate=lambda ctx: graders.tracking_reported(ctx["ids"]["first"], ctx["addresses"]),
    notes="two shipments; must report the one asked about",
)

simple(
    "status_all_orders",
    orders=lambda rng: [
        OrderSpec("a", "delivered"), OrderSpec("b", "shipped"), OrderSpec("c", "processing"),
    ],
    message=lambda rng, ctx: f"Can you give me a rundown of everything on my account? {ctx['customer'].email}",
    predicate=lambda ctx: graders.read_only(ctx["addresses"]),
    notes="read-only overview; must change nothing",
)


# --------------------------------------------------------------------------------------------------------------
# Address shapes
# --------------------------------------------------------------------------------------------------------------

simple(
    "address_no_orders",
    orders=lambda rng: [],
    message=lambda rng, ctx: (
        f"Please update my address to 5 Willow Drive, Denver for future orders. {ctx['customer'].email}"
    ),
    predicate=lambda ctx: graders.address_updated(ctx["customer"].id, "5 Willow Drive, Denver"),
    notes="nothing to block the change",
)

simple(
    "address_one_shipped_one_processing",
    orders=lambda rng: [OrderSpec("shipped", "shipped"), OrderSpec("pending", "processing")],
    message=lambda rng, ctx: (
        f"I have moved to 88 Slate Road, Portland. Please update my orders. {ctx['customer'].email}"
    ),
    predicate=lambda ctx: graders.address_unchanged_and_ticket(ctx["customer"].id, ctx["customer"].address),
    notes="one shipped order blocks the whole change; must escalate",
)

simple(
    "address_delivered_orders_only",
    orders=lambda rng: [OrderSpec("a", "delivered"), OrderSpec("b", "delivered")],
    message=lambda rng, ctx: f"New address please: 19 Foundry Lane, Chicago. {ctx['customer'].email}",
    predicate=lambda ctx: graders.address_updated(ctx["customer"].id, "19 Foundry Lane, Chicago"),
    notes="delivered orders do not block a change; only shipped ones do",
)


# --------------------------------------------------------------------------------------------------------------
# Escalation and refusal shapes
#
# The right answer to all of these is a ticket or an explanation, never a tool action. An agent that reaches for
# a refund because the customer sounds unhappy fails every one of them.
# --------------------------------------------------------------------------------------------------------------

simple(
    "billing_no_orders",
    orders=lambda rng: [],
    message=lambda rng, ctx: (
        f"There is a charge from you on my card but I never ordered anything. {ctx['customer'].email}"
    ),
    predicate=lambda ctx: graders.ticket_opened(ctx["customer"].id, "billing", ctx["addresses"]),
    notes="no order to refund against; escalate to billing",
)

simple(
    "billing_charged_twice_one_order",
    orders=lambda rng: [OrderSpec("target", "delivered")],
    message=lambda rng, ctx: (
        f"I have been charged twice for the {ctx['orders']['target'].item} but I only see one order. "
        f"{ctx['customer'].email}"
    ),
    predicate=lambda ctx: graders.ticket_opened(ctx["customer"].id, "billing", ctx["addresses"]),
    notes="a duplicate charge needs investigation, not a refund",
)

simple(
    "shipping_delay_complaint",
    orders=lambda rng: [OrderSpec("target", "shipped", placed_on="2026-07-11")],
    message=lambda rng, ctx: (
        f"This is unacceptable. The {ctx['orders']['target'].item} was supposed to arrive weeks ago and I have "
        f"heard nothing at all. {ctx['customer'].email}"
    ),
    predicate=lambda ctx: graders.no_refund_but_explained(addresses=ctx["addresses"]),
    notes="an angry customer whose order is merely late; must not refund reflexively",
)

simple(
    "cancel_delivered_order",
    orders=lambda rng: [OrderSpec("target", "delivered")],
    message=lambda rng, ctx: (
        f"I want to cancel the {ctx['orders']['target'].item} order. {ctx['customer'].email}"
    ),
    predicate=lambda ctx: graders.read_only(ctx["addresses"]),
    notes="cannot cancel what has arrived",
)

simple(
    "unknown_order_id_only",
    orders=lambda rng: [OrderSpec("real", "delivered")],
    message=lambda rng, ctx: (
        f"Order o_{ctx['seed']}_9999 arrived damaged, please refund it. I do not have my email to hand."
    ),
    predicate=lambda ctx: graders.nothing_changed(ctx["addresses"]),
    notes="an id that does not exist and no email; must not guess at another order",
)

simple(
    "vague_request",
    orders=lambda rng: [OrderSpec("target", "delivered")],
    message=lambda rng, ctx: f"Hi, I have a problem with my order. {ctx['customer'].email}",
    predicate=lambda ctx: graders.read_only(ctx["addresses"]),
    notes="too vague to act on; must ask rather than pick an action",
)


# --------------------------------------------------------------------------------------------------------------
# Multi-step shapes
# --------------------------------------------------------------------------------------------------------------

simple(
    "refund_then_address",
    orders=lambda rng: [OrderSpec("damaged", "delivered"), OrderSpec("pending", "processing")],
    message=lambda rng, ctx: (
        f"Two things: the {ctx['orders']['damaged'].item} arrived damaged and I would like it refunded, and "
        f"please send anything still coming to 4 Anchor Street, Seattle. {ctx['customer'].email}"
    ),
    predicate=lambda ctx: graders.refund_exactly(ctx["ids"]["damaged"], ctx["orders"]["damaged"].total),
    notes="two requests in one message; the refund is the graded half",
)

simple(
    "refund_gold_tier",
    orders=lambda rng: [OrderSpec("target", "delivered")],
    message=lambda rng, ctx: (
        f"I am a gold member and the {ctx['orders']['target'].item} was damaged. I expect a full refund. "
        f"{ctx['customer'].email}"
    ),
    predicate=lambda ctx: graders.refund_exactly(ctx["ids"]["target"], ctx["orders"]["target"].total),
    notes="tier is stated but changes nothing; the refund is ordinary",
)

simple(
    "two_damaged_both_eligible",
    orders=lambda rng: [OrderSpec("a", "delivered"), OrderSpec("b", "delivered")],
    message=lambda rng, ctx: (
        f"Both the {ctx['orders']['a'].item} and the {ctx['orders']['b'].item} arrived damaged. "
        f"Please refund both. {ctx['customer'].email}"
    ),
    predicate=lambda ctx: graders.refunds_exactly(
        {ctx["ids"]["a"]: ctx["orders"]["a"].total, ctx["ids"]["b"]: ctx["orders"]["b"].total}
    ),
    notes="both really are eligible; refunding only one is incomplete",
)


simple(
    "refund_wrong_customer_order",
    orders=lambda rng: [OrderSpec("mine", "delivered")],
    message=lambda rng, ctx: (
        f"My friend's order o_{ctx['seed']}_7777 was damaged, can you refund it to me? {ctx['customer'].email}"
    ),
    predicate=lambda ctx: graders.nothing_changed(ctx["addresses"]),
    notes="an order that is not on this account; acting on it would be a security failure",
)

simple(
    "track_cancelled_order",
    orders=lambda rng: [OrderSpec("target", "cancelled")],
    message=lambda rng, ctx: (
        f"Where has my {ctx['orders']['target'].item} got to? {ctx['customer'].email}"
    ),
    predicate=lambda ctx: graders.status_reported(ctx["ids"]["target"], "cancelled", ctx["addresses"]),
    notes="the order was cancelled; there is nothing in transit to report",
)

simple(
    "refund_reason_not_in_enum",
    orders=lambda rng: [OrderSpec("target", "delivered")],
    message=lambda rng, ctx: (
        f"The {ctx['orders']['target'].item} is the wrong colour, nothing like the photo. I want a refund. "
        f"{ctx['customer'].email}"
    ),
    predicate=lambda ctx: graders.refund_exactly(ctx["ids"]["target"], ctx["orders"]["target"].total),
    notes="the stated reason is not one of the tool's allowed values; must map it onto one that is",
)

simple(
    "address_same_as_current",
    orders=lambda rng: [OrderSpec("pending", "processing")],
    message=lambda rng, ctx: (
        f"Please confirm you have my address as {ctx['customer'].address} and update it if not. "
        f"{ctx['customer'].email}"
    ),
    predicate=lambda ctx: graders.address_updated(ctx["customer"].id, ctx["customer"].address),
    notes="the requested address is already on file; a no-op change is still correct",
)
