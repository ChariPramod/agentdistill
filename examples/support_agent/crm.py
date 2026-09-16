"""A small, real CRM backed by SQLite.

Deterministic: the same seed produces the same database, and the same sequence of calls produces the same
results. That determinism is what lets the eval harness replay a trajectory and lets a predicate check the end
state exactly.

It is a *real* stateful service, not a lookup table of canned strings. Tools fail when the data says they should:
refunding an unshipped order, refunding twice, changing the address on a shipped order. Those failures are part
of the trajectory and are where a teacher's behaviour becomes worth imitating.
"""

from __future__ import annotations

import hashlib
import json
import random
import sqlite3
from dataclasses import dataclass
from typing import Any

SCHEMA = """
CREATE TABLE customers (
  id      TEXT PRIMARY KEY,
  email   TEXT NOT NULL UNIQUE,
  name    TEXT NOT NULL,
  address TEXT NOT NULL,
  tier    TEXT NOT NULL
);
CREATE TABLE orders (
  id          TEXT PRIMARY KEY,
  customer_id TEXT NOT NULL REFERENCES customers(id),
  status      TEXT NOT NULL CHECK (status IN ('processing','shipped','delivered','cancelled')),
  total       REAL NOT NULL,
  item        TEXT NOT NULL,
  placed_on   TEXT NOT NULL,
  carrier     TEXT,
  tracking    TEXT
);
CREATE TABLE refunds (
  id       TEXT PRIMARY KEY,
  order_id TEXT NOT NULL REFERENCES orders(id),
  amount   REAL NOT NULL,
  reason   TEXT NOT NULL
);
CREATE TABLE tickets (
  id          TEXT PRIMARY KEY,
  customer_id TEXT NOT NULL REFERENCES customers(id),
  category    TEXT NOT NULL,
  summary     TEXT NOT NULL
);
"""

REFUND_REASONS = ["damaged", "late", "wrong_item", "unwanted"]
TICKET_CATEGORIES = ["billing", "shipping", "product", "other"]

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_customer",
            "description": "Look up a customer by email address. Returns the customer id, name, address, and tier.",
            "parameters": {
                "type": "object",
                "properties": {"email": {"type": "string", "description": "The customer's email address."}},
                "required": ["email"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_orders",
            "description": "List a customer's orders, most recent first. Optionally filter by status.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string"},
                    "status": {
                        "type": "string",
                        "enum": ["processing", "shipped", "delivered", "cancelled"],
                        "description": "Only return orders in this status.",
                    },
                },
                "required": ["customer_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_order",
            "description": "Get one order by id, including its status, total, carrier, and tracking number.",
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
            "name": "issue_refund",
            "description": (
                "Refund an order. Only delivered or shipped orders can be refunded, the amount may not exceed "
                "the order total, and an order can only be refunded once."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "amount": {"type": "number", "description": "Amount in dollars."},
                    "reason": {"type": "string", "enum": REFUND_REASONS},
                },
                "required": ["order_id", "amount", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_address",
            "description": (
                "Change a customer's shipping address. Fails if any of their orders has already shipped, since "
                "a shipped order cannot be redirected."
            ),
            "parameters": {
                "type": "object",
                "properties": {"customer_id": {"type": "string"}, "address": {"type": "string"}},
                "required": ["customer_id", "address"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_ticket",
            "description": "Open a support ticket for a human to follow up on.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string"},
                    "category": {"type": "string", "enum": TICKET_CATEGORIES},
                    "summary": {"type": "string"},
                },
                "required": ["customer_id", "category", "summary"],
            },
        },
    },
]

TOOL_NAMES = [t["function"]["name"] for t in TOOLS]


class ToolError(Exception):
    """A tool refused the call. Surfaced to the model as a tool result, not raised to the caller."""


@dataclass
class Customer:
    id: str
    email: str
    name: str
    address: str
    tier: str


FIRST = ["Ada", "Bo", "Cleo", "Dev", "Elif", "Farid", "Greta", "Hana", "Ivo", "Jun", "Kira", "Luca"]
LAST = ["Nowak", "Oyelaran", "Park", "Quinn", "Ramos", "Silva", "Tan", "Ueda", "Vargas", "Weiss"]
ITEMS = ["desk lamp", "wool blanket", "ceramic mug", "hiking boots", "espresso grinder", "wall clock",
         "yoga mat", "bluetooth speaker", "cast iron pan", "linen shirt"]
STREETS = ["12 Main Street", "40 Oak Avenue", "7 Birch Lane", "221 Harbor Road", "98 Cedar Court"]
CITIES = ["Boston", "Austin", "Denver", "Seattle", "Miami", "Chicago", "Portland", "Atlanta"]


class CRM:
    """The tool surface. `call(name, args)` is the only entry point the agent uses."""

    def __init__(self, conn: sqlite3.Connection, seed: int) -> None:
        self.conn = conn
        self.conn.row_factory = sqlite3.Row
        self.seed = seed
        #: Every call made, in order. The harness and the graders read this.
        self.calls: list[tuple[str, dict]] = []

    # ----------------------------------------------------------------------------------------------------------
    # construction
    # ----------------------------------------------------------------------------------------------------------

    @classmethod
    def empty(cls, seed: int = 0) -> CRM:
        conn = sqlite3.connect(":memory:")
        conn.executescript(SCHEMA)
        return cls(conn, seed)

    @classmethod
    def from_seed(cls, seed: int, n_customers: int = 3, orders_per_customer: tuple[int, int] = (1, 4)) -> CRM:
        """Build a populated database. Same seed, same database, always."""
        crm = cls.empty(seed)
        rng = random.Random(seed)
        for ci in range(n_customers):
            first, last = rng.choice(FIRST), rng.choice(LAST)
            cust = Customer(
                id=f"c_{seed}_{ci}",
                email=f"{first.lower()}.{last.lower()}{rng.randint(1, 99)}@example.com",
                name=f"{first} {last}",
                address=f"{rng.choice(STREETS)}, {rng.choice(CITIES)}",
                tier=rng.choice(["standard", "standard", "standard", "gold"]),
            )
            crm.add_customer(cust)
            for oi in range(rng.randint(*orders_per_customer)):
                status = rng.choice(["processing", "shipped", "delivered", "delivered"])
                crm.add_order(
                    order_id=f"o_{seed}_{ci}{oi}",
                    customer_id=cust.id,
                    status=status,
                    total=round(rng.uniform(12, 240), 2),
                    item=rng.choice(ITEMS),
                    placed_on=f"2026-0{rng.randint(6, 9)}-{rng.randint(10, 28)}",
                    carrier=rng.choice(["UPS", "DHL", "USPS"]) if status in ("shipped", "delivered") else None,
                )
        return crm

    def add_customer(self, c: Customer) -> None:
        self.conn.execute(
            "INSERT INTO customers (id, email, name, address, tier) VALUES (?,?,?,?,?)",
            (c.id, c.email, c.name, c.address, c.tier),
        )
        self.conn.commit()

    def add_order(
        self,
        order_id: str,
        customer_id: str,
        status: str,
        total: float,
        item: str,
        placed_on: str,
        carrier: str | None = None,
        tracking: str | None = None,
    ) -> None:
        if carrier and not tracking:
            # A stable digest, never the built-in hash(): PYTHONHASHSEED is randomized per process, so hash()
            # would give a rebuilt database a different tracking number than the recording had. The replay
            # grader rebuilds state in a *different process* from the one that recorded it, and an unstable id
            # there silently fails every predicate that checks a tracking number.
            digest = hashlib.sha256(order_id.encode()).hexdigest()
            tracking = f"1Z{int(digest[:12], 16) % 10**10:010d}"
        self.conn.execute(
            "INSERT INTO orders (id, customer_id, status, total, item, placed_on, carrier, tracking) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (order_id, customer_id, status, total, item, placed_on, carrier, tracking),
        )
        self.conn.commit()

    # ----------------------------------------------------------------------------------------------------------
    # the tool surface
    # ----------------------------------------------------------------------------------------------------------

    def call(self, name: str, args: dict) -> Any:
        """Dispatch a tool call. Records it, then runs it.

        Raises `ToolError` for a refusal the model should see and react to; raises `ToolError` for an unknown
        tool too, since a hallucinated tool name is exactly the kind of mistake a trajectory should contain.
        """
        self.calls.append((name, dict(args)))
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            raise ToolError(f"unknown tool {name!r}; available tools are {', '.join(TOOL_NAMES)}")
        return handler(**args)

    def _tool_get_customer(self, email: str) -> dict:
        row = self.conn.execute(
            "SELECT * FROM customers WHERE lower(email) = lower(?)", (email.strip(),)
        ).fetchone()
        if row is None:
            raise ToolError(f"no customer with email {email!r}")
        return {"customer_id": row["id"], "name": row["name"], "address": row["address"], "tier": row["tier"]}

    def _tool_list_orders(self, customer_id: str, status: str | None = None) -> dict:
        if not self._customer_exists(customer_id):
            raise ToolError(f"no customer with id {customer_id!r}")
        sql = "SELECT * FROM orders WHERE customer_id = ?"
        params: list[Any] = [customer_id]
        if status is not None:
            sql += " AND status = ?"
            params.append(status)
        sql += " ORDER BY placed_on DESC, id"
        rows = self.conn.execute(sql, params).fetchall()
        return {
            "orders": [
                {"order_id": r["id"], "status": r["status"], "total": r["total"], "item": r["item"],
                 "placed_on": r["placed_on"]}
                for r in rows
            ]
        }

    def _tool_get_order(self, order_id: str) -> dict:
        row = self.conn.execute("SELECT * FROM orders WHERE id = ?", (order_id,)).fetchone()
        if row is None:
            raise ToolError(f"no order with id {order_id!r}")
        refunded = self.conn.execute(
            "SELECT COALESCE(SUM(amount), 0) AS a FROM refunds WHERE order_id = ?", (order_id,)
        ).fetchone()["a"]
        return {
            "order_id": row["id"], "customer_id": row["customer_id"], "status": row["status"],
            "total": row["total"], "item": row["item"], "placed_on": row["placed_on"],
            "carrier": row["carrier"], "tracking": row["tracking"], "refunded_so_far": round(refunded, 2),
        }

    def _tool_issue_refund(self, order_id: str, amount: float, reason: str) -> dict:
        row = self.conn.execute("SELECT * FROM orders WHERE id = ?", (order_id,)).fetchone()
        if row is None:
            raise ToolError(f"no order with id {order_id!r}")
        if reason not in REFUND_REASONS:
            raise ToolError(f"reason must be one of {REFUND_REASONS}, got {reason!r}")
        if row["status"] not in ("shipped", "delivered"):
            raise ToolError(
                f"order {order_id} is {row['status']}, and only shipped or delivered orders can be refunded"
            )
        already = self.conn.execute(
            "SELECT COUNT(*) AS n FROM refunds WHERE order_id = ?", (order_id,)
        ).fetchone()["n"]
        if already:
            raise ToolError(f"order {order_id} has already been refunded")
        if amount <= 0:
            raise ToolError("refund amount must be positive")
        if round(amount, 2) > round(row["total"], 2):
            raise ToolError(f"refund of {amount} exceeds the order total of {row['total']}")
        refund_id = f"r_{order_id}_{already + 1}"
        self.conn.execute(
            "INSERT INTO refunds (id, order_id, amount, reason) VALUES (?,?,?,?)",
            (refund_id, order_id, round(float(amount), 2), reason),
        )
        self.conn.commit()
        return {"ok": True, "refund_id": refund_id, "amount": round(float(amount), 2), "order_id": order_id}

    def _tool_update_address(self, customer_id: str, address: str) -> dict:
        if not self._customer_exists(customer_id):
            raise ToolError(f"no customer with id {customer_id!r}")
        shipped = self.conn.execute(
            "SELECT id FROM orders WHERE customer_id = ? AND status = 'shipped'", (customer_id,)
        ).fetchall()
        if shipped:
            raise ToolError(
                f"order {shipped[0]['id']} has already shipped and cannot be redirected; "
                f"the address was not changed"
            )
        self.conn.execute("UPDATE customers SET address = ? WHERE id = ?", (address.strip(), customer_id))
        self.conn.commit()
        return {"ok": True, "customer_id": customer_id, "address": address.strip()}

    def _tool_create_ticket(self, customer_id: str, category: str, summary: str) -> dict:
        if not self._customer_exists(customer_id):
            raise ToolError(f"no customer with id {customer_id!r}")
        if category not in TICKET_CATEGORIES:
            raise ToolError(f"category must be one of {TICKET_CATEGORIES}, got {category!r}")
        n = self.conn.execute("SELECT COUNT(*) AS n FROM tickets").fetchone()["n"]
        ticket_id = f"t_{n + 1}"
        self.conn.execute(
            "INSERT INTO tickets (id, customer_id, category, summary) VALUES (?,?,?,?)",
            (ticket_id, customer_id, category, summary),
        )
        self.conn.commit()
        return {"ok": True, "ticket_id": ticket_id}

    def _customer_exists(self, customer_id: str) -> bool:
        return self.conn.execute("SELECT 1 FROM customers WHERE id = ?", (customer_id,)).fetchone() is not None

    # ----------------------------------------------------------------------------------------------------------
    # state, for predicates
    # ----------------------------------------------------------------------------------------------------------

    def customers(self) -> list[dict]:
        return [dict(r) for r in self.conn.execute("SELECT * FROM customers ORDER BY id")]

    def orders(self) -> list[dict]:
        return [dict(r) for r in self.conn.execute("SELECT * FROM orders ORDER BY id")]

    def refunds(self) -> list[dict]:
        return [dict(r) for r in self.conn.execute("SELECT * FROM refunds ORDER BY id")]

    def tickets(self) -> list[dict]:
        return [dict(r) for r in self.conn.execute("SELECT * FROM tickets ORDER BY id")]

    def customer(self, customer_id: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM customers WHERE id = ?", (customer_id,)).fetchone()
        return dict(row) if row else None

    def order(self, order_id: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM orders WHERE id = ?", (order_id,)).fetchone()
        return dict(row) if row else None

    def state(self) -> dict:
        """The whole database, for debugging and for state comparisons in tests."""
        return {
            "customers": self.customers(),
            "orders": self.orders(),
            "refunds": self.refunds(),
            "tickets": self.tickets(),
        }

    def state_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.state(), sort_keys=True).encode()).hexdigest()[:16]
