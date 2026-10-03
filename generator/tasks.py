"""The three demo task types and their tool registries.

Each task is a goal sentence plus an ordered plan of tool calls. Tool arguments
are sourced from upstream state wherever possible, which is what makes
`arg_novelty` a real signal: on a healthy run almost every argument token has
already appeared in some earlier step's output, and on a hallucinated_argument
run it has not.

Every tool result carries a `summary` string. Embeddings are taken over that
text, so `semantic_deviation` measures whether a step's output is actually on
topic for the run's goal.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

ArgsFn = Callable[[dict[str, Any], random.Random], dict[str, Any]]
ResultFn = Callable[[dict[str, Any], dict[str, Any], random.Random], dict[str, Any]]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    args_fn: ArgsFn
    result_fn: ResultFn
    # Declared output types. schema_violation breaks one of these on purpose.
    output_schema: dict[str, str]
    # Typical latency band in ms, before any fault or anomaly scaling.
    latency: tuple[int, int] = (120, 700)
    # Fields merged into agent state after a successful call.
    state_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class TaskType:
    name: str
    goal: str
    plan: tuple[str, ...]
    tools: dict[str, ToolSpec]
    distractors: dict[str, ToolSpec]

    @property
    def terminal_tool(self) -> str:
        return self.plan[-1]

    def tool(self, name: str) -> ToolSpec:
        return self.tools.get(name) or self.distractors[name]


def _pick(rng: random.Random, seq: list[Any]) -> Any:
    return seq[rng.randrange(len(seq))]


# ---------------------------------------------------------------------------
# Task 1: travel_booking
# ---------------------------------------------------------------------------

_AIRLINES = ["LH", "TP", "FR", "KL", "AF"]


def _search_flights_args(state, rng):
    return {"origin": "BER", "destination": "LIS", "date": "2026-03-12", "passengers": 1}


def _search_flights_result(args, state, rng):
    offers = []
    for _ in range(rng.randint(3, 5)):
        offers.append(
            {
                "flight_no": f"{_pick(rng, _AIRLINES)}{rng.randint(100, 999)}",
                "price": round(rng.uniform(96.0, 412.0), 2),
                "currency": _pick(rng, ["EUR", "USD", "GBP"]),
                "refundable": rng.random() < 0.5,
            }
        )
    cheapest = min(offers, key=lambda o: o["price"])
    return {
        "offers": offers,
        "count": len(offers),
        "cheapest_flight_no": cheapest["flight_no"],
        "cheapest_price": cheapest["price"],
        "currency": cheapest["currency"],
        "summary": (
            f"Found {len(offers)} flights from BER to LIS on 2026-03-12. "
            f"Cheapest is {cheapest['flight_no']} at {cheapest['price']} {cheapest['currency']}."
        ),
    }


def _fare_rules_args(state, rng):
    return {"flight_no": state.get("cheapest_flight_no", "LH404")}


def _fare_rules_result(args, state, rng):
    return {
        "flight_no": args["flight_no"],
        "refundable": True,
        "change_fee": round(rng.uniform(0.0, 60.0), 2),
        "summary": (
            f"Fare rules for {args['flight_no']}: refundable, "
            f"change fee {round(rng.uniform(0, 60), 2)} EUR."
        ),
    }


def _convert_args(state, rng):
    return {
        "amount": state.get("cheapest_price", 180.0),
        "from_currency": state.get("currency", "USD"),
        "to_currency": "EUR",
    }


def _convert_result(args, state, rng):
    rate = round(rng.uniform(0.86, 1.09), 4)
    amount = round(float(args["amount"]) * rate, 2)
    return {
        "amount": amount,
        "currency": "EUR",
        "rate": rate,
        "quoted_at": "2026-03-10T09:14:00Z",
        "summary": (
            f"Converted {args['amount']} {args['from_currency']} to {amount} EUR "
            f"at rate {rate}."
        ),
    }


def _seats_args(state, rng):
    return {"flight_no": state.get("cheapest_flight_no", "LH404"), "cabin": "economy"}


def _seats_result(args, state, rng):
    seats = rng.randint(2, 28)
    return {
        "flight_no": args["flight_no"],
        "seats_left": seats,
        "cabin": args["cabin"],
        "summary": f"{seats} economy seats remain on {args['flight_no']}.",
    }


def _book_args(state, rng):
    return {
        "flight_no": state.get("cheapest_flight_no", "LH404"),
        "passenger": "A. Moreau",
        "total_eur": state.get("amount", 164.20),
    }


def _book_result(args, state, rng):
    ref = "".join(_pick(rng, list("ABCDEFGHJKLMNPQRSTUVWXYZ23456789")) for _ in range(6))
    return {
        "booking_ref": ref,
        "status": "confirmed",
        "total_eur": args["total_eur"],
        "summary": (
            f"Booked {args['flight_no']} for {args['passenger']}, "
            f"reference {ref}, total {args['total_eur']} EUR."
        ),
    }


TRAVEL = TaskType(
    name="travel_booking",
    goal=(
        "Book the cheapest refundable flight from Berlin to Lisbon on 12 March "
        "for one passenger and report the total price in euros."
    ),
    plan=("search_flights", "get_fare_rules", "convert_currency", "check_seat_availability", "book_flight"),
    tools={
        "search_flights": ToolSpec(
            "search_flights", _search_flights_args, _search_flights_result,
            {"offers": "list", "count": "int", "cheapest_price": "float", "currency": "str"},
            (280, 900), ("cheapest_flight_no", "cheapest_price", "currency"),
        ),
        "get_fare_rules": ToolSpec(
            "get_fare_rules", _fare_rules_args, _fare_rules_result,
            {"flight_no": "str", "refundable": "bool", "change_fee": "float"},
            (120, 420), ("refundable",),
        ),
        "convert_currency": ToolSpec(
            "convert_currency", _convert_args, _convert_result,
            {"amount": "float", "currency": "str", "rate": "float", "quoted_at": "str"},
            (90, 300), ("amount", "rate"),
        ),
        "check_seat_availability": ToolSpec(
            "check_seat_availability", _seats_args, _seats_result,
            {"flight_no": "str", "seats_left": "int"},
            (110, 380), ("seats_left",),
        ),
        "book_flight": ToolSpec(
            "book_flight", _book_args, _book_result,
            {"booking_ref": "str", "status": "str", "total_eur": "float"},
            (400, 1200), ("booking_ref", "status"),
        ),
    },
    distractors={
        "search_hotels": ToolSpec(
            "search_hotels",
            lambda s, r: {"city": "Lisbon", "checkin": "2026-03-12", "nights": 2},
            lambda a, s, r: {
                "properties": [{"name": "Baixa Suites", "nightly_eur": round(r.uniform(70, 240), 2)}],
                "count": r.randint(1, 9),
                "summary": "Returned hotel availability in Lisbon for two nights.",
            },
            {"properties": "list", "count": "int"}, (200, 700),
        ),
        "get_weather": ToolSpec(
            "get_weather",
            lambda s, r: {"city": "Lisbon", "date": "2026-03-12"},
            lambda a, s, r: {
                "temp_c": round(r.uniform(9, 23), 1), "conditions": "partly cloudy",
                "summary": "Returned the Lisbon weather forecast for 12 March.",
            },
            {"temp_c": "float", "conditions": "str"}, (80, 260),
        ),
    },
)


# ---------------------------------------------------------------------------
# Task 2: invoice_reconciliation
# ---------------------------------------------------------------------------


def _invoices_args(state, rng):
    return {"vendor": "Northwind Supply", "period": "2026-03"}


def _invoices_result(args, state, rng):
    invoices = [
        {
            "invoice_id": f"INV-{rng.randint(41000, 41999)}",
            "amount": round(rng.uniform(220.0, 4800.0), 2),
            "currency": "EUR",
            "issued_at": f"2026-03-{rng.randint(1, 28):02d}",
        }
        for _ in range(rng.randint(3, 6))
    ]
    total = round(sum(i["amount"] for i in invoices), 2)
    return {
        "invoices": invoices,
        "count": len(invoices),
        "invoice_total": total,
        "first_invoice_id": invoices[0]["invoice_id"],
        "summary": (
            f"Fetched {len(invoices)} March invoices for Northwind Supply "
            f"totalling {total} EUR."
        ),
    }


def _vendor_args(state, rng):
    return {"vendor": "Northwind Supply"}


def _vendor_result(args, state, rng):
    vid = f"VEND-{rng.randint(300, 399)}"
    return {
        "vendor_id": vid,
        "name": "Northwind Supply",
        "tax_id": f"DE{rng.randint(10**8, 10**9 - 1)}",
        "payment_terms": "net 30",
        "summary": f"Vendor Northwind Supply resolved to {vid}, payment terms net 30.",
    }


def _payments_args(state, rng):
    return {"vendor_id": state.get("vendor_id", "VEND-301"), "period": "2026-03"}


def _payments_result(args, state, rng):
    payments = [
        {
            "payment_id": f"PAY-{rng.randint(7000, 7999)}",
            "amount": round(rng.uniform(200.0, 4600.0), 2),
            "paid_at": f"2026-03-{rng.randint(1, 28):02d}",
        }
        for _ in range(rng.randint(2, 5))
    ]
    total = round(sum(p["amount"] for p in payments), 2)
    return {
        "payments": payments,
        "count": len(payments),
        "payment_total": total,
        "summary": (
            f"Fetched {len(payments)} payments for {args['vendor_id']} in March "
            f"totalling {total} EUR."
        ),
    }


def _match_args(state, rng):
    return {
        "invoice_total": state.get("invoice_total", 0.0),
        "payment_total": state.get("payment_total", 0.0),
        "vendor_id": state.get("vendor_id", "VEND-301"),
    }


def _match_result(args, state, rng):
    delta = round(float(args["invoice_total"]) - float(args["payment_total"]), 2)
    return {
        "matched": rng.randint(1, 4),
        "unmatched": [f"INV-{rng.randint(41000, 41999)}"] if abs(delta) > 50 else [],
        "delta": delta,
        "summary": (
            f"Reconciled invoices against payments for {args['vendor_id']}: "
            f"delta {delta} EUR."
        ),
    }


def _post_args(state, rng):
    return {
        "vendor_id": state.get("vendor_id", "VEND-301"),
        "delta": state.get("delta", 0.0),
        "note": "March reconciliation",
    }


def _post_result(args, state, rng):
    rid = f"REC-{rng.randint(900, 999)}"
    return {
        "report_id": rid,
        "status": "posted",
        "delta": args["delta"],
        "summary": (
            f"Posted reconciliation {rid} for {args['vendor_id']} "
            f"with a discrepancy of {args['delta']} EUR."
        ),
    }


INVOICE = TaskType(
    name="invoice_reconciliation",
    goal=(
        "Reconcile the March invoices for vendor Northwind Supply against "
        "recorded payments and report any discrepancy over fifty euros."
    ),
    plan=("fetch_invoices", "lookup_vendor", "fetch_payments", "match_records", "post_reconciliation"),
    tools={
        "fetch_invoices": ToolSpec(
            "fetch_invoices", _invoices_args, _invoices_result,
            {"invoices": "list", "count": "int", "invoice_total": "float"},
            (240, 860), ("invoice_total", "first_invoice_id"),
        ),
        "lookup_vendor": ToolSpec(
            "lookup_vendor", _vendor_args, _vendor_result,
            {"vendor_id": "str", "name": "str", "tax_id": "str", "payment_terms": "str"},
            (100, 340), ("vendor_id",),
        ),
        "fetch_payments": ToolSpec(
            "fetch_payments", _payments_args, _payments_result,
            {"payments": "list", "count": "int", "payment_total": "float"},
            (220, 780), ("payment_total",),
        ),
        "match_records": ToolSpec(
            "match_records", _match_args, _match_result,
            {"matched": "int", "unmatched": "list", "delta": "float"},
            (150, 520), ("delta",),
        ),
        "post_reconciliation": ToolSpec(
            "post_reconciliation", _post_args, _post_result,
            {"report_id": "str", "status": "str", "delta": "float"},
            (380, 1100), ("report_id", "status"),
        ),
    },
    distractors={
        "fetch_purchase_orders": ToolSpec(
            "fetch_purchase_orders",
            lambda s, r: {"vendor": "Northwind Supply", "period": "2026-03"},
            lambda a, s, r: {
                "purchase_orders": [{"po_id": f"PO-{r.randint(500, 599)}"} for _ in range(r.randint(1, 4))],
                "count": r.randint(1, 4),
                "summary": "Returned open purchase orders, which are not payments.",
            },
            {"purchase_orders": "list", "count": "int"}, (210, 690),
        ),
        "export_ledger": ToolSpec(
            "export_ledger",
            lambda s, r: {"period": "2026-03", "format": "csv"},
            lambda a, s, r: {
                "export_id": f"EXP-{r.randint(100, 199)}", "rows": r.randint(400, 9000),
                "summary": "Exported the full general ledger for March as CSV.",
            },
            {"export_id": "str", "rows": "int"}, (600, 1800),
        ),
    },
)


# ---------------------------------------------------------------------------
# Task 3: support_triage
# ---------------------------------------------------------------------------


def _ticket_args(state, rng):
    return {"ticket_id": 4821}


def _ticket_result(args, state, rng):
    cid = f"CUST-{rng.randint(2000, 2999)}"
    return {
        "ticket_id": 4821,
        "subject": "Scheduled exports stopped running",
        "body": "Our nightly CSV export has not produced a file since Tuesday.",
        "customer_id": cid,
        "severity": _pick(rng, ["low", "normal", "high"]),
        "feature": "scheduled_exports",
        "summary": (
            f"Ticket 4821 from {cid}: scheduled CSV exports have not run since Tuesday."
        ),
    }


def _kb_args(state, rng):
    return {"query": "scheduled exports not running"}


def _kb_result(args, state, rng):
    articles = [
        {"article_id": f"KB-{rng.randint(100, 199)}", "title": "Why scheduled exports stall",
         "score": round(rng.uniform(0.61, 0.95), 3)}
        for _ in range(rng.randint(2, 4))
    ]
    best = max(articles, key=lambda a: a["score"])
    return {
        "articles": articles,
        "count": len(articles),
        "top_article_id": best["article_id"],
        "summary": (
            f"Knowledge base returned {len(articles)} articles on stalled scheduled "
            f"exports, best match {best['article_id']}."
        ),
    }


def _account_args(state, rng):
    return {"customer_id": state.get("customer_id", "CUST-2100")}


def _account_result(args, state, rng):
    plan = _pick(rng, ["starter", "growth", "scale"])
    return {
        "customer_id": args["customer_id"],
        "plan": plan,
        "seats": rng.randint(3, 120),
        "status": "active",
        "renewed_at": "2026-01-18",
        "summary": f"Account {args['customer_id']} is active on the {plan} plan.",
    }


def _entitlement_args(state, rng):
    return {
        "customer_id": state.get("customer_id", "CUST-2100"),
        "feature": state.get("feature", "scheduled_exports"),
    }


def _entitlement_result(args, state, rng):
    return {
        "feature": args["feature"],
        "entitled": True,
        "plan": state.get("plan", "growth"),
        "summary": (
            f"Confirmed {args['customer_id']} is entitled to {args['feature']} "
            f"on the {state.get('plan', 'growth')} plan."
        ),
    }


def _reply_args(state, rng):
    return {
        "ticket_id": 4821,
        "body": "Your plan covers scheduled exports. We have requeued the stalled job.",
    }


def _reply_result(args, state, rng):
    mid = f"MSG-{rng.randint(80000, 89999)}"
    return {
        "message_id": mid,
        "status": "sent",
        "ticket_id": args["ticket_id"],
        "summary": f"Sent resolution reply {mid} on ticket {args['ticket_id']}.",
    }


SUPPORT = TaskType(
    name="support_triage",
    goal=(
        "Triage support ticket 4821, confirm the customer's plan covers the "
        "failing scheduled export feature, and send a resolution reply."
    ),
    plan=("fetch_ticket", "search_kb", "get_account_status", "check_feature_entitlement", "send_reply"),
    tools={
        "fetch_ticket": ToolSpec(
            "fetch_ticket", _ticket_args, _ticket_result,
            {"ticket_id": "int", "subject": "str", "customer_id": "str", "severity": "str"},
            (110, 380), ("customer_id", "feature"),
        ),
        "search_kb": ToolSpec(
            "search_kb", _kb_args, _kb_result,
            {"articles": "list", "count": "int", "top_article_id": "str"},
            (200, 740), ("top_article_id",),
        ),
        "get_account_status": ToolSpec(
            "get_account_status", _account_args, _account_result,
            {"customer_id": "str", "plan": "str", "seats": "int", "status": "str"},
            (130, 450), ("plan", "status"),
        ),
        "check_feature_entitlement": ToolSpec(
            "check_feature_entitlement", _entitlement_args, _entitlement_result,
            {"feature": "str", "entitled": "bool", "plan": "str"},
            (100, 320), ("entitled",),
        ),
        "send_reply": ToolSpec(
            "send_reply", _reply_args, _reply_result,
            {"message_id": "str", "status": "str", "ticket_id": "int"},
            (320, 1000), ("message_id", "status"),
        ),
    },
    distractors={
        "escalate_to_engineering": ToolSpec(
            "escalate_to_engineering",
            lambda s, r: {"ticket_id": 4821, "team": "platform"},
            lambda a, s, r: {
                "escalation_id": f"ESC-{r.randint(600, 699)}", "queue": "platform",
                "summary": "Escalated the ticket to the platform engineering queue.",
            },
            {"escalation_id": "str", "queue": "str"}, (180, 620),
        ),
        "issue_refund": ToolSpec(
            "issue_refund",
            lambda s, r: {"customer_id": s.get("customer_id", "CUST-2100"), "amount": 49.0},
            lambda a, s, r: {
                "refund_id": f"REF-{r.randint(400, 499)}", "amount": a["amount"], "status": "issued",
                "summary": "Issued a account credit refund to the customer.",
            },
            {"refund_id": "str", "amount": "float", "status": "str"}, (300, 950),
        ),
    },
)


TASK_TYPES: tuple[TaskType, ...] = (TRAVEL, INVOICE, SUPPORT)
TASKS_BY_NAME: dict[str, TaskType] = {t.name: t for t in TASK_TYPES}
