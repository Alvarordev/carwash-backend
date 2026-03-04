from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from app.db import get_client
from app.whatsapp.meta import send_text_message

logger = logging.getLogger(__name__)


# ── Template rendering ──────────────────────────────────────────────────────────

def render_template(body: str, order: dict, customer: dict, items: list[dict]) -> str:
    service_list = ", ".join(item["service_name"] for item in items)
    vehicle_plate = (order.get("vehicle") or {}).get("plate", "")
    return (
        body
        .replace("{{customer_name}}", f"{customer.get('first_name', '')} {customer.get('last_name', '')}".strip())
        .replace("{{order_number}}", order.get("order_number", ""))
        .replace("{{vehicle_plate}}", vehicle_plate)
        .replace("{{service_list}}", service_list)
    )


# ── Fetch helpers ───────────────────────────────────────────────────────────────

def _fetch_order_details(order_id: str) -> dict | None:
    """Fetch order with vehicle and customer joined."""
    db = get_client()
    res = (
        db.table("orders")
        .select("*, vehicle:vehicles(plate), customer:customers(id, first_name, last_name, phone)")
        .eq("id", order_id)
        .single()
        .execute()
    )
    return res.data


def _fetch_order_items(order_id: str) -> list[dict]:
    db = get_client()
    res = db.table("order_items").select("service_id, service_name").eq("order_id", order_id).execute()
    return res.data or []


def _fetch_whatsapp_config(company_id: str) -> dict | None:
    db = get_client()
    res = (
        db.table("whatsapp_config")
        .select("*")
        .eq("company_id", company_id)
        .eq("is_active", True)
        .maybe_single()
        .execute()
    )
    return res.data


def _fetch_delivery_template(company_id: str) -> dict | None:
    db = get_client()
    res = (
        db.table("whatsapp_templates")
        .select("*")
        .eq("company_id", company_id)
        .eq("trigger_type", "delivery")
        .eq("is_active", True)
        .limit(1)
        .execute()
    )
    return res.data[0] if res.data else None


def _fetch_service_rules(company_id: str, service_ids: list[str]) -> list[dict]:
    if not service_ids:
        return []
    db = get_client()
    res = (
        db.table("whatsapp_service_rules")
        .select("*")
        .eq("company_id", company_id)
        .eq("is_active", True)
        .in_("service_id", service_ids)
        .execute()
    )
    return res.data or []


# ── Log helpers ─────────────────────────────────────────────────────────────────

def _log_message(
    company_id: str,
    order_id: str | None,
    phone: str,
    template_body: str,
    status: str,
    meta_message_id: str | None = None,
    error: str | None = None,
) -> None:
    db = get_client()
    now = datetime.now(timezone.utc).isoformat()
    db.table("whatsapp_message_log").insert({
        "company_id": company_id,
        "order_id": order_id,
        "phone": phone,
        "template_body": template_body,
        "sent_at": now if status == "sent" else None,
        "status": status,
        "meta_message_id": meta_message_id,
        "error": error,
    }).execute()


# ── Core delivery flow ──────────────────────────────────────────────────────────

async def handle_order_delivered(order_id: str, company_id: str) -> None:
    """Called when an order transitions to 'Entregado'."""
    order = _fetch_order_details(order_id)
    if not order:
        logger.warning("handle_order_delivered: order %s not found", order_id)
        return

    customer = order.get("customer") or {}
    phone = customer.get("phone")
    items = _fetch_order_items(order_id)

    # ── 1. Send immediate delivery message ────────────────────────────────────
    if phone:
        config = _fetch_whatsapp_config(company_id)
        template = _fetch_delivery_template(company_id)

        if config and template:
            message_body = render_template(template["body"], order, customer, items)
            try:
                result = await send_text_message(
                    phone_number_id=config["phone_number_id"],
                    access_token=config["access_token"],
                    to=phone,
                    body=message_body,
                )
                meta_id = (result.get("messages") or [{}])[0].get("id")
                _log_message(company_id, order_id, phone, message_body, "sent", meta_message_id=meta_id)
                logger.info("Delivery WhatsApp sent to %s for order %s", phone, order_id)
            except Exception as exc:
                _log_message(company_id, order_id, phone, message_body, "failed", error=str(exc))
                logger.error("Failed to send WhatsApp to %s: %s", phone, exc)
        else:
            logger.info("No WhatsApp config or delivery template for company %s — skipping delivery message", company_id)
    else:
        logger.info("Order %s has no customer phone — skipping delivery message", order_id)

    # ── 2. Schedule follow-up messages ────────────────────────────────────────
    service_ids = [item["service_id"] for item in items if item.get("service_id")]
    rules = _fetch_service_rules(company_id, service_ids)
    if not rules:
        return

    # Use order updated_at as base, fall back to now
    base_dt_str = order.get("updated_at") or datetime.now(timezone.utc).isoformat()
    try:
        base_dt = datetime.fromisoformat(base_dt_str.replace("Z", "+00:00"))
    except ValueError:
        base_dt = datetime.now(timezone.utc)

    effective_phone = phone or ""
    if not effective_phone:
        logger.info("Order %s has no phone — scheduled messages require a phone number, skipping", order_id)
        return

    db = get_client()
    rows = []
    for rule in rules:
        scheduled_at = (base_dt + timedelta(days=rule["delay_days"])).isoformat()
        rows.append({
            "company_id": company_id,
            "order_id": order_id,
            "phone": effective_phone,
            "template_id": rule["template_id"],
            "scheduled_at": scheduled_at,
            "status": "pending",
        })

    if rows:
        db.table("whatsapp_scheduled_messages").insert(rows).execute()
        logger.info("Scheduled %d follow-up message(s) for order %s", len(rows), order_id)


# ── Process scheduled message queue ────────────────────────────────────────────

async def process_scheduled_messages() -> dict:
    """Send all pending scheduled messages where scheduled_at <= now()."""
    db = get_client()
    now = datetime.now(timezone.utc).isoformat()

    res = (
        db.table("whatsapp_scheduled_messages")
        .select("*, template:whatsapp_templates(body)")
        .eq("status", "pending")
        .lte("scheduled_at", now)
        .execute()
    )
    pending = res.data or []

    sent_count = 0
    failed_count = 0

    for msg in pending:
        company_id = msg["company_id"]
        order_id = msg["order_id"]
        phone = msg["phone"]
        template = msg.get("template") or {}
        template_body_raw = template.get("body", "")

        config = _fetch_whatsapp_config(company_id)
        if not config:
            _mark_scheduled(db, msg["id"], "failed", error="No active WhatsApp config")
            failed_count += 1
            continue

        # Render template with order details
        order = _fetch_order_details(order_id) or {}
        customer = order.get("customer") or {}
        items = _fetch_order_items(order_id)
        message_body = render_template(template_body_raw, order, customer, items)

        try:
            result = await send_text_message(
                phone_number_id=config["phone_number_id"],
                access_token=config["access_token"],
                to=phone,
                body=message_body,
            )
            meta_id = (result.get("messages") or [{}])[0].get("id")
            _mark_scheduled(db, msg["id"], "sent")
            _log_message(company_id, order_id, phone, message_body, "sent", meta_message_id=meta_id)
            sent_count += 1
        except Exception as exc:
            err = str(exc)
            _mark_scheduled(db, msg["id"], "failed", error=err)
            _log_message(company_id, order_id, phone, message_body, "failed", error=err)
            failed_count += 1
            logger.error("Scheduled message %s failed: %s", msg["id"], exc)

    return {"processed": len(pending), "sent": sent_count, "failed": failed_count}


def _mark_scheduled(db, msg_id: str, status: str, error: str | None = None) -> None:
    now = datetime.now(timezone.utc).isoformat()
    update: dict = {"status": status}
    if status == "sent":
        update["sent_at"] = now
    if error:
        update["error"] = error
    db.table("whatsapp_scheduled_messages").update(update).eq("id", msg_id).execute()
