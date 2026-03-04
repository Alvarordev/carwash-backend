from __future__ import annotations

import logging
import os
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status

from app.auth import verify_jwt
from app.db import get_client
from app.whatsapp.models import (
    MessageLogOut,
    ScheduledMessageOut,
    ServiceRuleCreate,
    ServiceRuleOut,
    TemplateCreate,
    TemplateOut,
    TemplateUpdate,
    WebhookPayload,
    WhatsappConfigOut,
    WhatsappConfigUpsert,
)
from app.whatsapp.service import handle_order_delivered, process_scheduled_messages

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Helper: extract company_id from JWT payload ────────────────────────────────

def _company_id_from_jwt(payload: dict) -> str:
    # Supabase stores the user's app_metadata or user_metadata with company_id
    company_id = (
        (payload.get("app_metadata") or {}).get("company_id")
        or (payload.get("user_metadata") or {}).get("company_id")
    )
    if not company_id:
        raise HTTPException(status_code=400, detail="company_id not found in JWT claims")
    return str(company_id)


# ══════════════════════════════════════════════════════════════════════════════
# Webhook (called by Supabase Database Webhooks — no JWT, uses secret header)
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/webhooks/order-status-changed", status_code=200)
async def order_status_changed(
    payload: WebhookPayload,
    x_webhook_secret: Optional[str] = Header(None),
):
    expected_secret = os.environ.get("WEBHOOK_SECRET", "")
    if not expected_secret or x_webhook_secret != expected_secret:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid webhook secret")

    if payload.type != "UPDATE":
        return {"detail": "ignored"}

    record = payload.record
    if record.get("status") != "Entregado":
        return {"detail": "ignored"}

    order_id = record.get("id")
    company_id = record.get("company_id")
    if not order_id or not company_id:
        raise HTTPException(status_code=400, detail="Missing order id or company_id in payload")

    await handle_order_delivered(order_id, company_id)
    return {"detail": "processed"}


# ══════════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/whatsapp/config", response_model=WhatsappConfigOut)
def get_config(payload: dict = Depends(verify_jwt)):
    company_id = _company_id_from_jwt(payload)
    db = get_client()
    res = db.table("whatsapp_config").select("*").eq("company_id", company_id).maybe_single().execute()
    if not res.data:
        raise HTTPException(status_code=404, detail="No WhatsApp config found")
    return res.data


@router.post("/whatsapp/config", response_model=WhatsappConfigOut)
def upsert_config(body: WhatsappConfigUpsert, payload: dict = Depends(verify_jwt)):
    company_id = _company_id_from_jwt(payload)
    db = get_client()
    data = {
        "company_id": company_id,
        "phone_number_id": body.phone_number_id,
        "access_token": body.access_token,
        "is_active": body.is_active,
        "updated_at": "now()",
    }
    res = (
        db.table("whatsapp_config")
        .upsert(data, on_conflict="company_id")
        .execute()
    )
    return res.data[0]


# ══════════════════════════════════════════════════════════════════════════════
# Templates
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/whatsapp/templates", response_model=list[TemplateOut])
def list_templates(payload: dict = Depends(verify_jwt)):
    company_id = _company_id_from_jwt(payload)
    db = get_client()
    res = db.table("whatsapp_templates").select("*").eq("company_id", company_id).execute()
    return res.data or []


@router.post("/whatsapp/templates", response_model=TemplateOut, status_code=201)
def create_template(body: TemplateCreate, payload: dict = Depends(verify_jwt)):
    company_id = _company_id_from_jwt(payload)
    db = get_client()
    res = db.table("whatsapp_templates").insert({
        "company_id": company_id,
        "name": body.name,
        "body": body.body,
        "trigger_type": body.trigger_type.value,
        "is_active": body.is_active,
    }).execute()
    return res.data[0]


@router.put("/whatsapp/templates/{template_id}", response_model=TemplateOut)
def update_template(template_id: UUID, body: TemplateUpdate, payload: dict = Depends(verify_jwt)):
    company_id = _company_id_from_jwt(payload)
    db = get_client()
    updates = body.model_dump(exclude_none=True)
    if "trigger_type" in updates:
        updates["trigger_type"] = updates["trigger_type"].value
    updates["updated_at"] = "now()"
    res = (
        db.table("whatsapp_templates")
        .update(updates)
        .eq("id", str(template_id))
        .eq("company_id", company_id)
        .execute()
    )
    if not res.data:
        raise HTTPException(status_code=404, detail="Template not found")
    return res.data[0]


@router.delete("/whatsapp/templates/{template_id}", status_code=204)
def delete_template(template_id: UUID, payload: dict = Depends(verify_jwt)):
    company_id = _company_id_from_jwt(payload)
    db = get_client()
    db.table("whatsapp_templates").delete().eq("id", str(template_id)).eq("company_id", company_id).execute()


# ══════════════════════════════════════════════════════════════════════════════
# Service Rules
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/whatsapp/service-rules", response_model=list[ServiceRuleOut])
def list_service_rules(payload: dict = Depends(verify_jwt)):
    company_id = _company_id_from_jwt(payload)
    db = get_client()
    res = db.table("whatsapp_service_rules").select("*").eq("company_id", company_id).execute()
    return res.data or []


@router.post("/whatsapp/service-rules", response_model=ServiceRuleOut, status_code=201)
def create_service_rule(body: ServiceRuleCreate, payload: dict = Depends(verify_jwt)):
    company_id = _company_id_from_jwt(payload)
    db = get_client()
    res = db.table("whatsapp_service_rules").insert({
        "company_id": company_id,
        "service_id": str(body.service_id),
        "template_id": str(body.template_id),
        "delay_days": body.delay_days,
        "is_active": body.is_active,
    }).execute()
    return res.data[0]


@router.delete("/whatsapp/service-rules/{rule_id}", status_code=204)
def delete_service_rule(rule_id: UUID, payload: dict = Depends(verify_jwt)):
    company_id = _company_id_from_jwt(payload)
    db = get_client()
    db.table("whatsapp_service_rules").delete().eq("id", str(rule_id)).eq("company_id", company_id).execute()


# ══════════════════════════════════════════════════════════════════════════════
# Scheduled Messages
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/whatsapp/scheduled-messages", response_model=list[ScheduledMessageOut])
def list_scheduled_messages(
    status_filter: Optional[str] = Query(None, alias="status"),
    payload: dict = Depends(verify_jwt),
):
    company_id = _company_id_from_jwt(payload)
    db = get_client()
    query = db.table("whatsapp_scheduled_messages").select("*").eq("company_id", company_id)
    if status_filter:
        query = query.eq("status", status_filter)
    res = query.order("scheduled_at", desc=True).execute()
    return res.data or []


@router.post("/whatsapp/scheduled-messages/process")
async def process_messages(payload: dict = Depends(verify_jwt)):
    # No company scoping — processes all pending messages across companies.
    # Callers should have service-role access (e.g. cron job with a valid JWT).
    result = await process_scheduled_messages()
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Message Log
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/whatsapp/message-log", response_model=list[MessageLogOut])
def list_message_log(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    payload: dict = Depends(verify_jwt),
):
    company_id = _company_id_from_jwt(payload)
    db = get_client()
    offset = (page - 1) * page_size
    res = (
        db.table("whatsapp_message_log")
        .select("*")
        .eq("company_id", company_id)
        .order("created_at", desc=True)
        .range(offset, offset + page_size - 1)
        .execute()
    )
    return res.data or []
