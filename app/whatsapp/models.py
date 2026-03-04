from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID

from pydantic import BaseModel


# ── Enums ──────────────────────────────────────────────────────────────────────

class TriggerType(str, Enum):
    delivery = "delivery"
    scheduled = "scheduled"


class MessageStatus(str, Enum):
    pending = "pending"
    sent = "sent"
    failed = "failed"


# ── Config ─────────────────────────────────────────────────────────────────────

class WhatsappConfigUpsert(BaseModel):
    phone_number_id: str
    access_token: str
    is_active: bool = True


class WhatsappConfigOut(BaseModel):
    id: UUID
    company_id: UUID
    phone_number_id: str
    is_active: bool
    created_at: datetime
    updated_at: datetime


# ── Templates ──────────────────────────────────────────────────────────────────

class TemplateCreate(BaseModel):
    name: str
    body: str
    trigger_type: TriggerType
    is_active: bool = True


class TemplateUpdate(BaseModel):
    name: Optional[str] = None
    body: Optional[str] = None
    trigger_type: Optional[TriggerType] = None
    is_active: Optional[bool] = None


class TemplateOut(BaseModel):
    id: UUID
    company_id: UUID
    name: str
    body: str
    trigger_type: TriggerType
    is_active: bool
    created_at: datetime
    updated_at: datetime


# ── Service Rules ──────────────────────────────────────────────────────────────

class ServiceRuleCreate(BaseModel):
    service_id: UUID
    template_id: UUID
    delay_days: int = 30
    is_active: bool = True


class ServiceRuleOut(BaseModel):
    id: UUID
    company_id: UUID
    service_id: UUID
    template_id: UUID
    delay_days: int
    is_active: bool
    created_at: datetime
    updated_at: datetime


# ── Scheduled Messages ─────────────────────────────────────────────────────────

class ScheduledMessageOut(BaseModel):
    id: UUID
    company_id: UUID
    order_id: UUID
    phone: str
    template_id: UUID
    scheduled_at: datetime
    sent_at: Optional[datetime]
    status: MessageStatus
    error: Optional[str]
    created_at: datetime


# ── Message Log ────────────────────────────────────────────────────────────────

class MessageLogOut(BaseModel):
    id: UUID
    company_id: UUID
    order_id: Optional[UUID]
    phone: str
    template_body: str
    sent_at: Optional[datetime]
    status: MessageStatus
    meta_message_id: Optional[str]
    error: Optional[str]
    created_at: datetime


# ── Webhook ────────────────────────────────────────────────────────────────────

class WebhookPayload(BaseModel):
    type: str
    record: dict
    old_record: Optional[dict] = None
