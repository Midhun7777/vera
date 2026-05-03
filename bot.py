#!/usr/bin/env python3
"""
Vera Merchant AI Assistant — Bot Server
=========================================

FastAPI HTTP server implementing the 5-endpoint contract for the
magicpin AI Challenge judge harness.

Endpoints:
    GET  /v1/healthz    — liveness probe
    GET  /v1/metadata   — team identity
    POST /v1/context    — receive context pushes
    POST /v1/tick       — periodic wake-up; bot initiates conversations
    POST /v1/reply      — handle merchant/customer replies

Usage:
    # Set your LLM provider + API key
    export LLM_PROVIDER=gemini
    export LLM_API_KEY=your_key_here

    # Run the server
    uvicorn bot:app --host 0.0.0.0 --port 8080
"""

import os
import time
import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request
from pydantic import BaseModel

from composer import compose
from conversation_handlers import (
    ConversationState,
    handle_reply,
    is_auto_reply,
    is_hostile,
    is_commitment,
)

# ---------------------------------------------------------------------------
# App Init
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Vera — Merchant AI Assistant",
    description="magicpin AI Challenge Bot",
    version="1.0.0",
)

START_TIME = time.time()

# ---------------------------------------------------------------------------
# In-Memory Stores
# ---------------------------------------------------------------------------

# Context store: (scope, context_id) -> {version, payload}
contexts: Dict[tuple, dict] = {}

# Conversation state store: conversation_id -> ConversationState
conversations: Dict[str, ConversationState] = {}

# Suppression keys already used (to prevent duplicate sends)
suppressed_keys: set = set()

# Ended conversation IDs (merchant opted out / auto-reply hell)
ended_conversations: set = set()

# Merchants who opted out (suppress all future sends)
opted_out_merchants: set = set()


# ---------------------------------------------------------------------------
# Helper: Retrieve contexts
# ---------------------------------------------------------------------------

def get_context(scope: str, context_id: str) -> Optional[dict]:
    """Get a context payload by scope and ID."""
    entry = contexts.get((scope, context_id))
    return entry["payload"] if entry else None


def get_category_for_merchant(merchant: dict) -> dict:
    """Look up the CategoryContext for a merchant."""
    cat_slug = merchant.get("category_slug", "")
    return get_context("category", cat_slug) or {}


def count_contexts() -> dict:
    """Count contexts by scope."""
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _) in contexts:
        counts[scope] = counts.get(scope, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Endpoint: GET /v1/healthz
# ---------------------------------------------------------------------------

@app.get("/v1/healthz")
async def healthz():
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START_TIME),
        "contexts_loaded": count_contexts(),
    }


# ---------------------------------------------------------------------------
# Endpoint: GET /v1/metadata
# ---------------------------------------------------------------------------

@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": "Vera AI",
        "team_members": ["Builder"],
        "model": os.getenv("LLM_MODEL", "gemini-2.0-flash"),
        "approach": "LLM composer with trigger-kind dispatch, category-voice enforcement, multi-turn conversation handlers with auto-reply/intent/hostile detection",
        "contact_email": "builder@example.com",
        "version": "1.0.0",
        "submitted_at": datetime.utcnow().isoformat() + "Z",
    }


# ---------------------------------------------------------------------------
# Endpoint: POST /v1/context
# ---------------------------------------------------------------------------

class ContextBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: Dict[str, Any]
    delivered_at: str


@app.post("/v1/context")
async def push_context(body: ContextBody):
    key = (body.scope, body.context_id)

    # Validate scope
    if body.scope not in ("category", "merchant", "customer", "trigger"):
        return {"accepted": False, "reason": "invalid_scope", "details": f"Unknown scope: {body.scope}"}

    # Idempotency check
    current = contexts.get(key)
    if current and current["version"] >= body.version:
        return {
            "accepted": False,
            "reason": "stale_version",
            "current_version": current["version"],
        }

    # Store (atomic replace)
    contexts[key] = {"version": body.version, "payload": body.payload}

    return {
        "accepted": True,
        "ack_id": f"ack_{body.context_id}_v{body.version}",
        "stored_at": datetime.utcnow().isoformat() + "Z",
    }


# ---------------------------------------------------------------------------
# Endpoint: POST /v1/tick
# ---------------------------------------------------------------------------

class TickBody(BaseModel):
    now: str
    available_triggers: List[str] = []


@app.post("/v1/tick")
async def tick(body: TickBody):
    actions = []

    for trg_id in body.available_triggers:
        # Get trigger context
        trg_payload = get_context("trigger", trg_id)
        if not trg_payload:
            continue

        merchant_id = trg_payload.get("merchant_id")
        customer_id = trg_payload.get("customer_id")

        # Skip opted-out merchants
        if merchant_id in opted_out_merchants:
            continue

        # Suppression check
        supp_key = trg_payload.get("suppression_key", "")
        if supp_key and supp_key in suppressed_keys:
            continue

        # Get merchant context
        merchant = get_context("merchant", merchant_id)
        if not merchant:
            continue

        # Get category context
        category = get_category_for_merchant(merchant)
        if not category:
            continue

        # Get customer context (if customer-scoped trigger)
        customer = None
        if customer_id:
            customer = get_context("customer", customer_id)

        # Generate conversation_id
        conv_id = f"conv_{merchant_id}_{trg_id}"

        # Skip if conversation already ended
        if conv_id in ended_conversations:
            continue

        try:
            # Compose the message
            composed = compose(category, merchant, trg_payload, customer)

            if not composed.get("body"):
                continue

            # Build template params from body
            identity = merchant.get("identity", {})
            owner_name = identity.get("owner_first_name", identity.get("name", ""))

            # Determine template name based on trigger kind
            kind = trg_payload.get("kind", "generic")
            scope = trg_payload.get("scope", "merchant")
            if scope == "customer":
                template_name = f"merchant_{kind}_v1"
            else:
                template_name = f"vera_{kind}_v1"

            action = {
                "conversation_id": conv_id,
                "merchant_id": merchant_id,
                "customer_id": customer_id,
                "send_as": composed.get("send_as", "vera"),
                "trigger_id": trg_id,
                "template_name": template_name,
                "template_params": [owner_name, composed["body"][:100], ""],
                "body": composed["body"],
                "cta": composed.get("cta", "open_ended"),
                "suppression_key": composed.get("suppression_key", supp_key),
                "rationale": composed.get("rationale", ""),
            }

            actions.append(action)

            # Mark suppression key as used
            if supp_key:
                suppressed_keys.add(supp_key)

            # Initialize conversation state
            state = ConversationState(conv_id, merchant_id)
            state.add_bot_turn(composed["body"], "send")
            conversations[conv_id] = state

        except Exception as e:
            # Log error but don't crash — return empty action for this trigger
            print(f"[ERROR] Compose failed for {trg_id}: {e}")
            continue

    return {"actions": actions}


# ---------------------------------------------------------------------------
# Endpoint: POST /v1/reply
# ---------------------------------------------------------------------------

class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: str
    turn_number: int


@app.post("/v1/reply")
async def reply(body: ReplyBody):
    conv_id = body.conversation_id
    merchant_id = body.merchant_id or ""

    # Get or create conversation state
    if conv_id not in conversations:
        conversations[conv_id] = ConversationState(conv_id, merchant_id)

    state = conversations[conv_id]

    # Check if conversation already ended
    if state.ended or conv_id in ended_conversations:
        return {
            "action": "end",
            "rationale": "Conversation was previously ended."
        }

    # Get merchant and category contexts
    merchant = get_context("merchant", merchant_id) or {}
    category = get_category_for_merchant(merchant) if merchant else {}

    # Handle the reply
    result = handle_reply(
        state=state,
        merchant_message=body.message,
        merchant=merchant,
        category=category,
        turn_number=body.turn_number,
    )

    # Track ended conversations
    if result.get("action") == "end":
        state.ended = True
        ended_conversations.add(conv_id)

    # Track opt-outs
    if is_hostile(body.message):
        opted_out_merchants.add(merchant_id)

    return result


# ---------------------------------------------------------------------------
# Optional: POST /v1/teardown (wipe state)
# ---------------------------------------------------------------------------

@app.post("/v1/teardown")
async def teardown():
    """Wipe all state — called at end of test."""
    contexts.clear()
    conversations.clear()
    suppressed_keys.clear()
    ended_conversations.clear()
    opted_out_merchants.clear()
    return {"status": "wiped"}


# ---------------------------------------------------------------------------
# Run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    print(f"Starting Vera bot on port {port}...")
    print(f"LLM Provider: {os.getenv('LLM_PROVIDER', 'gemini')}")
    print(f"LLM Model: {os.getenv('LLM_MODEL', 'auto')}")
    uvicorn.run(app, host="0.0.0.0", port=port)
