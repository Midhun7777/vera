#!/usr/bin/env python3
"""
Vera — Multi-Turn Conversation Handlers
=========================================

Handles merchant replies: auto-reply detection, intent transitions,
hostile messages, off-topic redirection, and graceful exits.
"""

import re
import json
from typing import Dict, List, Optional
from composer import _call_llm, _build_system_prompt

# ---------------------------------------------------------------------------
# Auto-Reply Detection
# ---------------------------------------------------------------------------

AUTO_REPLY_PATTERNS = [
    r"thank(?:s| you) for contacting",
    r"our team will (?:respond|get back|reply)",
    r"we(?:'ll| will) get back to you",
    r"this is an automated",
    r"automated (?:reply|response|message|assistant)",
    r"i(?:'m| am) an automated assistant",
    r"please leave (?:a |your )?message",
    r"currently (?:unavailable|away|busy|not available)",
    r"business hours",
    r"aapki (?:jaankari|madad) ke liye (?:bahut-bahut |)(?:shukriya|dhanyavaad)",
    r"hamari team (?:tak|ko) pahuncha",
]

AUTO_REPLY_COMPILED = [re.compile(p, re.IGNORECASE) for p in AUTO_REPLY_PATTERNS]


def is_auto_reply(message: str) -> bool:
    """Detect if a message is a WhatsApp Business canned auto-reply."""
    msg_lower = message.strip().lower()
    for pattern in AUTO_REPLY_COMPILED:
        if pattern.search(msg_lower):
            return True
    return False


# ---------------------------------------------------------------------------
# Intent Detection
# ---------------------------------------------------------------------------

COMMITMENT_PHRASES = [
    r"(?:ok(?:ay)?|yes|yeah|haan|ha|ji)\s*(?:let'?s|lets)\s*do\s*(?:it|this)",
    r"(?:let'?s|lets)\s*(?:go|do it|proceed|start)",
    r"(?:go\s*ahead|proceed|confirm|done|approved)",
    r"(?:yes|haan|ha|ji)\s*(?:please|sure|definitely|do it)",
    r"(?:sounds?\s*good|looks?\s*good|works?\s*for me)",
    r"(?:i'?m|i am)\s*(?:in|ready|interested|down)",
    r"(?:what'?s?\s*next|next\s*step|how\s*do\s*(?:we|i)\s*(?:start|proceed))",
    r"(?:sign\s*me\s*up|count\s*me\s*in)",
    r"(?:mujhe|humko|hume)\s*(?:join|start|shuru)\s*(?:karna|karni|karo)",
    r"(?:chalo|chalein|shuru\s*karo|kar\s*do)",
]

COMMITMENT_COMPILED = [re.compile(p, re.IGNORECASE) for p in COMMITMENT_PHRASES]


def is_commitment(message: str) -> bool:
    """Detect if a merchant is committing to action (intent transition)."""
    for pattern in COMMITMENT_COMPILED:
        if pattern.search(message):
            return True
    return False


# ---------------------------------------------------------------------------
# Hostile / Not-Interested Detection
# ---------------------------------------------------------------------------

HOSTILE_PATTERNS = [
    r"stop\s*(?:messaging|sending|bothering|contacting|this)",
    r"(?:not|no(?:t)?)\s*interested",
    r"(?:leave\s*me\s*alone|don'?t\s*(?:message|contact|bother)\s*me)",
    r"(?:unsubscribe|opt\s*out|remove\s*me)",
    r"(?:useless|spam|waste\s*(?:of\s*)?time|scam)",
    r"(?:band\s*karo|mat\s*bhejo|rahne\s*do|nahi\s*chahiye)",
    r"STOP",
]

HOSTILE_COMPILED = [re.compile(p, re.IGNORECASE) for p in HOSTILE_PATTERNS]


def is_hostile(message: str) -> bool:
    """Detect hostile or opt-out messages."""
    for pattern in HOSTILE_COMPILED:
        if pattern.search(message):
            return True
    return False


# ---------------------------------------------------------------------------
# Off-Topic Detection
# ---------------------------------------------------------------------------

OFF_TOPIC_PATTERNS = [
    r"(?:gst|tax|income\s*tax|itr)\s*(?:filing|return|help)",
    r"(?:passport|visa|aadhar|pan\s*card)",
    r"(?:loan|emi|insurance|mutual\s*fund)",
    r"(?:weather|cricket\s*score|news)",
]

OFF_TOPIC_COMPILED = [re.compile(p, re.IGNORECASE) for p in OFF_TOPIC_PATTERNS]


def is_off_topic(message: str) -> bool:
    """Detect off-topic questions."""
    for pattern in OFF_TOPIC_COMPILED:
        if pattern.search(message):
            return True
    return False


# ---------------------------------------------------------------------------
# Conversation State
# ---------------------------------------------------------------------------

class ConversationState:
    """Tracks state for a multi-turn conversation."""

    def __init__(self, conversation_id: str, merchant_id: str):
        self.conversation_id = conversation_id
        self.merchant_id = merchant_id
        self.turns: List[Dict] = []
        self.auto_reply_count: int = 0
        self.last_auto_reply_text: str = ""
        self.intent_committed: bool = False
        self.ended: bool = False
        self.last_bot_body: str = ""

    def add_merchant_turn(self, message: str, turn_number: int):
        self.turns.append({"from": "merchant", "message": message, "turn": turn_number})

    def add_bot_turn(self, body: str, action: str):
        self.turns.append({"from": "bot", "body": body, "action": action})
        if action == "send":
            self.last_bot_body = body


# ---------------------------------------------------------------------------
# Reply Handler
# ---------------------------------------------------------------------------

def handle_reply(
    state: ConversationState,
    merchant_message: str,
    merchant: dict,
    category: dict,
    turn_number: int,
) -> dict:
    """
    Given conversation state + merchant's latest message, produce the reply.

    Returns dict with keys: action (send/wait/end), body (if send), cta, rationale
    """
    state.add_merchant_turn(merchant_message, turn_number)

    # 1. Check if conversation already ended
    if state.ended:
        return {
            "action": "end",
            "rationale": "Conversation was already ended."
        }

    # 2. Hostile / opt-out detection — immediate exit
    if is_hostile(merchant_message):
        state.ended = True
        return {
            "action": "send",
            "body": "Apologies — I won't message again. If anything changes, you can restart with 'Hi Vera'. 🙏",
            "cta": "none",
            "rationale": "Merchant expressed frustration or opted out. Graceful one-line exit with re-entry path."
        }

    # 3. Auto-reply detection
    if is_auto_reply(merchant_message):
        state.auto_reply_count += 1
        state.last_auto_reply_text = merchant_message

        if state.auto_reply_count >= 3:
            state.ended = True
            return {
                "action": "end",
                "rationale": f"Auto-reply detected {state.auto_reply_count} times. No real engagement signal; closing conversation."
            }
        elif state.auto_reply_count == 2:
            return {
                "action": "wait",
                "wait_seconds": 86400,
                "rationale": "Same auto-reply twice in a row — owner not at phone. Wait 24h before retry."
            }
        else:
            return {
                "action": "send",
                "body": "Looks like an auto-reply 😊 When the owner sees this, just reply 'Yes' and I'll pick up from here.",
                "cta": "binary_yes_no",
                "rationale": "Detected auto-reply (canned 'Thank you for contacting' phrasing). One explicit prompt to flag for the owner."
            }

    # Reset auto-reply counter on real message
    state.auto_reply_count = 0

    # 4. Intent transition — merchant commits to action
    if is_commitment(merchant_message):
        state.intent_committed = True
        # Use LLM to compose a concrete action response
        return _compose_action_response(state, merchant_message, merchant, category)

    # 5. Off-topic detection — redirect politely
    if is_off_topic(merchant_message):
        return _compose_redirect_response(state, merchant_message, merchant, category)

    # 6. General engaged reply — use LLM
    return _compose_engaged_reply(state, merchant_message, merchant, category)


def _compose_action_response(state: ConversationState, message: str, merchant: dict, category: dict) -> dict:
    """Compose a concrete action response when merchant commits."""
    system = _build_system_prompt(category)
    identity = merchant.get("identity", {})
    name = identity.get("owner_first_name", identity.get("name", "there"))

    conversation_context = "\n".join([
        f"[{t['from']}] {t.get('message', t.get('body', ''))[:150]}"
        for t in state.turns[-4:]
    ])

    user_prompt = f"""The merchant has COMMITTED to action. They said: "{message}"

CONVERSATION SO FAR:
{conversation_context}

MERCHANT: {identity.get('name')} ({identity.get('locality')}, {identity.get('city')})
Active Offers: {json.dumps([o.get('title') for o in merchant.get('offers', []) if o.get('status') == 'active'])}
Customer Aggregate: {json.dumps(merchant.get('customer_aggregate', {}))}

CRITICAL: Switch from qualifying to ACTION mode immediately. DO NOT ask another qualifying question.
- State what you are doing NOW (drafting, sending, scheduling)
- Provide specific deliverables with numbers/timelines
- End with a binary CONFIRM/CANCEL CTA

Return JSON: {{"body": "...", "cta": "binary_confirm_cancel", "rationale": "..."}}"""

    try:
        raw = _call_llm(system, user_prompt)
        # Parse JSON
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
        match = re.search(r'\{[\s\S]*\}', raw)
        if match:
            data = json.loads(match.group())
            result = {
                "action": "send",
                "body": data.get("body", f"Great, {name}! Working on it now — I'll have the draft ready in 2 minutes."),
                "cta": data.get("cta", "binary_confirm_cancel"),
                "rationale": data.get("rationale", "Merchant committed; switching to action mode with concrete deliverables.")
            }
            state.add_bot_turn(result["body"], "send")
            return result
    except Exception:
        pass

    # Fallback
    result = {
        "action": "send",
        "body": f"Great, {name}! Working on it now — I'll have the draft ready in 2 minutes. Reply CONFIRM to proceed.",
        "cta": "binary_confirm_cancel",
        "rationale": "Merchant committed to action; switching from qualifying to execution mode."
    }
    state.add_bot_turn(result["body"], "send")
    return result


def _compose_redirect_response(state: ConversationState, message: str, merchant: dict, category: dict) -> dict:
    """Redirect an off-topic question back to mission."""
    last_topic = ""
    for t in reversed(state.turns):
        if t.get("from") == "bot" and t.get("body"):
            last_topic = t["body"][:80]
            break

    body = f"I'll have to leave that to your CA/specialist — that's outside what I can help with directly. Coming back to the earlier topic — shall I continue with that?"
    result = {
        "action": "send",
        "body": body,
        "cta": "open_ended",
        "rationale": "Out-of-scope ask politely declined; redirects back to the original trigger without losing thread."
    }
    state.add_bot_turn(result["body"], "send")
    return result


def _compose_engaged_reply(state: ConversationState, message: str, merchant: dict, category: dict) -> dict:
    """Use LLM to compose an engaged reply to a real merchant message."""
    system = _build_system_prompt(category)
    identity = merchant.get("identity", {})

    conversation_context = "\n".join([
        f"[{t['from']}] {t.get('message', t.get('body', ''))[:150]}"
        for t in state.turns[-5:]
    ])

    user_prompt = f"""Continue this conversation with the merchant. Their latest message: "{message}"

CONVERSATION SO FAR:
{conversation_context}

MERCHANT: {identity.get('name')} ({identity.get('locality')}, {identity.get('city')})
Languages: {identity.get('languages', ['en'])}
Active Offers: {json.dumps([o.get('title') for o in merchant.get('offers', []) if o.get('status') == 'active'])}
Performance: views={merchant.get('performance', {}).get('views', '?')}, ctr={merchant.get('performance', {}).get('ctr', '?')}
Signals: {merchant.get('signals', [])}

RULES:
- Respond directly to what the merchant said
- Be helpful, specific, concise
- If they asked a question, answer it with data from context
- If they agreed, take action
- End with one clear next step
- DO NOT repeat anything you've already said
- DO NOT re-introduce yourself

Return JSON: {{"body": "...", "cta": "open_ended or binary_yes_no", "rationale": "..."}}"""

    try:
        raw = _call_llm(system, user_prompt)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
        match = re.search(r'\{[\s\S]*\}', raw)
        if match:
            data = json.loads(match.group())
            body = data.get("body", "")

            # Anti-repetition check
            if body == state.last_bot_body:
                body += " (anything else I can help with?)"

            result = {
                "action": "send",
                "body": body,
                "cta": data.get("cta", "open_ended"),
                "rationale": data.get("rationale", "Engaged reply to merchant message.")
            }
            state.add_bot_turn(result["body"], "send")
            return result
    except Exception:
        pass

    # Fallback
    result = {
        "action": "send",
        "body": "Got it — let me look into that. I'll have an update for you shortly.",
        "cta": "open_ended",
        "rationale": "Acknowledged merchant message; LLM unavailable for detailed reply."
    }
    state.add_bot_turn(result["body"], "send")
    return result
