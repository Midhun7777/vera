#!/usr/bin/env python3
"""
Vera Merchant AI Assistant — LLM-Powered Message Composer
==========================================================

Composes WhatsApp messages for merchants and their customers using
the 4-context framework (Category, Merchant, Trigger, Customer).

Supports multiple LLM providers: Gemini, OpenAI, Anthropic, DeepSeek, Groq.
"""

import os
import json
import re
from urllib import request as urlrequest
from typing import Optional, Dict, Any

# ---------------------------------------------------------------------------
# LLM Configuration (env vars)
# ---------------------------------------------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "")
LLM_TIMEOUT = 45

# ---------------------------------------------------------------------------
# LLM Call Dispatch
# ---------------------------------------------------------------------------

def _call_llm(system_prompt: str, user_prompt: str) -> str:
    """Route LLM call to the configured provider."""
    provider = LLM_PROVIDER.lower()
    if provider == "gemini":
        return _call_gemini(system_prompt, user_prompt)
    elif provider == "openai":
        return _call_openai(system_prompt, user_prompt)
    elif provider == "anthropic":
        return _call_anthropic(system_prompt, user_prompt)
    elif provider == "deepseek":
        return _call_deepseek(system_prompt, user_prompt)
    elif provider == "groq":
        return _call_groq(system_prompt, user_prompt)
    elif provider == "openrouter":
        return _call_openrouter(system_prompt, user_prompt)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider}")


def _call_gemini(system: str, user: str) -> str:
    model = LLM_MODEL or "gemini-2.0-flash"
    full_prompt = f"{system}\n\n{user}" if system else user
    body = json.dumps({
        "contents": [{"parts": [{"text": full_prompt}]}],
        "generationConfig": {"temperature": 0.15, "maxOutputTokens": 1200}
    }).encode("utf-8")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={LLM_API_KEY}"
    req = urlrequest.Request(url, data=body, headers={"Content-Type": "application/json"})
    resp = urlrequest.urlopen(req, timeout=LLM_TIMEOUT)
    data = json.loads(resp.read().decode("utf-8"))
    return data["candidates"][0]["content"]["parts"][0]["text"]


def _call_openai(system: str, user: str) -> str:
    model = LLM_MODEL or "gpt-4o-mini"
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    body = json.dumps({"model": model, "messages": messages, "temperature": 0.15, "max_tokens": 1200}).encode("utf-8")
    req = urlrequest.Request(
        "https://api.openai.com/v1/chat/completions", data=body,
        headers={"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"}
    )
    resp = urlrequest.urlopen(req, timeout=LLM_TIMEOUT)
    data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def _call_anthropic(system: str, user: str) -> str:
    model = LLM_MODEL or "claude-3-5-sonnet-20241022"
    body_dict = {"model": model, "max_tokens": 1200, "messages": [{"role": "user", "content": user}]}
    if system:
        body_dict["system"] = system
    req = urlrequest.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(body_dict).encode("utf-8"),
        headers={"x-api-key": LLM_API_KEY, "Content-Type": "application/json", "anthropic-version": "2023-06-01"}
    )
    resp = urlrequest.urlopen(req, timeout=LLM_TIMEOUT)
    data = json.loads(resp.read().decode("utf-8"))
    return data["content"][0]["text"]


def _call_deepseek(system: str, user: str) -> str:
    model = LLM_MODEL or "deepseek-chat"
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    req = urlrequest.Request(
        "https://api.deepseek.com/v1/chat/completions",
        data=json.dumps({"model": model, "messages": messages, "temperature": 0.15, "max_tokens": 1200}).encode("utf-8"),
        headers={"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"}
    )
    resp = urlrequest.urlopen(req, timeout=LLM_TIMEOUT)
    data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def _call_groq(system: str, user: str) -> str:
    model = LLM_MODEL or "llama-3.1-70b-versatile"
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    req = urlrequest.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=json.dumps({"model": model, "messages": messages, "temperature": 0.15, "max_tokens": 1200}).encode("utf-8"),
        headers={"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"}
    )
    resp = urlrequest.urlopen(req, timeout=LLM_TIMEOUT)
    data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def _call_openrouter(system: str, user: str) -> str:
    model = LLM_MODEL or "meta-llama/llama-3.3-70b-instruct:free"
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    req = urlrequest.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps({"model": model, "messages": messages, "temperature": 0.15, "max_tokens": 1200}).encode("utf-8"),
        headers={"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json",
                 "HTTP-Referer": "https://magicpin.com"}
    )
    resp = urlrequest.urlopen(req, timeout=LLM_TIMEOUT)
    data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# System Prompt Builder (per-category voice enforcement)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_BASE = """You are Vera, magicpin's AI merchant assistant. You compose WhatsApp messages for Indian merchants.

ABSOLUTE RULES:
1. NEVER fabricate data not in the contexts provided. No fake stats, citations, competitor names, or offers.
2. Keep the message concise — WhatsApp style. No preambles like "I hope you're doing well".
3. Use a SINGLE primary CTA at the end. Binary YES/STOP for actions, open-ended for information.
4. Anchor on VERIFIABLE facts: numbers, dates, source citations, prices from the context.
5. Match the merchant's language preference. Hindi-English code-mix is natural and encouraged.
6. Use the owner's first name when available.
7. DO NOT use URLs in messages.
8. DO NOT re-introduce yourself after the first message.

OUTPUT FORMAT — return ONLY valid JSON with these exact keys:
{
  "body": "the WhatsApp message body",
  "cta": "binary_yes_no | open_ended | multi_choice_slot | none",
  "send_as": "vera | merchant_on_behalf",
  "suppression_key": "from trigger",
  "rationale": "2-3 sentences explaining why this message, what compulsion levers used"
}"""


def _build_system_prompt(category: dict) -> str:
    """Build a category-aware system prompt with voice rules."""
    voice = category.get("voice", {})
    tone = voice.get("tone", "professional")
    taboos = voice.get("vocab_taboo", [])
    vocab = voice.get("vocab_allowed", [])
    tone_examples = voice.get("tone_examples", [])

    cat_rules = f"""
CATEGORY: {category.get('slug', 'unknown')} ({category.get('display_name', '')})
VOICE TONE: {tone}
ALLOWED VOCABULARY: {', '.join(vocab[:15])}
TABOO WORDS (NEVER USE): {', '.join(taboos)}
TONE EXAMPLES: {chr(10).join(f'  - {e}' for e in tone_examples)}

PEER BENCHMARKS: {json.dumps(category.get('peer_stats', {}), indent=2)}
"""
    return SYSTEM_PROMPT_BASE + cat_rules


# ---------------------------------------------------------------------------
# Trigger-Kind-Specific Prompt Variants
# ---------------------------------------------------------------------------

TRIGGER_PROMPTS = {
    "research_digest": """Compose a message sharing this research/digest item with the merchant.
- Lead with the source citation (journal name, page, date)
- Anchor on trial size, percentage, or key finding
- Connect to the merchant's specific patient/customer cohort if data exists
- Offer to draft a patient/customer-facing version they can share
- Tone: peer-to-peer, curious, not promotional""",

    "regulation_change": """Compose a compliance alert message.
- Lead with the regulatory body and deadline
- Explain what changed and what action is needed
- Be precise about what's affected (equipment, processes, records)
- Offer concrete help (audit checklist, template)
- Tone: urgent but not alarmist""",

    "recall_due": """Compose a recall/appointment reminder sent ON BEHALF of the merchant to their customer.
- Use send_as: merchant_on_behalf
- Address the customer by name
- State when their last visit was and that the recall window is due
- Offer specific available slots from the trigger payload
- Include the service price from merchant offers
- Match the customer's language preference
- Tone: warm, clinical where appropriate, no overclaims""",

    "perf_dip": """Compose a performance alert about a metric drop.
- State the specific metric and percentage drop
- Provide context (is it seasonal? peer comparison?)
- Suggest ONE specific action
- Offer to help implement it
- Tone: matter-of-fact, not alarming""",

    "perf_spike": """Compose a positive performance notification.
- Celebrate the specific metric improvement with numbers
- Attribute the likely driver if known
- Suggest how to capitalize on the momentum
- Tone: encouraging, data-driven""",

    "renewal_due": """Compose a subscription renewal reminder.
- State days remaining and plan name
- Reference their current performance to show value
- Don't pressure — frame as continuation of momentum
- Single binary CTA: renew YES/NO""",

    "festival_upcoming": """Compose a festival-related opportunity message.
- Name the festival and date
- Provide category-specific preparation advice
- Suggest a specific offer or campaign tied to the festival
- Offer to draft the campaign content""",

    "ipl_match_today": """Compose an IPL match-day message.
- Reference the specific match, venue, and time
- Provide DATA-DRIVEN insight (not generic hype)
- Suggest whether to push or pull back based on day-of-week patterns
- Reference existing offers if applicable""",

    "review_theme_emerged": """Compose a review trend alert.
- State the specific theme, sentiment, and count
- Include the actual customer quote if available
- Suggest one concrete action to address it
- Tone: collaborative problem-solving""",

    "milestone_reached": """Compose a milestone celebration message.
- State the specific milestone number
- Provide peer comparison for context
- Suggest how to leverage it (social proof, GBP post)
- Keep it brief and celebratory""",

    "curious_ask_due": """Compose a curiosity-driven engagement message.
- Ask the merchant a genuine question about their business
- Offer a specific deliverable in return for their answer (GBP post, WhatsApp reply template)
- Keep it low-stakes, no commitment required
- Tone: curious peer, not interrogating""",

    "winback_eligible": """Compose a re-engagement message for a lapsed subscription merchant.
- Acknowledge the gap without guilt-tripping
- Show what they've been missing (performance data since expiry)
- Offer a specific re-activation step
- Tone: helpful, not salesy""",

    "active_planning_intent": """Compose a response to a merchant's planning question.
- Provide a CONCRETE draft/plan they can edit (pricing tiers, schedule, content)
- Reference their locality and business specifics
- Include specific numbers (prices, counts, time windows)
- Offer the logical next step
- Tone: collaborative, operator-level""",

    "supply_alert": """Compose an urgent supply/recall alert.
- List specific batch numbers and manufacturer
- State the risk level clearly (no alarmism)
- Provide affected customer count if derivable from context
- Offer to draft customer communication
- Tone: precise, trustworthy, urgent""",

    "chronic_refill_due": """Compose a chronic medication refill reminder ON BEHALF of the merchant to the customer.
- Use send_as: merchant_on_behalf
- List specific molecules/medicines
- State when stock runs out
- Include price + any applicable discounts (senior, delivery)
- Offer delivery if available
- Match language preference — respectful tone for seniors""",

    "customer_lapsed_hard": """Compose a winback message for a lapsed customer ON BEHALF of the merchant.
- Use send_as: merchant_on_behalf
- No shame or guilt — normalize the gap
- Reference their previous focus/services
- Offer something new and relevant
- Single binary CTA with explicit no-commitment framing""",

    "wedding_package_followup": """Compose a bridal/wedding follow-up message ON BEHALF of the merchant.
- Use send_as: merchant_on_behalf
- Reference days to wedding
- Suggest the next logical step in the bridal journey
- Include specific pricing and slot
- Tone: warm, excited, practical""",

    "trial_followup": """Compose a trial follow-up message ON BEHALF of the merchant.
- Use send_as: merchant_on_behalf
- Reference the trial date and experience
- Offer next session with specific slot
- Low-pressure CTA""",

    "seasonal_perf_dip": """Compose a seasonal context message.
- Acknowledge the dip but REFRAME it as expected/normal
- Provide the seasonal pattern data
- Suggest what to do INSTEAD of panicking (retention, save budget)
- Tone: coaching, reassuring""",

    "category_seasonal": """Compose a seasonal demand shift advisory.
- List specific products/services trending up and down with numbers
- Suggest shelf/menu/schedule adjustments
- Offer to help implement changes
- Tone: practical, data-driven""",

    "dormant_with_vera": """Compose a re-engagement message for a merchant who hasn't talked to Vera.
- Don't guilt-trip about the silence
- Lead with something NEW and valuable (a digest item, a perf insight)
- Ask a low-stakes question
- Tone: casual peer check-in""",

    "gbp_unverified": """Compose a Google Business Profile verification nudge.
- Explain the specific uplift (percentage from context)
- State the verification path (postcard/phone)
- Offer to walk them through it
- Tone: helpful, not nagging""",

    "cde_opportunity": """Compose a professional development opportunity message.
- Name the event, date, credits, and cost
- Connect to the merchant's practice focus if possible
- Single CTA to register or get more info
- Tone: peer recommendation""",

    "competitor_opened": """Compose a competitive intelligence alert.
- Name the competitor and distance
- State their offer if available
- Suggest differentiation strategy (don't copy on price)
- Tone: voyeur-curiosity, not alarming""",

    "appointment_tomorrow": """Compose an appointment reminder ON BEHALF of the merchant.
- Use send_as: merchant_on_behalf
- State date, time, service
- Offer rescheduling option
- Brief and functional""",

    "customer_lapsed_soft": """Compose a soft re-engagement for a mildly lapsed customer ON BEHALF of merchant.
- Use send_as: merchant_on_behalf
- Gentle nudge with something new or a timely reason to return
- Reference their last service
- Low-friction CTA""",
}

DEFAULT_TRIGGER_PROMPT = """Compose a message appropriate for this trigger kind.
- Anchor on specific data from the contexts
- Use the merchant's name and relevant performance data
- Include one clear CTA
- Match the voice and tone for this category"""


# ---------------------------------------------------------------------------
# Core Compose Function
# ---------------------------------------------------------------------------

def compose(
    category: dict,
    merchant: dict,
    trigger: dict,
    customer: Optional[dict] = None
) -> dict:
    """
    Compose a WhatsApp message from the 4-context framework.

    Returns dict with keys: body, cta, send_as, suppression_key, rationale
    """
    system_prompt = _build_system_prompt(category)
    trigger_kind = trigger.get("kind", "unknown")
    trigger_instruction = TRIGGER_PROMPTS.get(trigger_kind, DEFAULT_TRIGGER_PROMPT)

    # Build the user prompt with all contexts serialized
    user_prompt = _build_user_prompt(category, merchant, trigger, customer, trigger_instruction)

    try:
        raw_response = _call_llm(system_prompt, user_prompt)
        result = _parse_compose_response(raw_response, trigger)
        result = _validate_and_fix(result, category, merchant, trigger, customer)
        return result
    except Exception as e:
        # Fallback if LLM fails
        return _fallback_compose(category, merchant, trigger, customer, str(e))


def _build_user_prompt(
    category: dict, merchant: dict, trigger: dict,
    customer: Optional[dict], trigger_instruction: str
) -> str:
    """Build the structured user prompt with all 4 contexts."""

    identity = merchant.get("identity", {})
    perf = merchant.get("performance", {})
    delta = perf.get("delta_7d", {})
    sub = merchant.get("subscription", {})
    offers = [o for o in merchant.get("offers", []) if o.get("status") == "active"]
    conv_history = merchant.get("conversation_history", [])
    signals = merchant.get("signals", [])
    cust_agg = merchant.get("customer_aggregate", {})
    review_themes = merchant.get("review_themes", [])

    # Digest items from category
    digest_items = category.get("digest", [])
    # Find the relevant digest item if trigger references one
    trigger_payload = trigger.get("payload", {})
    top_item_id = trigger_payload.get("top_item_id") or trigger_payload.get("digest_item_id") or trigger_payload.get("alert_id")
    relevant_digest = None
    if top_item_id:
        for d in digest_items:
            if d.get("id") == top_item_id:
                relevant_digest = d
                break

    prompt = f"""=== TASK ===
{trigger_instruction}

=== CATEGORY CONTEXT ===
Category: {category.get('slug')} ({category.get('display_name', '')})
Peer Stats: avg_rating={category.get('peer_stats', {}).get('avg_rating', '?')}, avg_ctr={category.get('peer_stats', {}).get('avg_ctr', '?')}, avg_views_30d={category.get('peer_stats', {}).get('avg_views_30d', '?')}, avg_calls_30d={category.get('peer_stats', {}).get('avg_calls_30d', '?')}
Seasonal Beats: {json.dumps(category.get('seasonal_beats', []))}
Trend Signals: {json.dumps(category.get('trend_signals', [])[:3])}
"""

    if relevant_digest:
        prompt += f"""
Relevant Digest Item:
  ID: {relevant_digest.get('id')}
  Title: {relevant_digest.get('title')}
  Source: {relevant_digest.get('source')}
  Summary: {relevant_digest.get('summary', '')}
  Actionable: {relevant_digest.get('actionable', '')}
  Trial Size: {relevant_digest.get('trial_n', 'N/A')}
  Patient Segment: {relevant_digest.get('patient_segment', 'N/A')}
"""
    elif digest_items:
        prompt += f"\nAll Digest Items: {json.dumps(digest_items[:3])}\n"

    prompt += f"""
=== MERCHANT CONTEXT ===
Name: {identity.get('name', '?')}
Owner First Name: {identity.get('owner_first_name', '?')}
City: {identity.get('city', '?')}, Locality: {identity.get('locality', '?')}
Verified: {identity.get('verified', '?')}
Languages: {identity.get('languages', ['en'])}
Subscription: {sub.get('status', '?')} ({sub.get('plan', '?')}), days_remaining={sub.get('days_remaining', '?')}
Performance (30d): views={perf.get('views', '?')}, calls={perf.get('calls', '?')}, directions={perf.get('directions', '?')}, ctr={perf.get('ctr', '?')}, leads={perf.get('leads', '?')}
7-day Delta: views {delta.get('views_pct', '?')}, calls {delta.get('calls_pct', '?')}
Active Offers: {json.dumps([o.get('title') for o in offers]) if offers else 'None'}
Customer Aggregate: {json.dumps(cust_agg)}
Signals: {signals}
Review Themes: {json.dumps(review_themes[:3]) if review_themes else 'None'}
"""

    if conv_history:
        prompt += f"\nRecent Conversation History (last {min(3, len(conv_history))} turns):\n"
        for turn in conv_history[-3:]:
            prompt += f"  [{turn.get('from', '?')}] {turn.get('body', '')[:150]}  (engagement: {turn.get('engagement', '?')})\n"

    prompt += f"""
=== TRIGGER CONTEXT ===
ID: {trigger.get('id', '?')}
Kind: {trigger.get('kind', '?')}
Scope: {trigger.get('scope', '?')}
Source: {trigger.get('source', '?')}
Urgency: {trigger.get('urgency', '?')}
Suppression Key: {trigger.get('suppression_key', '')}
Payload: {json.dumps(trigger_payload)}
"""

    if customer:
        cust_identity = customer.get("identity", {})
        cust_rel = customer.get("relationship", {})
        cust_prefs = customer.get("preferences", {})
        prompt += f"""
=== CUSTOMER CONTEXT (this is a customer-facing message) ===
Name: {cust_identity.get('name', '?')}
Language Preference: {cust_identity.get('language_pref', 'en')}
Age Band: {cust_identity.get('age_band', '?')}
Senior Citizen: {cust_identity.get('senior_citizen', False)}
State: {customer.get('state', '?')}
Relationship: first_visit={cust_rel.get('first_visit', '?')}, last_visit={cust_rel.get('last_visit', '?')}, visits={cust_rel.get('visits_total', '?')}
Services Received: {cust_rel.get('services_received', [])}
Preferences: {json.dumps(cust_prefs)}
Consent Scope: {customer.get('consent', {}).get('scope', [])}

IMPORTANT: This message is sent FROM the merchant's WhatsApp number to the customer.
Set send_as to "merchant_on_behalf".
"""

    prompt += """
=== RESPOND WITH JSON ONLY ===
Return a single JSON object with keys: body, cta, send_as, suppression_key, rationale
"""
    return prompt


def _parse_compose_response(raw: str, trigger: dict) -> dict:
    """Parse the LLM's JSON response, handling markdown fences and partial JSON."""
    # Strip markdown code fences
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

    # Find JSON object
    match = re.search(r'\{[\s\S]*\}', raw)
    if not match:
        raise ValueError(f"No JSON found in LLM response: {raw[:200]}")

    data = json.loads(match.group())

    return {
        "body": data.get("body", ""),
        "cta": data.get("cta", "open_ended"),
        "send_as": data.get("send_as", "vera"),
        "suppression_key": data.get("suppression_key", trigger.get("suppression_key", "")),
        "rationale": data.get("rationale", ""),
    }


def _validate_and_fix(result: dict, category: dict, merchant: dict, trigger: dict, customer: Optional[dict]) -> dict:
    """Post-LLM validation and fixups."""
    body = result.get("body", "")

    # Fix send_as for customer-scoped triggers
    if customer or trigger.get("scope") == "customer":
        result["send_as"] = "merchant_on_behalf"

    # Ensure suppression_key is populated
    if not result.get("suppression_key"):
        result["suppression_key"] = trigger.get("suppression_key", f"{trigger.get('kind', 'unknown')}:{trigger.get('id', 'unknown')}")

    # Check for taboo words
    taboos = category.get("voice", {}).get("vocab_taboo", [])
    body_lower = body.lower()
    for taboo in taboos:
        if taboo.lower() in body_lower:
            body = body.replace(taboo, "").replace(taboo.lower(), "")
            result["body"] = body

    return result


def _fallback_compose(category: dict, merchant: dict, trigger: dict, customer: Optional[dict], error: str) -> dict:
    """Generate a basic message without LLM when the API call fails."""
    identity = merchant.get("identity", {})
    name = identity.get("owner_first_name") or identity.get("name", "there")
    trigger_kind = trigger.get("kind", "update")
    scope = trigger.get("scope", "merchant")

    if scope == "customer" and customer:
        cust_name = customer.get("identity", {}).get("name", "there")
        body = f"Hi {cust_name}, {identity.get('name', 'your clinic')} here — we have an update for you. Reply YES for details."
        send_as = "merchant_on_behalf"
    else:
        body = f"Hi {name}, quick update regarding your business on magicpin. Reply YES if you'd like details."
        send_as = "vera"

    return {
        "body": body,
        "cta": "binary_yes_no",
        "send_as": send_as,
        "suppression_key": trigger.get("suppression_key", ""),
        "rationale": f"Fallback message due to LLM error: {error[:100]}",
    }
