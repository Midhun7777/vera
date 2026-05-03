# Vera — Merchant AI Assistant (magicpin AI Challenge)

## Approach

### Architecture: 4-Context LLM Composer with Multi-Turn Handlers

The bot is a stateful FastAPI server that composes WhatsApp messages using the 4-context framework (Category, Merchant, Trigger, Customer) with an LLM at its core.

**Three modules:**

1. **`bot.py`** — HTTP server implementing all 5 endpoints. In-memory stores for contexts, conversations, suppression keys, and opt-out state.

2. **`composer.py`** — LLM-powered message composer with **trigger-kind dispatch** (20+ specialized prompt variants for research_digest, recall_due, perf_dip, etc.) and **category-voice enforcement** (voice rules, taboo words, vocabulary injected into the system prompt per category).

3. **`conversation_handlers.py`** — Multi-turn reply logic with:
   - **Auto-reply detection** — regex patterns + repetition counting. Backs off after 1st detection, waits 24h after 2nd, ends after 3rd.
   - **Intent transition** — commitment phrase detection triggers immediate switch to action mode (no more qualifying questions).
   - **Hostile handling** — graceful one-line exit with re-entry path.
   - **Off-topic redirection** — polite decline + redirect to original topic.

### Key Design Decisions

- **Trigger-kind prompt variants** — Instead of one generic prompt, each trigger kind (research_digest, perf_dip, recall_due, etc.) has a specialized instruction that guides the LLM to use the right framing, data anchors, and CTA shape. This is the single biggest quality driver.

- **Category voice enforcement** — The system prompt includes per-category tone rules, allowed/taboo vocabulary, and tone examples. Post-LLM validation strips any taboo words that leak through.

- **Suppression dedup** — Each trigger has a `suppression_key`; once used, the key is blocked for the session. Prevents re-sending the same nudge.

- **Conversation state per conv_id** — Tracks auto-reply counts, intent commitment, turn history. Enables multi-turn behavior without re-processing.

### Tradeoffs

- **In-memory state** — simple but doesn't survive restarts. Fine for the test window; production would use Redis.
- **Single LLM call per compose** — no multi-step chain or retrieval. Keeps latency under 30s but limits reasoning depth.
- **No URL generation** — URLs are penalized in the rubric, so all output is text-only.

### What Would Help Most

- **Real merchant conversation logs** — to fine-tune the tone per category beyond the case studies.
- **A/B testing framework** — to iterate on prompt variants with real engagement data.
- **Category-specific digest pipelines** — automated sourcing from PubMed, Google Trends, etc. instead of static seed data.

## How to Run

```bash
pip install -r requirements.txt
export LLM_PROVIDER=gemini      # or openai, anthropic, deepseek, groq, openrouter
export LLM_API_KEY=your_key
python bot.py                    # starts on port 8080
```

## Generate Submission

```bash
python generate_submission.py    # creates submission.jsonl (30 lines)
```

## Test with Judge

```bash
python judge_simulator.py        # runs all scenarios against localhost:8080
```
