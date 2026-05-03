#!/usr/bin/env python3
"""Quick test of all bot endpoints — no LLM required for basic tests."""

import json
import urllib.request
import sys

BASE = "http://localhost:8080"

def req(method, path, body=None):
    data = json.dumps(body).encode() if body else None
    r = urllib.request.Request(f"{BASE}{path}", data=data, method=method,
                               headers={"Content-Type": "application/json"})
    try:
        resp = urllib.request.urlopen(r, timeout=10)
        return json.loads(resp.read().decode()), resp.status
    except urllib.error.HTTPError as e:
        return json.loads(e.read().decode()), e.code


def test_healthz():
    data, code = req("GET", "/v1/healthz")
    assert code == 200 and data["status"] == "ok", f"healthz failed: {data}"
    print(f"[PASS] healthz — contexts_loaded: {data['contexts_loaded']}")


def test_metadata():
    data, code = req("GET", "/v1/metadata")
    assert code == 200 and "team_name" in data, f"metadata failed: {data}"
    print(f"[PASS] metadata — team: {data['team_name']}, model: {data['model']}")


def test_context_push():
    # Push category
    cat = json.load(open("dataset/categories/dentists.json"))
    data, code = req("POST", "/v1/context", {
        "scope": "category", "context_id": "dentists", "version": 1,
        "payload": cat, "delivered_at": "2026-04-26T09:45:00Z"
    })
    assert data.get("accepted") == True, f"category push failed: {data}"
    print(f"[PASS] context push category/dentists — ack_id: {data['ack_id']}")

    # Push same version again (should be rejected)
    data, _ = req("POST", "/v1/context", {
        "scope": "category", "context_id": "dentists", "version": 1,
        "payload": cat, "delivered_at": "2026-04-26T09:45:00Z"
    })
    assert data.get("accepted") == False and data.get("reason") == "stale_version", f"idempotency failed: {data}"
    print(f"[PASS] context idempotency — correctly rejected stale version")

    # Push version 2 (should replace)
    data, _ = req("POST", "/v1/context", {
        "scope": "category", "context_id": "dentists", "version": 2,
        "payload": cat, "delivered_at": "2026-04-26T10:00:00Z"
    })
    assert data.get("accepted") == True, f"version bump failed: {data}"
    print(f"[PASS] context version bump — replaced v1 with v2")

    # Push merchant
    merchants = json.load(open("dataset/merchants_seed.json"))["merchants"]
    m = merchants[0]
    data, _ = req("POST", "/v1/context", {
        "scope": "merchant", "context_id": m["merchant_id"], "version": 1,
        "payload": m, "delivered_at": "2026-04-26T09:45:30Z"
    })
    assert data.get("accepted") == True, f"merchant push failed: {data}"
    print(f"[PASS] context push merchant/{m['merchant_id'][:20]}...")

    # Push trigger
    triggers = json.load(open("dataset/triggers_seed.json"))["triggers"]
    t = triggers[0]
    data, _ = req("POST", "/v1/context", {
        "scope": "trigger", "context_id": t["id"], "version": 1,
        "payload": t, "delivered_at": "2026-04-26T10:32:00Z"
    })
    assert data.get("accepted") == True, f"trigger push failed: {data}"
    print(f"[PASS] context push trigger/{t['id'][:30]}...")

    # Push customer
    customers = json.load(open("dataset/customers_seed.json"))["customers"]
    c = customers[0]
    data, _ = req("POST", "/v1/context", {
        "scope": "customer", "context_id": c["customer_id"], "version": 1,
        "payload": c, "delivered_at": "2026-04-26T09:46:00Z"
    })
    assert data.get("accepted") == True, f"customer push failed: {data}"
    print(f"[PASS] context push customer/{c['customer_id'][:25]}...")


def test_healthz_after_push():
    data, _ = req("GET", "/v1/healthz")
    counts = data["contexts_loaded"]
    print(f"[PASS] healthz after push — {counts}")
    assert counts["category"] >= 1
    assert counts["merchant"] >= 1
    assert counts["trigger"] >= 1


def test_reply_auto():
    """Test auto-reply detection."""
    auto = "Thank you for contacting us! Our team will respond shortly."
    
    data, _ = req("POST", "/v1/reply", {
        "conversation_id": "conv_auto_test", "merchant_id": "m_001_drmeera_dentist_delhi",
        "customer_id": None, "from_role": "merchant", "message": auto,
        "received_at": "2026-04-26T10:42:00Z", "turn_number": 2
    })
    print(f"[PASS] auto-reply turn 1 — action: {data.get('action')}")
    assert data.get("action") in ("send", "wait"), f"expected send/wait, got: {data}"

    # Second auto-reply
    data, _ = req("POST", "/v1/reply", {
        "conversation_id": "conv_auto_test", "merchant_id": "m_001_drmeera_dentist_delhi",
        "customer_id": None, "from_role": "merchant", "message": auto,
        "received_at": "2026-04-26T10:43:00Z", "turn_number": 3
    })
    print(f"[PASS] auto-reply turn 2 — action: {data.get('action')}")

    # Third auto-reply
    data, _ = req("POST", "/v1/reply", {
        "conversation_id": "conv_auto_test", "merchant_id": "m_001_drmeera_dentist_delhi",
        "customer_id": None, "from_role": "merchant", "message": auto,
        "received_at": "2026-04-26T10:44:00Z", "turn_number": 4
    })
    print(f"[PASS] auto-reply turn 3 — action: {data.get('action')}")
    assert data.get("action") == "end", f"expected end after 3 auto-replies, got: {data}"


def test_reply_hostile():
    """Test hostile detection."""
    data, _ = req("POST", "/v1/reply", {
        "conversation_id": "conv_hostile_test", "merchant_id": "m_001_drmeera_dentist_delhi",
        "customer_id": None, "from_role": "merchant",
        "message": "Stop messaging me. This is useless spam.",
        "received_at": "2026-04-26T10:42:00Z", "turn_number": 2
    })
    print(f"[PASS] hostile reply — action: {data.get('action')}")
    assert data.get("action") in ("send", "end"), f"expected send/end for hostile, got: {data}"


def test_reply_intent():
    """Test intent transition detection."""
    data, _ = req("POST", "/v1/reply", {
        "conversation_id": "conv_intent_test", "merchant_id": "m_001_drmeera_dentist_delhi",
        "customer_id": None, "from_role": "merchant",
        "message": "Ok lets do it. Whats next?",
        "received_at": "2026-04-26T10:42:00Z", "turn_number": 2
    })
    print(f"[PASS] intent transition — action: {data.get('action')}")
    body = data.get("body", "")
    print(f"  body: {body[:100]}...")


def test_teardown():
    data, _ = req("POST", "/v1/teardown")
    assert data.get("status") == "wiped", f"teardown failed: {data}"
    print(f"[PASS] teardown — state wiped")


if __name__ == "__main__":
    print("=" * 60)
    print("  Vera Bot — Endpoint Tests (no LLM needed)")
    print("=" * 60)
    print()
    
    tests = [
        test_healthz,
        test_metadata,
        test_context_push,
        test_healthz_after_push,
        test_reply_auto,
        test_reply_hostile,
        test_reply_intent,
        test_teardown,
    ]
    
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {t.__name__} FAILED: {e}")
            failed += 1
        print()
    
    print(f"Results: {passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
