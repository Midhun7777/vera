#!/usr/bin/env python3
"""
Generate submission.jsonl — 30 test pair messages for the magicpin AI Challenge.

This script:
1. Loads the seed dataset (categories, merchants, customers, triggers)
2. Optionally runs generate_dataset.py to expand to full dataset
3. Creates 30 (merchant, trigger) test pairs
4. Calls composer.compose() for each pair
5. Writes submission.jsonl

Usage:
    export LLM_PROVIDER=gemini
    export LLM_API_KEY=your_key_here
    python generate_submission.py
"""

import json
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from composer import compose


def load_dataset(dataset_dir: Path) -> tuple:
    """Load the expanded dataset."""
    categories = {}
    merchants = {}
    customers = {}
    triggers = {}

    expanded_dir = dataset_dir / "expanded"

    # Load categories
    cat_dir = expanded_dir / "categories"
    if cat_dir.exists():
        for f in cat_dir.glob("*.json"):
            with open(f, encoding="utf-8") as fp:
                data = json.load(fp)
                categories[data.get("slug", f.stem)] = data

    # Load merchants
    merch_dir = expanded_dir / "merchants"
    if merch_dir.exists():
        for f in merch_dir.glob("*.json"):
            with open(f, encoding="utf-8") as fp:
                data = json.load(fp)
                merchants[data["merchant_id"]] = data

    # Load customers
    cust_dir = expanded_dir / "customers"
    if cust_dir.exists():
        for f in cust_dir.glob("*.json"):
            with open(f, encoding="utf-8") as fp:
                data = json.load(fp)
                customers[data["customer_id"]] = data

    # Load triggers
    trig_dir = expanded_dir / "triggers"
    if trig_dir.exists():
        for f in trig_dir.glob("*.json"):
            with open(f, encoding="utf-8") as fp:
                data = json.load(fp)
                triggers[data["id"]] = data

    return categories, merchants, customers, triggers


def get_test_pairs(triggers: dict) -> list:
    """Create 30 test pairs from available triggers — 2 per kind, covering all kinds."""
    by_kind = {}
    for t in triggers.values():
        by_kind.setdefault(t["kind"], []).append(t)

    pairs = []
    test_id = 1
    for kind in sorted(by_kind.keys()):
        for t in by_kind[kind][:2]:
            pairs.append({
                "test_id": f"T{test_id:02d}",
                "trigger_id": t["id"],
                "merchant_id": t["merchant_id"],
                "customer_id": t.get("customer_id"),
            })
            test_id += 1
            if len(pairs) >= 30:
                break
        if len(pairs) >= 30:
            break

    # If we don't have 30 yet, pad with remaining triggers
    used_trg_ids = {p["trigger_id"] for p in pairs}
    for t in triggers.values():
        if len(pairs) >= 30:
            break
        if t["id"] not in used_trg_ids:
            pairs.append({
                "test_id": f"T{test_id:02d}",
                "trigger_id": t["id"],
                "merchant_id": t["merchant_id"],
                "customer_id": t.get("customer_id"),
            })
            test_id += 1
            used_trg_ids.add(t["id"])

    return pairs[:30]


def main():
    dataset_dir = Path(__file__).parent / "dataset"
    output_file = Path(__file__).parent / "submission.jsonl"

    print("Loading dataset...")
    categories, merchants, customers, triggers = load_dataset(dataset_dir)
    print(f"  Loaded: {len(categories)} categories, {len(merchants)} merchants, "
          f"{len(customers)} customers, {len(triggers)} triggers")

    print("\nCreating test pairs...")
    pairs = get_test_pairs(triggers)
    print(f"  {len(pairs)} test pairs ready")

    print("\nComposing messages (this calls the LLM for each pair)...\n")

    results = []
    for i, pair in enumerate(pairs):
        test_id = pair["test_id"]
        trg_id = pair["trigger_id"]
        mid = pair["merchant_id"]
        cid = pair.get("customer_id")

        trigger = triggers.get(trg_id, {})
        merchant = merchants.get(mid, {})
        category = categories.get(merchant.get("category_slug", ""), {})
        customer = customers.get(cid) if cid else None

        print(f"  [{test_id}] {trigger.get('kind', '?'):30s} -> {merchant.get('identity', {}).get('name', mid)[:30]}", end="")

        try:
            start = time.time()
            composed = compose(category, merchant, trigger, customer)
            elapsed = time.time() - start

            result = {
                "test_id": test_id,
                "body": composed["body"],
                "cta": composed["cta"],
                "send_as": composed["send_as"],
                "suppression_key": composed["suppression_key"],
                "rationale": composed["rationale"],
            }
            results.append(result)
            print(f"  [PASS] ({elapsed:.1f}s)")
        except Exception as e:
            print(f"  [FAIL] Error: {e}")
            results.append({
                "test_id": test_id,
                "body": f"Hi, we have an update for you. Reply YES for details.",
                "cta": "binary_yes_no",
                "send_as": "vera",
                "suppression_key": trigger.get("suppression_key", ""),
                "rationale": f"Fallback due to error: {str(e)[:100]}",
            })
        
        # Sleep to avoid hitting OpenRouter's free tier rate limits (429 Too Many Requests)
        time.sleep(4)

    # Write JSONL
    with open(output_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n[PASS] Wrote {len(results)} lines to {output_file}")
    print(f"  Preview of first entry:")
    if results:
        preview = results[0]
        print(f"    test_id: {preview['test_id']}")
        print(f"    body: {preview['body'][:120]}...")
        print(f"    cta: {preview['cta']}")
        print(f"    send_as: {preview['send_as']}")


if __name__ == "__main__":
    main()
