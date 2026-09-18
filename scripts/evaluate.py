"""
Evaluation runner — multi-turn pipeline accuracy test.

Usage
─────
    python scripts/evaluate.py

Tests the LIVE decision pipeline against a set of representative cases.
Each case specifies:
  - An initial message
  - Optional follow-up messages (simulating the multi-turn flow)
  - The expected final action

The script does NOT look up answers from tickets.csv — it runs real
Gemini + RAG decisions.

Requirements
────────────
  - FastAPI backend running at API_BASE (default http://localhost:8000)
  - Knowledge base ingested
  - GEMINI_API_KEY configured
"""
import json
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

API_BASE = "http://localhost:8000"
EVAL_EMAIL = "eval_runner@evaltest.com"
EVAL_PASSWORD = "eval_runner_secret_2024x!"

# ── Test cases ────────────────────────────────────────────────────────────────
# Each case:
#   initial_message : str           — first user message
#   follow_ups      : list[str]     — subsequent user messages (may be empty)
#   expected_action : str           — the expected FINAL action
#   description     : str           — human-readable label

TEST_CASES = [
    {
        "id": "E01",
        "description": "High-value damage — photos required then approved",
        "initial_message": "My ₹3,500 order arrived damaged yesterday. The outer box and product are both crushed.",
        "follow_ups": ["Order value is ₹3,500, delivered 1 day ago. I have uploaded photos of the damage."],
        "expected_final_action": "APPROVE_REFUND_OR_REPLACEMENT",
        "expected_intermediate_action": "REQUEST_PHOTOS",
    },
    {
        "id": "E02",
        "description": "Low-value damage — immediate approval",
        "initial_message": "My ₹800 food item arrived broken. Delivered 2 days ago.",
        "follow_ups": [],
        "expected_final_action": "APPROVE_REFUND_OR_REPLACEMENT",
        "expected_intermediate_action": None,
    },
    {
        "id": "E03",
        "description": "Change of mind — unopened non-food within 14 days",
        "initial_message": "I changed my mind. The non-food product is still unopened. Arrived 10 days ago.",
        "follow_ups": [],
        "expected_final_action": "APPROVE_RETURN",
        "expected_intermediate_action": None,
    },
    {
        "id": "E04",
        "description": "Shipping delay 9 days — investigation",
        "initial_message": "My parcel has not arrived. It was dispatched 9 days ago.",
        "follow_ups": [],
        "expected_final_action": "OPEN_SHIPPING_INVESTIGATION",
        "expected_intermediate_action": None,
    },
    {
        "id": "E05",
        "description": "Wrong item — within 7 days",
        "initial_message": "I ordered strawberry but received chocolate. Delivered 2 days ago.",
        "follow_ups": [],
        "expected_final_action": "REPLACE_CORRECT_ITEM",
        "expected_intermediate_action": None,
    },
    {
        "id": "E06",
        "description": "Vague message — needs more information",
        "initial_message": "I want to return this.",
        "follow_ups": [],
        "expected_final_action": "NEEDS_MORE_INFORMATION",
        "expected_intermediate_action": None,
        "action_aliases": ["REQUEST_MORE_INFORMATION"],   # same intent, different name
    },
    {
        "id": "E07",
        "description": "Pre-dispatch cancellation",
        "initial_message": "Please cancel my order. It has not been dispatched yet.",
        "follow_ups": [],
        "expected_final_action": "CANCEL_AND_REFUND",
        "expected_intermediate_action": None,
    },
    {
        "id": "E08",
        "description": "Food return rejected",
        "initial_message": "I want to return this food product. I changed my mind.",
        "follow_ups": [],
        "expected_final_action": "REJECT_FOOD_RETURN",
        "expected_intermediate_action": None,
    },
    {
        "id": "E09",
        "description": "Multi-turn: defective product, high value → evidence requested → approved",
        "initial_message": "My ₹3,999 device is defective and stops working after a few minutes.",
        "follow_ups": ["Delivered 9 days ago. I can provide a video of the defect."],
        "expected_final_action": "REQUEST_DEFECT_EVIDENCE",
        "expected_intermediate_action": "REQUEST_DEFECT_EVIDENCE",
    },
    {
        "id": "E10",
        "description": "Long shipping delay — replacement or refund",
        "initial_message": "My parcel has still not arrived and it was dispatched 18 days ago.",
        "follow_ups": [],
        "expected_final_action": "OFFER_REPLACEMENT_OR_REFUND",
        "expected_intermediate_action": None,
    },
]


def _auth() -> str:
    try:
        requests.post(f"{API_BASE}/register",
                      json={"email": EVAL_EMAIL, "password": EVAL_PASSWORD}, timeout=10)
        r = requests.post(f"{API_BASE}/login",
                          json={"email": EVAL_EMAIL, "password": EVAL_PASSWORD}, timeout=10)
        if r.status_code != 200:
            print(f"ERROR: Login failed: {r.text}", file=sys.stderr)
            sys.exit(1)
        return r.json()["access_token"]
    except requests.ConnectionError:
        print(f"ERROR: Cannot connect to {API_BASE}. Is FastAPI running?", file=sys.stderr)
        sys.exit(1)


def _post_ticket(token: str, message: str) -> dict | None:
    try:
        r = requests.post(f"{API_BASE}/tickets", json={"message": message},
                          headers={"Authorization": f"Bearer {token}"}, timeout=90)
        return r.json() if r.status_code == 201 else {"error": f"HTTP {r.status_code}: {r.text[:100]}"}
    except Exception as e:
        return {"error": str(e)}


def _post_followup(token: str, ticket_id: int, content: str) -> dict | None:
    try:
        r = requests.post(f"{API_BASE}/tickets/{ticket_id}/messages",
                          json={"content": content},
                          headers={"Authorization": f"Bearer {token}"}, timeout=90)
        return r.json() if r.status_code == 200 else {"error": f"HTTP {r.status_code}: {r.text[:100]}"}
    except Exception as e:
        return {"error": str(e)}


def run():
    print(f"Evaluation runner — {len(TEST_CASES)} test cases")
    print(f"API: {API_BASE}\n")

    token = _auth()
    print("Authenticated.\n")

    correct = 0
    failures = []

    for case in TEST_CASES:
        cid = case["id"]
        desc = case["description"]
        print(f"[{cid}] {desc}")

        # Submit initial ticket
        resp = _post_ticket(token, case["initial_message"])
        time.sleep(4)  # respect free-tier rate limits between calls
        if not resp or "error" in resp:
            print(f"  ✗ TICKET ERROR: {resp}")
            failures.append({**case, "predicted": "API_ERROR", "error": str(resp)})
            continue

        ticket_id = resp["ticket"]["id"]
        current_decision = resp["decision"]

        # Check intermediate action if expected
        if case.get("expected_intermediate_action"):
            inter_pred = current_decision.get("action", "")
            inter_exp = case["expected_intermediate_action"]
            match_icon = "✓" if inter_pred == inter_exp else "~"
            print(f"  {match_icon} Intermediate: predicted={inter_pred} expected={inter_exp}")

        # Send follow-ups
        for fu in case.get("follow_ups", []):
            time.sleep(4)
            fu_resp = _post_followup(token, ticket_id, fu)
            if not fu_resp or "error" in fu_resp:
                print(f"  ✗ FOLLOW-UP ERROR: {fu_resp}")
                current_decision = None
                break
            current_decision = fu_resp.get("decision", current_decision)

        if current_decision is None:
            failures.append({**case, "predicted": "PIPELINE_ERROR"})
            continue

        predicted = current_decision.get("action", "NO_DECISION")
        expected = case["expected_final_action"]
        aliases = case.get("action_aliases", [])
        match = (predicted == expected) or (predicted in aliases)
        icon = "✓" if match else "✗"
        print(f"  {icon} Final: predicted={predicted}  expected={expected}  ({current_decision.get('confidence', 0):.0%})")
        if match:
            correct += 1
        else:
            failures.append({**case, "predicted": predicted})

        print()

    total = len(TEST_CASES)
    print("=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total  : {total}")
    print(f"Correct: {correct}")
    print(f"Wrong  : {total - correct}")
    print(f"Accuracy: {correct/total*100:.1f}%")

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  {f['id']}: expected={f['expected_final_action']}  predicted={f.get('predicted','?')}")

    print()


if __name__ == "__main__":
    run()
