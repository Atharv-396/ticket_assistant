"""Quick pipeline smoke test — submits the 5 HR sample cases and prints results."""
import sys
import requests

BASE = "http://localhost:8000"

# Register + login a fresh test user
requests.post(f"{BASE}/register",
    json={"email": "pipeline_test@example.com", "password": "testpass123"})
r = requests.post(f"{BASE}/login",
    json={"email": "pipeline_test@example.com", "password": "testpass123"}, timeout=5)
token = r.json()["access_token"]
print("Authenticated OK\n")

test_cases = [
    ("S01", "My 3500 rupee order arrived damaged yesterday.", "REQUEST_PHOTOS"),
    ("S02", "I changed my mind about this unopened non-food product. It arrived 10 days ago.", "APPROVE_RETURN"),
    ("S03", "My parcel has still not arrived and it was dispatched 9 days ago.", "OPEN_SHIPPING_INVESTIGATION"),
    ("S04", "I ordered strawberry but received chocolate 2 days ago.", "REPLACE_CORRECT_ITEM"),
    ("S05", "I want to return this.", "NEEDS_MORE_INFORMATION"),
]

correct = 0
for case_id, message, expected in test_cases:
    r = requests.post(f"{BASE}/tickets",
        json={"message": message},
        headers={"Authorization": f"Bearer {token}"},
        timeout=60)
    if r.status_code == 201:
        d = r.json().get("decision", {})
        action = d.get("action", "NO_DECISION")
        confidence = d.get("confidence", 0)
        match = "✓" if action == expected else "✗"
        if action == expected:
            correct += 1
        print(f"{match} {case_id}")
        print(f"   Expected  : {expected}")
        print(f"   Predicted : {action}  ({confidence:.0%} confidence)")
        print(f"   Reason    : {d.get('reason','')[:80]}")
        print()
    else:
        print(f"✗ {case_id}: HTTP {r.status_code} — {r.json()}")

print(f"Result: {correct}/{len(test_cases)} correct")
