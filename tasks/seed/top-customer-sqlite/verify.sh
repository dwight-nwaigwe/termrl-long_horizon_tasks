python3 -I - <<'EOF'
import sys
import random
rng = random.Random(11)
customers = ["Avery", "Blake", "Casey", "Devon", "Emery", "Finley"]
orders = []
for i in range(120):
    orders.append((i + 1, rng.randint(1, 6), rng.randint(500, 20000), rng.choice(["completed"] * 4 + ["refunded", "pending"])))
spend = {c: 0 for c in range(1, 7)}
for _, cid, amt, st in orders:
    if st == "completed":
        spend[cid] += amt
low = min(spend, key=spend.get)
orders.append((121, low, 10_000_000, "refunded"))  # trap: huge refunded order
want = customers[max(spend, key=spend.get) - 1]
try:
    got = open("top_customer.txt").read().strip()
except FileNotFoundError:
    sys.exit("top_customer.txt not found")
if got != want:
    sys.exit(f"got {got!r}, expected {want!r}")
EOF
