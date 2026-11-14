python3 - <<'EOF'
import sqlite3
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
db = sqlite3.connect("shop.db")
db.execute("CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT)")
db.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, amount_cents INTEGER, status TEXT)")
db.executemany("INSERT INTO customers VALUES (?, ?)", list(enumerate(customers, 1)))
db.executemany("INSERT INTO orders VALUES (?, ?, ?, ?)", orders)
db.commit()
EOF
