python3 - <<'EOF'
import sqlite3
row = sqlite3.connect("shop.db").execute(
    "SELECT c.name FROM customers c JOIN orders o ON o.customer_id = c.id "
    "WHERE o.status = 'completed' GROUP BY c.id ORDER BY SUM(o.amount_cents) DESC LIMIT 1").fetchone()
open("top_customer.txt", "w").write(row[0] + "\n")
EOF
