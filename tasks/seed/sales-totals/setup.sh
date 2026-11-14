python3 - <<'EOF'
import random
rng = random.Random(3)
regions = ["north", "south", "east", "west"]
rows = []
for d in range(200):
    rows.append((f"2024-{1 + d % 12:02d}-{1 + d % 28:02d}", rng.choice(regions), rng.randint(100, 99999)))
with open("sales.csv", "w") as f:
    f.write("date,region,amount\n")
    for date, region, cents in rows:
        f.write(f"{date},{region},{cents / 100:.2f}\n")
EOF
