python3 -I - <<'EOF'
import sys
from collections import defaultdict
import random
rng = random.Random(3)
regions = ["north", "south", "east", "west"]
rows = []
for d in range(200):
    rows.append((f"2024-{1 + d % 12:02d}-{1 + d % 28:02d}", rng.choice(regions), rng.randint(100, 99999)))
tot = defaultdict(int)
for _, region, cents in rows:
    tot[region] += cents
want = ["region,total"] + [f"{k},{tot[k] // 100}.{tot[k] % 100:02d}" for k in sorted(tot)]
try:
    have = [l.strip() for l in open("totals.csv").read().strip().splitlines()]
except FileNotFoundError:
    sys.exit("totals.csv not found")
if have != want:
    sys.exit(f"totals.csv is {have}, expected {want}")
EOF
