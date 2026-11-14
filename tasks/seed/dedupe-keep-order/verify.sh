python3 -I - <<'EOF'
import sys
import random
rng = random.Random(13)
pool = ["Zoe", "Liam", "Mia", "Noah", "Ava", "Omar", "Ivy", "Theo", "Lena", "Ravi", "Kai", "Nora"]
names = [rng.choice(pool) for _ in range(60)]
want = list(dict.fromkeys(names))
try:
    have = [l.strip() for l in open("unique.txt").read().strip().splitlines()]
except FileNotFoundError:
    sys.exit("unique.txt not found")
if have != want:
    sys.exit(f"got {have}, expected {want}")
EOF
