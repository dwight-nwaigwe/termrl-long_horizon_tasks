python3 - <<'EOF'
import csv
from collections import defaultdict
tot = defaultdict(float)
for r in csv.DictReader(open("sales.csv")):
    tot[r["region"]] += float(r["amount"])
with open("totals.csv", "w") as f:
    f.write("region,total\n")
    for k in sorted(tot):
        f.write(f"{k},{tot[k]:.2f}\n")
EOF
