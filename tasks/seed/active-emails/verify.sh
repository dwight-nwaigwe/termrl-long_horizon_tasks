python3 -I - <<'EOF'
import sys
import random
rng = random.Random(5)
first = ["ana", "bo", "chen", "dara", "eli", "fay", "gus", "hana"]
domains = ["example.com", "mail.org", "corp.net"]
users = []
for i in range(40):
    name = rng.choice(first)
    email = f"{name}{rng.randint(1, 6)}@{rng.choice(domains)}"
    email = "".join(c.upper() if rng.random() < 0.3 else c for c in email)
    users.append({"id": i, "name": name.title(), "email": email, "active": rng.random() < 0.6})
want = sorted({u["email"].lower() for u in users if u["active"]})
try:
    have = [l.strip() for l in open("active_emails.txt").read().strip().splitlines()]
except FileNotFoundError:
    sys.exit("active_emails.txt not found")
if have != want:
    sys.exit(f"got {len(have)} lines, expected {len(want)}; first diff: {[x for x in zip(have, want) if x[0] != x[1]][:1]}")
EOF
