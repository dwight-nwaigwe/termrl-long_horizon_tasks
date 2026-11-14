python3 - <<'EOF'
import json
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
json.dump(users, open("users.json", "w"), indent=2)
EOF
