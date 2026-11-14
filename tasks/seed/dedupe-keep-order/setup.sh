python3 - <<'EOF'
import random
rng = random.Random(13)
pool = ["Zoe", "Liam", "Mia", "Noah", "Ava", "Omar", "Ivy", "Theo", "Lena", "Ravi", "Kai", "Nora"]
names = [rng.choice(pool) for _ in range(60)]
open("names.txt", "w").write("\n".join(names) + "\n")
EOF
