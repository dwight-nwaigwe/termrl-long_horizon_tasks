python3 - <<'EOF'
import os
sizes = {"a/small.bin": 1000, "a/big1.bin": 2 * 1048576, "b/c/huge.dat": 3 * 1048576,
         "b/exact.bin": 1048576, "b/c/d/medium.log": 1048576 + 1, "e/tiny.txt": 10, "notes.txt": 500}
for rel, n in sizes.items():
    p = os.path.join("data", rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "wb") as f:
        f.write(b"x" * n)
EOF
