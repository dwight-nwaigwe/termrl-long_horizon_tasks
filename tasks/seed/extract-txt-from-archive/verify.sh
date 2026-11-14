python3 -I - <<'EOF'
import os, sys
want = {"docs/a.txt": "alpha\n", "notes/b.txt": "bravo\n", "notes/deep/c.txt": "charlie\n"}
have = {}
for root, _, files in os.walk("restored"):
    for f in files:
        p = os.path.join(root, f)
        have[os.path.relpath(p, "restored")] = open(p, errors="replace").read()
if set(have) != set(want):
    sys.exit(f"restored/ contains {sorted(have)}, expected {sorted(want)}")
if have != want:
    sys.exit("file contents differ from the archive")
EOF
