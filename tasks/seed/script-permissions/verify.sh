python3 -I - <<'EOF'
import os, stat, sys
want = {"deploy.sh": 0o700, "backup.sh": 0o700, "tools/cleanup.sh": 0o700,
        "README.md": 0o600, "config.env": 0o600, "tools/list.txt": 0o600, ".": 0o755, "tools": 0o755}
for rel, mode in want.items():
    p = os.path.join("scripts", rel)
    if not os.path.exists(p):
        sys.exit(f"{p} missing")
    got = stat.S_IMODE(os.stat(p).st_mode)
    if got != mode:
        sys.exit(f"{p} has mode {oct(got)}, expected {oct(mode)}")
EOF
