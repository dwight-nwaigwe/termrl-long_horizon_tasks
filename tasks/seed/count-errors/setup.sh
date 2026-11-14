mkdir -p logs
python3 - <<'EOF'
lines = []
for i in range(1, 301):
    lvl = "ERROR" if i % 7 == 0 else ("WARN" if i % 5 == 0 else "INFO")
    msg = "disk ERROR rate nominal" if (lvl == "INFO" and i % 11 == 0) else f"request {i} handled"
    lines.append(f"2024-03-01T10:{i // 60:02d}:{i % 60:02d} {lvl} {msg}")
open("logs/app.log", "w").write("\n".join(lines) + "\n")
EOF
