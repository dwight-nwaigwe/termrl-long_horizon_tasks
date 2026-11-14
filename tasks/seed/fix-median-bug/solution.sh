python3 - <<'EOF'
p = "calc/stats.py"
s = open(p).read()
s = s.replace("def median(xs):\n", "def median(xs):\n    xs = sorted(xs)\n")
open(p, "w").write(s)
EOF
