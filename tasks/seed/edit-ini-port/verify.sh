python3 -I - <<'EOF'
import configparser, sys
text = open("config/app.ini").read()
if "; main application config" not in text:
    sys.exit("comment line removed")
c = configparser.ConfigParser()
c.read_string(text)
checks = {("server", "port"): "9090", ("server", "host"): "0.0.0.0", ("database", "port"): "5432",
          ("database", "host"): "db.internal", ("database", "name"): "shop"}
for (sec, key), want in checks.items():
    got = c.get(sec, key, fallback=None)
    if got != want:
        sys.exit(f"[{sec}] {key} = {got}, expected {want}")
EOF
