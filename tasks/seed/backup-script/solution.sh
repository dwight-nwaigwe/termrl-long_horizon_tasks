cat > backup.sh <<'EOF'
#!/usr/bin/env bash
if [ $# -lt 1 ] || [ ! -d "$1" ]; then
  echo "usage: $0 DIR (DIR must be a directory)" >&2
  exit 1
fi
d="${1%/}"
tar -czf "$(basename "$d")-backup.tar.gz" "$d"
EOF
chmod +x backup.sh
