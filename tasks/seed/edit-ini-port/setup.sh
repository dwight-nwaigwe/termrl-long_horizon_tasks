mkdir -p config
cat > config/app.ini <<'EOF'
; main application config
[server]
host = 0.0.0.0
port = 8080

[database]
host = db.internal
port = 5432
name = shop
EOF
