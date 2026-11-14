mkdir -p scripts/tools
for f in deploy.sh backup.sh README.md config.env tools/cleanup.sh tools/list.txt; do echo "# $f" > "scripts/$f"; done
chmod 644 scripts/* scripts/tools/* 2>/dev/null
chmod 666 scripts/config.env
chmod 755 scripts scripts/tools
