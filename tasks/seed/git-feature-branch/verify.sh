cd project 2>/dev/null || { echo "project/ missing"; exit 1; }
fail() { echo "$1"; exit 1; }
git rev-parse -q --verify refs/heads/feature/readme >/dev/null || fail "branch feature/readme not found"
[ "$(git log -1 --format=%s feature/readme)" = "Update README" ] || fail "wrong commit message"
[ "$(git rev-list --count main)" = "1" ] || fail "main moved"
[ "$(git rev-parse feature/readme~1)" = "$(git rev-parse main)" ] || fail "feature/readme is not one commit on top of main"
git show feature/readme:README.md | grep -q "Usage: python3 app.py" || fail "README change not committed"
[ -z "$(git status --porcelain)" ] || fail "working tree not clean"
