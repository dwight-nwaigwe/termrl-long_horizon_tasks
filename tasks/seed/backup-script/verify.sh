fail() { echo "$1"; exit 1; }
[ -f backup.sh ] && [ -x backup.sh ] || fail "backup.sh missing or not executable"
t=$(mktemp -d) && cp backup.sh "$t/" && cd "$t" || fail "setup error"
mkdir -p mydata/sub && echo hello > mydata/sub/f.txt
timeout 30 ./backup.sh mydata >/dev/null 2>&1 || fail "./backup.sh mydata exited non-zero"
[ -f mydata-backup.tar.gz ] || fail "mydata-backup.tar.gz not created"
tar -tzf mydata-backup.tar.gz | grep -q 'mydata/sub/f.txt' || fail "archive does not contain mydata/sub/f.txt"
out=$(timeout 10 ./backup.sh 2>err1); rc=$?
[ $rc -eq 1 ] || fail "no-argument call exited $rc, expected 1"
[ -z "$out" ] && [ -s err1 ] || fail "no-argument call must print to stderr only"
out=$(timeout 10 ./backup.sh missing_dir 2>err2); rc=$?
[ $rc -eq 1 ] || fail "missing-dir call exited $rc, expected 1"
[ -z "$out" ] && [ -s err2 ] || fail "missing-dir call must print to stderr only"
