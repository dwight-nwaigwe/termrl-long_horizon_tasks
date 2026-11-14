cd calc 2>/dev/null || { echo "calc/ missing"; exit 1; }
grep -q 'self.assertEqual(median(\[10, 2, 8, 4\]), 6)' test_stats.py || { echo "test file was modified"; exit 1; }
python3 -I - <<'EOF'
import sys
sys.path.insert(0, ".")
from stats import mean, median
cases = [([3, 1, 2], 2), ([4, 1, 3, 2], 2.5), ([5], 5), ([7, 7, 1, 100, 3], 7), ([-1, -5], -3)]
for xs, want in cases:
    orig = list(xs)
    got = median(xs)
    if got != want:
        sys.exit(f"median({orig}) = {got}, expected {want}")
    if xs != orig:
        sys.exit("median() modified its input")
if mean([1, 2, 3, 4]) != 2.5:
    sys.exit("mean() broken")
EOF
[ $? -eq 0 ] || exit 1
python3 -m unittest -q 2>&1 | tail -3
python3 -m unittest -q >/dev/null 2>&1 || { echo "unit tests fail"; exit 1; }
