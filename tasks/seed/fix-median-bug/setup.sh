mkdir -p calc
cat > calc/stats.py <<'EOF'
def mean(xs):
    return sum(xs) / len(xs)


def median(xs):
    n = len(xs)
    mid = n // 2
    if n % 2 == 1:
        return xs[mid]
    return (xs[mid - 1] + xs[mid]) / 2
EOF
cat > calc/test_stats.py <<'EOF'
import unittest

from stats import mean, median


class TestStats(unittest.TestCase):
    def test_mean(self):
        self.assertEqual(mean([1, 2, 3, 4]), 2.5)

    def test_median_odd(self):
        self.assertEqual(median([9, 1, 5]), 5)

    def test_median_even(self):
        self.assertEqual(median([10, 2, 8, 4]), 6)


if __name__ == "__main__":
    unittest.main()
EOF
