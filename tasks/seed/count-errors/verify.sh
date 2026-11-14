got=$(tr -d '[:space:]' < error_count.txt 2>/dev/null)
[ "$got" = "42" ] || { echo "expected 42, got '$got'"; exit 1; }
