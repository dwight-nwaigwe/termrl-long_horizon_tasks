python3 -I - <<'EOF'
import os, sys
want = {"IMG_001.jpg": "IMG_001.JPG", "IMG_002.jpg": "IMG_002.jpeg", "IMG_003.jpg": "IMG_003.jpg",
        "IMG_004.jpg": "IMG_004.JPEG", "beach.jpg": "beach.Jpg", "notes.txt": "notes.txt"}
have = set(os.listdir("photos")) if os.path.isdir("photos") else set()
if have != set(want):
    sys.exit(f"photos/ contains {sorted(have)}, expected {sorted(want)}")
for new, old in want.items():
    if open(f"photos/{new}").read() != old + "\n":
        sys.exit(f"content of {new} changed")
EOF
