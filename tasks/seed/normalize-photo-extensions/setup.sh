mkdir -p photos
for f in IMG_001.JPG IMG_002.jpeg IMG_003.jpg IMG_004.JPEG beach.Jpg notes.txt; do
  echo "$f" > "photos/$f"
done
