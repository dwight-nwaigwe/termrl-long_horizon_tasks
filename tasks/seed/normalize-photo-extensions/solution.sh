cd photos
for f in *; do
  case "${f,,}" in
    *.jpg|*.jpeg) base="${f%.*}"; [ "$f" = "$base.jpg" ] || mv -- "$f" "$base.jpg" ;;
  esac
done
