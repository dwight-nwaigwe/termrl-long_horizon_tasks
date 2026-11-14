mkdir -p stage/docs stage/notes/deep
echo "alpha" > stage/docs/a.txt
printf '\x89PNG fake' > stage/docs/img.png
echo "bravo" > stage/notes/b.txt
echo "charlie" > stage/notes/deep/c.txt
echo "x,y" > stage/data.csv
echo "not text" > stage/notes/deep/c.txt.bak
tar -czf backup.tar.gz -C stage docs notes data.csv
rm -rf stage
