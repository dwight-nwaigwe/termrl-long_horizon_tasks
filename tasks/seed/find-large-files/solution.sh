(cd data && find . -type f -size +1048576c | sed 's|^\./||' | LC_ALL=C sort) > large_files.txt
