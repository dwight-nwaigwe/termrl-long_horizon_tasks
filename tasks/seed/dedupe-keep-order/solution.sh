awk '!seen[$0]++' names.txt > unique.txt
