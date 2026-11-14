want=$'a/big1.bin\nb/c/d/medium.log\nb/c/huge.dat'
have=$(sed 's/[[:space:]]*$//' large_files.txt 2>/dev/null | sed '/^$/d')
[ "$have" = "$want" ] || { echo "large_files.txt is wrong:"; echo "$have"; exit 1; }
