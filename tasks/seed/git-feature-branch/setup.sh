mkdir project && cd project
git init -q -b main
git config user.name "Dev"
git config user.email "dev@example.com"
printf '# Project\n' > README.md
printf 'print("hi")\n' > app.py
git add . && git commit -qm "Initial commit"
printf '# Project\n\nUsage: python3 app.py\n' > README.md
