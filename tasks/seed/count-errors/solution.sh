awk '$2 == "ERROR"' logs/app.log | wc -l > error_count.txt
