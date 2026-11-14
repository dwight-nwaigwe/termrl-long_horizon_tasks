jq -r '.[] | select(.active == true) | .email | ascii_downcase' users.json | LC_ALL=C sort -u > active_emails.txt
