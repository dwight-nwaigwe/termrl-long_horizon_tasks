find scripts -type f -name '*.sh' -exec chmod 700 {} +
find scripts -type f ! -name '*.sh' -exec chmod 600 {} +
