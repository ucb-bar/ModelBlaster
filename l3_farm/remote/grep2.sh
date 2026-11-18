# Args: <file> <pattern> [max lines]. grep a file on the node (read-only).
grep -n -E "$2" "$1" 2>/dev/null | tail -${3:-40} | cut -c1-400
