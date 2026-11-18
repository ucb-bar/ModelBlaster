# Args: <instance farm dir under /tmp/krisd/fsched_*> <name>... Removes those problem dirs of a smoke-test instance.
F=$1; shift
case "$F" in /tmp/krisd/fsched_*) ;; *) echo refused; exit 1;; esac
for n in "$@"; do rm -rf "$F/$n"; done; echo removed
