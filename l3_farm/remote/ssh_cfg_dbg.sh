# Which ssh options apply to a pool host (prints only host-key related settings). Args: <host>
ssh -G $1 </dev/null 2>/dev/null | grep -iE "^(userknownhostsfile|stricthostkeychecking|globalknownhostsfile|hashknownhosts|batchmode)"
