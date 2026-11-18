#!/bin/bash
# Node-side template of the batch "holder" job (prototype; see analysis/batch_interface/REPORT.md).
# Submitted by batch_nodes.submit_holder() with  sbatch --signal=B:USR1@<warn_s> ...  and fed through sbatch's stdin.
# Contract (env, set by submit_holder via --export):
#   MB_WORKDIR  node-local dir for the markers (default /tmp/krisd/batchproto)   -> $D = $MB_WORKDIR/$SLURM_JOB_ID
#   MB_HOOK     optional absolute path of a node-side script run on the warning (e.g. the farm's pack2.sh wrapper)
#   MB_LIFE_S   optional: exit by ourselves after this many seconds (tests); default: hold until the Slurm limit
# Markers written in $D (the farm's probe reads them through `srun --jobid=J --overlap`):
#   started      epoch + host at script start          warn      epoch of the USR1 warning (= expiry approaching)
#   hook.rc      exit code of $MB_HOOK                 term      epoch of SIGTERM (the limit was reached / scancel)
#   exit         epoch + reason when the script ends
# What keeps the node: the batch script itself. While it is alive the allocation is alive; srun --overlap steps from
# the frontend ride on it. When it exits (rc 0 or not) Slurm ends the job and kills every step on the node.
D=${MB_WORKDIR:-/tmp/krisd/batchproto}/$SLURM_JOB_ID
mkdir -p "$D"
echo "$(date +%s) $(hostname) job=$SLURM_JOB_ID" > "$D/started"
on_warn() {                       # USR1 arrives @warn_s seconds before the limit (Slurm resolution is ~60 s)
  echo "$(date +%s)" > "$D/warn"
  if [ -n "$MB_HOOK" ] && [ -x "$MB_HOOK" ]; then "$MB_HOOK" "$D" "$SLURM_JOB_ID" > "$D/hook.log" 2>&1; echo $? > "$D/hook.rc"; fi
}
on_term() {                       # SIGTERM = limit reached (SIGKILL follows after KillWait) or scancel
  echo "$(date +%s)" > "$D/term"
  echo "$(date +%s) term" > "$D/exit"
  exit 0
}
trap on_warn USR1
trap on_term TERM INT
# bash only runs a trap between commands, so never block in a foreground child: sleep in the background and `wait`
# (wait returns at once when a trapped signal arrives; loop until the sleep really ends).
if [ -n "$MB_LIFE_S" ]; then sleep "$MB_LIFE_S" & else sleep infinity & fi
S=$!
while kill -0 $S 2>/dev/null; do wait $S; done
echo "$(date +%s) script-exit" > "$D/exit"
exit 0
