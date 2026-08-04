#!/usr/bin/env bash
#
# Collect everything we'd want to see after a QRB5165 board crash or
# wedge. Runs against a live board (via SSH) and pulls back:
#
#   - dmesg.{0,current} (last 2 boots)
#   - kern.log filtered to qnn / adsprpc / fastrpc / subsys / oops / panic
#   - syslog tail filtered the same way
#   - listing of /system/rfs/msm/cdsp/ramdumps/pd_dump_/frpc/
#     (the cDSP user-PD crash dumps — see
#      qnn_models/QRB5165_MULTIGRAPH_CDSP_CRASH_FORENSICS.md)
#   - /data/vendor/ssrdump/ contents (sub-system restart dumps)
#   - uptime, load, memory
#
# Output: a single tarball at $OUT (default: qrb_diagnostics_<host>_<ts>.tar.gz).
#
# Usage:
#   bash collect_board_diagnostics.sh                       # uses $QNN_BOARD_HOST or default
#   bash collect_board_diagnostics.sh root@<BOARD_IP>
#   bash collect_board_diagnostics.sh qrb_cloud out_dir/
#
# Notes:
#   - Does NOT pull the actual ramdump .elfs (they're 18 MB+ each; just lists names + sizes).
#     Pull a specific one manually if you need to disassemble.
#   - Safe to run repeatedly; never modifies anything on the board.

set -uo pipefail

BOARD="${1:-${QNN_BOARD_HOST:-root@<BOARD_IP>}}"
OUT_DIR="${2:-.}"
TS=$(date +%Y%m%d_%H%M%S)
HOST_SLUG=$(echo "$BOARD" | sed 's/[^a-zA-Z0-9]/_/g')
OUT="$OUT_DIR/qrb_diagnostics_${HOST_SLUG}_${TS}.tar.gz"
TMP_REMOTE=/tmp/qnn_diag_$$

echo "==> collecting diagnostics from $BOARD → $OUT"

ssh -o ConnectTimeout=10 "$BOARD" bash <<EOF
set -uo pipefail
rm -rf $TMP_REMOTE
mkdir -p $TMP_REMOTE

# --- system state ---
{
    echo "===== uptime ====="
    uptime
    echo
    echo "===== date ====="
    date
    echo
    echo "===== meminfo ====="
    head -5 /proc/meminfo
    echo
    echo "===== mounts (relevant) ====="
    mount | grep -E "(^/|tmpfs)" | head -20
} > $TMP_REMOTE/00_sysinfo.txt 2>&1

# --- dmesg (current + last boot) ---
cp /var/log/dmesg     $TMP_REMOTE/10_dmesg.current 2>/dev/null || dmesg > $TMP_REMOTE/10_dmesg.current
[ -f /var/log/dmesg.0 ]    && cp /var/log/dmesg.0    $TMP_REMOTE/11_dmesg.previous
[ -f /var/log/dmesg.1.gz ] && cp /var/log/dmesg.1.gz $TMP_REMOTE/12_dmesg.2_prior.gz

# --- kern.log / syslog filtered ---
grep -iE "cdsp|adsp|fastrpc|qnn|hexagon|subsys|ssr|q6v5|oops|panic|BUG:|SError|wdog|hardware error|signal" \
    /var/log/kern.log 2>/dev/null | tail -2000 > $TMP_REMOTE/20_kernlog_filtered.txt
grep -iE "cdsp|adsp|fastrpc|qnn|hexagon|subsys|ssr|q6v5|oops|panic|BUG:|crash" \
    /var/log/syslog 2>/dev/null   | tail -2000 > $TMP_REMOTE/21_syslog_filtered.txt

# --- ramdump listing (cDSP user-PD crashes) ---
echo "===== /system/rfs/msm/cdsp/ramdumps (cDSP user-PD crash dumps) =====" \
    > $TMP_REMOTE/30_ramdumps.txt
ls -laR /system/rfs/msm/cdsp/ramdumps/ 2>/dev/null >> $TMP_REMOTE/30_ramdumps.txt
echo "" >> $TMP_REMOTE/30_ramdumps.txt
echo "===== count by binary =====" >> $TMP_REMOTE/30_ramdumps.txt
ls /system/rfs/msm/cdsp/ramdumps/pd_dump_*/frpc/ 2>/dev/null \
    | awk '{print \$NF}' | sed 's/^[a-f0-9]* //; s/\.[0-9]*\.elf//' \
    | sort | uniq -c | sort -rn >> $TMP_REMOTE/30_ramdumps.txt

# --- ssrdump (subsystem restart dumps) ---
echo "===== /data/vendor/ssrdump =====" > $TMP_REMOTE/31_ssrdump.txt
ls -la /data/vendor/ssrdump/ 2>/dev/null >> $TMP_REMOTE/31_ssrdump.txt
echo "" >> $TMP_REMOTE/31_ssrdump.txt
echo "===== /persist =====" >> $TMP_REMOTE/31_ssrdump.txt
ls -laR /persist 2>/dev/null | head -50 >> $TMP_REMOTE/31_ssrdump.txt

# --- recent QNN-related processes / state ---
{
    echo "===== ps aux (qnn-related) ====="
    ps aux 2>/dev/null | grep -iE "qnn|qair|qrt|fastrpc|adsprpc" | grep -v grep
    echo
    echo "===== /dev/fastrpc* ====="
    ls -la /dev/*fastrpc* 2>/dev/null
    echo
    echo "===== /sys/kernel/debug/fastrpc (if accessible) ====="
    ls /sys/kernel/debug/fastrpc/ 2>/dev/null
} > $TMP_REMOTE/40_qnn_state.txt 2>&1

# --- /tmp ctx-gen logs (qnn-context-binary-generator stderr) ---
{
    echo "===== /tmp/_ctxgen_*.log tails ====="
    for f in /tmp/_ctxgen_*.log /tmp/_bld_*.log /tmp/_mg_*.log; do
        [ -f "\$f" ] || continue
        echo "--- \$f ---"
        tail -10 "\$f"
        echo
    done
} > $TMP_REMOTE/50_ctxgen_logs.txt 2>&1

tar -czf $TMP_REMOTE.tar.gz -C $TMP_REMOTE .
echo "DIAG_TARBALL=$TMP_REMOTE.tar.gz"
EOF

if [ $? -ne 0 ]; then
    echo "  ! ssh collection failed; board may be wedged. Tarball not produced."
    exit 1
fi

scp -q "$BOARD:$TMP_REMOTE.tar.gz" "$OUT"
ssh "$BOARD" "rm -rf $TMP_REMOTE $TMP_REMOTE.tar.gz" 2>/dev/null

if [ ! -f "$OUT" ]; then
    echo "  ! tarball not created — diagnostics collection failed"
    exit 1
fi

echo "==> wrote $OUT  ($(du -h "$OUT" | cut -f1))"
echo ""
echo "Contents:"
tar -tzf "$OUT" | sed 's/^/    /'
echo ""
echo "To inspect:"
echo "  tar -xzf $OUT -C ./diag_extracted"
echo "  less diag_extracted/20_kernlog_filtered.txt"
