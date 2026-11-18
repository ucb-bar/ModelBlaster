# Args: <trace dir> <sha>
D=$1; S=$2; f=$(ls $D/blobs/${S:0:2}/$S* 2>/dev/null | head -1); [ -z "$f" ] && f=$(find $D/blobs -name "$S*" | head -1)
case "$f" in *.gz|*.zst) (zcat "$f" 2>/dev/null || zstd -dc "$f");; *) cat "$f";; esac
