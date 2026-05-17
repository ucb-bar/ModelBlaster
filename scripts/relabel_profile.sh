#!/usr/bin/env bash
# Relabel an existing gen/profile/<hw>/<src_cpu>/... tree to a different
# hw / cpu naming scheme — used when the actual provenance label
# (e.g. "spike", "rvv") doesn't match the target HW label XPU-RT wants
# (e.g. "generic_riscv64", "RVV").
#
# Renames both the path components AND the inner spec dirs (which embed
# the same labels), keeping CSV contents untouched.
#
# Usage:
#   bash modelblaster/scripts/relabel_profile.sh \
#       --src-hw=rvv     --dst-hw=RVV \
#       --src-cpu=spike  --dst-cpu=generic_riscv64
set -euo pipefail

SRC_HW=""
DST_HW=""
SRC_CPU=""
DST_CPU=""
ROOT="${ROOT:-gen/profile}"

for arg in "$@"; do
    case "$arg" in
        --src-hw=*)  SRC_HW="${arg#*=}" ;;
        --dst-hw=*)  DST_HW="${arg#*=}" ;;
        --src-cpu=*) SRC_CPU="${arg#*=}" ;;
        --dst-cpu=*) DST_CPU="${arg#*=}" ;;
        --root=*)    ROOT="${arg#*=}" ;;
        *) echo "unknown arg: $arg" >&2; exit 2 ;;
    esac
done

if [[ -z "${SRC_HW}" || -z "${DST_HW}" || -z "${SRC_CPU}" || -z "${DST_CPU}" ]]; then
    echo "usage: $0 --src-hw=<...> --dst-hw=<...> --src-cpu=<...> --dst-cpu=<...>" >&2
    exit 2
fi

src_root="${ROOT}/${SRC_HW}/${SRC_CPU}"
dst_root="${ROOT}/${DST_HW}/${DST_CPU}"

if [[ ! -d "${src_root}" ]]; then
    echo "[relabel] no source dir at ${src_root}, nothing to do"
    exit 0
fi

mkdir -p "$(dirname "${dst_root}")"
mv "${src_root}" "${dst_root}"
echo "[relabel] ${src_root}  ->  ${dst_root}"

# The inner "spec" subdir bakes (model)_(cpu)_(hw)_(model.quant) into
# its name; rename so it tracks the new labels too.
shopt -s nullglob
for spec_dir in "${dst_root}"/*/*/*/; do
    spec_dir="${spec_dir%/}"
    spec_name="$(basename "${spec_dir}")"
    new_name="${spec_name//_${SRC_CPU}_/_${DST_CPU}_}"
    new_name="${new_name//_${SRC_HW}_/_${DST_HW}_}"
    if [[ "${spec_name}" != "${new_name}" ]]; then
        new_dir="$(dirname "${spec_dir}")/${new_name}"
        mv "${spec_dir}" "${new_dir}"
        echo "[relabel]   spec: ${spec_name}  ->  ${new_name}"
    fi
done

# Drop now-empty parent dirs (e.g. gen/profile/<src_hw>/) so the tree is
# clean.
rmdir "${ROOT}/${SRC_HW}" 2>/dev/null || true
echo "[relabel] done."
