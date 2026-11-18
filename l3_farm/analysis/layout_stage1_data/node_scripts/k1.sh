# standalone kernel build + correctness + sanitizer (+ bench when asked)
set -e
cd /tmp/krisd/x142; export HOME=/tmp/krisd/x142/home TRITON_CACHE_DIR=/tmp/krisd/x142/home/triton TMPDIR=/tmp/krisd/x142/tmp; mkdir -p $HOME $TMPDIR
export PATH=/home/scratch.svc_compute_arch/release/cuda_toolkit/public/12.6.1/x86_64/u22.04/bin:$PATH
. /tmp/krisd/venv/bin/activate
cat /tmp/krisd/x142/setup.status
cd /tmp/krisd/x142/kt; mkdir -p out
nvidia-smi --query-gpu=name,clocks.max.mem,clocks.mem --format=csv,noheader
python - <<'PY'
import kernel_lib as KL
KL.build("mb_cuda_kernels_f16.cu", "libnew.so"); KL.build("naive_f16.cu", "libnaive.so"); print("built")
PY
python kernel_test.py ./libnew.so cases_real.json --small 1500 --out out/correct_new.json 2>&1 | tail -80
