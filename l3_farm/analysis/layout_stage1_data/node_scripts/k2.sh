# slot-alone compile, arch variants, sanitizers, sanity bench
cd /tmp/krisd/x142; export HOME=/tmp/krisd/x142/home TRITON_CACHE_DIR=/tmp/krisd/x142/home/triton TMPDIR=/tmp/krisd/x142/tmp; mkdir -p $HOME $TMPDIR
export PATH=/home/scratch.svc_compute_arch/release/cuda_toolkit/public/12.6.1/x86_64/u22.04/bin:$PATH
. /tmp/krisd/venv/bin/activate
cd kt; mkdir -p out
echo "== arch compile (library TU): sm_75 sm_80 sm_89 sm_90"
for a in sm_75 sm_80 sm_89 sm_90; do nvcc -O3 -arch=$a -c -std=c++17 -o /dev/null mb_cuda_kernels_f16.cu && echo "$a ok"; done
echo "== ptxas resource usage of k_bmm_strided (sm_89)"
nvcc -O3 -arch=sm_89 -Xptxas -v -c -o /dev/null mb_cuda_kernels_f16.cu 2>&1 | grep -A2 "k_bmm_strided" | head -6
echo "== sanitizers (small cases, no real shapes)"
for t in memcheck racecheck synccheck initcheck; do
  echo "-- $t"; compute-sanitizer --tool $t --print-limit 5 python kernel_test.py ./libnew.so cases_real.json --no-real --small 40 --seed 7 2>&1 | tail -6
done
echo "== sanity bench"
python bench.py ./libnew.so ./libnaive.so cases_real.json out/bench.json 2>&1 | tail -20
