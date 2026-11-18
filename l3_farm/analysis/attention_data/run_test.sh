set -e
D=/tmp/krisd/attn; cd $D
export PATH=/home/scratch.svc_compute_arch/release/cuda_toolkit/public/12.6.1/x86_64/u22.04/bin:$PATH
export CUDA_HOME=/home/scratch.svc_compute_arch/release/cuda_toolkit/public/12.6.1/x86_64/u22.04
. /tmp/krisd/venv/bin/activate
rm -rf pkg && mkdir -p pkg/modelblaster && tar xzf proto.tgz -C pkg/modelblaster
export PYTHONPATH=$D/pkg TMPDIR=$D/cache
echo "others: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)"
nvcc -O3 -arch=sm_89 -I pkg/modelblaster/gpu_eval/harness/cuda -o seed_bench seed_bench.cu 2>&1 | grep -v warning | head -5 || true
./seed_bench || true
rm -rf work && mkdir -p work/irs
for m in mini43 mini50 mini31 mini44; do
  python -m modelblaster.pipeline.extract_graph --bench-file minis/$m.py --bench-target-mb 0 --quant fp16 --out-dir work/irs/$m > work/extract_$m.log 2>&1 || { echo "extract $m failed"; tail -5 work/extract_$m.log; }
done
python test_attn.py $D/work mini43 mini50 mini31 mini44 2>work/test.err || { echo rc=$?; tail -30 work/test.err; }
