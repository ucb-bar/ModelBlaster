#!/bin/bash
# Run the installation of the dependencies
git clone --recurse-submodules https://github.com/SakanaAI/robust-kbench.git
nvidia-smi

# Install python3.11, pip, development headers, and CUDA toolkit with non-interactive settings
export DEBIAN_FRONTEND=noninteractive
export NEEDRESTART_MODE=a
apt-get update
apt-get install -y python3.11
apt-get install -y python3.11-dev
apt-get install -y python3-pip
apt-get install -y build-essential

# Use existing CUDA installation (detected CUDA 12.4)
echo "Using existing CUDA installation..."

# Set up CUDA environment variables to use existing CUDA
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

# Add python3.11 to PATH for current and future sessions
PY311_PATH=$(dirname $(which python3.11))
export PATH="$PY311_PATH:$PATH"
echo "export PATH=\"$PY311_PATH:\$PATH\"" >> ~/.bashrc

# Set up alternatives for both python3 and python to use python3.11
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 2
update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Also create symlinks as backup (in case alternatives don't work)
if [ ! -f /usr/local/bin/python ]; then
    ln -sf /usr/bin/python3.11 /usr/local/bin/python
fi

# Apply aliases for current session
alias python=python3.11
alias python3=python3.11

# Verify the setup
echo "Python version check:"
python3.11 --version
python3 --version || echo "python3 not working"
python --version || echo "python not working"

echo "Python 3.11 setup completed"

# Verify CUDA installation
echo "Verifying CUDA installation:"
echo "CUDA_HOME: $CUDA_HOME"
nvcc --version || echo "CUDA compiler not found"
ls -la /usr/local/cuda/include/cuda.h || echo "CUDA headers not found"
ls -la /usr/local/cuda-12.4/include/cuda.h || echo "CUDA 12.4 headers not found"

# Install the dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Fix setuptools version for PyCUDA compatibility
pip install --upgrade pip setuptools wheel
pip install "setuptools<70" wheel

cd robust-kbench
pip install -e .
echo "Robust-kbench dependencies installed"

# Check CUDA version with PyTorch
echo "Checking CUDA version with PyTorch:"
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'Number of GPUs: {torch.cuda.device_count()}'); print(f'GPU names: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}')"

python run_kernel.py --task_dir tasks/llama_ffw --cuda_code_path highlighted/llama_ffw/forward/kernel.cu
python run_kernel.py --task_dir tasks/llama_rmsnorm --cuda_code_path highlighted/llama_rmsnorm/forward/kernel.cu
python run_kernel.py --task_dir tasks/mnist_conv_relu_pool --cuda_code_path highlighted/mnist_conv_relu_pool/forward/kernel.cu
python run_kernel.py --task_dir tasks/mnist_cross_entropy --cuda_code_path highlighted/mnist_cross_entropy/forward/kernel.cu
python run_kernel.py --task_dir tasks/mnist_linear --cuda_code_path highlighted/mnist_linear/forward/kernel.cu
python run_kernel.py --task_dir tasks/mnist_linear_relu --cuda_code_path highlighted/mnist_linear_relu/forward/kernel.cu
python run_kernel.py --task_dir tasks/resnet_block --cuda_code_path highlighted/resnet_block/forward/kernel.cu
python run_kernel.py --task_dir tasks/mnist_cross_entropy --cuda_code_path highlighted/mnist_cross_entropy/backward/kernel.cu --backward
python run_kernel.py --task_dir tasks/mnist_linear --cuda_code_path highlighted/mnist_linear/backward/kernel.cu --backward
python run_kernel.py --task_dir tasks/mnist_linear_relu --cuda_code_path highlighted/mnist_linear_relu/backward/kernel.cu --backward
python run_kernel.py --task_dir tasks/mnist_pool --cuda_code_path highlighted/mnist_pool/backward/kernel.cu --backward
python run_kernel.py --task_dir tasks/layernorm --cuda_code_path highlighted/layernorm/forward/kernel.cu
echo "Evaluation completed"