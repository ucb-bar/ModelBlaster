# Runs detached on a node: venv + torch. Writes /tmp/krisd/farm/setup.status (running|done|failed).
F=/tmp/krisd/farm; mkdir -p $F; cd /tmp/krisd
echo running > $F/setup.status
export PATH=/home/utils/Python-3.12.5/bin:$PATH
if [ ! -x venv/bin/python ]; then python3 -m venv venv || { echo failed > $F/setup.status; exit 1; }; fi
. venv/bin/activate
pip -q install --upgrade pip
pip -q install torch --index-url https://download.pytorch.org/whl/cu126 &&
pip -q install numpy pyyaml requests pillow fastapi uvicorn openai boto3 aiohttp pydantic pandas loguru python-dotenv psutil einops || { echo failed > $F/setup.status; exit 1; }
python -c "import torch;assert torch.cuda.is_available()" || { echo failed > $F/setup.status; exit 1; }
echo done > $F/setup.status
