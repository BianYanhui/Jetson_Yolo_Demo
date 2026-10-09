#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY || true

python3 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install torch torchvision --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
pip install -r requirements.txt

if [[ ! -d third_party/YOLOv6/.git ]]; then
  mkdir -p third_party
  git clone --depth 1 git@github.com:meituan/YOLOv6.git third_party/YOLOv6
fi

mkdir -p weights
if [[ ! -f weights/yolov6n.pt ]]; then
  curl -L --retry 3 -o weights/yolov6n.pt \
    "https://ghfast.top/https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6n.pt"
fi

echo "Setup done. Ensure cudss is installed: sudo apt-get install -y cudss-cuda-12"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
