# Installation

## Requirements

- Python 3.10+
- CUDA 12.x + compatible GPU (tested on RTX 3080, sm_86)
- `ninja` (for building CUDA extensions)

## Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Upgrade pip and pin setuptools (needed for tinycudann build)
pip install --upgrade pip
pip install setuptools==69.5.1 ninja

# Install PyTorch (CUDA 12.x)
pip install torch torchvision

# Install main dependencies
pip install pytorch-lightning omegaconf==2.2.3 scipy matplotlib opencv-python \
    imageio imageio-ffmpeg tensorboard Pillow "trimesh[easy]" PyMCubes pyransac3d

# Install CUDA-only packages
pip install nerfacc==0.3.3 torch_efficient_distloss
```

## tinycudann (requires compilation)

tinycudann must be built from source. The pip install often fails due to `pkg_resources` issues with modern setuptools, so we use `setup.py install` directly.

```bash
# Clone
cd /tmp
git clone --recursive https://github.com/NVlabs/tiny-cuda-nn.git
cd tiny-cuda-nn/bindings/torch

# Build for your GPU architecture
# Common values: 70 (V100), 75 (T4), 80 (A100), 86 (RTX 3080/3090), 89 (RTX 4090)
TCNN_CUDA_ARCHITECTURES=86 python setup.py install

# Verify
python -c "import tinycudann; print('OK')"
```

If `pip install "git+https://..."` fails with `ModuleNotFoundError: No module named 'pkg_resources'`, use the manual clone + `setup.py install` method above.

## Verify installation

```bash
python -c "
import tinycudann; print('tinycudann OK')
import nerfacc; print('nerfacc OK')
from torch_efficient_distloss import flatten_eff_distloss; print('distloss OK')
import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')
"
```

## Run tests (CPU-only, no CUDA deps needed)

```bash
pip install pytest
python -m pytest tests/ -v
```

Tests mock CUDA-only dependencies (`tinycudann`, `nerfacc`, `torch_efficient_distloss`) so they run on CPU.
