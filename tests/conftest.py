"""Mock CUDA-only dependencies so tests can run on CPU-only environments."""
import sys
import types
from unittest.mock import MagicMock


def _make_mock_package(name, submodules=None):
    """Create a mock module that acts as a proper package with submodules."""
    mock = MagicMock()
    mock.__spec__ = None
    mock.__path__ = []
    mock.__package__ = name
    sys.modules[name] = mock
    for sub in (submodules or []):
        full = f"{name}.{sub}"
        sub_mock = MagicMock()
        sub_mock.__spec__ = None
        setattr(mock, sub, sub_mock)
        sys.modules[full] = sub_mock


# CUDA-only packages that can't be pip-installed without a GPU
_make_mock_package("tinycudann")
_make_mock_package("nerfacc", submodules=["intersection"])
_make_mock_package("torch_efficient_distloss")

# Optional heavy dependency
_make_mock_package("OpenEXR")
