<div align="center">
<h1>Open-RNb</h1>
<h3>Open-source Reflectance and Normal-based<br>
Multi-View 3D Reconstruction</h3>

<p>
<b>A fully open-source reimplementation of
<a href="https://robinbruneau.github.io/publications/rnb_neus2.html">RNb-NeuS2</a></b><br>
No proprietary CUDA libraries &mdash; runs out of the box with standard PyTorch + tiny-cuda-nn.
</p>

[**Robin Bruneau**](https://robinbruneau.github.io/)<sup><span>&#9733;</span></sup> · [**Baptiste Brument**](https://bbrument.github.io/)<sup><span>&#9733;</span></sup>
<br>
[**Yvain Quéau**](https://yqueau.github.io/) · [**Jean Mélou**](https://www.irit.fr/~Jean.Melou/) · [**François Lauze**](https://loutchoa.github.io/) · [**Jean-Denis Durou**](https://cv.hal.science/jean-denis-durou) · [**Lilian Calvet**](https://scholar.google.com/citations?user=6JewdrMAAAAJ&hl=en)

<span>&#9733;</span> corresponding authors

<div style="display: flex; gap: 10px; justify-content: center; align-items: center;">
    <a href='https://arxiv.org/abs/2506.04115'><img src='https://img.shields.io/badge/arXiv-RNb--NeuS2-red' alt='Paper PDF' height="30"></a>
    <a href='https://robinbruneau.github.io/publications/rnb_neus2.html'><img src='https://img.shields.io/badge/Project_Page-RNb--NeuS2-green' alt='Project Page' height="30"></a>
    <a href='https://robinbruneau.github.io/publications/rnb_neus.html'><img src='https://img.shields.io/badge/Project_Page-RNb--NeuS-blue' alt='Project Page' height="30"></a>
</div>
</div>

## Overview

This repository is a **clean, open-source reimplementation** of the [RNb-NeuS2](https://robinbruneau.github.io/publications/rnb_neus2.html) method.

**RNb-NeuS** reconstructs high-quality 3D surfaces from multi-view normal and reflectance (albedo) maps estimated by photometric stereo methods such as [SDM-UniPS](https://github.com/satoshi-ikehata/SDM-UniPS-CVPR2023/) and [Uni-MS-PS](https://github.com/Clement-Hardy/Uni-MS-PS).

Built on [instant-nsr-pl](https://github.com/bennyguo/instant-nsr-pl) with [NeuS](https://lingjie0206.github.io/papers/NeuS/) as the underlying signed distance function (SDF) representation, the method combines normal supervision with a two-phase albedo scaling pipeline to produce accurate geometry even when per-view reflectance maps have inconsistent scales.

## Features

- Two dataset backends: **IDR** (cameras.npz) and **SfM** (Meshroom / AliceVision JSON)
- Two-phase training with automatic albedo scaling
- Scene normalization: `scale_mat`, `point_cloud`, `silhouette`, `camera`, or `auto`
- PLY mesh export with optional vertex colors

## Meshroom Plugin

A ready-to-use [Meshroom](https://github.com/alicevision/Meshroom) plugin is available at [**meshroomHub/mrRNbNeuS**](https://github.com/meshroomHub/mrRNbNeuS/). It wraps RNb-NeuS as a native Meshroom node so you can integrate neural surface reconstruction directly into your photogrammetry pipeline without command-line usage.

## Requirements

- Python 3.10+
- CUDA 12.x + NVIDIA GPU (RTX 2080 Ti or newer)
- `tinycudann`, `nerfacc 0.3.3`, `torch_efficient_distloss`

See [docs/install.md](docs/install.md) for detailed setup instructions.

## Data

### IDR format (cameras.npz)

```
data/<scene>/
    albedo/       000.png, 001.png, ...
    normal/       000.png, 001.png, ...
    mask/         000.png, 001.png, ...
    cameras.npz
```

Pre-built datasets (DiLiGenT-MV, LUCES-MV, Skoltech3D) with normals and
reflectance from [SDM-UniPS](https://github.com/satoshi-ikehata/SDM-UniPS-CVPR2023/) and [Uni-MS-PS](https://github.com/Clement-Hardy/Uni-MS-PS) are available on [Google Drive](https://drive.google.com/drive/folders/1TbOrB38klLpG41bXzI7B1A01qsbEbz9h?usp=sharing).

### SfM format (Meshroom JSON)

Provide separate `.sfm` / `.json` files for normals, albedos, and masks.
Views are matched by `viewId` across files.

```yaml
dataset:
  name: sfm
  normal_sfm: path/to/normalSfm.json
  albedo_sfm: path/to/albedoSfm.json   # optional
  mask_sfm:   path/to/maskSfm.json     # optional
```

## Training

### Single command

```bash
# IDR dataset
python launch.py --config configs/idr.yaml --gpu 0 --train \
    dataset.scene=golden_snail \
    dataset.root_dir=./data/golden_snail

# SfM dataset
python launch.py --config configs/sfm.yaml --gpu 0 --train \
    dataset.scene=golden_snail \
    dataset.normal_sfm=data/golden_snail/normalSfm.json \
    dataset.albedo_sfm=data/golden_snail/albedoSfm.json \
    dataset.mask_sfm=data/golden_snail/maskSfm.json
```

### Two-phase training (albedo scaling)

When albedos are available, training automatically uses two phases:

1. **Phase 1** (geometry warmup): `no_albedo=True`, white albedos, rendering loss
   trains only the SDF
2. Intermediate mesh extraction + scene renormalization + albedo ratio computation
3. **Phase 2** (full training): fresh model with scaled albedos

Control via config:
```yaml
system:
  albedo_scaling:
    enabled: null              # null=auto, true/false to force
    warmup_ratio: 0.1          # Phase 1 = 10% of total steps
    n_samples: 2000            # Pixel samples per view for ratios
    intermediate_mesh_resolution: 512
    sphere_scale_p2: 1.5       # Phase 2 bounding sphere radius
```

### Key options

| Option | Default | Description |
|--------|---------|-------------|
| `trainer.max_steps` | 20000 | Total training iterations |
| `dataset.scaling_mode` | `auto` (SfM) / `scale_mat` (IDR) | Scene normalization method |
| `dataset.sphere_scale` | 1.0 | Phase 1 bounding sphere radius |
| `model.geometry.isosurface.resolution` | 512 | Marching cubes grid resolution |
| `system.save_images` | false | Save validation/test images |

## Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

Tests mock CUDA dependencies and run on CPU.

## Acknowledgements

This work is supported by [**DOPAMIn**](https://www.cnrsinnovation.com/actualite/une-seconde-promotion-pour-le-programme-open-7-nouveaux-logiciels-scientifiques-a-valoriser/) (*Diffusion Open de Photogrammétrie par AliceVision/Meshroom pour l'Industrie*), selected in the 2024 cohort of the [**OPEN**](https://www.cnrsinnovation.com/open/) programme run by [CNRS Innovation](https://www.cnrsinnovation.com/). OPEN supports the valorization of open-source scientific software by providing dedicated developer resources, governance expertise, and industry partnership support.

**Lead researcher:** [Jean-Denis Durou](https://cv.hal.science/jean-denis-durou), [IRIT](https://www.irit.fr/) (INP-Toulouse)

## Citation

- [RNb-NeuS2](https://robinbruneau.github.io/publications/rnb_neus2.html)

```bibtex
@article{bruneau25,
    title={{Multi-view Surface Reconstruction Using Normal and Reflectance Cues}},
    author={Robin Bruneau and Baptiste Brument and Yvain Qu{\'e}au and Jean M{\'e}lou and Fran{\c{c}}ois Bernard Lauze and Jean-Denis Durou and Lilian Calvet},
    journal={International Journal of Computer Vision (IJCV)},
    year={2025},
    eprint={2506.04115},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2506.04115},
}
```

- [RNb-NeuS](https://robinbruneau.github.io/publications/rnb_neus.html)

```bibtex
@inproceedings{brument24,
    title={{RNb-NeuS: Reflectance and Normal-based Multi-View 3D Reconstruction}},
    author={Baptiste Brument and Robin Bruneau and Yvain Qu{\'e}au and Jean M{\'e}lou and Fran{\c{c}}ois Lauze and Jean-Denis Durou and Lilian Calvet},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year={2024}
}
```

- [instant-nsr-pl](https://github.com/bennyguo/instant-nsr-pl)

```bibtex
@misc{instant-nsr-pl,
    Author = {Yuan-Chen Guo},
    Year = {2022},
    Note = {https://github.com/bennyguo/instant-nsr-pl},
    Title = {Instant Neural Surface Reconstruction}
}
```
