# RNb-NeuS (based on Instant-nsr-pl)

This repository contains a concise and extensible implementation of NeuS / RNb-NeuS
for neural surface reconstruction based on Instant-NGP and the Pytorch-Lightning 
framework. 


## Requirements
**Note:**
- To utilize multiresolution hash encoding or fully fused networks provided by tiny-cuda-nn, you should have least an RTX 2080Ti, see [https://github.com/NVlabs/tiny-cuda-nn#requirements](https://github.com/NVlabs/tiny-cuda-nn#requirements) for more details.
- Multi-GPU training is currently not supported on Windows (see [#4](https://github.com/bennyguo/instant-nsr-pl/issues/4)).
### Environments
- Install PyTorch>=1.10 [here](https://pytorch.org/get-started/locally/) based the package management tool you used and your cuda version (older PyTorch versions may work but have not been tested)
- Install tiny-cuda-nn PyTorch extension: `pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch`
- `pip install -r requirements.txt`


## Data

We provide the [DiLiGenT-MV, LUCES-MV and Skoltech3D](https://drive.google.com/drive/folders/1TbOrB38klLpG41bXzI7B1A01qsbEbz9h?usp=sharing) datasets with normals and reflectance maps estimated using [SDM-UniPS](https://github.com/satoshi-ikehata/SDM-UniPS-CVPR2023/) and [Uni-MS-PS](https://github.com/Clement-Hardy/Uni-MS-PS). This link contains also the cleaned resulting meshs and groundtruths.

### Data Convention

Organize your data in the `./data/` folder following this structure:
```plaintext
./data/FOLDER/
    albedo/          # (Mandatory for the moment)
        000.png
        001.png
        002.png
    normal/          # (Mandatory)
        000.png
        001.png
        002.png
    mask/            # (Mandatory)
        000.png
        001.png
        002.png
    cameras.npz
```

## Training

### Preprocess the Data

```bash
bash ./train.sh
```
## Citation

- [RNb-NeuS2](https://robinbruneau.github.io/publications/rnb_neus2.html)

```bibtex
@misc{Bruneau25,
    title={{Multi-view Surface Reconstruction Using Normal and Reflectance Cues}},
    author={Robin Bruneau and Baptiste Brument and Yvain Quéau and Jean Mélou and François Bernard Lauze and Jean-Denis
    Durou and Lilian Calvet},
    year={2025},
    eprint={2506.04115},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2506.04115%7D,
}
```

- [RNb-NeuS](https://robinbruneau.github.io/publications/rnb_neus.html)

```bibtex
@inproceedings{Brument24,
    title={{RNb-NeuS: Reflectance and Normal-based Multi-View 3D Reconstruction}},
    author={Baptiste Brument and Robin Bruneau and Yvain Quéau and Jean Mélou and François Lauze and Jean-Denis Durou and Lilian Calvet},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year={2024}
}
```

- [Instant-nsr-pl](https://github.com/bennyguo/instant-nsr-pl)

```
@misc{instant-nsr-pl,
    Author = {Yuan-Chen Guo},
    Year = {2022},
    Note = {https://github.com/bennyguo/instant-nsr-pl},
    Title = {Instant Neural Surface Reconstruction}
}
```
