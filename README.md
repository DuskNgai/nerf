# Neural Radiance Fields Related Code

This repository contains code related to Neural Radiance Fields (NeRF) and its extensions.

- [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/abs/2003.08934)
- [Plenoxels: Radiance Fields without Neural Networks](https://arxiv.org/abs/2112.05131)
- [TensoRF: Tensorial Radiance Fields](https://arxiv.org/abs/2203.09517)

## Setup

First, download the codebase and the submodules:

```bash
git clone https://github.com/DuskNgai/nerf.git -o nerf && cd nerf
git submodule update --init --recursive
```

Second, install the dependencies by **manually installing them**:
- Install dependencies manually:
    ```bash
    conda create -n nerf python=3.11
    conda activate nerf
    conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
    conda install ipykernel lightning matplotlib "numpy<2.0.0" pandas rich scipy tensorboard
    pip install fvcore mrcfile omegaconf timm
    ```
