---
layout: page
title: Get Started
description: Get started with fastmoe with docker or direct source
---

Get Started
============

## Docker

### Environment Preparation

#### On host machine

First, you need to setup the environment on the host machine.

- [NVIDIA driver](https://github.com/NVIDIA/nvidia-docker/wiki/Frequently-Asked-Questions#how-do-i-install-the-nvidia-driver)

- [Docker](https://www.docker.com/) and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

Then, we recommend the [official PyTorch docker image](https://hub.docker.com/r/pytorch/pytorch), as the environment is well-setup there. Note that you should use the image with **"devel"** in its tag, rather than "runtime". Theoretically, Pytorch environment on your host machine is not needed.

For example, you can run `docker pull pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel` to get a Pytorch docker image.

#### Inside the docker

Run a docker container with commands like:

```shell
docker run --name pytorch -it pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel
```

And use bash to interact with it:

```shell
docker exec -ti pytorch /bin/bash
```

For distributed expert feature, NCCL is required. Inside the docker, you can first check if the NCCL is installed, such as:

```shell
$ apt list --installed | grep nccl
libnccl-dev/unknown,now 2.8.4-1+cuda11.2 amd64 [installed]
libnccl2/unknown,now 2.8.4-1+cuda11.2 amd64 [installed]
```

If not, you can follow the [official documentation](https://docs.nvidia.com/deeplearning/nccl/install-guide/index.html) to install the right version according to CUDA version (which can be inspected by `nvcc -V`) in your docker. After that, you need to setup NCCL in your conda environment, following [this](https://anaconda.org/anaconda/nccl).

Finally, you can check NCCL simply with `torch.cuda.nccl.version()` in Python. Additionally, there is an [official repo](https://github.com/NVIDIA/nccl-tests/) for testing NCCL, and it is up to you.

### Installation

Enter our repo directory inside the well-prepared docker container. By default, the distributed expert feature is disabled. So you need to set environment variable `USE_NCCL=1` to enable it. Use `python setup.py install` to easily install our FastMOE, and you can check the installation with:

```shell
$ conda list | grep fastmoe
fastmoe	0.1.1	pypi_0	pypi
```

Finally, enjoy using FastMoE for training!