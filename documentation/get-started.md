---
layout: page
title: Get Started
description: Get started with FastMoE
---

Get Started
============

You can get started with FastMoE with docker or in a direct way.

## Docker

### Environment Setup

#### On host machine

First, you need to setup the environment on the host machine.

- [NVIDIA driver](https://github.com/NVIDIA/nvidia-docker/wiki/Frequently-Asked-Questions#how-do-i-install-the-nvidia-driver)

- [Docker](https://www.docker.com/) and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

Then, we recommend the [official PyTorch docker image](https://hub.docker.com/r/pytorch/pytorch), as the environment is well-setup there. Note that you should use the image with `devel` in its tag, rather than `runtime`. Theoretically, Pytorch environment on your host machine is not needed.

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

Enter our repo directory inside the well-prepared docker container. By default, the distributed expert feature is disabled. So you need to set environment variable `USE_NCCL=1` to enable it. Use `python setup.py install` to easily install our FastMoE, and you can check the installation with:

```shell
$ conda list | grep fastmoe
fastmoe	0.1.1	pypi_0	pypi
```

Finally, enjoy using FastMoE for training!

## Direct way

### Preparations

To use FastMoe, CUDA and PyTorch are required.

1. CUDA Tookit is available at <https://developer.nvidia.com/cuda-downloads>. Select your operating system and follow instructions on the website to install CUDA.
   Notice: version of CUDA must match the version of nvidia driver. If you're not sure whether you have installed nvidia driver or you don't know its version, you may use `nvidia-smi` to get information about nvidia driver.

2. Add CUDA to the list of environmental variables. If you work with Linux, use command `vi ~/.bashrc`  and add the following content to the end of file (replace X.X with version of CUDA you've downloaded):

```
export PATH=$PATH:/usr/local/cuda-X.X/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-X.X/lib64
export CUDA_HOME=$CUDA_HOME:/usr/local/cuda-X.X
```

Then don't forget to use `source ~/.bashrc` to update the configurations.
So far, CUDA has been installed successfully, you can use `nvcc --version` to check its version.

3. PyTorch can be installed with pip. Version `>=1.8.0` is required if you want to use Megatron. After installation, run the following Python code:

```
import torch
torch.cuda.is_available()
torch.cuda.decive_count()
```

If result of `torch.cuda.is_available()` is `True` and `torch.cuda.decive_count()` returns number of your device, then conguatulations! CUDA and PyTorch run successfully on your device.

### NCCL

1. If you want to enable distributed expert feature, please download NCCL at <https://developer.nvidia.com/nccl/nccl-legacy-downloads>. Version of  NCCL should be no less than `2.7.5` and match the version of PyTorch. You can use function `torch.cuda.nccl.version()` to see the version of NCCL required.

2. Install the 'deb' file. If you use Ubuntu or Debian, just use the following commands (nccl_repo_file is your file, XXX and X.X mean version of NCCL and CUDA):

```
sudo dpkg -i nccl_repo_file.deb
sudo apt update
sudo apt install libnccl2=XXX+cudaX.X libnccl-dev=XXX+cudaX.X
```

### FastMoe Installation

1. Clone the repo of FastMoe from <https://github.com/laekov/fastmoe>, and use the following command to install:

```
python3 setup.py install
```

2. If you need NCCL, set environmental variable `USE_NCCL=1` before installation. For example, you may use command as follows:

```
export USE_NCCL=1
```

3. Installation finishes. Enjoy FastMoe now! You can try excuting `benchmark_mlp.py` in directory `tests`.