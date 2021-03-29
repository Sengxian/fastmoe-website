---
layout: page
title: 开始使用
description: FastMoE的安装与使用
---

## 准备工作

为了使用 FastMoe ，您需要首先安装 CUDA 和 PyTorch 。

1. 前往 CUDA 官网 <https://developer.nvidia.com/cuda-downloads> ，选择对应的操作系统并根据该网站的提示安装 CUDA ，注意安装的 CUDA 版本必须与 nvidia 显卡驱动版本匹配。如果您不确定是否已安装 nvidia 显卡驱动或不清楚其版本，可以在终端执行命令 `nvidia-smi` 查看 GPU 信息和驱动版本。

2. 把 CUDA 添加到环境变量中，具体地，在 Linux 系统下您可以在终端执行命令 `vi ~/.bashrc` ，在文件末尾添加如下内容（其中 X.X 为您安装的 CUDA 版本）：

```
export PATH=$PATH:/usr/local/cuda-X.X/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-X.X/lib64
export CUDA_HOME=$CUDA_HOME:/usr/local/cuda-X.X
```

然后在终端执行命令 `source ~/.bashrc`。
至此， CUDA 已安装完毕，您可以执行命令 `nvcc --version` 查看 CUDA 是否安装成功及其版本。

3. PyTorch 可以直接通过 pip 安装，请选择 `>=1.8.0` 的版本以适配 Megatron 框架，安装完毕后您可以在 Python 环境中运行如下代码：

```
import torch
torch.cuda.is_available()
torch.cuda.decive_count()
```

如果 `torch.cuda.is_available()` 的运行结果为 `True` 并且 `torch.cuda.decive_count()` 成功返回了设备个数的值，那么可以确认 CUDA 和 PyTorch 都已安装成功。

## NCCL

1. 如果您希望使用多卡通信框架 NCCL ，可以前往 NCCL 官网 <https://developer.nvidia.com/nccl/nccl-legacy-downloads> 下载，注意 NCCL 版本应 `>=2.7.5` ，另外， NCCL 版本需要与 PyTorch 版本对应，具体地，您可以在 Python 环境中运行 `torch.cuda.nccl.version()` 查看所需要安装的 NCCL 版本。

2. 下载完成后，在终端进行安装。在 Ubuntu 或 Debian 系统下，您可以在下载目录下执行如下命令（ nccl_repo_file 为您下载的 deb 文件， XXX 和 X.X 分别为 NCCL 和 CUDA 版本）：

```
sudo dpkg -i nccl_repo_file.deb
sudo apt update
sudo apt install libnccl2=XXX+cudaX.X libnccl-dev=XXX+cudaX.X
```

## FastMoe安装

1. 从 <https://github.com/laekov/fastmoe> 下载 FastMoe 仓库，在仓库目录下执行如下命令：

```
python3 setup.py install
```

2. 如果您希望使用 NCCL ，请在安装前设置环境变量 `USE_NCCL=1` 。例如您可以在终端执行如下命令：

```
export USE_NCCL=1
```

3. 安装完成后，您可以执行 `tests` 目录下的 `benchmark_mlp.py` ，如果成功运行，那么恭喜， FastMoe 已安装完成。

