---
layout: post
title: Setup Tensorflow and Keras with CUDA Support - A fast and pain-free approach with Miniconda
date: 2021-09-02 18:00:00 +0300
description: This short tutorial shows you how to setup Tensorflow with GPU support on Linux and Miniconda
img: /post5/teaser.png
tags: [Keras, Tensorflow, Deep Learning, GPU Setup, CUDA, Anaconda, Miniconda]
---

As a quick and easy to use deep learning framework I love to use Keras. With few lines of code one can utilize modern deep learning models with transfer learning on custom datasets. But it can be a giant pain to setup correctly on a Linux machine if any of the dependencies do not match (Ubuntu version, Tensorflow, CUDA, CuDNN...). I have spent way too many hours on trying to get it to work on PopOS with the regular pip-Versions, therefore this is a super short tutorial on how to set it up correctly. I assume that you already setup Linux with the correct nVidia drivers.

## Table of Contents

- [Table of Contents](#table-of-contents)
- [TL;DR](#tldr)
- [Setup Miniconda](#setup-miniconda)
- [Test your Setup](#test-your-setup)

## TL;DR

```bash
bash Miniconda3-py39_4.10.3-Linux-x86_64.sh
conda create --name keras-project
source /home/markus/miniconda3/bin/activate
conda activate keras-project
conda install python=3.8
conda install -c anaconda cudatoolkit=11.0
pip install tensorflow==2.4
```

## Setup Miniconda

First, download Miniconda from here: https://docs.conda.io/en/latest/miniconda.html

Open your terminal and change to the folder where your Miniconda was downloaded to. Start the installation process and accept the installation conditions.

```bash
bash Miniconda3-py39_4.10.3-Linux-x86_64.sh
```

Restart your terminal. Create a new environment with a fitting name, like `keras-project` and activate it:

```bash
conda create --name keras-project
source /home/YOURUSERNAME/miniconda3/bin/activate
conda activate keras-project
```

Now we can install any Python version that we want. **Important:** Check which version of Python and Tensorflow you will need here: https://www.tensorflow.org/install/source#linux

In this tutorial we will use Python 3.8 with Tensorflow 2.4 and CUDA 11.0. Therefore we will install all these dependencies accordingly:

```bash
conda install python=3.8
conda install -c anaconda cudatoolkit=11.0
pip install tensorflow==2.4
```

## Test your Setup

With the following Python code you should be able to check if your Tensorflow-Installation is using the GPU. Run this code within your active conda environment:

```python
import tensorflow as tf
tf.config.list_physical_devices('GPU')
```

```bash
2021-09-02 17:58:11.357479: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.11.0
2021-09-02 17:58:12.430604: I tensorflow/compiler/jit/xla_cpu_device.cc:41] Not creating XLA devices, tf_xla_enable_xla_devices not set
2021-09-02 17:58:12.431227: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcuda.so.1
2021-09-02 17:58:12.459596: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:941] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-09-02 17:58:12.459997: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1720] Found device 0 with properties:
pciBusID: 0000:01:00.0 name: NVIDIA GeForce RTX 2070 with Max-Q Design computeCapability: 7.5
coreClock: 1.185GHz coreCount: 36 deviceMemorySize: 7.79GiB deviceMemoryBandwidth: 357.69GiB/s
2021-09-02 17:58:12.460037: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.11.0
2021-09-02 17:58:12.467855: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublas.so.11
2021-09-02 17:58:12.468007: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublasLt.so.11
2021-09-02 17:58:12.473083: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcufft.so.10
2021-09-02 17:58:12.475665: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcurand.so.10
2021-09-02 17:58:12.484575: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcusolver.so.10
2021-09-02 17:58:12.487250: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcusparse.so.11
2021-09-02 17:58:12.490611: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudnn.so.8
```

Your output should look similar to above, if there are any errors you might have to restart this process.

After a successful setup I would personally create a YAML file as a basic starter for any future Keras projects. This can be done with

```bash
conda env export --name keras-project > tensorflow-base-env.yml
```

The YAML file can be used as a starter for a new environment with

```bash
conda env create --name keras-project-new --file tensorflow-base-env.yml
```

For more tips on how to use conda as a package manager, use the [cheat sheet](https://docs.conda.io/projects/conda/en/latest/_downloads/843d9e0198f2a193a3484886fa28163c/conda-cheatsheet.pdf).

The YAML file should look like this:

```yaml
name: keras-project
channels:
  - anaconda
  - defaults
dependencies:
  - _libgcc_mutex=0.1=main
  - _openmp_mutex=4.5=1_gnu
  - ca-certificates=2020.10.14=0
  - certifi=2020.6.20=py38_0
  - cudatoolkit=11.0.221=h6bb024c_0
  - ld_impl_linux-64=2.35.1=h7274673_9
  - libffi=3.3=he6710b0_2
  - libgcc-ng=9.3.0=h5101ec6_17
  - libgomp=9.3.0=h5101ec6_17
  - libstdcxx-ng=9.3.0=hd4cf53a_17
  - ncurses=6.2=he6710b0_1
  - openssl=1.1.1k=h27cfd23_0
  - pip=21.0.1=py38h06a4308_0
  - python=3.8.11=h12debd9_0_cpython
  - readline=8.1=h27cfd23_0
  - setuptools=52.0.0=py38h06a4308_0
  - sqlite=3.36.0=hc218d9a_0
  - tk=8.6.10=hbc83047_0
  - wheel=0.37.0=pyhd3eb1b0_0
  - xz=5.2.5=h7b6447c_0
  - zlib=1.2.11=h7b6447c_3
  - pip:
      - absl-py==0.13.0
      - astunparse==1.6.3
      - cachetools==4.2.2
      - charset-normalizer==2.0.4
      - flatbuffers==1.12
      - gast==0.3.3
      - google-auth==1.35.0
      - google-auth-oauthlib==0.4.6
      - google-pasta==0.2.0
      - grpcio==1.32.0
      - h5py==2.10.0
      - idna==3.2
      - keras-preprocessing==1.1.2
      - markdown==3.3.4
      - numpy==1.19.5
      - oauthlib==3.1.1
      - opt-einsum==3.3.0
      - protobuf==3.17.3
      - pyasn1==0.4.8
      - pyasn1-modules==0.2.8
      - requests==2.26.0
      - requests-oauthlib==1.3.0
      - rsa==4.7.2
      - six==1.15.0
      - tensorboard==2.6.0
      - tensorboard-data-server==0.6.1
      - tensorboard-plugin-wit==1.8.0
      - tensorflow==2.4.0
      - tensorflow-estimator==2.4.0
      - termcolor==1.1.0
      - typing-extensions==3.7.4.3
      - urllib3==1.26.6
      - werkzeug==2.0.1
      - wrapt==1.12.1
prefix: /home/YOURUSERNAME/miniconda3/envs/keras-project
```

It it still does not work... just switch to PyTorch and PyTorch Lightning if possible.
