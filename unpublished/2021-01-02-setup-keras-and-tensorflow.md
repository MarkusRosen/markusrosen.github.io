---
layout: post
title: Setup Tensorflow and Keras with CUDA Support - A fast and pain-free approach with Miniconda
date: 2021-09-02 18:00:00 +0300
description: This short tutorial shows you how to setup Tensorflow with GPU support on Linux and Miniconda
img: /post4/teaser.png
tags: [Keras, Tensorflow, Deep Learning, GPU Setup, CUDA, Anaconda, Miniconda]
---

As a quick and easy to use deep learning framework I love to use Keras. With few lines of code one can utilize modern deep learning models with transfer learning on custom datasets. But it can be a giant pain to setup correctly on a Linux machine if any of the dependencies do not match (Ubuntu version, Tensorflow, CUDA, CuDNN...). I have spent way too many hours on trying to get it to work on PopOS with the regular pip-Versions, therefore this is a super short tutorial on how to set it up correctly.

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

In this tutorial we will use Python 3.8 with Tensorflow 2.4. Therefore we will install all these dependencies accordingly:

```bash
conda install python=3.8
conda install -c anaconda cudatoolkit=11.0
pip install tensorflow==2.4
```

## Test your Setup

With the following Python code you should be able to check if your Tensorflow-Installation is using the GPU. Run this code within your active conda environment:

```python

```

```python

```

Your output should look similar to this, if there are any errors you might have to restart this process.

After a successful setup I would personally create a YAML file as a basic starter for any future Keras projects. This can be done with

```bash

```

The YAML file can be used as a starter for a new environment with

```bash

```

The YAML file should now look like this:

```yaml

```

It it still does not work... just switch to PyTorch and PyTorch Lightning if possible.
