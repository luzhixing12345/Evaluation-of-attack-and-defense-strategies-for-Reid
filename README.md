Evaluation and Study of Attack and Defense Strategies for Pedestrian Re-identification
===


# Introduction:

This project is a national innovation project of the school of computer  science of Wuhan University in 2021. The project team consists of four members and our academic advisor.

Our code is developed and expanded for the second time based on [fastreid](https://github.com/JDAI-CV/fast-reid). While trying to maintain the structural integrity of the source code, we have redesigned the evaluation framework for our purpose.

# Installation

## Requirements


- Linux or macOS with python ≥ 3.6
- PyTorch ≥ 1.6
- torchvision that matches the Pytorch installation. You can install them together at [pytorch.org](https://pytorch.org/) to make sure of this.
- [yacs](https://github.com/rbgirshick/yacs)
- Cython (optional to compile evaluation code)
- tensorboard (needed for visualization): `pip install tensorboard`
- gdown (for automatically downloading pre-train model)
- sklearn
- termcolor
- tabulate
- [faiss](https://github.com/facebookresearch/faiss) `pip install faiss-cpu`
- openpyxl
- scikit-image



## Set up with Conda

```shell script
conda create -n fastreid python=3.7
conda activate fastreid
conda install pytorch==1.6.0 torchvision tensorboard -c pytorch
pip install -r docs/requirements.txt
conda install Cython
conda install yaml
conda install -c conda-forge tensorboardx
```

# Use:

All detailed usage of the framework in [Use.md](Use.md)

Main issues you may encounter will be listed in [Issues.md](Issues.md). If it doesn't solve your problem, please leave your issue and I will reply as soon as possible.

# pretrained model

You can use our framework to train a model yourself, or directly download our pretrained model to save your time

see more in [Model_zoo.md](Model_zoo.md)

# result

# Reference
