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



# Set up with Conda



```shell script
conda create -n fastreid python=3.7
conda activate fastreid
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/pro
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
conda install pytorch==1.6.0 torchvision tensorboard -c pytorch
pip install -r docs/requirements.txt
conda install Cython
conda install yaml
conda install -c conda-forge tensorboardx
```



# Start

## activate environment

```
conda activate fastreid
```

## start the project

注：所有执行命令行中的“ \ " 都是表示python交互式模式下的换行符号，如不需要换行删掉” \ "



<!--以如下命令为例-->

```shell script
python3 tools/train_net.py --config-file ./configs/Market1501/bagtricks_R50.yml --attack-by --defend-by\
MODEL.WEIGHTS ./model/market_bot_R50.pth MODEL.ATTACKMETHOD "FGSM" MODEL.DEFENCEMETHOD "GRAD"  MODEL.DEVICE "cuda:0"
```

```
主程序为train_net.py

config-file 对应的是数据集和网络结构，其后面为该文件的位置

attack-by 表示用某种方法进行攻击
defend-by 表示用某种方法进行防御

MODEL.WEIGHTS 表示模型的权重，后面的地址为你需要测评的已经训练好的模型的位置
MODEL.ATTACKMETHOD 表示攻击的方法， 与attack-by配合使用，后面为你所使用的攻击算法名称
MODEL.DEFENCEMETHOD 表示防御的方法， 与defend-by配合使用，后面为你所使用的防御算法名称
MODEL.DEVICE "cuda:0" 表示使用单个GPU训练，如需使用多个GPU或者多台计算机完成训练，请查阅源fastreid中运行文件，这里不再赘述

请注意 --attack-by --defend-by任意选择是否使用，只需要与后面的对应位置是否出现匹配即可
但后面的参数个数必须为偶数，且一一对应
```



