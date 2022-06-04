# Evaluation and Study of Attack and Defense Strategies for Pedestrian Re-identification

[![GitHub license](https://img.shields.io/github/license/luzhixing12345/Evaluation-of-attack-and-defense-strategies-for-Reid)](https://github.com/luzhixing12345/Evaluation-of-attack-and-defense-strategies-for-Reid/blob/master/LICENSE)
![Language](https://img.shields.io/badge/python-3.7-blue.svg)
![GitHub last commit](https://img.shields.io/github/last-commit/luzhixing12345/Evaluation-of-attack-and-defense-strategies-for-Reid)

## Introduction

该项目是武汉大学计算机学院2021年国家级创新项目 "面向行人重识别的攻防测评及研究" 的代码仓库,用于评测目前主流的针对行人重识别领域的攻击/防御算法.

我们的代码基于[fastreid](https://github.com/JDAI-CV/fast-reid)进行二次开发扩展,在尽量保持源代码结构完整性的同时,重新设计了评估我们的目的的框架

如果您尚不了解行人重识别以及fastreid框架,可以参考[博客]() # todo

## Installation : Set up with Conda

```shell script
conda create -n fastreid python=3.7
conda activate fastreid
conda install pytorch==1.6.0 torchvision tensorboard -c pytorch
pip install -r docs/requirements.txt
conda install Cython
conda install yaml
conda install scikit-image=0.15.0
conda install -c conda-forge tensorboardx
```

> 请务必使用 conda 下载 scikit-image=0.15.0版本,不要使用 pip install

## Dataset

我们在Market1501和DukeMTMC两个数据集上测试了实验结果,关于数据集的下载和保存位置请参考[dataset](datasets/README.md)

下载地址失效的话也可以下载我们上传的数据集

- [Market1501](https://github.com/luzhixing12345/Evaluation-of-attack-and-defense-strategies-for-Reid/releases/download/v0.0.2/Market-1501-v15.09.15.zip)
- [DukeMTMC](https://github.com/luzhixing12345/Evaluation-of-attack-and-defense-strategies-for-Reid/releases/download/v0.0.2/DukeMTMC-reID.zip)

请下载数据集,解压后放到 `datasets/` 目录下

## Use

具体的使用方法详见 [Use.md](Use.md)

您可能遇到的主要问题会在[Issues.md](Issues.md)中列出，如果没有解决您的问题，请留下您的问题，我会尽快回复。

## Pretrained model

您可以使用我们的框架自己训练模型，也可以直接下载我们的预训练模型以节省您的时间,详见[Model_zoo.md](Model_zoo.md)

## Visualization result

## Reference

|paper|github|
|:--:|:--:|
|Adversarial Metric Attack and Defense for Person Re-identification|https://github.com/SongBaiHust/Adversarial_Metric_Attack|
|Vulnerability of Person Re-Identification Models to Metric Adversarial Attacks|https://github.com/qbouniot/adv-reid|
|Transferable Controllable and Inconspicuous Adversarial Attacks on Person Re-identification|https://github.com/whj363636/Adversarial-attack-on-Person-ReID-With-Deep-Mis-Ranking|
|Adversarial Ranking Attack and Defense|https://github.com/cdluminate/robrank|

## Contributor

项目团队由四名成员和我们的学术导师组成

- 指导老师: 梁超
- 队长: 曾俊淇
- 队员: 陆知行
- 队员: 袁梦莹
- 队员: 马晓雅

<a href="https://github.com/luzhixing12345/Evaluation-of-attack-and-defense-strategies-for-Reid/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=luzhixing12345/Evaluation-of-attack-and-defense-strategies-for-Reid" />
</a>

Made with [contrib.rocks](https://contrib.rocks).

## Conclusion

如何想要在本项目上做再次修改,我同样记录了针对完整的fastreid框架的修改内容,可以参考[博客]() #TODO

如果对本项目感兴趣的话,可以看看这个项目背后的[故事]() #TODO

再次感谢您对本项目的关注
