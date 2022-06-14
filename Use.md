# Use

## 使用方法

该框架可以在两个数据集上,使用两个模型架构,训练Reid模型结构. 并且使用5种攻击算法,4种防御算法得到攻防结果.

下面我将使用一个简单的实例来帮助您理解如何使用,您可以放心,笔者均经过亲自测试,以下的例子都可以直接复制运行.

我们的框架包括三个部分

- **Step1：训练一个可以进行跨境行人重新识别的 Reid 模型**
- **Step2：使用一种攻击算法攻击 Reid 模型**
- **Step3：使用一种防御算法防御 Reid 模型**

1. 首先您需要训练一个用于行人重识别的 Reid 模型

   > 使用Market1501数据集,使用 BOT 模型

   ```bash
   python3 tools/train_net.py --config-file configs/Market1501/bagtricks_R50.yml --train
   ```

   初次训练大概会花费您一段时间,当然可以在这里找到我们所有预训练的[模型](Model_zoo.md) #todo

2. 接下来我们对于这个模型进行攻击

   > 使用QA-的攻击方式,使用FGSM攻击算法

   ```bash
   python3 tools/train_net.py --config-file configs/Market1501/bagtricks_R50.yml --attack QA-:FGSM
   ```

3. 接下来我们对于这个攻击算法进行防御

   > 使用ADV防御算法

   ```bash
   python3 tools/train_net.py --config-file configs/Market1501/bagtricks_R50.yml --attack QA-:FGSM:OD --defense ADV
   ```

4. 当然你可以将2 3两步合并为一步,我也推荐你这样做

   ```bash
   python3 tools/train_net.py --config-file configs/Market1501/bagtricks_R50.yml --attack QA-:FGSM --defense ADV
   ```

5. 接下来你可以查看生成的 `result.xlsx` 文件,里面有这次任务保存的结果

```bash
python3 tools/train_net.py --config-file ./configs/DukeMTMC/bagtricks_R50.yml --attack QA-:FGSM 
```

如果您想要完整的了解如何使用我们的全部内容,希望您可以有耐心的看完下方冗长的命令行可选参数.我们也在最后提供了一些具体的例子,帮助您理解如何使用

## 命令行传参

- `--config-file {DATASET}/{MODEL_CONFIGURATION}` (must):加载基础配置信息

  - DATASET: 两个数据集 `DukeMTMC` 和 `Market1501`
  - MODEL_CONFIGURATION: 两种模型结构 `BOT` for `bagtricksR50.yml` 和 `AGW` for `AGW_R50.yml`

  examples:

  ```bash
  --config-file configs/Market1501/bagtricks_R50.yml
  --config-file configs/DukeMTMC/AGW_R50.yml
  ```

- `--train` (optional):  在步骤 1 中训练 Reid 模型
- `--attack {ATTACK_TYPE}:{ATTACK_ALGORITHM}:{OPTION}` (optional): 使用攻击算法攻击 Reid 模型

  - ATTACK_TYPE: 检索攻击算法有`QA`(查询攻击)和`GA`(图库攻击)，在检索攻击中你还需要指出你的攻击方向
  
    - `+` 表示将负(ground false)样本拉近图像
    - `-` 表示将正样本(ground truth)推离图像
  
    所以可以使用 [`QA+`,`QA-`,`GA+`,`GA-`]

  - ATTACK_ALGORITHM: 攻击算法, 该框架融合了以下攻击算法
    - `FGSM`
    - `IFGSM`
    - `MIFGSM`
    - `ODFA`
    - `MISR`
    - `MUAP`
    - `SSAE`
    - `FNA`
  - OPTION: 可选参数, 这一部分用于不同情况下的执行方式

    - 当你使用攻击算法`SSAE`或`MISR`时，它们都需要**先在训练集中训练一个攻击模型，而且训练时间很长**.所以第一次使用这种攻击算法需要花费很长时间首先训练它，在接下来的评估中，您可以使用 `P` 来使用您的预训练攻击模型，以避免浪费时间。
    - 如果不想攻击纯Reid模型,只想针对训练一个防御的Reid模型，使用`OD`，是(only defence)的缩写

    examples

    ```bash
    --attack QA+:FGSM
    --attack GA-:ODFA
    --attack QA-:SSAE:P
    ```

- `--defense {DEFENSE_ALGORITHM}:{P}` (optional): 使用防御算法来防御 Reid 模型

  - DEFENSE_ALGORITHM: 防御算法,该框架融合了以下防御算法
    - `ADV`
    - `GOAT`
    - `EST`
    - `SES`

    值得注意的是,ADV防御算法是对抗性防御,所以它必须和一个攻击算法一起出现.

    - 正确:

      ```bash
      python3 tools/train_net.py --config-file ./configs/DukeMTMC/bagtricks_R50.yml --attack QA-:FGSM --defense ADV
      ```

    - 错误,缺少具体的防御的攻击算法对象

      ```bash
      python3 tools/train_net.py --config-file ./configs/DukeMTMC/bagtricks_R50.yml --defense ADV
      ```

  - 第一次训练你的防御模型后，你可以使用`P`加载预训练的防御模型以节约时间

  examples

  ```bash
  --defense EST:P
  ```

- `--log` (optional)(default True):

  运行结束后,每个结果都将默认保存在根路径下的excel中为`./result.xlsx`。除了终端输出，你所有的工作都会记录在文件`./log.txt`中，并有简短清晰的注释

## 注意事项

值得注意的是,对于某一个基础的Reid模型,**我们只需要在一开始训练一次**,之后我们就可以使用这个预训练模型进行其他的操作了

对于一个防御模型来说,EST SES GOAT均与攻击算法无关,**我们也不需要多次训练相同的防御模型,只需要训练一次防御模型**,之后就可以直接使用了

对于ADV防御算法来说,它的防御模型需要结合不同的攻击算法,初次训练耗时很长(这个是真的很长),但训练结束之后我们就可以使用这个预训练模型进行其他的操作了

综上,请参考命令行传参,使用 `P` 导入预训练模型.

我们也将攻击模块和防御模块分离开了,可以测评 单独攻击/单独防御/攻击防御后的模型

## 一些例子

1. 在QA-中通过MUAP攻击Market1501-agw的reid模型，在ADV中进行防御，使用预训练的防御模型

   ```bash
   python3 tools/train_net.py --config-file ./configs/Market1501/AGW_R50.yml --attack QA-:MUAP --defense ADV:P
   ```

2. 在 QA+ 中通过 SSAE 攻击 reid 模型，使用预训练的 SSAE 攻击模型，并在 GOAT 中进行防御

   ```bash
   python3 tools/train_net.py --config-file ./configs/DukeMTMC/bagtricks_R50.yml --attack QA+:SSAE:P --defense GOAT 
   ```

3. 只攻击防御模型，防御模型预训练

   ```bash
   python3 tools/train_net.py --config-file ./configs/Market1501/bagtricks_R50.yml --attack GA-:IFGSM:OD --defense SES:P
   ```

**如果您在工作中遇到任何问题，请先检查您的命令行输入用法是否正确，然后在[Issues.md](Issues.md)中查找是否有相关解答。如果问题仍然存在，请留下您的问题,我会尽快回复**
