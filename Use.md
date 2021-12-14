# How to use our framework

## Our framework includes three parts:

- **Step1: train a Reid model which can do Cross border pedestrian re-identification**
- **Step2: attack the Reid model by using one of attack algorithm**
- **Step3: defense the Reid model by using one of defense algorithm**

## cmdline arguments

- `--config-file` (must):load base configuration information, including datasets/batch size/model structure selection. 
 **Need one argument where the configuration file is**.
 
- `--train` (optional):  **only** train a Reid model in step 1. **No argument**.

- `--attack` (optional): use attack algorithm to attack the Reid model. **Need one argument. And the argument consists of two parts** 
  - The first part is `attack type`. 
    - There are `T`(target attack) and `UT`(untarget attack) for classfication attack algorithm(CA).
    - There are `QA`(query attack) and `GA`(gallery attack) for retrieval attack algorithm, and in retrieval attack you also need to point your attack direction
        - `+` means to pull the negtive (ground false) samples closer to the images
        - `-` means to push the positive(ground truth) samples away from the images
        so your choice must be in [`QA+`,`QA-`,`GA+`,`GA-`]
  - The second part is `attack algorithm`
    - classfication : [`C-FGSM`,`C-IFGSM`,`C-MIFGSM`,`CW`]
    - retrieval :[`R-FGSM`,`R-IFGSM`,`R-MIFGSM`,`ODFA`,`MISR`,`MUAP`,`SSAE`,`SMA`,`FNA`]
  - **attention**: `attack type` and `attack algorithm` must match, both in classification or retrieval, and two parts should sperated by `:`.
  
- `--defense` (optional): use defense algorithm to defense the Reid model. **Need one argument.** The `defense algorithm` should be chosen in 
  - general defense algorithm [`ADV`,`GRA`,`DIST`] 
  - retrieval defense algorithm [`GOAT`,`EST`,`SES`,`PNP`]
  but you don't need to distinguish between defense types
  
- `-record` (optional) : to record your evaluation result. It will be saved in an excel in the root path as `./result.xlsx`. Except long python terminal outputs, all your works will be recorded in file `./log.txt` with a short and clear notes, you don't need to worry about that `log.txt` will be recovered because all the notes would be written at the end of the file. **No argument**

- `--pretrained` (optional): use pretrained attack model. **No argument**. 

   **Pay attention that** when you are using the attack algorithm `SSAE` or `MISR` , they both needs to train in training set first, it takes a long time to train. So the first time you use this attack algorithm you need to spend a long time on training it first, and **in the following evaluation you can add the argument in the command line to use your pretrained attack model** to avoid wasting time, or **directly download our pretrained attack model in [Model_zoo.md](Model_zoo.md) to save your first training time**

## examples

#### 1.train a reid model in DukeMTMC
```bash
python3 tools/train_net.py --config-file ./configs/DukeMTMC/bagtricks_R50.yml --train MODEL.DEVICE 'cuda:0'
```
#### 2.attack the reid model by SSAE in QA+ with a pretrained SSAE attack model without record
```bash
python3 tools/train_net.py --config-file ./configs/DukeMTMC/bagtricks_R50.yml --attack QA+:SSAE --pretrained MODEL.DEVICE 'cuda:0'
```
#### 3.attack and defense and record
```bash
python3 tools/train_net.py --config-file ./configs/Market1501/bagtricks_R50.yml --attack GA-:R-IFGSM --defense GRA --record MODEL.DEVICE 'cuda:0'
```
```bash
python3 tools/train_net.py --config-file ./configs/Market1501/bagtricks_R50.yml --attack T:C-IFGSM --defense GOAT --record MODEL.DEVICE 'cuda:0'
```

**If you meet any issue in your work, please first check out whether your command line input usage is right, and then look up in [Issues.md](Issues.md). If the problem still exists, leave your issue and I will reply as soon as possible.**