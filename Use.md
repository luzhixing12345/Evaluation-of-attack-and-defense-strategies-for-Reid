# How to use our framework

## Our framework includes three parts:

- **Step1: train a Reid model which can do cross border pedestrian re-identification**
- **Step2: attack the Reid model by using one of attack algorithm**
- **Step3: defense the Reid model by using one of defense algorithm**

## cmdline arguments

- `--config-file` (must):load base configuration information, including datasets/batch size/model structure selection. 
 **Need one argument where the configuration file is**.
  - two datasets including `DukeMTMC` and `Market1501`
  - two model structure `BOT` for `bagtricksR50.yml` and `AGW` for `AGW_R50.yml` 
 
- `--train` (optional):  **only** train a Reid model in step 1. **No argument**.

- `--attack` (optional): use attack algorithm to attack the Reid model. **Need one argument. And the argument consists of three parts** 
  - The first part is `attack type`. 
  There are `QA`(query attack) and `GA`(gallery attack) for retrieval attack algorithm, and in retrieval attack you also need to point your attack direction
    - `+` means to pull the negtive (ground false) samples closer to the images
    - `-` means to push the positive(ground truth) samples away from the images
  <br>so your choice must be in [`QA+`,`QA-`,`GA+`,`GA-`]
  - The second part is `attack algorithm`
    - `FGSM`
    - `IFGSM`
    - `MIFGSM`
    - `ODFA`
    - `MISR`
    - `MUAP`
    - `SSAE`
    - `FNA`
  - The third part is flexible
    - When you are using the attack algorithm `SSAE` or `MISR` , they both needs to train in training set first, and it takes a long time to train. So the first time you use this attack algorithm you need to spend a long time on training it first, and in the following evaluation you can use `P` to use your pretrained attack model to avoid wasting time.
    - If you don't want to attack the pure Reid model, and just want to attack the defense Reid model,use `OD`, which is the abbreviation of (only defense)
    - Including the above two cases `P-OD`
  - Use `:` to connect each part.
  
- `--defense` (optional): use defense algorithm to defense the Reid model. **Need one argument.** The `defense algorithm` should be chosen in 
  - The first part,defense type
     - `ADV`
     - `GOAT`
     - `EST`
     - `SES`
  - The second part,after training your defense model first time, you could use `:` to load the pretrained defense model
  
Each result will be saved in an excel in the root path as `./result.xlsx`. Except long python terminal outputs, all your works will be recorded in file `./log.txt` with a short and clear notes, you don't need to worry about that `log.txt` will be recovered because all the notes would be written at the end of the file. **No argument**


## examples
basic format for train
```bash
python3 tools/train_net.py --config-file ./configs/DATASETS/FILE --train 
```
basic format for attack and defense
```bash
python3 tools/train_net.py --config-file ./configs/DATASETS/FILE --attack X:X:X --defense X 
```
#### 1.train a reid model in DukeMTMC bot
```bash
python3 tools/train_net.py --config-file ./configs/DukeMTMC/bagtricks_R50.yml --train 
```
#### 2.attack a reid model of DukeMTMC bot by FGSM in QA-
```bash
python3 tools/train_net.py --config-file ./configs/DukeMTMC/bagtricks_R50.yml --attack QA-:FGSM 
```
#### 3.attack a reid model of Market1501 agw by MUAP in QA-, and defense in ADV,use pretrained defense model
```bash
python3 tools/train_net.py --config-file ./configs/Market1501/AGW_R50.yml --attack QA-:MUAP --defense ADV: 
```
#### 4.attack the reid model by SSAE in QA+ with a pretrained SSAE attack model without record, and defense in GOAT
```bash
python3 tools/train_net.py --config-file ./configs/DukeMTMC/bagtricks_R50.yml --attack QA+:SSAE:P --defense GOAT 
```
#### 5.only attack the defense model, with defense model pretrained
```bash
python3 tools/train_net.py --config-file ./configs/Market1501/bagtricks_R50.yml --attack GA-:IFGSM:OD --defense SES:
```

**If you meet any issue in your work, please first check out whether your command line input usage is right, and then look up in [Issues.md](Issues.md). If the problem still exists, leave your issue and I will reply as soon as possible.**
