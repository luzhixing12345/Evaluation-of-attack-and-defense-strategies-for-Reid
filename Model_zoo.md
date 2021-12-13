

# Reid model 

### Market1501 Baselines

- **BoT**:

| Method | Pretrained | Rank@1 | mAP | mINP | download |
| :---: | :---: | :---: |:---: | :---: | :---: |
| [BoT(R50)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/Market1501/bagtricks_R50.yml) | ImageNet | 94.4% | 86.1% | 59.4% | [model](https://github.com/luzhixing12345/Evaluation-of-attack-and-defense-strategies-for-Reid/releases/download/v0.0.0/model_Market1501_bot.pth) |
| [BoT(R50-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/Market1501/bagtricks_R50-ibn.yml) | ImageNet | 94.9% | 87.6% | 64.1% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/market_bot_R50-ibn.pth) |
| [BoT(S50)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/Market1501/bagtricks_S50.yml) | ImageNet | 95.2% | 88.7% | 66.9% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/market_bot_S50.pth) |
| [BoT(R101-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/Market1501/bagtricks_R101-ibn.yml) | ImageNet| 95.4% | 88.9% | 67.4% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/market_bot_R101-ibn.pth) |

- **AGW**:

| Method | Pretrained | Rank@1 | mAP | mINP | download |
| :---: | :---: | :---: |:---: | :---: |:---: |
| [AGW(R50)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/Market1501/AGW_R50.yml) | ImageNet | 95.3% | 88.2% | 66.3% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/market_agw_R50.pth) |
| [AGW(R50-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/Market1501/AGW_R50-ibn.yml) | ImageNet | 95.1% | 88.7% | 67.1% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/market_agw_R50-ibn.pth) |
| [AGW(S50)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/Market1501/AGW_S50.yml) | ImageNet | 95.3% | 89.3% | 68.5% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/market_agw_S50.pth) |
| [AGW(R101-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/Market1501/AGW_R101-ibn.yml) | ImageNet | 95.5% | 89.5% | 69.5% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/market_agw_R101-ibn.pth) |

### DukeMTMC Baseline

**BoT**:

| Method | Pretrained | Rank@1 | mAP | mINP | download |
| :---: | :---: | :---: |:---: | :---: | :---: |
| [BoT(R50)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/DukeMTMC/bagtricks_R50.yml) | ImageNet | 87.2% | 77.0% | 42.1% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/duke_bot_R50.pth) |
| [BoT(R50-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/DukeMTMC/bagtricks_R50-ibn.yml) | ImageNet | 89.3% | 79.6% | 45.2% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/duke_bot_R50-ibn.pth) |
| [BoT(S50)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/DukeMTMC/bagtricks_S50.yml) | ImageNet | 90.0% | 80.13% | 45.8% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/duke_bot_S50.pth) |
| [BoT(R101-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/DukeMTMC/bagtricks_R101-ibn.yml) | ImageNet| 91.2% | 81.2% | 47.5% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/duke_bot_R101-ibn.pth) |

**AGW**:

| Method | Pretrained | Rank@1 | mAP | mINP | download |
| :---: | :---: | :---: |:---: | :---: | :---:|
| [AGW(R50)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/DukeMTMC/AGW_R50.yml) | ImageNet | 89.0% | 79.9% | 46.1% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/duke_agw_R50.pth) |
| [AGW(R50-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/DukeMTMC/AGW_R50-ibn.yml) | ImageNet | 90.5% | 80.8% | 47.6% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/duke_agw_R50-ibn.pth) |
| [AGW(S50)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/DukeMTMC/AGW_S50.yml) | ImageNet | 90.9% | 82.4% | 49.2% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/duke_agw_S50.pth) |
| [AGW(R101-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/DukeMTMC/AGW_R101-ibn.yml) | ImageNet | 91.7% | 82.3% | 50.0% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/duke_agw_R101-ibn.pth) |

# Attack algorithm model

**SSAE**:

| Method | Pretrained | Rank@1 | mAP | mINP | download |
| :---: | :---: | :---: |:---: | :---: | :---:|
| [AGW(R50)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/DukeMTMC/AGW_R50.yml) | ImageNet | 89.0% | 79.9% | 46.1% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/duke_agw_R50.pth) |
| [AGW(R50-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/DukeMTMC/AGW_R50-ibn.yml) | ImageNet | 90.5% | 80.8% | 47.6% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/duke_agw_R50-ibn.pth) |
| [AGW(S50)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/DukeMTMC/AGW_S50.yml) | ImageNet | 90.9% | 82.4% | 49.2% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/duke_agw_S50.pth) |
| [AGW(R101-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/DukeMTMC/AGW_R101-ibn.yml) | ImageNet | 91.7% | 82.3% | 50.0% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/duke_agw_R101-ibn.pth) |

**MISR**:

| Method | Pretrained | Rank@1 | mAP | mINP | download |
| :---: | :---: | :---: |:---: | :---: | :---:|
| [AGW(R50)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/DukeMTMC/AGW_R50.yml) | ImageNet | 89.0% | 79.9% | 46.1% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/duke_agw_R50.pth) |
| [AGW(R50-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/DukeMTMC/AGW_R50-ibn.yml) | ImageNet | 90.5% | 80.8% | 47.6% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/duke_agw_R50-ibn.pth) |
| [AGW(S50)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/DukeMTMC/AGW_S50.yml) | ImageNet | 90.9% | 82.4% | 49.2% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/duke_agw_S50.pth) |
| [AGW(R101-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/DukeMTMC/AGW_R101-ibn.yml) | ImageNet | 91.7% | 82.3% | 50.0% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/duke_agw_R101-ibn.pth) |

# Defense algorithm model

**GOAT**:

| Method | Pretrained | Rank@1 | mAP | mINP | download |
| :---: | :---: | :---: |:---: | :---: | :---:|
| [AGW(R50)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/DukeMTMC/AGW_R50.yml) | ImageNet | 89.0% | 79.9% | 46.1% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/duke_agw_R50.pth) |
| [AGW(R50-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/DukeMTMC/AGW_R50-ibn.yml) | ImageNet | 90.5% | 80.8% | 47.6% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/duke_agw_R50-ibn.pth) |
| [AGW(S50)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/DukeMTMC/AGW_S50.yml) | ImageNet | 90.9% | 82.4% | 49.2% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/duke_agw_S50.pth) |
| [AGW(R101-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/DukeMTMC/AGW_R101-ibn.yml) | ImageNet | 91.7% | 82.3% | 50.0% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/duke_agw_R101-ibn.pth) |

**DISTALL**:

| Method | Pretrained | Rank@1 | mAP | mINP | download |
| :---: | :---: | :---: |:---: | :---: | :---:|
| [AGW(R50)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/DukeMTMC/AGW_R50.yml) | ImageNet | 89.0% | 79.9% | 46.1% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/duke_agw_R50.pth) |
| [AGW(R50-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/DukeMTMC/AGW_R50-ibn.yml) | ImageNet | 90.5% | 80.8% | 47.6% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/duke_agw_R50-ibn.pth) |
| [AGW(S50)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/DukeMTMC/AGW_S50.yml) | ImageNet | 90.9% | 82.4% | 49.2% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/duke_agw_S50.pth) |
| [AGW(R101-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/DukeMTMC/AGW_R101-ibn.yml) | ImageNet | 91.7% | 82.3% | 50.0% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/duke_agw_R101-ibn.pth) |
