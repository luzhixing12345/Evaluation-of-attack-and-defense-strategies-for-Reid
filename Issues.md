

## This document is used to record some common issues which may occur when you use our framework


### 1. **RuntimeError: CUDA out of memory**

It occurs when your GPU has no availale memory for you to malloc, try to decrease `batch size` while training or testing by adding the following parse argument
default batch size for training is `64` for C/Q/GA part is `128`
```
python3 tools/train_net.py --config-file xxx.yml --attack xxx:xxx --defense xxx --record MODEL.DEVICE 'cuda:0' SOLVER.IMS_PER_BATCH 32
```
### 2. **requests.exceptions.ConnectionError: HTTPSConnectionPool(host='download.pytorch.org', port=443):Max retries exceeded with url: /models/resnet50-19c8e357.pth ......**
- The first solution: connect your computer with Internet
- The second solution: if your running environment can't connect with Internet, you could download `resnet50-19c8e357.pth` from [here]() and move it under `.cache/torch/checkpoints/resnet50-19c8e357.pth`