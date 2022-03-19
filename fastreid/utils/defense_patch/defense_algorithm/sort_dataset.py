# @copyright CEA-LIST/DIASI/SIALV/LVA (2020)
# @author CEA-LIST/DIASI/SIALV/LVA <quentin.bouniot@cea.fr>
# @license CECILL

import os
from shutil import copyfile
from pathlib import Path
from torchvision import transforms,datasets
from collections import defaultdict

def sort_datasets(datasets_name):
    print('Sort datasets for GOAT')
    # You only need to change this line to your dataset download path
    if datasets_name == 'Market1501':
        download_path = Path('./datasets/Market-1501-v15.09.15')
    elif datasets_name == 'DukeMTMC':
        download_path = Path('./datasets/DukeMTMC-reID')
    else:
        raise KeyError
    if not os.path.isdir(download_path):
        print('please change the download_path')
        
    save_path = download_path / 'sorted'
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
        
    #train
    train_path = download_path / 'bounding_box_train'
    train_save_path = save_path / 'train'
    if not os.path.isdir(train_save_path):
        os.mkdir(train_save_path)

    for root, dirs, files in os.walk(train_path, topdown=True):
        for name in files:
            if not (name[-3:]=='jpg'or name[-3:]=='png'):
                continue
            ID  = name.split('_')
            src_path = train_path / name
            dst_path = train_save_path / ID[0]
            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)
            copyfile(src_path, dst_path / name)
    print('finish dataset sort for GOAT defense')
    
    sorted_dataset = datasets.ImageFolder(os.path.join(download_path, 'sorted', 'train'), transform=train_transforms)

    idx = defaultdict(list)
    for image,label in sorted_dataset:
        idx[label].append(image)
    
    return idx
        
        
train_transforms = transforms.Compose([
    # transforms.Resize((288,144)),
    transforms.Resize((256,128), interpolation=3),
    transforms.Pad(10),
    transforms.RandomCrop((256,128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    # RandomErasing()
    ])