# An Effective CNN and Transformer Fusion Network for Camouflaged Object Detection


> **Authors:** 
> Dongdong Zhang,
> Chunping Wang,
> Huiying Wang,
> Qiang Fu,
> Zhaorui Li.

## 1. Preface

- This repository provides code for "_**An Effective CNN and Transformer Fusion Network for Camouflaged Object Detection**_"

## 2. Proposed Baseline

### 2.1. Training/Testing

The training and testing experiments are conducted using [PyTorch](https://github.com/pytorch/pytorch) with 
a single NVIDIA  GeForce RTX 3090 GPU of 24 GB Memory.

1. Configuring your environment (Prerequisites):
    
    + Creating a virtual environment in terminal: `conda create -n CTFNet python=3.7`.
    
    + Installing necessary packages: `pip install -r requirements.txt`.

1. Downloading necessary data:

    + downloading testing dataset and move it into `./data/TestDataset/`, 
    which can be found in this [download link (Google Drive)](https://drive.google.com/file/d/1SLRB5Wg1Hdy7CQ74s3mTQ3ChhjFRSFdZ/view?usp=sharing).
    
    + downloading training dataset and move it into `./data/TrainDataset/`, 
    which can be found in this [download link (Google Drive)](https://drive.google.com/file/d/1Kifp7I0n9dlWKXXNIbN7kgyokoRY4Yz7/view?usp=sharing).
    
    + downloading pretrained weights and move it into `./checkpoints/CTF-Net/Net_epoch_best.pth`, 
    which can be found in this [download link (Baidu Pan)](https://pan.baidu.com/s/1id9-vZy3bReN90PDXKfu4Q    hyu2).
    
    + downloading Res2Net weights and move it into `./models/res2net50_v1b_26w_4s-3cf99910.pth`[download link (Google Drive)](https://drive.google.com/file/d/1_1N-cx1UpRQo7Ybsjno1PAg4KE1T9e5J/view?usp=sharing).

    + downloading PVTv2 weights and move it into `./pvt_v2_b2.pth`[download link (Baidu Pan)](https://pan.baidu.com/s/1n5d-q4Wj3EN7kLxNv6Xg1A   a8of).
   
1. Training Configuration:

    + Assigning your costumed path, like `--train_save` and `--train_path` in `etrain.py`.

1. Testing Configuration:

    + After you download all the pre-trained model and testing dataset, just run `etest.py` to generate the final prediction map: 
    replace your trained model directory (`--pth_path`).

### 2.2 Evaluating your trained model:

Assigning your costumed path, like `pred_root` and `model_lst` in `MyEval.py`.

Just run `MyEval.py` to evaluate the trained model.

> pre-computed maps of CTF-Net can be found in [download link (Baidu Pan)](https://pan.baidu.com/s/1IPxvXD49Oq5yyjhNyiRqpw    irmx).


