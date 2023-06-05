# Federated Semi-supervised Medical Image Classification via Inter-client Relation Matching
by [Quande Liu](https://github.com/liuquande), [Hongzheng Yang](https://github.com/HongZhengYang), [Qi Dou](http://www.cse.cuhk.edu.hk/~qdou/), [Pheng-Ann Heng](http://www.cse.cuhk.edu.hk/~pheng/).

## Introduction

Pytorch implementation for MICCAI 2021 paper "Federated Semi-supervised Medical Image Classification via Inter-client Relation Matching"

![](figure/miccai2021_fedirm.png)
## Usage
1. create conda environment

       conda create -f environment.yml
       conda activate fedIRM

2. Install dependencies:

   1. install pytorch==1.8.0 torchvision==0.9.0 (via conda, recommend)

3. download the dataset from [kaggle](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection) and preprocess it follow this [notebook](https://www.kaggle.com/guiferviz/prepare-dataset-resizing-and-saving-as-png). You can download the preprocessed the dataset from [notebook](https://drive.google.com/drive/folders/1bhe_0KvdxEli7-6ZrQ9ahaDPpSnvF4UW?usp=share_link).

5. modify the corresponding data path in options.py

4. train the model

       python train_main.py
## Citation

If this repository is useful for your research, please cite:

    @article{liu2021federated,
      title={Federated Semi-supervised Medical Image Classification via Inter-client Relation Matching},
      author={Liu, Quande and Yang, Hongzheng and Dou, Qi and Heng, Pheng-Ann},
      journal={International Conference on Medical Image Computing and Computer Assisted Intervention},
      year={2021}
    }

### Questions

Please contact 'qdliu0226@gmail.com' or 'hzyang05@gmail.com'
