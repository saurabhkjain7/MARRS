#MARRS
Code release for our paper "MARRS: Modern Backbones Assisted Co-Training for Rapid and Robust Semi-Supervised Domain Adaptation'' accepted by CVPRW 2023.

[Paper](https://openaccess.thecvf.com/content/CVPR2023W/ECV/html/Jain_MARRS_Modern_Backbones_Assisted_Co-Training_for_Rapid_and_Robust_Semi-Supervised_CVPRW_2023_paper.html) 

## Requirements
Python 3.8.10, Pytorch 1.9.0, Torch Vision 0.10.0. Use the provided requirements.txt file to create virtual environment.

## Data preparation
[Office Dataset](https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE/view?pli=1&resourcekey=0-gNMHVtZfRAyO_t2_WrOunA),
[OfficeHome Dataset](http://hemanthdv.org/OfficeHome-Dataset/), [DomainNet dataset](http://ai.bu.edu/M3SDA/)

**Note**: For DomainNet dataset download the original version not cleaned version

Prepare OfficeHome dataset like this in images directory.
```
./images/office_home/Real
./images/office_home/Clipart
./images/office_home/Product
./images/office_home/Art
```

Prepare DomaiNet dataset like this in images directory.
```
./images/dnet/real
./images/dnet/painting
./images/dnet/clipart
./images/dnet/sketch
```

Prepare Office-31 dataset like this in images directory.
```
./images/dnet/amazon
./images/dnet/dslr
./images/dnet/webcam
```

## Stage-1: Feature extraction stage

Use the `features_extract.sh` script to extract and store features using modern backbones in the feature_weights folder.

```
sh scripts/features_extract.sh $gpu-id 
```

## Stage-2: Classifier training stage

Use the `train.sh` script to train classifier using features extracted from the first stage.

Ex. 3-shot domain adaptation on the OfficeHome dataset.

```
sh scripts/train.sh $gpu-id 3 office_home 
```

## Knowledge distillation 

Use the `train_student.sh` script to train a small network using base classifiers trained from second stage.

Ex. 3-shot domain adaptation on the OfficeHome dataset.

```
sh scripts/train_student.sh $gpu-id 3 office_home 
```

## Reference codes
Part of our codes are taken from the following links:
1. MME: https://github.com/VisionLearningGroup/SSDA_MME
2. PACE: https://github.com/Chris210634/PACE-Domain-Adaptation/tree/main

## Reference
This repository is contributed by [Saurabh Kumar Jain](http://www.cse.iitm.ac.in/profile.php?arg=Mjc4MQ==).
If you consider using this code or its derivatives, please consider citing:

```
@InProceedings{Jain_2023_CVPR,
    author    = {Jain, Saurabh Kumar and Das, Sukhendu},
    title     = {MARRS: Modern Backbones Assisted Co-Training for Rapid and Robust Semi-Supervised Domain Adaptation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2023},
    pages     = {4580-4589}
}

```