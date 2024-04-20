#!/bin/bash

### For dominnet dataset
# python features_extract.py --dataset multi --augmentation none  --data_root images/dnet --image_list_root data/multi --gpu_id $1
# python features_extract.py --dataset multi --augmentation perspective  --data_root images/dnet --image_list_root data/multi --gpu_id $1
# python features_extract.py --dataset multi --augmentation randaugment  --data_root images/dnet --image_list_root data/multi --gpu_id $1

### For office_home dataset
python features_extract.py --dataset office_home --augmentation none --data_root images/office_home --image_list_root data/office_home --gpu_id $1
python features_extract.py --dataset office_home --augmentation perspective --data_root images/office_home --image_list_root data/office_home --gpu_id $1
python features_extract.py --dataset office_home --augmentation randaugment --data_root images/office_home --image_list_root data/office_home --gpu_id $1

### For office dataset
# python features_extract.py --dataset office --augmentation none  --data_root images/office --image_list_root data/office --gpu_id $1
# python features_extract.py --dataset office --augmentation perspective  --data_root images/office --image_list_root data/office --gpu_id $1
# python features_extract.py --dataset office --augmentation randaugment  --data_root images/office --image_list_root data/office --gpu_id $1

