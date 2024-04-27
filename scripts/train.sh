#!/bin/bash
d1="multi"
if [ "$3" = "$d1" ]; then
    python train.py --aug1 perspective --aug2 none --epsilon 0.1 --gpu_id "$1" --shots "$2" --dataset "$3" --coral2
else
    python train.py --aug1 perspective --aug2 none --epsilon 0.001 --gpu_id "$1" --shots "$2" --dataset "$3" --coral2
fi