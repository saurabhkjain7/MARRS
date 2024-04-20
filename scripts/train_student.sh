#!/bin/bash
if  [ $3 == 'multi' ]
then
    python train.py --aug1 perspective --aug2 none --epsilon 0.1 --gpu_id $1 --shots $2 --dataset $3  --coral2 --kd --distill_steps 11000  
else
    python train.py --aug1 perspective --aug2 none --epsilon 0.001 --gpu_id $1 --shots $2 --dataset $3  --coral2 --kd --distill_steps 2600  
fi

