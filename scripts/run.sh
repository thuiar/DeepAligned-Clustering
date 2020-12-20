#!/usr/bin bash

for s in 0 1 2 3 4 5 6 7 8 9
do 
    python DeepAligned.py \
        --dataset clinc \
        --known_cls_ratio 0.75 \
        --cluster_num_factor 1 \
        --seed $s \
        --freeze_bert_parameters \
        --pretrain
done
