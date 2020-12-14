
#!/usr/bin bash

for s in 4
do 
    for d in clinc
    do
        for k in 0.5
        do

            python aba.py \
                --dataset clinc \
                --known_cls_ratio $k \
                --cluster_num_factor 2 \
                --seed $s \
                --num_train_epochs 100 \
                --num_pretrain_epochs 100 \
                --lr 5e-5 \
                --lr_pre 5e-5 \
                --gpu_id 0 \
                --save_results \
                --freeze_bert_parameters \
                --pretrain
                
        done
    done
done

