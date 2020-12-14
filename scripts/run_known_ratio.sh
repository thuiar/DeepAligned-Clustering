
#!/usr/bin bash

for s in 0 1 2 3 4 5 6 7 8 9
do 
    for k in 0.25 0.5
    do
        
        python aba.py \
            --dataset stackoverflow \
            --known_cls_ratio $k \
            --cluster_num_factor 1 \
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
