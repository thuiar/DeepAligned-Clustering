
#!/usr/bin bash

for s in 3 4 5 6 7 8 9
do
    
    python DeepInit.py \
        --dataset banking \
        --known_cls_ratio 0.75 \
        --cluster_num_factor 1 \
        --seed $s \
        --method wopre_init_zhl \
        --num_train_epochs 100 \
        --num_pretrain_epochs 100 \
        --lr 5e-5 \
        --lr_pre 5e-5 \
        --gpu_id 1 \
        --save_results \
        --freeze_bert_parameters 
        
done

# for s in 5 6 7 8 9
# do
    
#     python DeepInit.py \
#         --dataset clinc \
#         --known_cls_ratio 0.75 \
#         --cluster_num_factor 1 \
#         --seed $s \
#         --method wopre_init \
#         --num_train_epochs 100 \
#         --num_pretrain_epochs 100 \
#         --lr 5e-5 \
#         --lr_pre 5e-5 \
#         --gpu_id 1 \
#         --save_results \
#         --freeze_bert_parameters 
        
# done