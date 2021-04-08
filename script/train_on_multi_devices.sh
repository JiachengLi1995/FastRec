DATASET=${2}

CUDA_VISIBLE_DEVICES=$1 python large_gpu.py \
    --data_path "data/${DATASET}" \
    --train_batch_size 2048 \
    --val_batch_size 512 \
    --test_batch_size 512 \
    --test_negative_sampler_code 'random' \
    --test_negative_sample_size 1000 \
    --test_negative_sampling_seed 98765 \
    --device 'cuda' \
    --device_idx $1 \
    --emb_device_idx "{'cuda:0':(0,20), 'cuda:1':(20,50)}" \
    --optimizer 'Adam' \
    --lr 1e-3 \
    --weight_decay 0 \
    --num_epochs 10 \
    --global_epochs 20 \
    --local_epochs 5 \
    --subset_size 5000000 \
    --best_metric 'NDCG@10' \
    --model_init_seed 0 \
    --trm_dropout 0.3 \
    --trm_att_dropout 0.3 \
    --trm_hidden_dim 50 \
    --trm_max_len 50 \
    --trm_num_blocks 2  \
    --trm_num_heads 1 \
    --verbose 1 
