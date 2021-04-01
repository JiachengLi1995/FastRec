DATASET=${2}

CUDA_VISIBLE_DEVICES=$2 python ${3}.py \
    --data_path "data/${DATASET}" \
    --train_batch_size 2048 \
    --val_batch_size 512 \
    --test_batch_size 512 \
    --test_negative_sampler_code 'random' \
    --test_negative_sample_size 1000 \
    --test_negative_sampling_seed 98765 \
    --device 'cuda' \
    --device_idx $2 \
    --optimizer 'Adam' \
    --lr 1e-3 \
    --weight_decay 0 \
    --num_epochs 10 \
    --global_epochs 1000 \
    --local_epochs 10 \
    --subset_size 5000000 \
    --best_metric 'NDCG@10' \
    --model_init_seed 0 \
    --trm_dropout 0.3 \
    --trm_att_dropout 0.1 \
    --trm_hidden_dim 50 \
    --trm_max_len 50 \
    --trm_num_blocks 2  \
    --trm_num_heads 1 \
    --verbose 1 