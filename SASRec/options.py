from datasets import DATASETS
from dataloaders import DATALOADERS
from models import MODELS
from trainers import TRAINERS

import argparse

parser = argparse.ArgumentParser(description='Bert4Rec')

################
# Test
################
parser.add_argument('--test_model_path', type=str, default=None)
parser.add_argument('--load_pretrained_weights', type=str, default=None)
parser.add_argument('--cold_start_threshold', type=int, default=None)

################
# Dataset
################
parser.add_argument('--dataset_code', type=str, default='ml-20m', choices=DATASETS.keys())
parser.add_argument('--split', type=str, default='leave_one_out', help='How to split the datasets')
parser.add_argument('--dataset_split_seed', type=int, default=98765)
parser.add_argument('--data_path', type=str, default=None)
parser.add_argument('--side_info', type=str, default=None)

################
# Dataloader
################
parser.add_argument('--dataloader_code', type=str, default='bert', choices=DATALOADERS.keys())
parser.add_argument('--dataloader_random_seed', type=float, default=0.0)
parser.add_argument('--train_batch_size', type=int, default=64)
parser.add_argument('--val_batch_size', type=int, default=64)
parser.add_argument('--test_batch_size', type=int, default=64)
parser.add_argument('--mask_align', type=int, default=1)

################
# NegativeSampler
################
parser.add_argument('--train_negative_sampler_code', type=str, default='random', choices=['popular', 'random'],
                    help='Method to sample negative items for training. Not used in bert')
parser.add_argument('--train_negative_sample_size', type=int, default=100)
parser.add_argument('--train_negative_sampling_seed', type=int, default=None)
parser.add_argument('--test_negative_sampler_code', type=str, default='random', choices=['popular', 'random'],
                    help='Method to sample negative items for evaluation')
parser.add_argument('--test_negative_sample_size', type=int, default=100)
parser.add_argument('--test_negative_sampling_seed', type=int, default=None)

################
# Trainer
################
parser.add_argument('--trainer_code', type=str, default='bert', choices=TRAINERS.keys())
parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
parser.add_argument('--num_gpu', type=int, default=1)
parser.add_argument('--device_idx', type=str, default='0')
# optimizer #
parser.add_argument('--optimizer', type=str, default='AdamW', choices=['SGD', 'AdamW','Adam'])
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=0, help='l2 regularization')
parser.add_argument('--momentum', type=float, default=None, help='SGD momentum')
parser.add_argument('--warmup_steps', type=int, default=100, help='Warmup Steps')
parser.add_argument('--adam_epsilon', type=float, default=1e-6, help='Adam Epsilon')
parser.add_argument('--loss_code', type=str, default=None, help='Types of loss')
parser.add_argument('--num_samples', type=int, default=1, help='Negatives number')
parser.add_argument('--query_weight', type=float, default=1.0, help='weight of query prediction loss')
parser.add_argument('--visual_weight', type=float, default=1.0, help='weight of visual prediction loss')
parser.add_argument('--keyword_weight', type=float, default=1.0, help='weight of keyword prediction loss')

# lr scheduler #
parser.add_argument('--enable_lr_schedule', type=str, default="linear", help='Enable lr scheduler')
parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs for training')
parser.add_argument('--stop_epochs', type=int, default=None, help='Number of epochs to stop, smaller than num_epochs')
parser.add_argument('--verbose', type=int, default=10)
# evaluation #
parser.add_argument('--metric_ks', nargs='+', type=int, default=[5, 10, 20], help='ks for Metric@k')
parser.add_argument('--best_metric', type=str, default='NDCG@10', help='Metric for determining the best model')

################
# Model
################
parser.add_argument('--model_code', type=str, default='bert', choices=MODELS.keys())
parser.add_argument('--model_init_seed', type=int, default=None)
# BERT #
parser.add_argument('--bert_max_len', type=int, default=None, help='Length of sequence for bert')
parser.add_argument('--bert_hidden_dim', type=int, default=None, help='Size of hidden vectors (d_model)')
parser.add_argument('--bert_num_blocks', type=int, default=None, help='Number of transformer layers')
parser.add_argument('--shared_bert_num_blocks', type=int, default=1, help='Number of shared transformer layers')
parser.add_argument('--separated_bert_num_blocks', type=int, default=1, help='Number of seperated transformer layers')
parser.add_argument('--att_stride', type=int, default=None, help='stride for co-att sliding window')
parser.add_argument('--init_val', type=float, default=None, help='initial value for hyper-param opt')
parser.add_argument('--co_att_strides', type=str, default="1", help='stride for co-att sliding window')
parser.add_argument('--bert_num_heads', type=int, default=None, help='Number of heads for multi-attention')
parser.add_argument('--bert_dropout', type=float, default=None, help='Dropout probability to use throughout the model')
parser.add_argument('--bert_att_dropout', type=float, default=0.2, help='Dropout probability to use throughout the attention scores')
parser.add_argument('--cross_att_dropout', type=float, default=0, help='Dropout probability to use throughout the attention scores')
parser.add_argument('--bert_pos_dropout', type=float, default=0.2, help='Dropout probability to use throughout the positional embedding')
parser.add_argument('--bert_mask_prob', type=float, default=None, help='Probability for masking items in the training sequence')

# VBERT #
parser.add_argument('--fusion_code', type=str, default='simple', help='Types of fusion')

# SSE-PT #
parser.add_argument('--item_hidden_dim', type=int, default=None, help='Size of item hidden vectors (d_model)')
parser.add_argument('--user_hidden_dim', type=int, default=None, help='Size of user hidden vectors (d_model)')
parser.add_argument('--user_replace_prob', type=float, default=None, help='user replace prob')
parser.add_argument('--item_replace_prob', type=float, default=None, help='item replace prob')

# Keyword #
parser.add_argument('--bert_keyword_max_len', type=int, default=50, help='Max length of keywords for one item')

# Ablation #
parser.add_argument('--ablation', type=str, default='no_odl', help='Types of fusion')
parser.add_argument('--case_study', type=int, default=0, help='Whether save outputs for case study')

################
# Experiment
################
parser.add_argument('--experiment_dir', type=str, default='experiments')
parser.add_argument('--experiment_description', type=str, default='test')
parser.add_argument('--hyper_search', type=int, default=0)
parser.add_argument('--attention_map', type=int, default=0)
parser.add_argument('--personalized', type=int, default=0)

args = parser.parse_args()

args.side_info = args.side_info.split(",") if args.side_info is not None else None
args.co_att_strides = [int(i) for i in args.co_att_strides.split(",")] if args.co_att_strides is not None else None
