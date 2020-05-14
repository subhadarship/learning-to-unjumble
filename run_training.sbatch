#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --time=24:30:00
#SBATCH --mem=88GB
#SBATCH --mail-type=END
#SBATCH --mail-user=jh6926@nyu.edu

NETID=jh6926

cd ./unjumble

DATA_DIR_TRAIN=/scratch/${NETID}/nlu/data/wikidump/train.txt
DATA_DIR_VAL=/scratch/${NETID}/nlu/data/wikidump/val.txt

# run roberta training with masked token-modification-discrimination head
!python run_language_modeling.py \
--output_dir ../models/roberta_token_discrimination \
--tensorboard_log_dir ../tb/roberta_token_discrimination \
--model_type roberta \
--model_name_or_path roberta-base \
--token_discrimination \
--do_train \
--gradient_accumulation_steps 64 \
--save_steps 50 \
--max_steps 1000 \
--weight_decay 0.01 \
--warmup_steps 100 \
--learning_rate 1e-4 \
--per_gpu_train_batch_size 16 \
--per_gpu_eval_batch_size 16 \
--train_data_file $TRAIN_DATA_PATH \
--eval_data_file $VAL_DATA_PATH \
--jumble_probability 0.15 \
--line_by_line \
--logging_steps 1 \
--do_eval \
--eval_all_checkpoints
