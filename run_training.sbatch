#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --time=24:30:00
#SBATCH --mem=88GB
#SBATCH --mail-type=END
#SBATCH --mail-user=aa7513@nyu.edu

NETID=aa7513


cd /scratch/aa7513/nlu_projects/learning-to-unjumble/unjumble

DATA_DIR_TRAIN=/scratch/${NETID}/nlu_projects/data/wikidump/train.txt
DATA_DIR_VAL=/scratch/${NETID}/nlu_projects/data/wikidump/val.txt

python run_language_modeling.py \
--output_dir ../models/roberta_token_discrimination_lr_e4 \
--tensorboard_log_dir ../tb/roberta_token_discrimination_lr_e4 \
--model_type roberta \
--model_name_or_path roberta-base \
--token_discrimination \
--do_train \
--gradient_accumulation_steps 256 \
--save_steps 100 \
--max_steps 1000 \
--weight_decay 0.01 \
--warmup_steps 100 \
--learning_rate 1e-4 \
--per_gpu_train_batch_size 4 \
--per_gpu_eval_batch_size 4 \
--train_data_file $DATA_DIR_TRAIN \
--eval_data_file $DATA_DIR_VAL \
--jumble_probability 0.15 \
--line_by_line \
--logging_steps 1 \
--do_eval \
--eval_all_checkpoints \
