#!/bin/bash

cd ../compute_glue_scores
echo "WORKING DIR: $PWD"

# prepare glue data
GLUE_DATA_DIR=../data/glue_data
mkdir -p $GLUE_DATA_DIR
python download_glue_data.py --data_dir ${GLUE_DATA_DIR} --tasks RTE,STS-B,WNLI,SST-2,QNLI,CoLA,MRPC

for TASK in CoLA SST-2 QNLI RTE STS-B WNLI MRPC; do
  for SEED in 100 200 300 400 500; do
    for MODEL_NAME in roberta_tokdisc_p0.15 roberta_mlm roberta_tokdisc_p0.3; do
      MODEL_DIR=../models/${MODEL_NAME}
      OUTPUT_DIR=../models/${MODEL_NAME}_glue_${TASK}_seed{SEED}
      python run_glue.py \
      --model_type roberta \
      --model_name_or_path $MODEL_DIR \
      --task_name $TASK \
      --do_train \
      --do_eval \
      --data_dir ${GLUE_DATA_DIR}/${TASK} \
      --max_seq_length 128 \
      --per_gpu_eval_batch_size=64 \
      --per_gpu_train_batch_size=64 \
      --learning_rate 2e-5 \
      --num_train_epochs 3 \
      --output_dir ${OUTPUT_DIR} \
      --seed ${SEED} \
      --overwrite_output_dir
    done
  done
done
