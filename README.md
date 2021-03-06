<img src="https://media.giphy.com/media/xUOxeQdcBbmybIAjNm/giphy.gif" width="250" height="250" />

learning to unjumble as a pretraining objective for RoBERTa

## GLUE Results Using RoBERTa with Jumbled Token Discrimination Loss

#### jumbling probability = `0.15`, peak lr = `1e-4`, steps = `1000`
- Score on QNLI task: see [`notebooks_roberta_jumbled_token_discrimination_lr_e-4_prob_0.15/qnli.ipynb`](https://github.com/subhadarship/learning-to-unjumble/tree/master/notebooks_GLUE/notebooks_roberta_jumbled_token_discrimination_lr_e-4_prob_0.15/qnli.ipynb)
- Score on RTE task: see [`notebooks_roberta_jumbled_token_discrimination_lr_e-4_prob_0.15/rte.ipynb`](https://github.com/subhadarship/learning-to-unjumble/tree/master/notebooks_GLUE/notebooks_roberta_jumbled_token_discrimination_lr_e-4_prob_0.15/rte.ipynb)
- Score on CoLA task: see [`notebooks_roberta_jumbled_token_discrimination_lr_e-4_prob_0.15/cola.ipynb`](https://github.com/subhadarship/learning-to-unjumble/tree/master/notebooks_GLUE/notebooks_roberta_jumbled_token_discrimination_lr_e-4_prob_0.15/cola.ipynb)
- Score on SST task: see [`notebooks_roberta_jumbled_token_discrimination_lr_e-4_prob_0.15/sst-2.ipynb`](https://github.com/subhadarship/learning-to-unjumble/tree/master/notebooks_GLUE/notebooks_roberta_jumbled_token_discrimination_lr_e-4_prob_0.15/sst-2.ipynb)
- Score on MRPC task: see [`notebooks_roberta_jumbled_token_discrimination_lr_e-4_prob_0.15/mrpc.ipynb`](https://github.com/subhadarship/learning-to-unjumble/tree/master/notebooks_GLUE/notebooks_roberta_jumbled_token_discrimination_lr_e-4_prob_0.15/mrpc.ipynb)
- Score on STS-B task: see [`notebooks_roberta_jumbled_token_discrimination_lr_e-4_prob_0.15/sts-b.ipynb`](https://github.com/subhadarship/learning-to-unjumble/tree/master/notebooks_GLUE/notebooks_roberta_jumbled_token_discrimination_lr_e-4_prob_0.15/sts-b.ipynb)
- Score on WNLI task: see [`notebooks_roberta_jumbled_token_discrimination_lr_e-4_prob_0.15/wnli.ipynb`](https://github.com/subhadarship/learning-to-unjumble/tree/master/notebooks_GLUE/notebooks_roberta_jumbled_token_discrimination_lr_e-4_prob_0.15/wnli.ipynb)

#### jumbling probability = `0.30`, peak lr = `1e-4`, steps = `1000`
- Score on QNLI task: see [`notebooks_roberta_jumbled_token_discrimination_lr_e-4_prob_0.3/qnli.ipynb`](https://github.com/subhadarship/learning-to-unjumble/tree/master/notebooks_GLUE/notebooks_roberta_jumbled_token_discrimination_lr_e-4_prob_0.3/qnli.ipynb)
- Score on RTE task: see [`notebooks_roberta_jumbled_token_discrimination_lr_e-4_prob_0.3/rte.ipynb`](https://github.com/subhadarship/learning-to-unjumble/tree/master/notebooks_GLUE/notebooks_roberta_jumbled_token_discrimination_lr_e-4_prob_0.3/rte.ipynb)
- Score on CoLA task: see [`notebooks_roberta_jumbled_token_discrimination_lr_e-4_prob_0.3/cola.ipynb`](https://github.com/subhadarship/learning-to-unjumble/tree/master/notebooks_GLUE/notebooks_roberta_jumbled_token_discrimination_lr_e-4_prob_0.3/cola.ipynb)
- Score on SST task: see [`notebooks_roberta_jumbled_token_discrimination_lr_e-4_prob_0.3/sst-2.ipynb`](https://github.com/subhadarship/learning-to-unjumble/tree/master/notebooks_GLUE/notebooks_roberta_jumbled_token_discrimination_lr_e-4_prob_0.3/sst-2.ipynb)
- Score on MRPC task: see [`notebooks_roberta_jumbled_token_discrimination_lr_e-4_prob_0.3/mrpc.ipynb`](https://github.com/subhadarship/learning-to-unjumble/tree/master/notebooks_GLUE/notebooks_roberta_jumbled_token_discrimination_lr_e-4_prob_0.3/mrpc.ipynb)
- Score on STS-B task: see [`notebooks_roberta_jumbled_token_discrimination_lr_e-4_prob_0.3/sts-b.ipynb`](https://github.com/subhadarship/learning-to-unjumble/tree/master/notebooks_GLUE/notebooks_roberta_jumbled_token_discrimination_lr_e-4_prob_0.3/sts-b.ipynb)
- Score on WNLI task: see [`notebooks_roberta_jumbled_token_discrimination_lr_e-4_prob_0.3/wnli.ipynb`](https://github.com/subhadarship/learning-to-unjumble/tree/master/notebooks_GLUE/notebooks_roberta_jumbled_token_discrimination_lr_e-4_prob_0.3/wnli.ipynb)

## GLUE Results Using RoBERTa with Masked Token Discrimination Loss [BASELINE]

#### masking probability = `0.15`, peak lr = `1e-4`, steps = `1000`
- Score on QNLI task: see [`baseline/qnli.ipynb`](https://github.com/subhadarship/learning-to-unjumble/tree/master/notebooks_GLUE/baseline/qnli.ipynb)
- Score on RTE task: see [`baseline/rte.ipynb`](https://github.com/subhadarship/learning-to-unjumble/tree/master/notebooks_GLUE/baseline/rte.ipynb)
- Score on CoLA task: see [`baseline/cola.ipynb`](https://github.com/subhadarship/learning-to-unjumble/tree/master/notebooks_GLUE/baseline/cola.ipynb)
- Score on SST task: see [`baseline/sst-2.ipynb`](https://github.com/subhadarship/learning-to-unjumble/tree/master/notebooks_GLUE/baseline/sst-2.ipynb)
- Score on MRPC task: see [`baseline/mrpc.ipynb`](https://github.com/subhadarship/learning-to-unjumble/tree/master/notebooks_GLUE/baseline/mrpc.ipynb)
- Score on STS-B task: see [`baseline/sts-b.ipynb`](https://github.com/subhadarship/learning-to-unjumble/tree/master/notebooks_GLUE/baseline/sts-b.ipynb)
- Score on WNLI task: see [`baseline/wnli.ipynb`](https://github.com/subhadarship/learning-to-unjumble/tree/master/notebooks_GLUE/baseline/wnli.ipynb)


## Additional GLUE Results

### Pretrained RoBERTa
- roberta score on QNLI task: see [`notebooks_roberta/roberta_qnli.ipynb`](https://github.com/subhadarship/learning-to-unjumble/tree/master/notebooks_GLUE/notebooks_roberta/roberta_qnli.ipynb)
- roberta score on RTE task: see [`notebooks_roberta/roberta_rte.ipynb`](https://github.com/subhadarship/learning-to-unjumble/tree/master/notebooks_GLUE/notebooks_roberta/roberta_rte.ipynb)
- roberta score on CoLA task: see [`notebooks_roberta/roberta_cola.ipynb`](https://github.com/subhadarship/learning-to-unjumble/tree/master/notebooks_GLUE/notebooks_roberta/roberta_cola.ipynb)
- roberta score on SST task: see [`notebooks_roberta/roberta_sst.ipynb`](https://github.com/subhadarship/learning-to-unjumble/tree/master/notebooks_GLUE/notebooks_roberta/roberta_sst.ipynb)
- roberta score on MRPC task: see [`notebooks_roberta/Roberta_MRPC_Baseline.ipynb`](https://github.com/subhadarship/learning-to-unjumble/blob/master/notebooks_GLUE/notebooks_roberta/Roberta_MRPC_Baseline.ipynb)
- roberta score on STS-B task: see [`notebooks_roberta/Roberta_STS_B_Baseline.ipynb`](https://github.com/subhadarship/learning-to-unjumble/blob/master/notebooks_GLUE/notebooks_roberta/Roberta_STS_B_Baseline.ipynb)
- roberta score on WNLI task: see [`notebooks_roberta/roberta wnli.ipynb`](https://github.com/subhadarship/learning-to-unjumble/blob/master/notebooks_GLUE/notebooks_roberta/roberta%20wnli.ipynb)

### Pretrained ELECTRA
- electra score on QNLI task: see [`notebooks_electra/electra_qnli.ipynb`](https://github.com/subhadarship/learning-to-unjumble/tree/master/notebooks_GLUE/notebooks_electra/electra_qnli.ipynb)
- electra score on RTE task: see [`notebooks_electra/electra_rte.ipynb`](https://github.com/subhadarship/learning-to-unjumble/tree/master/notebooks_GLUE/notebooks_electra/electra_rte.ipynb)


## Training and Evaluation Command Lines

### Train with MLM Loss

```shell script
# make sure transformers version is 2.7.0
!pip install transformers==2.7.0

!cd ./unjumble

"""
# use wikidump data or wikitext data
"""
# download data
from torchtext.datasets import WikiText103
WikiText103.download('./data')

# run roberta training with MLM loss
python run_language_modeling.py \
--output_dir ./models/roberta_mlm \
--model_type roberta \
--mlm \
--do_train \
--do_eval \
--save_steps 2000 \
--per_gpu_train_batch_size 8 \
--evaluate_during_training \
--train_data_file data/wikitext-103/wikitext-103/wiki.train.tokens \
--line_by_line \
--eval_data_file data/wikitext-103/wikitext-103/wiki.test.tokens \
--model_name_or_path roberta-base

```

### Train with jumbled token discrimination loss

```shell script
# make sure transformers version is 2.7.0
!pip install transformers==2.7.0

!cd ./unjumble

# download data
TRAIN_DATA_PATH=../../data/wikidump/train.txt
VAL_DATA_PATH=../../data/wikidump/val.txt

# run roberta training with jumbled token-modification-discrimination head
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
--learning_rate 5e-5 \
--per_gpu_train_batch_size 16 \
--per_gpu_eval_batch_size 16 \
--train_data_file $TRAIN_DATA_PATH \
--eval_data_file $VAL_DATA_PATH \
--jumble_probability 0.15 \
--line_by_line \
--logging_steps 1 \
--do_eval \
--eval_all_checkpoints

```

### Train with POS based jumbled token discrimination loss

```shell script
# make sure transformers version is 2.7.0
!pip install transformers==2.7.0

!cd ./unjumble

# download data
TRAIN_DATA_PATH=../../data/wikidump/train.txt
VAL_DATA_PATH=../../data/wikidump/val.txt

# run roberta training with jumbled token-modification-discrimination head
!python run_language_modeling.py \
--output_dir ../models/roberta_token_discrimination \
--tensorboard_log_dir ../tb/roberta_token_discrimination \
--model_type roberta \
--model_name_or_path roberta-base \
--token_discrimination \
--pos \  # perform POS based jumbling (only Nouns and Adjectives are jumbled)
--do_train \
--gradient_accumulation_steps 64 \
--save_steps 50 \
--max_steps 1000 \
--weight_decay 0.01 \
--warmup_steps 100 \
--learning_rate 5e-5 \
--per_gpu_train_batch_size 16 \
--per_gpu_eval_batch_size 16 \
--train_data_file $TRAIN_DATA_PATH \
--eval_data_file $VAL_DATA_PATH \
--jumble_probability 0.15 \
--line_by_line \
--logging_steps 1 \
--do_eval \
--eval_all_checkpoints

```

### Train with masked token discrimination loss

```shell script
# make sure transformers version is 2.7.0
!pip install transformers==2.7.0

!cd ./unjumble

# download data
TRAIN_DATA_PATH=../../data/wikidump/train.txt
VAL_DATA_PATH=../../data/wikidump/val.txt

# run roberta training with masked token-modification-discrimination head
!python run_language_modeling.py \
--output_dir ../models/roberta_MASK_token_discrimination \
--tensorboard_log_dir ../tb/roberta_MASK_token_discrimination \
--model_type roberta \
--model_name_or_path roberta-base \
--mask_token_discrimination \  # NOTE THIS AND..
--do_train \
--gradient_accumulation_steps 64 \
--save_steps 50 \
--max_steps 1000 \
--weight_decay 0.01 \
--warmup_steps 100 \
--learning_rate 5e-5 \
--per_gpu_train_batch_size 16 \
--per_gpu_eval_batch_size 16 \
--train_data_file $TRAIN_DATA_PATH \
--eval_data_file $VAL_DATA_PATH \
--mask_probability 0.15 \  # ..THIS
--line_by_line \
--logging_steps 1 \
--do_eval \
--eval_all_checkpoints

```

### Running on Prince
```shell script
# Load these modules every time you log in
module purge
module load anaconda3/5.3.1
module load cuda/10.0.130
module load gcc/6.3.0

# Activate your environment

NETID=aa7513

source activate /scratch/${NETID}/nlu_projects/env

# Git pull/clone the repo

cd /scratch/${NETID}/

sbatch run_training.sbatch
```

### Compute GLUE scores
```shell script
# make sure transformers version is 2.8.0
!pip install transformers==2.8.0

cd ./compute_glue_scores

GLUE_DIR=../data/glue
TASK_NAME=QNLI  # specify GLUE task

# download GLUE data
!python download_glue_data.py --data_dir $GLUE_DIR --tasks $TASK_NAME

# specify the model directory
# the model directory may be a checkpoint directory
# it should contain config.json, merges.txt, pytorch_model.bin, special_tokens_map.json, tokenizer_config.json, training_args.bin, vocab.json
# it SHOULD NOT contain optimizer.pt and scheduler.pt
MODEL_DIR=../models/roberta_token_discrimination

OUTPUT_DIR=../models/glue/$TASK_NAME

# run glue
!python run_glue.py \
    --model_type roberta \
    --model_name_or_path $MODEL_DIR \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --data_dir $GLUE_DIR/$TASK_NAME \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=64   \
    --per_gpu_train_batch_size=64   \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --output_dir $OUTPUT_DIR \
    --overwrite_output_dir

```