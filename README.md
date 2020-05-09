<img src="https://media.giphy.com/media/xUOxeQdcBbmybIAjNm/giphy.gif" width="250" height="250" />

learning to unjumble as a pretraining objective for RoBERTa

## BASELINE using RoBERTa
- roberta score on QNLI task: see [`notebooks_roberta/roberta_qnli.ipynb`](https://github.com/subhadarship/learning-to-unjumble/tree/master/notebooks_roberta/roberta_qnli.ipynb)
- roberta score on RTE task: see [`notebooks_roberta/roberta_rte.ipynb`](https://github.com/subhadarship/learning-to-unjumble/tree/master/notebooks_roberta/roberta_rte.ipynb)
- roberta score on CoLA task: see [`notebooks_roberta/roberta_cola.ipynb`](https://github.com/subhadarship/learning-to-unjumble/tree/master/notebooks_roberta/roberta_cola.ipynb)
- roberta score on SST task: see [`notebooks_roberta/roberta_sst.ipynb`](https://github.com/subhadarship/learning-to-unjumble/tree/master/notebooks_roberta/roberta_sst.ipynb)
- roberta score on MRPC task: see [`notebooks_roberta/Roberta_MRPC_Baseline.ipynb`](https://github.com/subhadarship/learning-to-unjumble/blob/master/notebooks_roberta/Roberta_MRPC_Baseline.ipynb)
- roberta score on STS-B task: see [`notebooks_roberta/Roberta_STS_B_Baseline.ipynb`](https://github.com/subhadarship/learning-to-unjumble/blob/master/notebooks_roberta/Roberta_STS_B_Baseline.ipynb)
- roberta score on WNLI task: see [`notebooks_roberta/roberta wnli.ipynb`](https://github.com/subhadarship/learning-to-unjumble/blob/master/notebooks_roberta/roberta%20wnli.ipynb)

## BASELINE using ELECTRA
- electra score on QNLI task: see [`notebooks_electra/electra_qnli.ipynb`](https://github.com/subhadarship/learning-to-unjumble/tree/master/notebooks_electra/electra_qnli.ipynb)
- electra score on RTE task: see [`notebooks_electra/electra_rte.ipynb`](https://github.com/subhadarship/learning-to-unjumble/tree/master/notebooks_electra/electra_rte.ipynb)

## Train with MLM Loss

```jupyter
# make sure transformers version is 2.7.0
!pip install transformers==2.7.0

!cd ./unjumble

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

## Train with token-modification-discrimination head

```jupyter
# make sure transformers version is 2.7.0
!pip install transformers==2.7.0

!cd ./unjumble

# download data
from torchtext.datasets import WikiText103
WikiText103.download('./data')

# run roberta training with token-modification-discrimination head
python run_language_modeling.py \
--output_dir ./models/roberta_token_mod_disc_head \
--model_type roberta \
--electra_loss \
--do_train \
--do_eval \
--save_steps 2000 \
--per_gpu_train_batch_size 8 \
--evaluate_during_training \
--train_data_file data/wikitext-103/wikitext-103/wiki.train.tokens \
--line_by_line \
--eval_data_file data/wikitext-103/wikitext-103/wiki.test.tokens \
--model_name_or_path roberta-base
--prob 0.15
--logging_steps 100

```
