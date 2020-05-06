import logging
import os
import pickle
from itertools import groupby
from operator import itemgetter
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    PreTrainedTokenizer,
)
from transformers import RobertaTokenizer

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)


def get_mapping_from_subwords(subwords):
    mapping_idx = -1
    mapping_list = []
    for subword in subwords:
        if subword.startswith('Ġ'):
            mapping_idx = mapping_idx + 1
        mapping_list.append(mapping_idx)
    return mapping_list


def scramble(seq, mapping, prob=0.15):
    seq_to_shuffle = ['<unjumble>'.join([c for i, c in group]) for key, group in
                      groupby(zip(mapping, seq), itemgetter(0))]
    seq_to_shuffle = np.array(seq_to_shuffle)
    num_permute = int(len(list(set(mapping))) * prob)
    full_permutation = np.random.permutation(len(seq_to_shuffle))
    inverse_full_permutation = np.argsort(full_permutation)
    partial_permutation = np.random.permutation(num_permute)
    seq_to_shuffle = seq_to_shuffle[full_permutation]
    seq_to_shuffle = np.concatenate(
        (seq_to_shuffle[:num_permute][partial_permutation], seq_to_shuffle[num_permute:]))
    seq_to_shuffle = seq_to_shuffle[inverse_full_permutation]
    seq_to_shuffle = [s.split('<unjumble>') for s in seq_to_shuffle]
    seq_to_shuffle = [y for x in seq_to_shuffle for y in x]
    return seq_to_shuffle


def mask_tokens(inputs: torch.Tensor, tokenizer: PreTrainedTokenizer, args) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )

    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, args.mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


class TextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, file_path: str, block_size=512):
        assert os.path.isfile(file_path)

        block_size = block_size - (tokenizer.max_len - tokenizer.max_len_single_sentence)

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory, args.model_type + "_cached_lm_" + str(block_size) + "_" + filename
        )

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", directory)

            self.examples = []
            with open(file_path, encoding="utf-8") as f:
                text = f.read()

            tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

            for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
                self.examples.append(tokenizer.build_inputs_with_special_tokens(tokenized_text[i: i + block_size]))
            # Note that we are loosing the last truncated example here for the sake of simplicity (no padding)
            # If your dataset is small, first you should loook for a bigger one :-) and second you
            # can change this behavior by adding (model specific) padding.

            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)


class LineByLineTextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, file_path: str, block_size=512):
        assert os.path.isfile(file_path)
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        self.examples = tokenizer.batch_encode_plus(lines, add_special_tokens=True, max_length=block_size)["input_ids"]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)


class LineByLineTextDatasetForJumbled(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args,
                 file_path: str,
                 block_size=512,
                 prob=0.15):
        assert os.path.isfile(file_path)
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            self.lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        # tokenize
        self.tokens = [
            tokenizer.tokenize(
                line, add_special_tokens=True, max_length=block_size
            ) for line in tqdm(self.lines)
        ]

        self.token_ids = [
            [tokenizer.bos_token_id] +
            tokenizer.convert_tokens_to_ids(token) +
            [tokenizer.eos_token_id] \
            for token in self.tokens
        ]  # token here is a set of tokens for a sequence actually

        # obtain mapping lists
        self.mapping_lists = [get_mapping_from_subwords(token)
                              for token in tqdm(self.tokens)]

        # jumble
        self.jumbled_tokens = [
            scramble(
                token, mapping_list, prob
            )
            for token, mapping_list in
            tqdm(zip(self.tokens, self.mapping_lists), total=len(self.tokens))
        ]

        # obtain jumbled token ids
        self.jumbled_tokens_ids = [
            [tokenizer.bos_token_id] +
            tokenizer.convert_tokens_to_ids(jumbled_tokens) +
            [tokenizer.eos_token_id] \
            for jumbled_tokens in self.jumbled_tokens
        ]

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, i):
        return (
            torch.tensor(self.jumbled_tokens_ids[i], dtype=torch.long),
            torch.tensor(self.tokens_ids[i], dtype=torch.long)
        )


class LineByLineTextDatasetForElectra(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args,
                 file_path: str,
                 block_size=512,
                 prob=0.15):
        assert os.path.isfile(file_path)
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            self.lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        # tokenize
        self.tokens = [
            tokenizer.tokenize(
                line, add_special_tokens=True, max_length=block_size
            ) for line in tqdm(self.lines)
        ]

        self.token_ids = [
            [tokenizer.bos_token_id] +
            tokenizer.convert_tokens_to_ids(token) +
            [tokenizer.eos_token_id] \
            for token in self.tokens
        ]  # token here is a set of tokens for a sequence actually

        # obtain mapping lists
        self.mapping_lists = [get_mapping_from_subwords(token)
                              for token in tqdm(self.tokens)]

        # jumble
        self.jumbled_tokens = [
            scramble(
                token, mapping_list, prob
            )
            for token, mapping_list in
            tqdm(zip(self.tokens, self.mapping_lists), total=len(self.tokens))
        ]

        # obtain jumbled token ids
        self.jumbled_tokens_ids = [
            [tokenizer.bos_token_id] +
            tokenizer.convert_tokens_to_ids(jumbled_tokens) +
            [tokenizer.eos_token_id] \
            for jumbled_tokens in self.jumbled_tokens
        ]

        # obtain label ids for electra loss
        self.labels = []
        for token_id, jumbled_token_id in zip(self.token_ids, self.jumbled_tokens_ids):
            self.labels.append(
                np.array(
                    np.array(token_id) == np.array(jumbled_token_id),
                    dtype=np.int
                ))

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, i):
        return (
            torch.tensor(self.jumbled_tokens_ids[i], dtype=torch.long),
            torch.tensor(self.labels[i], dtype=torch.long)
        )


def load_and_cache_examples(args, tokenizer, evaluate=False):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    if args.line_by_line and args.mlm:
        return LineByLineTextDataset(tokenizer, args, file_path=file_path, block_size=args.block_size)
    if args.line_by_line and args.electra_loss:
        return LineByLineTextDatasetForElectra(tokenizer, args, file_path=file_path, block_size=args.block_size,
                                               prob=args.prob)
    else:
        return TextDataset(tokenizer, args, file_path=file_path, block_size=args.block_size)


if __name__ == "__main__":
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    filepath = './data/wikitext-103/wikitext-103/wiki.valid.tokens'
    block_size = 512

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [
            line for line in tqdm(f.read().splitlines())
            if (len(line) > 0 and not line.isspace())
        ]

    lines_words = [line.split() for line in tqdm(lines)]
    """
    jumble after tokenization OR jumble before tokenization
    our preference: jumble after tokenization
    """

    tokens = [
        tokenizer.tokenize(
            line, add_special_tokens=True, max_length=block_size
        ) for line in tqdm(lines)
    ]

    """
    token_ids = tokenizer.batch_encode_plus(
        lines, add_special_tokens=True, max_length=block_size
    )
    """

    print(len(lines))
    print(lines[20])
    print(tokens[20])
    print(get_mapping_from_subwords(tokens[20]))

    mapping_lists = [get_mapping_from_subwords(token) for token in tqdm(tokens)]
    jumbled_tokens_50p = [
        scramble(
            token, mapping_list, prob=0.0
        )
        for token, mapping_list in
        tqdm(zip(tokens, mapping_lists), total=len(tokens))
    ]

    jumbled_tokens_ids_50p = [
        [tokenizer.bos_token_id] +
        tokenizer.convert_tokens_to_ids(jumbled_tokens) +
        [tokenizer.eos_token_id] \
        for jumbled_tokens in jumbled_tokens_50p
    ]

    print(jumbled_tokens_ids_50p[20])
    print(tokenizer.batch_encode_plus(
        lines,
        add_special_tokens=True,
        max_length=block_size
    )["input_ids"][20])

    """
    print(len(token_ids))
    print(token_ids[0])
    print(tokenizer.decode([0, 5457, 11858, 42292, 20577, 3916, 687, 5457, 2]))
    """
