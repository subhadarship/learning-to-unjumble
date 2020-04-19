import numpy as np
from tqdm import tqdm
from transformers import RobertaTokenizer


def get_mapping_from_subwords(subwords):
    mapping_idx = -1
    mapping_list = []
    for subword in subwords:
        if subword.startswith('Ġ'):
            mapping_idx = mapping_idx + 1
        mapping_list.append(mapping_idx)
    return mapping_list


def scramble(seq, prob=0.15):
    seq = np.array(seq)
    # print(seq)
    num_permute = int(len(seq) * prob)
    full_permutation = np.random.permutation(len(seq))
    inverse_full_permutation = np.argsort(full_permutation)
    partial_permutation = np.random.permutation(num_permute)
    seq = seq[full_permutation]
    # print(seq)
    seq = np.concatenate(
        (seq[:num_permute][partial_permutation], seq[num_permute:]))
    seq = seq[inverse_full_permutation]
    seq = list(seq)
    return seq


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
    """
    print(len(token_ids))
    print(token_ids[0])
    print(tokenizer.decode([0, 5457, 11858, 42292, 20577, 3916, 687, 5457, 2]))
    """
