import os
import random
from tqdm import tqdm
from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--extracted_data_dir',
        type=str,
        default='../data/extracted/wikicorpus_en/AA'
    )
    parser.add_argument(
        '--prepared_data_dir',
        type=str,
        default='../data/wikidump'
    )

    return parser.parse_args()


def split_data(dataset, val_ratio=0.1, random_seed=22):
    random.seed(random_seed)
    random.shuffle(dataset)
    num_val = int(len(dataset) * val_ratio)
    return dataset[num_val:], dataset[:num_val]


if __name__ == "__main__":
    args = get_args()
    file_paths = map(
        lambda filename: os.path.join(
            args.extracted_data_dir, filename
        ),
        os.listdir(args.extracted_data_dir)
    )

    # load all lines from all filepaths
    text = []
    [
        text.extend(
            open(file_path, 'r', encoding='utf-8')
                .readlines()
        )
        for file_path in file_paths
    ]

    # clean
    clean_text = []
    line_idx = 0
    while line_idx < len(text):
        line = text[line_idx]
        if line.startswith('<doc id='):
            line_idx = line_idx + 2
        elif line.startswith('</doc>'):
            line_idx = line_idx + 1
        elif len(line) == 0 or line.isspace():
            line_idx = line_idx + 1
        else:
            clean_text.append(line)
            line_idx = line_idx + 1

    # print number of lines before and after cleaning
    print('number of lines before cleaning:', len(text))
    print('number of lines after cleaning', len(clean_text))

    # split data into train and val
    train_text, val_text = split_data(clean_text)

    os.makedirs(args.prepared_data_dir, exist_ok=False)
    for split, data in zip(
            ['train', 'val'],
            [train_text, val_text]
    ):
        with open(
                f'{args.prepared_data_dir}/{split}.txt',
                'w',
                encoding='utf-8'
        ) as f_out:
            for line in tqdm(
                    data,
                    desc=f'writing {split}.txt',
                    unit=' lines'
            ):
                f_out.write(line)
