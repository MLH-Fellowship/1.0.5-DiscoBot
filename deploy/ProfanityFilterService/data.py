import collections
import os
import random

import torch
import torch.nn as nn
import torchtext
import torchtext.experimental
import torchtext.experimental.vectors
from torchtext.experimental.datasets.text_classification import \
    TextClassificationDataset

from helpers import sequential_transforms, to_tensor, vocab_func


def get_train_valid_split(raw_train, split_ratio=0.7):
    raw_train = list(raw_train)
    random.shuffle(raw_train)

    n_train_ex = int(len(raw_train) * split_ratio)
    train_data = raw_train[:n_train_ex]
    valid_data = raw_train[n_train_ex:]
    return train_data, valid_data


def gen_vocab(raw_data, tokenizer, **vocab_kwargs):
    token_freqs = collections.Counter()

    for label, text in raw_data:
        tokens = tokenizer.tokenize(text)
        token_freqs.update(tokens)

    vocab = torchtext.vocab.Vocab(token_freqs, **vocab_kwargs)

    return vocab


def process_raw(raw_data, tokenizer, vocab):
    raw_data = [(label, text) for (label, text) in raw_data]
    text_trans = sequential_transforms(
        tokenizer.tokenize, vocab_func(vocab), to_tensor(dtype=torch.long)
    )
    label_trans = sequential_transforms(to_tensor(dtype=torch.long))

    transforms = (label_trans, text_trans)

    return TextClassificationDataset(raw_data, vocab, transforms)


class Tokenizer:
    def __init__(self, fn="basic_english", lower=True, max_len=None):
        self.tokenize_fn = torchtext.data.utils.get_tokenizer(fn)
        self.lower = lower
        self.max_len = max_len

    def tokenize(self, s):
        tokens = self.tokenize_fn(s)

        if self.lower:
            tokens = [token.lower() for token in tokens]

        if self.max_len is not None:
            tokens = tokens[: self.max_len]

        return tokens


class Collator:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def collate(self, batch):
        labels, text = zip(*batch)
        labels, lengths = torch.LongTensor(labels), torch.LongTensor(
            [len(x) for x in text]
        )

        text = nn.utils.rnn.pad_sequence(text, padding_value=self.pad_idx)
        return labels, text, lengths


class Dataset:
    def __init__(self, max_len, max_size, batch_size, pad_token):
        self.MAX_LEN = max_len
        self.MAX_SIZE = max_size
        self.BATCH_SIZE = batch_size

        raw_train, raw_test = torchtext.experimental.datasets.raw.IMDB(root=os.path.join(os.path.abspath(os.path.dirname(__file__)), '.data'))
        raw_train, raw_valid = get_train_valid_split(raw_train)

        self.tokenizer = Tokenizer(max_len=max_len)
        self.vocab = gen_vocab(raw_train, self.tokenizer, max_size=max_size)
        self.collator = Collator(self.vocab[pad_token])

        self.train_data = process_raw(raw_train, self.tokenizer, self.vocab)
        self.test_data = process_raw(raw_test, self.tokenizer, self.vocab)
        self.valid_data = process_raw(raw_valid, self.tokenizer, self.vocab)

    def get_vocab(self):
        return self.vocab

    def get_tokenizer(self):
        return self.tokenizer

    def get_iterator(self):
        train_iterator = torch.utils.data.DataLoader(
            self.train_data,
            self.BATCH_SIZE,
            shuffle=True,
            collate_fn=self.collator.collate,
        )
        valid_iterator = torch.utils.data.DataLoader(
            self.valid_data,
            self.BATCH_SIZE,
            shuffle=False,
            collate_fn=self.collator.collate,
        )
        test_iterator = torch.utils.data.DataLoader(
            self.test_data,
            self.BATCH_SIZE,
            shuffle=False,
            collate_fn=self.collator.collate,
        )
        return train_iterator, test_iterator, valid_iterator
