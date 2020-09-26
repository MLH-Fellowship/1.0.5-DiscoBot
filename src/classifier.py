import collections
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
import torchtext.experimental
import torchtext.experimental.vectors
from torchtext.experimental.datasets.raw.text_classification import \
    RawTextIterableDataset
from torchtext.experimental.datasets.text_classification import \
    TextClassificationDataset

from helpers import *


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


def init_params(m: nn.Module):
    if isinstance(m, nn.Embedding):
        nn.init.uniform_(m.weight, -0.05, 0.05)
    elif isinstance(m, nn.LSTM):
        for n, p in m.named_parameters():
            if "weight_ih" in n:
                i, f, g, o = p.chunk(4)
                nn.init.xavier_uniform_(i)
                nn.init.xavier_uniform_(f)
                nn.init.xavier_uniform_(g)
                nn.init.xavier_uniform_(o)
            elif "weight_hh" in n:
                i, f, g, o = p.chunk(4)
                nn.init.orthogonal_(i)
                nn.init.orthogonal_(f)
                nn.init.orthogonal_(g)
                nn.init.orthogonal_(o)
            elif "bias" in n:
                i, f, g, o = p.chunk(4)
                nn.init.zeros_(i)
                nn.init.ones_(f)
                nn.init.zeros_(g)
                nn.init.zeros_(o)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


def get_pretrained_embedding(init_embed, pretrained_vectors, vocab, unk_token):
    pretrained_embedding = torch.FloatTensor(init_embed.weight.clone()).detach()
    pretrained_vocab = pretrained_vectors.vectors.get_stoi()

    unk_tokens = []

    for idx, token in enumerate(vocab.itos):
        if token in pretrained_vocab:
            pretrained_vector = pretrained_vectors[token]
            pretrained_embedding[idx] = pretrained_vector
        else:
            unk_tokens.append(token)

    return pretrained_embedding, unk_tokens


def train(model, iterator, optimizer, criterion, device):
    ep_loss, ep_acc = 0, 0

    model.train()

    for labels, text, lengths in iterator:
        labels, text = labels.to(device), text.to(device)

        optimizer.zero_grad()

        predictions = model(text, lengths)

        loss = criterion(predictions, labels)

        acc = calc_acc(predictions, labels)

        loss.backward()
        optimizer.step()

        ep_loss += loss.item()
        ep_acc += acc.item()

    return ep_loss / len(iterator), ep_acc / len(iterator)


def evaluate(model, iterator, criterion, device):
    ep_loss, ep_acc = 0, 0

    model.eval()

    with torch.no_grad():
        for labels, text, lengths in iterator:
            labels, text = labels.to(device), text.to(device)

            predictions = model(text, lengths)

            loss = criterion(predictions, labels)

            acc = calc_acc(predictions, labels)

            ep_loss += loss.item()
            ep_acc += acc.item()

    return ep_loss / len(iterator), ep_acc / len(iterator)


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


class BiLSTM(nn.Module):
    def __init__(
        self, input_dim, emb_dim, hid_dim, output_dim, n_layer, dropout, pad_idx
    ):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            emb_dim, hid_dim, num_layers=n_layer, bidirectional=True, dropout=dropout
        )
        self.fc = nn.Linear(2 * hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, lengths):
        # [seq_len, batch_size, emb_dim]
        embedded = self.dropout(self.embedding(text))
        # https://discuss.pytorch.org/t/simple-working-example-how-to-use-packing-for-variable-length-sequence-inputs-for-rnn/2120
        packed_emb = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths, enforce_sorted=False
        )
        packed_out, (hidden, cell) = self.lstm(packed_emb)

        # outputs : [seq_len, batch_size, n_direction * hid_dim]
        # hid : [n_layers * n_direction, batch_size, hid_dim]
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out)

        # [batch_size, hid_dim]
        hidden_fwd, hidden_bck = hidden[-2], hidden[-1]
        # [batch_size, hid_dim*2]
        hidden = torch.cat((hidden_fwd, hidden_bck), dim=1)
        # pred : [batch_size, output_dim]
        return self.fc(self.dropout(hidden))
