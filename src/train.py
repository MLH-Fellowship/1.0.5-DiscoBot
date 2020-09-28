import os
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
import torchtext.experimental
import torchtext.experimental.vectors

from classifier import BiLSTM
from data import Dataset
from helpers import (calc_acc, ep_time, get_config, get_pretrained_embedding,
                     init_params)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = get_config(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), "config.yml")
)
torch.manual_seed(config["seed"])
random.seed(config["seed"])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

MODEL_PATH = os.path.join(os.getcwd(), "models", config["model_name"] + ".pt")


dataloader = Dataset(
    config["max_len"], config["max_size"], config["batch_size"], config["pad_token"]
)
train_iterator, test_iterator, valid_iterator = dataloader.get_iterator()
vocab = dataloader.get_vocab()
tokenizer = dataloader.get_tokenizer()

pad_idx = vocab[config["pad_token"]]
input_dim = len(vocab)


def train(model, iterator, optimizer, criterion):
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


def evaluate(model, iterator, criterion):
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


class Model:
    def __init__(self):
        model = BiLSTM(
            input_dim,
            config["emb_dim"],
            config["hid_dim"],
            config["output_dim"],
            config["n_layers"],
            config["dropout"],
            pad_idx,
        )

        glove = torchtext.experimental.vectors.GloVe(name="6B", dim=config["emb_dim"])
        # for n,p in model.named_parameters():
        #     print(f'name:{n}\nshape:{p.shape}\n')
        model.apply(init_params)

        pretrained_embedding, _ = get_pretrained_embedding(
            model.embedding, glove, vocab, config["unk_token"]
        )

        model.embedding.weight.data.copy_(pretrained_embedding)
        model.embedding.weight.data[pad_idx] = torch.zeros(config["emb_dim"])

        self.optimizer = optim.Adam(model.parameters())

        criterion = nn.CrossEntropyLoss()

        self.model = model.to(device)
        self.criterion = criterion.to(device)

    def train_loop(self, train_iterator, valid_iterator):
        best_valid_loss = float("inf")
        print("Start training...")
        for epoch in range(config["n_epochs"]):

            start_time = time.monotonic()

            train_loss, train_acc = train(
                self.model, train_iterator, self.optimizer, self.criterion
            )
            valid_loss, valid_acc = evaluate(self.model, valid_iterator, self.criterion)

            end_time = time.monotonic()

            epoch_mins, epoch_secs = ep_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self.model.state_dict(), MODEL_PATH)

            print(f"Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s")
            print(f"\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%")
            print(f"\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%")


if __name__ == "__main__":
    m = Model()
    m.train_loop(train_iterator, valid_iterator)
