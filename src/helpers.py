import torch
import torch.nn as nn


def sequential_transforms(*transforms):
    def func(inputs):
        for transform in transforms:
            inputs = transform(inputs)
        return inputs

    return func


def to_tensor(dtype):
    def func(ids_list):
        return torch.tensor(ids_list).to(dtype)

    return func


def vocab_func(vocab):
    def func(tok_iter):
        return [vocab[tok] for tok in tok_iter]

    return func


def ep_time(start_time, end_time):
    elapsed = end_time - start_time
    elapsed_mins = int(elapsed / 60)
    elapsed_secs = int(elapsed - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def calc_acc(predictions, labels):
    top_pred = predictions.argmax(1, keepdim=True)
    correct = top_pred.eq(labels.view_as(top_pred)).sum()
    acc = correct.float() / labels.shape[0]
    return acc


def predict(tokenizer, vocab, model, device, sentence):
    model.eval()

    tokens = tokenizer.tokenize(sentence)
    length = torch.LongTensor([len(tokens)]).to(device)
    idx = [vocab.stoi[token] for token in tokens]
    tensor = torch.LongTensor(idx).unsqueeze(-1).to(device)

    prediction = model(tensor, length)
    probabilities = nn.functional.softmax(prediction, dim=-1)
    return probabilities.squeeze()[-1].item()
