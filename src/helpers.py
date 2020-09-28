import torch
import torch.nn as nn
import yaml


def get_config(fpath: str) -> dict:
    with open(fpath, "r") as f:
        parsed = yaml.load(f, Loader=yaml.FullLoader)
    f.close()
    return parsed


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
