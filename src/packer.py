import torch

from bento_service import ProfanityFilterService
from classifier import BiLSTM
from train import MODEL_PATH, config, device, input_dim, pad_idx, tokenizer

model = BiLSTM(
    input_dim,
    config["emb_dim"],
    config["hid_dim"],
    config["output_dim"],
    config["n_layers"],
    config["dropout"],
    pad_idx,
)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

bento_svc = ProfanityFilterService()
bento_svc.pack("model", model)
bento_svc.pack("tokenizer", tokenizer)

saved_path = bento_svc.save()

print(
    bento_svc.predict(
        {"text": "I love you so much. I hope you can achieve what you love one day"}
    )
)
print("-" * 50)
print(bento_svc.predict({"text": "Fuck off you son of a bit I fucking hate you"}))
print("-" * 50)
print("saved modelpath: %s" % saved_path)
