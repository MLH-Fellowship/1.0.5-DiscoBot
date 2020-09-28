import torch
import torch.nn as nn
from bentoml import BentoService, api, artifacts, env
from bentoml.adapters import JsonInput, JsonOutput
from bentoml.frameworks.pytorch import PytorchModelArtifact
from bentoml.service.artifacts.common import PickleArtifact

from train import vocab

device = torch.device("cpu")


@env(infer_pip_packages=True)
@artifacts([PytorchModelArtifact("model"), PickleArtifact("tokenizer")])
class ProfanityFilterService(BentoService):
    def model_pred(self, sentence):
        self.artifacts.model.eval()

        tokens = self.artifacts.tokenizer.tokenize(sentence)
        length = torch.LongTensor([len(tokens)]).to(device)
        idx = [vocab.stoi[token] for token in tokens]
        tensor = torch.LongTensor(idx).unsqueeze(-1).to(device)

        prediction = self.artifacts.model(tensor, length)
        probabilities = nn.functional.softmax(prediction, dim=-1)
        return probabilities.squeeze()[-1].item()

    @api(input=JsonInput(), output=JsonOutput())
    def predict(self, parsed_json):
        return self.model_pred(parsed_json["text"])
