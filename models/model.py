import torch
import torch.nn as nn


class UnimodalModel(nn.Module):
    def __init__(self, encoders: dict, classifier) -> None:
        super(UnimodalModel, self).__init__()
        self.encoders = nn.ModuleDict(encoders)
        self.classifier = classifier

    def forward(self, x):
        only_key = next(iter(self.encoders.keys()))
        encoded = self.encoders[only_key](x[only_key])
        out = self.classifier(encoded)
        return out
