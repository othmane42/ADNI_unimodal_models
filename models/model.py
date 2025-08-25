import torch
import pytorch_lightning as pl
from typing import List

from dataclasses import dataclass
import torch.nn as nn
import numpy as np

# import our library
# from sklearn.metrics import f1_score, top_k_accuracy_score
from sklearn.metrics import f1_score as sklearn_f1_score
from sklearn.metrics import roc_auc_score
from torchmetrics.functional import f1_score
from torchmetrics.functional.classification import accuracy
# initialize metric
from sklearn.metrics import classification_report

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


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
