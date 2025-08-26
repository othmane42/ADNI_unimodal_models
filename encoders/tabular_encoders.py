import torch
import torch.nn as nn
from .common_encoders import BaseEncoder
from transformers import AutoModel, AutoTokenizer


class TabEncoderMLP(BaseEncoder):
    def __init__(self, **kwargs):
        super(TabEncoderMLP, self).__init__(**kwargs)

        self.features_extractor = nn.Sequential(
            nn.LazyLinear(128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2)
        )

        if self.freeze:
            self.freeze_parameters()

    def forward(self, x):
        # extract features
        features = self.features_extractor(x)

        if self.classifier is not None:
            features = self.classifier(features)

        return features


class ClinicalBERTEncoder(BaseEncoder):
    def __init__(self, checkpoint: str, **kwargs):
        super(ClinicalBERTEncoder, self).__init__(**kwargs)
        self.features_extractor = AutoModel.from_pretrained(checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint, padding=True, truncation=True, max_length=256)

        self.freeze_parameters()

    def forward(self, x):
        inputs = self.tokenizer(x, return_tensors="pt",
                                padding=True, truncation=True)
        # Move tokenized inputs to GPU
        inputs = {k: v.to(self.features_extractor.device)
                  for k, v in inputs.items()}
        outputs = self.features_extractor(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]
        # embeddings = embeddings
        if self.classifier:
            self.classifier = self.classifier
            embeddings = self.classifier(embeddings)
        return embeddings
