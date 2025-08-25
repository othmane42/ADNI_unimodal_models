
import torch.nn as nn



class BaseEncoder(nn.Module):
    def __init__(self,freeze=True, include_head=False, num_classes=None):
        super(BaseEncoder, self).__init__()
        self.include_head = include_head
        self.features_extractor = None  # This should be defined in subclasses
        self.freeze=freeze
        if include_head and num_classes:
            print("defining the classifier")
            self.classifier = nn.LazyLinear(num_classes)
        else:
            self.classifier = None

    def freeze_parameters(self):
        if self.freeze and self.features_extractor is not None:
            print("gonna freeze params")
            for param in self.features_extractor.parameters():
               
                param.requires_grad = False
    def forward(self, x):
        x = self.features_extractor(x)
        if self.classifier:
            x = self.classifier(x)
        return x