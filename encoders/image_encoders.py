import torch

import torch.nn as nn
import torchvision.models as models
from transformers import ViTModel
import clip
from .common_encoders import BaseEncoder
import logging
from monai.networks.nets import resnet10, resnet18, resnet34, resnet50, DenseNet121, get_pretrained_resnet_medicalnet
import os
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


depth_map = {10: resnet10,
             18: resnet18,
             34: resnet34,
             50: resnet50}


class Resnet3D(BaseEncoder):
    def __init__(self, depth, **kwargs):
        super(Resnet3D, self).__init__(**kwargs)
        net = depth_map[depth](
            pretrained=False,
            spatial_dims=3,
            n_input_channels=1,
            num_classes=1
        )
        pretrained_state_dict = get_pretrained_resnet_medicalnet(
            resnet_depth=depth)
        net_dict = net.state_dict()
        pretrained_state_dict = {
            k.replace("module.", ""): v for k, v in pretrained_state_dict.items()}
        missing = tuple(
            {k for k in net_dict if k not in pretrained_state_dict})
        inside = tuple({k for k in pretrained_state_dict if k in net_dict})
        logging.debug(f"inside pretrained: {len(inside)}")
        unused = tuple({k for k in pretrained_state_dict if k not in net_dict})
        logging.debug(f"unused pretrained: {len(unused)}")
        assert len(inside) > len(missing)
        assert len(inside) > len(unused)
        pretrained_state_dict = {
            k: v for k, v in pretrained_state_dict.items() if k in net_dict}
        net.load_state_dict(pretrained_state_dict, strict=False)

        self.features_extractor = nn.Sequential(*list(net.children())[:-1])
        self.freeze_parameters()

    def forward(self, x):
        x = self.features_extractor(x).squeeze()
        if self.classifier:
            x = self.classifier(x)
        if len(x.shape) == 1:
            x = torch.unsqueeze(x, 0)
        return x


class DenseNet3D(BaseEncoder):
    def __init__(self, **kwargs):
        super(DenseNet3D, self).__init__(**kwargs)
        net = DenseNet121(spatial_dims=3, in_channels=1, out_channels=2)
        self.features_extractor = nn.Sequential(
            *list(net.children())[:-1],  # All layers except class_layers
            # Add just the flatten layer
            *list(net.class_layers.children())[:-1]
        )
        self.freeze_parameters()

    def forward(self, x):
        x = self.features_extractor(x).squeeze()
        if self.classifier:
            x = self.classifier(x)
        if len(x.shape) == 1:
            x = torch.unsqueeze(x, 0)
        return x


class VIT(BaseEncoder):
    def __init__(self, checkpoint: str = 'google/vit-base-patch16-224-in21k', **kwargs):
        super(VIT, self).__init__(**kwargs)
        self.features_extractor = ViTModel.from_pretrained(
            checkpoint,
        )
        self.freeze_parameters()

    def forward(self, x):
        x = self.features_extractor(
            **x, return_dict=True).last_hidden_state[:, 0, :]
        if self.classifier:
            x = self.classifier(x)
        return x


class VGG16(BaseEncoder):
    def __init__(self, **kwargs) -> None:
        super(VGG16, self).__init__(**kwargs)
        self.features_extractor = models.vgg16(pretrained=True)
        self.features_extractor.classifier = self.features_extractor.classifier[:-1]
        self.freeze_parameters()

    def forward(self, x):
        return super(VGG16, self).forward(x)


class InceptionV3(BaseEncoder):
    def __init__(self, output_layer="avgpool", **kwargs):
        super(InceptionV3, self).__init__(**kwargs)
        pretrained_inceptionv3 = models.inception_v3(
            pretrained=True, aux_logits=False)

        # Build feature extractor up to the specified output layer
        self.children_list = []
        for n, c in pretrained_inceptionv3.named_children():
            self.children_list.append(c)
            if n == output_layer:
                break

        self.features_extractor = nn.Sequential(*self.children_list)
        # Example fully connected layer (optional)
        self.fc = nn.LazyLinear(128)
        self.freeze_parameters()

    def forward(self, x):
        x = self.features_extractor(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        if self.classifier:
            x = self.classifier(x)
        return x


class CLIPImageEncoder(BaseEncoder):
    CHECKPOINT = "ViT-B/32"

    def __init__(self, checkpoint: str = CHECKPOINT, **kwargs):
        super(CLIPImageEncoder, self).__init__(**kwargs)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.features_extractor, self.preprocess = clip.load(
            checkpoint, device=self.device)
        self.freeze_parameters()

    def forward(self, x):
        # if isinstance(x, list)  or len(x.shape)>3:
        #     images = [self.preprocess(to_pil_image(image.squeeze())).unsqueeze(0).to(self.device) for image in x]
        #     images = torch.cat(images, 0)
        # else:
        #     images = self.preprocess(to_pil_image(x.squeeze())).unsqueeze(0).to(self.device)

        with torch.no_grad():
            x = self.features_extractor.encode_image(x.to(self.device))

        if self.classifier:
            x = self.classifier(x)
        return x


class ResNet50(BaseEncoder):
    def __init__(self, output_layer="avgpool", **kwargs):
        super(ResNet50, self).__init__(**kwargs)
        pretrained_resnet = models.resnet50(pretrained=True)

        # Build feature extractor up to the specified output layer
        self.children_list = []
        for n, c in pretrained_resnet.named_children():
            self.children_list.append(c)
            if n == output_layer:
                break

        self.features_extractor = nn.Sequential(*self.children_list)
        self.freeze_parameters()

    def forward(self, x):
        x = self.features_extractor(x)
        x = torch.flatten(x, 1)
        if self.classifier:
            x = self.classifier(x)
        return x
