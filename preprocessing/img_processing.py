
from torchvision import transforms
from typing import Callable , List
from PIL import Image
from torchvision.transforms import ToPILImage
import clip
import torch
def get_vgg16_preprocess() -> Callable:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

     
    
    def preprocess(images: List[Image.Image]) -> torch.Tensor:
        return torch.stack([transform(image) for image in images])

    return preprocess


def get_resnet50_processing() -> Callable:
    img_size = 224
    mean, std = (
            0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)


    transform_func = transforms.Compose(
                [transforms.Resize(256),
                    transforms.CenterCrop(img_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                    ])
    
    def preprocess(images: List[Image.Image]) -> torch.Tensor:
        return torch.stack([transform_func(image) for image in images])

    return preprocess


def get_clip_preprocessing() -> Callable:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, preprocess = clip.load("ViT-B/32", device=device)  # Load CLIP preprocessing
    to_pil_image = ToPILImage()

    def preprocess_images(images: List[torch.Tensor]) -> torch.Tensor:
        return torch.stack([preprocess(image) for image in images])

    return preprocess_images