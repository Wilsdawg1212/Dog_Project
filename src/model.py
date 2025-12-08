import torch
import torch.nn as nn
from transformers import ViTForImageClassification
from torchvision.models import resnet50, ResNet50_Weights
from transformers import AutoModelForImageClassification, AutoImageProcessor

DEFAULT_VIT_MODEL_NAME = "google/vit-base-patch16-224-in21k"

def get_normalization(model_type: str, hf_model_name: str = DEFAULT_VIT_MODEL_NAME):
    """
    Returns mean and std for proper image normalization.
    """
    if model_type == "resnet":
        weights = ResNet50_Weights.DEFAULT
        mean = weights.transforms().mean
        std  = weights.transforms().std
        return mean, std

    elif model_type == "vit":
        processor = AutoImageProcessor.from_pretrained(hf_model_name)
        return processor.image_mean, processor.image_std

    else:
        raise ValueError("Unknown model_type. Use 'resnet' or 'vit'.")
    
class DogBreedClassifier:
    def __init__(
        self,
        model_type: str,
        num_classes: int,
        hf_model_name: str = DEFAULT_VIT_MODEL_NAME,
        freeze_base: bool = False,
    ):
        super().__init__()

        self.model_type = model_type
        self.hf_model_name = hf_model_name

        if model_type == "resnet":
            weights = ResNet50_Weights.DEFAULT
            model = resnet50(weights=weights)

            #classifier head
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)

            if freeze_base:
                for name, param in model.named_parameters():
                    if not name.startswith("fc"):
                        param.requires_grad = False

            self.backbone = model
        
        elif model_type == "vit":
            model = AutoModelForImageClassification.from_pretrained(
                hf_model_name,
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
            )

            if freeze_base:
                for name, param in model.named_parameters():
                    if "classifier" not in name:
                        param.requires_grad = False

            self.backbone = model
        
        else:
            raise ValueError("model_type must be 'resnet' or 'vit'.")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.model_type == "resnet":
            logits = self.backbone(x)
            return logits
        
        elif self.model_type == "vit":
            outputs = self.backbone(pixel_values=x)
            logits = outputs.logits
            return logits
        
        else:
            raise ValueError("Invalid model_type in forward().")



def build_model(
    model_type: str,
    num_classes: int,
    hf_model_name: str = DEFAULT_VIT_MODEL_NAME,
    freeze_base: bool = False,
) -> nn.Module:
    """
    Convenience factory used by training scripts / notebooks.
    """
    return DogBreedClassifier(
        model_type=model_type,
        num_classes=num_classes,
        hf_model_name=hf_model_name,
        freeze_base=freeze_base,
    )