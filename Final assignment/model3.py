import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights



class Model(nn.Module):
    """
    DeepLabV3+ model with a ResNet-101 backbone for semantic segmentation.
    This wraps the torchvision model and exposes a unified interface.
    """

    def __init__(self, in_channels=3, n_classes=19):
        super(Model, self).__init__()

        # Load model with pretrained backbone (no pretrained weights on segmentation head)
        # self.model = deeplabv3_resnet101(pretrained=True, num_classes=n_classes)
        self.model = deeplabv3_resnet101(
        weights=DeepLabV3_ResNet101_Weights.DEFAULT,  # loads pretrained ResNet101 backbone
        num_classes=n_classes
        )

    def forward(self, x):
        return self.model(x)["out"]
