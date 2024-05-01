import torch
import torch.nn as nn
import torchvision.models as models


class LWModel(nn.Module):
    """
    this is lightweight module used as student in knowledge distillation
    """
    def __init__(self) -> None:
        super(LWModel, self).__init__()
        self.model_base = models.mobilenet_v2(model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2))
        # todo some layer acoording to
        
    def forward(self, x):
        out = self.model_base(x)
        return out