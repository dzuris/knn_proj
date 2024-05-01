import torch
import torch.nn as nn
import torchvision.models as models

import baseline as bs

class LWModel(nn.Module):
    """
    this is lightweight module used as student in knowledge distillation
    """
    def __init__(self, n_classes) -> None:
        super(LWModel, self).__init__()
        self.model = models.mobilenet_v2(weights = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2))
        
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_ftrs, n_classes)

        # self.final_layer = nn.Sequential(
        #     nn.BatchNorm1d(512),
        #     nn.Linear(512,n_class)
        # )
        # todo some layer acoording to
        
    def forward(self, x):
        out = self.model(x)
        # fin_out = self.final_layer(x, cam, view)
        return out
