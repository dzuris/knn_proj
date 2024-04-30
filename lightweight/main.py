# import eval.eval as bs # import baseline

import torch
import torch.nn as nn
import torch.utils.data as tdata

# model = torch.hub.load('XingangPan/IBN-Net', 'resnet50_ibn_a', pretrained=True)
model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
print(model.classifier)
# model.eval()
# print(sum(p.numel() for p in model.parameters()))
