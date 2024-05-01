# import the teacher model from the eval folder
import sys
sys.path.insert(0, 'eval')
import eval
import metric
import baseline
import triplet_sampler
# from eval import eval, metric, baseline, triplet_sampler

# import eval.eval as bs # import baseline
import torch
import torch.nn as nn
import torch.utils.data as tdata
import torchvision.models as models


# model = torch.hub.load('XingangPan/IBN-Net', 'resnet50_ibn_a', pretrained=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def getLeightWeightModel():
    # download mobilenet and set its weights
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2) # torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
    model = model.to(device)
    print(model.classifier)
    print('Number of parameters: ', sum(p.numel() for p in model.parameters()))
    return model

def getTeacherModel(weight_path, type='MBR_4G'):
    # get model according to the type 
    model = eval.get_model(type, device)
    model.load_state_dict(torch.load(weight_path, map_location='cpu')) 
    model = model.to(device)
    model.eval() # this model will always just produce results, not to be trained
    
if __name__ == '__main__':
    ()    
# set model for training
# model.train()
    


