DEVICE_NOTCPU = "mps" #"cpu"

import math
import warnings
import torch
import torch.nn as nn
import torchvision
from torchvision import models
import torch.nn.functional as F
from torch import Tensor
import numpy as np
import copy
from tqdm import tqdm
from multiprocessing import Pool
import time
import random
from collections import defaultdict
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import os
from torchvision import transforms
from torch.utils.data import DataLoader
from typing import OrderedDict
import torch.multiprocessing
import yaml

class IBN(nn.Module):
    r"""Instance-Batch Normalization layer from
    `"Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net"
    <https://arxiv.org/pdf/1807.09441.pdf>`

    Args:
        planes (int): Number of channels for the input tensor
        ratio (float): Ratio of instance normalization in the IBN layer
    """
    def __init__(self, planes, ratio=0.5):
        super(IBN, self).__init__()
        self.half = int(planes * ratio)
        self.IN = nn.InstanceNorm2d(self.half, affine=True)
        self.BN = nn.BatchNorm2d(planes - self.half)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out

class Bottleneck_IBN(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, ibn=None, stride=1, downsample=None):
        super(Bottleneck_IBN, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        if ibn == 'a':
            self.bn1 = IBN(planes)
        else:
            self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.IN = nn.InstanceNorm2d(planes * 4, affine=True) if ibn == 'b' else None
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.IN is not None:
            out = self.IN(out)
        out = self.relu(out)

        return out

class ResNet_IBN(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 ibn_cfg=('a', 'a', 'a', None),
                 num_classes=1000):
        self.inplanes = 64
        super(ResNet_IBN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        if ibn_cfg[0] == 'b':
            self.bn1 = nn.InstanceNorm2d(64, affine=True)
        else:
            self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], ibn=ibn_cfg[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, ibn=ibn_cfg[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, ibn=ibn_cfg[2])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, ibn=ibn_cfg[3])
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, ibn=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes,
                            None if ibn == 'b' else ibn,
                            stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                None if (ibn == 'b' and i < blocks-1) else ibn))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class MBR_model(nn.Module):         
    def __init__(self, class_num, n_branches, n_groups, losses="LBS", backbone="ibn", droprate=0, linear_num=False, return_f = True, circle_softmax=False, pretrain_ongroups=True, end_bot_g=False, group_conv_mhsa=False, group_conv_mhsa_2=False, x2g=False, x4g=False, LAI=False, n_cams=0, n_views=0):
        super(MBR_model, self).__init__()  

        self.modelup2L3 = base_branches(backbone=backbone)
        self.modelL4 = multi_branches(n_branches=n_branches, n_groups=n_groups, pretrain_ongroups=pretrain_ongroups, end_bot_g=end_bot_g, group_conv_mhsa=group_conv_mhsa, group_conv_mhsa_2=group_conv_mhsa_2, x2g=x2g, x4g=x4g)
        self.finalblock = FinalLayer(class_num=class_num, n_branches=n_branches, n_groups=n_groups, losses=losses, droprate=droprate, linear_num=linear_num, return_f=return_f, circle_softmax=circle_softmax, LAI=LAI, n_cams=n_cams, n_views=n_views, x2g=x2g, x4g=x4g)
        

    def forward(self, x,cam, view):
        mix = self.modelup2L3(x)
        output = self.modelL4(mix)
        preds, embs, ffs = self.finalblock(output, cam, view)

        return preds, embs, ffs, output

class base_branches(nn.Module):
    def __init__(self, backbone="ibn", stride=1):
        super(base_branches, self).__init__()
        if backbone == 'r50':
            model_ft = models.resnet50()
        elif backbone == '101ibn':
            model_ft = torch.hub.load('XingangPan/IBN-Net', 'resnet101_ibn_a', pretrained=True)# 'resnet50_ibn_a'
        elif backbone == '34ibn':
            model_ft = torch.hub.load('XingangPan/IBN-Net', 'resnet34_ibn_a', pretrained=True)# 'resnet50_ibn_a'
        else:
            if torch.cuda.is_available():
                model_ft = torch.hub.load('XingangPan/IBN-Net', 'resnet50_ibn_a', pretrained=True)
            else:    
                _model = ResNet_IBN(block=Bottleneck_IBN,
                        layers=[3, 4, 6, 3],
                        ibn_cfg=('a', 'a', 'a', None))
                _model.load_state_dict(torch.hub.load_state_dict_from_url('https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet50_ibn_a-d9d0bb7b.pth', map_location='cpu'))
                model_ft = _model
            
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            if backbone == "34ibn":
                model_ft.layer4[0].conv1.stride = (1,1)
            else:
                model_ft.layer4[0].conv2.stride = (1,1)

        self.model = torch.nn.Sequential(*(list(model_ft.children())[:-3])) 

    def forward(self, x):
        x = self.model(x)
        return x
    
class multi_branches(nn.Module):
    def __init__(self, n_branches, n_groups, pretrain_ongroups=True, end_bot_g=False, group_conv_mhsa=False, group_conv_mhsa_2=False, x2g = False, x4g=False):
        super(multi_branches, self).__init__()

        if torch.cuda.is_available():
            model_ft = torch.hub.load('XingangPan/IBN-Net', 'resnet50_ibn_a', pretrained=True)
        else:
            _model = ResNet_IBN(block=Bottleneck_IBN,
                        layers=[3, 4, 6, 3],
                        ibn_cfg=('a', 'a', 'a', None))
            _model.load_state_dict(torch.hub.load_state_dict_from_url('https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet50_ibn_a-d9d0bb7b.pth', map_location='cpu'))
            model_ft = _model

        model_ft= model_ft.layer4
        self.x2g = x2g
        self.x4g = x4g
        if n_groups > 0:
            convlist = [k.split('.') for k, m in model_ft.named_modules(remove_duplicate=False) if isinstance(m, nn.Conv2d)]
            for item in convlist:
                if item[1] == "downsample":
                    m = model_ft[int(item[0])].get_submodule(item[1])[0]
                else:
                    m = model_ft[int(item[0])].get_submodule(item[1]) #'.'.join(
                weight = m.weight[:int(m.weight.size(0)), :int(m.weight.size(1)/n_groups), :,:]
                if item[1] == "downsample":
                    getattr(model_ft[int(item[0])], item[1])[0] = nn.Conv2d(int(m.weight.size(1)), int(m.weight.size(0)), kernel_size=1, stride=1, groups=n_groups, bias=False).apply(weights_init_kaiming)
                    if pretrain_ongroups:
                        getattr(model_ft[int(item[0])], item[1])[0].weight.data = weight
                elif item[1] == "conv2":
                    setattr(model_ft[int(item[0])], item[1], nn.Conv2d(int(m.weight.size(1)), int(m.weight.size(0)), kernel_size=3, stride=1, padding=(1,1), groups=n_groups, bias=False).apply(weights_init_kaiming))
                    if pretrain_ongroups:
                        setattr(model_ft[int(item[0])].get_submodule(item[1]).weight, "data", weight)                        
                else:
                    setattr(model_ft[int(item[0])], item[1], nn.Conv2d(int(m.weight.size(1)), int(m.weight.size(0)), kernel_size=1, stride=1, groups=n_groups, bias=False).apply(weights_init_kaiming))
                    if pretrain_ongroups:
                        setattr(model_ft[int(item[0])].get_submodule(item[1]).weight, "data", weight)
        self.model = nn.ModuleList()

        if len(n_branches) > 0:
            if n_branches[0] == "2x":
                self.model.append(model_ft)
                self.model.append(copy.deepcopy(model_ft))
            else:
                for item in n_branches:
                    if item =="R50":
                        self.model.append(copy.deepcopy(model_ft))
                    elif item == "BoT":
                        layer_0 = Bottleneck_Transformer(1024, 512, resolution=[16, 16], use_mlp = False)
                        layer_1 = Bottleneck_Transformer(2048, 512, resolution=[16, 16], use_mlp = False)
                        layer_2 = Bottleneck_Transformer(2048, 512, resolution=[16, 16], use_mlp = False)
                        self.model.append(nn.Sequential(layer_0, layer_1, layer_2))
                    else:
                        print("No valid architecture selected for branching by expansion!")
        else:
            self.model.append(model_ft)


    def forward(self, x):
        output = []
        for cnt, branch in enumerate(self.model):
            if self.x2g and cnt>0:
                aux = torch.cat((x[:,int(x.shape[1]/2):,:,:], x[:,:int(x.shape[1]/2),:,:]), dim=1)
                output.append(branch(aux))
            elif self.x4g and cnt>0:
                aux = torch.cat((x[:,int(x.shape[1]/4):int(x.shape[1]/4*2),:,:], x[:, :int(x.shape[1]/4),:,:], x[:, int(x.shape[1]/4*3):,:,:], x[:, int(x.shape[1]/4*2):int(x.shape[1]/4*3),:,:]), dim=1)
                output.append(branch(aux))
            else:
                output.append(branch(x))
       
        return output

class FinalLayer(nn.Module):
    def __init__(self, class_num, n_branches, n_groups, losses="LBS", droprate=0, linear_num=False, return_f = True, circle_softmax=False, n_cams=0, n_views=0, LAI=False, x2g=False,x4g=False):
        super(FinalLayer, self).__init__()    
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.finalblocks = nn.ModuleList()
        self.withLAI = LAI
        if n_groups > 0:
            self.n_groups = n_groups
            for i in range(n_groups*(len(n_branches)+1)):
                if losses == "LBS":
                    if i%2==0:
                        self.finalblocks.append(ClassBlock(int(2048/n_groups), class_num, droprate, linear=linear_num, return_f = return_f, circle=circle_softmax))
                    else:
                        bn= nn.BatchNorm1d(int(2048/n_groups))
                        bn.bias.requires_grad_(False)  
                        bn.apply(weights_init_kaiming)
                        self.finalblocks.append(bn)
                else:
                    self.finalblocks.append(ClassBlock(int(2048/n_groups), class_num, droprate, linear=linear_num, return_f = return_f, circle=circle_softmax))
        else:
            self.n_groups = 1
            for i in range(len(n_branches)):
                if losses == "LBS":
                    if i%2==0:
                        self.finalblocks.append(ClassBlock(2048, class_num, droprate, linear=linear_num, return_f = return_f, circle=circle_softmax))
                    else:
                        bn= nn.BatchNorm1d(int(2048))
                        bn.bias.requires_grad_(False)  
                        bn.apply(weights_init_kaiming)
                        self.finalblocks.append(bn)
                else:
                    self.finalblocks.append(ClassBlock(2048, class_num, droprate, linear=linear_num, return_f = return_f, circle=circle_softmax))

        if losses == "LBS":
            self.LBS = True
        else:
            self.LBS = False

    def forward(self, x, cam, view):
        # if len(x) != len(self.finalblocks):
        #     print("Something is wrong")
        embs = []
        ffs = []
        preds = []
        for i in range(len(x)):
            emb = self.avg_pool(x[i]).squeeze(dim=-1).squeeze(dim=-1)
            for j in range(self.n_groups):
                aux_emb = emb[:,int(2048/self.n_groups*j):int(2048/self.n_groups*(j+1))]
                if self.LBS:
                    if (i+j)%2==0:
                        pred, ff = self.finalblocks[i+j](aux_emb)
                        ffs.append(ff)
                        preds.append(pred)
                    else:
                        ff = self.finalblocks[i+j](aux_emb)
                        embs.append(aux_emb)
                        ffs.append(ff)
                else:
                    aux_emb = emb[:,int(2048/self.n_groups*j):int(2048/self.n_groups*(j+1))]
                    pred, ff = self.finalblocks[i+j](aux_emb)
                    embs.append(aux_emb)
                    ffs.append(ff)
                    preds.append(pred)
                    
        return preds, embs, ffs

# Defines the new fc layer and classification layer
# |--MLP--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate=0.0, relu=False, bnorm=True, linear=False, return_f = True, circle=False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        self.circle = circle
        add_block = []
        if linear: ####MLP to reduce
            final_dim = linear
            add_block += [nn.Linear(input_dim, input_dim), nn.ReLU(), nn.Linear(input_dim, final_dim)]
        else:
            final_dim = input_dim
        if bnorm:
            tmp_block = nn.BatchNorm1d(final_dim)
            tmp_block.bias.requires_grad_(False) 
            add_block += [tmp_block]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate>0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(final_dim, class_num, bias=False)] # 
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier
    def forward(self, x):
        if x.dim()==4:
            x = x.squeeze().squeeze()
        if x.dim()==1:
            x = x.unsqueeze(0)
        x = self.add_block(x)
        if self.return_f:
            f = x
            if self.circle:
                x = F.normalize(x)
                self.classifier[0].weight.data = F.normalize(self.classifier[0].weight, dim=1)
                x = self.classifier(x)
                return x, f
            else:
                x = self.classifier(x)
            return x, f
        else:
            x = self.classifier(x)
            return x

class Bottleneck_Transformer(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, heads=4, resolution=None, use_mlp = False):
        super(Bottleneck_Transformer, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.ModuleList()
        self.conv2.append(MHSA(planes, width=int(resolution[0]), height=int(resolution[1]), heads=heads))
        if stride == 2:
            self.conv2.append(nn.AvgPool2d(2, 2))
        self.conv2 = nn.Sequential(*self.conv2)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(self.expansion*planes)
            )
        self.use_MLP = use_mlp
        if use_mlp:
            self.LayerNorm = torch.nn.InstanceNorm2d(in_planes)
            self.MLP_torch = torchvision.ops.MLP(in_planes, [512, 2048])

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        if self.use_MLP:
            residual = out
            out = self.LayerNorm(out)
            out = out.permute(0,3,2,1)
            out = self.MLP_torch(out)
            out = out.permute(0,3,2,1)
            out = out + residual
            # out = F.relu(out)
        return out
    
class MHSA(nn.Module):
    def __init__(self, n_dims, width=16, height=16, heads=4):
        super(MHSA, self).__init__()
        self.heads = heads
        ### , bias = False in conv2d
        self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1, bias = True)
        self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1, bias = True)
        self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1, bias = True)

        self.rel_h = nn.Parameter(torch.randn([1, heads, n_dims // heads, 1, height]), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn([1, heads, n_dims // heads, width, 1]), requires_grad=True)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        n_batch, C, width, height = x.size() # C // self.heads,
        q = self.query(x).view(n_batch, self.heads, C // self.heads, -1)
        k = self.key(x).view(n_batch, self.heads, C // self.heads, -1)
        v = self.value(x).view(n_batch, self.heads, C // self.heads, -1)

        content_content = torch.matmul(q.permute(0, 1, 3, 2), k)

        content_position = (self.rel_h + self.rel_w).view(1, self.heads, torch.div(C, self.heads, rounding_mode='floor'), -1).permute(0, 1, 3, 2)
        content_position = torch.matmul(content_position, q)

        energy = content_content + content_position
        attention = self.softmax(energy)

        out = torch.matmul(v, attention.permute(0, 1, 3, 2))
        out = out.view(n_batch, C, width, height)

        return out

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50, remove_junk=True):
    """Evaluation with veri776 metric
    Key: for each query identity, its gallery images from the same camera view are discarded.

    :param np.ndarray distmat:
    :param np.ndarray q_pids:
    :param np.ndarray g_pids:
    :param np.ndarray q_camids:
    :param np.ndarray g_camids:
    :param int max_rank:
    :param bool remove_junk:
    :return:
    """
    # compute cmc curve for each query
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in tqdm(range(num_q), desc='Computing CMC and mAP', bar_format='{l_bar}{bar:20}{r_bar}'):
        # get query pid and camid
        q_pid = q_pids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = np.argsort(distmat[q_idx])
        if remove_junk:
            q_camid = q_camids[q_idx]
            remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        else:
            remove = np.zeros_like(g_pids).astype(np.bool)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
    #     orig_cmc = matches[q_idx][keep]
        orig_cmc = (g_pids[order] == q_pid).astype(np.int32)[keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP

def train_collate_fn(batch):
    imgs, pids, camids, viewids = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)

    return torch.stack(imgs, dim=0), pids, camids, viewids 

        
class CustomDataSet4VERIWILD(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None, with_view=True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_info = pd.read_csv(csv_file, sep=' ', header=None)
        self.with_view = with_view
        self.root_dir = root_dir
        self.transform = transform

    def get_class(self, idx):
        return self.data_info.iloc[idx, 1]    

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.data_info.iloc[idx, 0])
        image = torchvision.io.read_image(img_name)

        vid = self.data_info.iloc[idx, 1]
        camid = self.data_info.iloc[idx, 2]
        
        view_id = 0 #self.data_info.iloc[idx, 3]

        if self.transform:
            img = self.transform((image.type(torch.FloatTensor))/255.0)
        if self.with_view :
            return img, vid, camid, view_id
        else:
            return img, vid, camid, 0


class CustomDataSet4VERIWILDv2(Dataset):
    """VeriWild 2.0 dataset."""

    def __init__(self, csv_file, root_dir, transform=None, with_view=True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_info = pd.read_csv(csv_file, sep=' ', header=None)
        self.with_view = with_view
        self.root_dir = root_dir
        self.transform = transform

    def get_class(self, idx):
        return self.data_info.iloc[idx, 1]    

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.data_info.iloc[idx, 0])
        image = torchvision.io.read_image(img_name)

        vid = self.data_info.iloc[idx, 1]
        camid = 0 #self.data_info.iloc[idx, 2]
        view_id = 0 # = self.data_info.iloc[idx, 3]

        if self.transform:
            img = self.transform((image.type(torch.FloatTensor))/255.0)
        if self.with_view:
            return img, vid, camid, view_id
        else:
            return img, vid, camid

class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        for index in range(len(self.data_source.data_info)):
            pid = self.data_source.get_class(index)
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length
        
class CustomDataSet4Market1501(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, image_list, root_dir, is_train=True, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        #self.data_info = pd.read_xml(csv_file, sep=' ', header=None)
        reader = open(image_list)
        lines = reader.readlines()
        self.data_info = []
        self.names = []
        self.labels = []
        self.cams = []
        if is_train == True:
            for line in lines:
                line = line.strip()
                self.names.append(line)
                self.labels.append(line.split('_')[0])
                self.cams.append(line.split('_')[1])  
            labels = sorted(set(self.labels))
            for pid, id in enumerate(labels):
                idxs = [i for i, v in enumerate(self.labels) if v==id] 
                for j in idxs:
                    self.labels[j] = pid
        else:
            for line in lines:
                line = line.strip()
                self.names.append(line)
                self.labels.append(line.split('_')[0])
                self.cams.append(line.split('_')[1])      
        self.data_info = self.names        
        self.root_dir = root_dir
        self.transform = transform

    def get_class(self, idx):
        return self.labels[idx]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.names[idx])
        image = torchvision.io.read_image(img_name)
        vid = np.int64(self.labels[idx])
        camid = np.int64(self.cams[idx].split('s')[0].replace('c', ""))


        if self.transform:
            img = self.transform((image.type(torch.FloatTensor))/255.0)

        return img, vid, camid     

class CustomDataSet4Veri776(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, image_list, root_dir, is_train=True, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        #self.data_info = pd.read_xml(csv_file, sep=' ', header=None)
        reader = open(image_list)
        lines = reader.readlines()
        self.data_info = []
        self.names = []
        self.labels = []
        self.cams = []
        if is_train == True:
            for line in lines:
                line = line.strip()
                self.names.append(line)
                self.labels.append(line.split('_')[0])
                self.cams.append(line.split('_')[1])     
            labels = sorted(set(self.labels))
            for pid, id in enumerate(labels):
                idxs = [i for i, v in enumerate(self.labels) if v==id] 
                for j in idxs:
                    self.labels[j] = pid
                # print(pid, id, 'debug')
        else:
            for line in lines:
                line = line.strip()
                self.names.append(line)
                self.labels.append(line.split('_')[0])
                self.cams.append(line.split('_')[1])      
        self.data_info = self.names        
        self.root_dir = root_dir
        self.transform = transform

    def get_class(self, idx):
        return self.labels[idx]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.names[idx])
        image = torchvision.io.read_image(img_name)
        vid = np.int64(self.labels[idx])
        camid = np.int64(self.cams[idx].replace('c', ""))


        if self.transform:
            img = self.transform((image.type(torch.FloatTensor))/255.0)

        return img, vid, camid, 0 

class CustomDataSet4Veri776_withviewpont(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, image_list, root_dir, viewpoint_train, viewpoint_test, is_train=True, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.viewpoint_train = pd.read_csv(viewpoint_train, sep=' ', header = None)
        self.viewpoint_test = pd.read_csv(viewpoint_test, sep=' ', header = None)
        reader = open(image_list)
        lines = reader.readlines()
        self.data_info = []
        self.names = []
        self.labels = []
        self.cams = []
        self.view = []
        conta_missing_images = 0
        if is_train == True:
            for line in lines:
                line = line.strip()
                view = self.viewpoint_train[self.viewpoint_train.iloc[:, 0] == line]
                if self.viewpoint_train[self.viewpoint_train.iloc[:, 0] == line].shape[0] ==0:
                    conta_missing_images += 1
                    continue
                view = int(view.iloc[0, -1])
                self.view.append(view)
                self.names.append(line)
                self.labels.append(line.split('_')[0])
                self.cams.append(line.split('_')[1]) 
            labels = sorted(set(self.labels))
            for pid, id in enumerate(labels):
                idxs = [i for i, v in enumerate(self.labels) if v==id] 
                for j in idxs:
                    self.labels[j] = pid
        else:
            for line in lines:
                line = line.strip()
                view = self.viewpoint_test[self.viewpoint_test.iloc[:, 0] == line]
                if self.viewpoint_test[self.viewpoint_test.iloc[:, 0] == line].shape[0] == 0:
                    conta_missing_images += 1
                    continue
                view = int(view.iloc[0, -1])
                self.view.append(view)
                self.names.append(line)
                self.labels.append(line.split('_')[0])
                self.cams.append(line.split('_')[1])      
        self.data_info = self.names        
        self.root_dir = root_dir
        self.transform = transform
        print('Missed viewpoint for ', conta_missing_images, ' images!')
    def get_class(self, idx):
        return self.labels[idx]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.names[idx])
        image = torchvision.io.read_image(img_name)
        vid = np.int64(self.labels[idx])
        camid = np.int64(self.cams[idx].replace('c', ""))-1
        viewid = np.int64(self.view[idx])


        if self.transform:
            img = self.transform((image.type(torch.FloatTensor))/255.0)

        return img, vid, camid, viewid     

class CustomDataSet4VehicleID_Random(Dataset):
    def __init__(self, lines, root_dir, is_train=True, mode=None, transform=None, teste=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_info = []
        self.names = []
        self.labels = []
        self.teste = teste
        if is_train == True:
            for line in lines:
                line = line.strip()
                name = line[:7] 
                vid = line[8:]
                self.names.append(name)
                self.labels.append(vid)   
            labels = sorted(set(self.labels))
            print("ncls: ",len(labels))
            for pid, id in enumerate(labels):
                idxs = [i for i, v in enumerate(self.labels) if v==id] 
                for j in idxs:
                    self.labels[j] = pid
        else:
            print("Dataload Test mode: ", mode)
            vid_container = set()
            for line in lines:
                line = line.strip()
                name = line[:7]
                vid = line[8:]
                # random.shuffle(dataset)
                if mode=='g':  
                    if vid not in vid_container:
                        vid_container.add(vid)
                        self.names.append(name)
                        self.labels.append(vid)
                else:
                    if vid not in vid_container:
                        vid_container.add(vid)
                    else:
                        self.names.append(name)
                        self.labels.append(vid)

        self.data_info = self.names        
        self.root_dir = root_dir
        self.transform = transform

    def get_class(self, idx):
        return self.labels[idx]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.names[idx]+ ".jpg")
        image = torchvision.io.read_image(img_name)
        vid = np.int64(self.labels[idx])
        ### no camera information
        camid = idx #np.int64(self.cams[idx].replace('c', ""))

        if self.transform:
            img = self.transform((image.type(torch.FloatTensor))/255.0)
        if self.teste:
            return img, vid, camid, 0
        else:
            return img, vid, camid

class CustomDataSet4VehicleID(Dataset):
    def __init__(self, image_list, root_dir, is_train=True, mode=None, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        reader = open(image_list)
        lines = reader.readlines()
        self.data_info = []
        self.names = []
        self.labels = []
        if is_train == True:
            for line in lines:
                line = line.strip()
                name = line[:7] 
                vid = line[8:]
                self.names.append(name)
                self.labels.append(vid)   
            labels = sorted(set(self.labels))
            print("ncls: ",len(labels))
            for pid, id in enumerate(labels):
                idxs = [i for i, v in enumerate(self.labels) if v==id] 
                for j in idxs:
                    self.labels[j] = pid
        else:
            print("Dataload Test mode: ", mode)
            vid_container = set()
            for line in lines:
                line = line.strip()
                name = line[:7]
                vid = line[8:]
                # random.shuffle(dataset)
                if mode=='g':  
                    if vid not in vid_container:
                        vid_container.add(vid)
                        self.names.append(name)
                        self.labels.append(vid)
                else:
                    if vid not in vid_container:
                        vid_container.add(vid)
                    else:
                        self.names.append(name)
                        self.labels.append(vid)

        self.data_info = self.names        
        self.root_dir = root_dir
        self.transform = transform

    def get_class(self, idx):
        return self.labels[idx]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.names[idx]+ ".jpg")
        image = torchvision.io.read_image(img_name)
        vid = np.int64(self.labels[idx])
        camid = idx #np.int64(self.cams[idx].replace('c', ""))

        if self.transform:
            img = self.transform((image.type(torch.FloatTensor))/255.0)

        return img, vid, camid, 0

def re_ranking(probFea, galFea, k1, k2, lambda_value, local_distmat=None, only_local=False):
    # if feature vector is numpy, you should use 'torch.tensor' transform it to tensor
    query_num = probFea.size(0)
    all_num = query_num + galFea.size(0)
    if only_local:
        original_dist = local_distmat
    else:
        feat = torch.cat([probFea,galFea])
        print('using GPU to compute original distance')
        distmat = torch.pow(feat,2).sum(dim=1, keepdim=True).expand(all_num,all_num) + \
                      torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(all_num, all_num).t()
        distmat.addmm_(1,-2,feat,feat.t())
        original_dist = distmat.cpu().numpy()
        del feat
        if not local_distmat is None:
            original_dist = original_dist + local_distmat
    gallery_num = original_dist.shape[0]
    original_dist = np.transpose(original_dist / np.max(original_dist, axis=0))
    V = np.zeros_like(original_dist).astype(np.float16)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    print('starting re_ranking')
    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i, :k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate, :int(np.around(k1 / 2)) + 1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,
                                               :int(np.around(k1 / 2)) + 1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(
                    candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)
    original_dist = original_dist[:query_num, ]
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float16)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:, i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float16)

    for i in range(query_num):
        temp_min = np.zeros(shape=[1, gallery_num], dtype=np.float16)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]],
                                                                               V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num, query_num:]
    return final_dist

def get_model(data, device):
    ### 4G hybryd with LBS     MBR-4G
    if data['model_arch'] =='MBR_4G':
        model = MBR_model(class_num=data['n_classes'], n_branches=[], losses="LBS", n_groups=4, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'])
    
    ### MBR-4B (4B hybrid LBS)
    if data['model_arch'] == 'MBR_4B':
        model = MBR_model(class_num=data['n_classes'], n_branches=["R50", "R50", "BoT", "BoT"], losses="LBS", n_groups=0, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'])

    return model.to(device)

def normalize_batch(batch, maximo=None, minimo = None):
    if maximo != None:
        return (batch - minimo.unsqueeze(-1).unsqueeze(-1)) / (maximo.unsqueeze(-1).unsqueeze(-1) - minimo.unsqueeze(-1).unsqueeze(-1))
    else:
        return (batch - torch.amin(batch, dim=(1, 2)).unsqueeze(-1).unsqueeze(-1)) / (torch.amax(batch, dim=(1, 2)).unsqueeze(-1).unsqueeze(-1) - torch.amin(batch, dim=(1, 2)).unsqueeze(-1).unsqueeze(-1))

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic= True
    torch.backends.cudnn.benchmark= False

def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_epoch(model, device, dataloader_q, dataloader_g, model_arch, remove_junk=True, scaler=None, re_rank=False):
    model.eval()
    re_escala = torchvision.transforms.Resize((256,256), antialias=True)

    qf = []
    gf = []
    q_camids = []
    g_camids = []
    q_vids = []
    g_vids = []
    q_images = []
    g_images =  []
    count_imgs = 0
    blend_ratio =0.3
    with torch.no_grad():
        for image, q_id, cam_id, view_id  in tqdm(dataloader_q, desc='Query infer (%)', bar_format='{l_bar}{bar:20}{r_bar}'):
            image = image.to(device)
            if scaler:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    _, _, ffs, activations = model(image, cam_id, view_id)
            else:
                _, _, ffs, activations = model(image, cam_id, view_id) #activations is list containing tensor with shape (48,2048,16,16)
                    
            count_imgs += activations[0].shape[0]
            end_vec = []
            for item in ffs:
                end_vec.append(F.normalize(item))
            _concatenated = torch.cat(end_vec, 1)
            qf.append(torch.cat(end_vec, 1))
            q_vids.append(q_id)
            q_camids.append(cam_id)

        del q_images
        count_imgs = 0
        for image, g_id, cam_id, view_id in tqdm(dataloader_g, desc='Gallery infer (%)', bar_format='{l_bar}{bar:20}{r_bar}'):
            image = image.to(device)
            if scaler:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    _, _, ffs, activations = model(image, cam_id, view_id)
            else:
                _, _, ffs, activations = model(image, cam_id, view_id)

            end_vec = []
            for item in ffs:
                end_vec.append(F.normalize(item))
            _concatenated = torch.cat(end_vec, 1)
            gf.append(torch.cat(end_vec, 1))
            g_vids.append(g_id)
            g_camids.append(cam_id)

            count_imgs += activations[0].shape[0]

        del g_images

    qf = torch.cat(qf, dim=0) 
    gf = torch.cat(gf, dim=0) 

    m, n = qf.shape[0], gf.shape[0]   
    if re_rank:
        distmat = re_ranking(qf, gf, k1=80, k2=16, lambda_value=0.3)
    else:
        distmat =  torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t() 
        distmat.addmm_(qf, gf.t(),beta=1, alpha=-2) 
        distmat = torch.sqrt(distmat).cpu().numpy() 

    q_camids = torch.cat(q_camids, dim=0).cpu().numpy()
    g_camids = torch.cat(g_camids, dim=0).cpu().numpy()
    q_vids = torch.cat(q_vids, dim=0).cpu().numpy()
    g_vids = torch.cat(g_vids, dim=0).cpu().numpy()   
    
    del qf, gf
    
    cmc, mAP = eval_func(distmat, q_vids, g_vids, q_camids, g_camids, remove_junk=remove_junk)
    print(f'mAP = {mAP},  CMC1= {cmc[0]}, CMC5= {cmc[4]}')

    return cmc, mAP

from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

class LightNN(nn.Module):
    def __init__(self, class_num, n_branches, n_groups, losses="LBS", backbone="ibn", droprate=0, linear_num=False, return_f = True, circle_softmax=False, pretrain_ongroups=True, end_bot_g=False, group_conv_mhsa=False, group_conv_mhsa_2=False, x2g=False, x4g=False, LAI=False, n_cams=0, n_views=0):
        super(LightNN, self).__init__()
        self.backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights)
        self.backbone.classifier = nn.Sequential()

        self.multiSimulator = nn.Sequential(
            nn.Linear(576, 2048),
            nn.BatchNorm1d(2048)
        )

        self.classifier = nn.Sequential(
            nn.Linear(2048, class_num, bias=True)
            # nn.Linear(2048, 1024, bias=True),
            # nn.Hardswish(),
            # nn.Dropout(p=0.2, inplace=True),
            # nn.Linear(1024, class_num, bias=True)
        )
        
        # self.model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights)
        # self.model.classifier = nn.Sequential(
        #     nn.Linear(576, 2048),
        #     nn.BatchNorm1d(2048)
        # )

    def forward(self, x,cam, view):
        # out = self.model(x)
        # return None, None, [out], [torch.empty(out.shape[0])]
        backboneOut = self.backbone(x)
        ffs = self.multiSimulator(backboneOut)
        pred = self.classifier(ffs)
        return pred, None, [ffs], [torch.empty(ffs.shape[0])]
    
import torch.optim as optim

def train_knowledge_distillation(teacher, student, train_loader, epochs, device, teacherGamma, teacherAlpha):#, learning_rate, T, soft_target_loss_weight, ce_loss_weight):
    TEMPERATURE = 1.
    ALPHA = 0.4

    optimizer = optim.Adam(student.parameters())#, lr=learning_rate)

    teacher.eval()  # Teacher set to evaluation mode
    student.train() # Student to train mode

    for epoch in range(epochs):
        running_loss = 0.0
        for image_batch, label, cam, view in tqdm(train_loader, desc='Epoch ' + str(epoch+1) +' (%)' , bar_format='{l_bar}{bar:20}{r_bar}'):
            image_batch = image_batch.to(device)
            # label = label.to(device)

            with torch.no_grad():
                teacher_preds, teacher_embs, teacher_ffs, teacher_activations = teacher(image_batch, cam, view)
            
            student_preds, _, student_ffs, _ = student(image_batch, cam, view)

            if type(teacher_preds) != list:
                teacher_preds = [teacher_preds]
            teacher_preds = teacher_preds[0]

            lossPred = F.kl_div(F.log_softmax(student_preds / TEMPERATURE, dim=1), F.softmax(teacher_preds / TEMPERATURE, dim=1), reduction='batchmean')
        
            end_vec = []
            for item in teacher_ffs:
                end_vec.append(F.normalize(item))
            teacher_ffs = torch.cat(end_vec, 1) #(48,2048)

            end_vec = []
            for item in student_ffs:
                end_vec.append(F.normalize(item))
            student_ffs = torch.cat(end_vec, 1) #(48,2048)

            lossFss = F.mse_loss(student_ffs, teacher_ffs)

            loss = ALPHA * lossPred + (1 - ALPHA) * lossFss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * image_batch.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}')

if __name__ == "__main__":
    args_path_weights = "cfg/"
    args_re_rank = False

    with open(args_path_weights + "config.yaml", "r") as stream:
            data = yaml.safe_load(stream)

    teste_transform = transforms.Compose([
                    transforms.Resize((data['y_length'],data['x_length']), antialias=True),
                    transforms.Normalize(data['n_mean'], data['n_std']),

    ])                  

    if data['half_precision']:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler=False

    if data['dataset'] == 'Veri776':
        data_q = CustomDataSet4Veri776_withviewpont(data['query_list_file'], data['query_dir'], data['train_keypoint'], data['test_keypoint'], is_train=False, transform=teste_transform)
        data_g = CustomDataSet4Veri776_withviewpont(data['gallery_list_file'], data['teste_dir'], data['train_keypoint'], data['test_keypoint'], is_train=False, transform=teste_transform)
        data_q = DataLoader(data_q, batch_size=data['BATCH_SIZE'], shuffle=False, num_workers=data['num_workers_teste'])
        data_g = DataLoader(data_g, batch_size=data['BATCH_SIZE'], shuffle=False, num_workers=data['num_workers_teste'])

    # Check if the GPU is available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(DEVICE_NOTCPU)
    print(f'Selected device: {device}')

    model = get_model(data, device)
    # One of the saved weights last.pt best_CMC.pt best_mAP.pt
    path_weights = args_path_weights + 'best_mAP.pt'

    try:
        model.load_state_dict(torch.load(path_weights, map_location='cpu')) 
    except RuntimeError:
    ### nn.Parallel adds "module." to the dict names. Although like said nn.Parallel can incur in weird results in some cases 
        tmp = torch.load(path_weights, map_location='cpu')
        tmp = OrderedDict((k.replace("module.", ""), v) for k, v in tmp.items())
        model.load_state_dict(tmp)
    # model = LightNN(class_num=data['n_classes'], n_branches=[], losses="LBS", n_groups=4, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'])
    # model.load_state_dict(torch.load('light.pth'))

    # model = model.to(device)
    # model.eval()

    # mean = False
    # l2 = True

    # cmc, mAP = test_epoch(model, device, data_q, data_g, data['model_arch'], remove_junk=True, scaler=scaler, re_rank=args_re_rank)
    # exit()

    teacher = model

    teacher_params_count = sum(p.numel() for p in teacher.parameters())
    print("teacher params", teacher_params_count)

    student = LightNN(class_num=data['n_classes'], n_branches=[], losses="LBS", n_groups=4, LAI=data['LAI'], n_cams=data['n_cams'], n_views=data['n_views'])

    student_params_count = sum(p.numel() for p in student.parameters())

    print("Student network has", teacher_params_count / student_params_count, "times less parameters")

    if data["LAI"]:
        raise NotImplementedError("Original model is better of LAI, see article")

    train_transform = transforms.Compose([
                        transforms.Resize((data['y_length'],data['x_length']), antialias=True),
                        transforms.Pad(10),
                        transforms.RandomCrop((data['y_length'], data['x_length'])),
                        transforms.RandomHorizontalFlip(p=data['p_hflip']),
                        transforms.Normalize(data['n_mean'], data['n_std']),
                        transforms.RandomErasing(p=data['p_rerase'], value=0),
        ]) 

    data_train = CustomDataSet4Veri776(data['train_list_file'], data['train_dir'], is_train=True, transform=train_transform)
    data_train = DataLoader(data_train, sampler=RandomIdentitySampler(data_train, data['BATCH_SIZE'], data['NUM_INSTANCES']), num_workers=data['num_workers_train'], batch_size = data['BATCH_SIZE'], collate_fn=train_collate_fn, pin_memory=True)        

    teacher.to(device)
    student.to(device)

    train_knowledge_distillation(teacher, student, data_train, 1, device, data['gamma_ce'], data['alpha_ce'])
    torch.save(student.state_dict(), "light.pth")
