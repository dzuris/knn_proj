# import the teacher model from the eval folder
import sys
sys.path.insert(0, 'eval')
import eval
import metric
import baseline
import triplet_sampler as trs
# from eval import eval, metric, baseline, triplet_sampler

import lwmodel as lw

# import eval.eval as bs # import baseline
import torch
import torch.nn as nn
import torch.utils.data as td
import torchvision.models as models
from torchvision import transforms

from tqdm import tqdm
import yaml
# some definitions necessary to be global

def getLeightweightModel(data):
    # model = torch.hub.load('XingangPan/IBN-Net', 'resnet50_ibn_a', pretrained=True)
    # download mobilenet and set its weights
    # model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2) # torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
    # print(model.classifier)
    # print('Number of parameters: ', sum(p.numel() for p in model.parameters()))
    model = lw.LWModel(data['n_classes'])
    return model.to(device)

def getTeacherModel(weight_path, data):
    """
    Args:
        weight_path : path to the file *.pt with weights for the model_type
        model_type : type of the used model [MBR_4G|MBR_4B]
    """
    # get model according to the type 
    model = eval.get_model(data, device)
    model.load_state_dict(torch.load(weight_path, map_location='cpu')) 
    model.eval() # this model will always just produce results, it wont be trained
    
    return model.to(device)

def getDatasetInParts(path, data, dataset='VERIWILD'):
    """
    Args:
        path : path to the root folder of the dataset (not to exact files!)
        dataset : [VERIWILD|VehicleID|Veri776]
    Return:
        data_train, data_g, data_q
    """
    
    ## Dataset Loading       
    if dataset == "VehicleID":
        data_q = trs.CustomDataSet4VehicleID(path+'/train_test_split/test_list_800.txt', path+'/image/', is_train=False, mode="q", transform=test_transform)
        data_g = trs.CustomDataSet4VehicleID(path+'/train_test_split/test_list_800.txt', path+'/image/', is_train=False, mode="g", transform=test_transform)
        data_train = trs.CustomDataSet4VehicleID(path+"/train_test_split/train_list.txt", path+'/image/', is_train=True, transform=train_transform)
        data_train = trs.DataLoader(data_train, sampler=trs.RandomIdentitySampler(data_train, data['BATCH_SIZE'], data['NUM_INSTANCES']), num_workers=data['num_workers_train'], batch_size = data['BATCH_SIZE'], collate_fn=trs.train_collate_fn, pin_memory=True)#
        data_q = trs.DataLoader(data_q, batch_size=data['BATCH_SIZE'], shuffle=False, num_workers=data['num_workers_teste'])
        data_g = trs.DataLoader(data_g, batch_size=data['BATCH_SIZE'], shuffle=False, num_workers=data['num_workers_teste'])
    if dataset == 'VERIWILD':
        data_q = trs.CustomDataSet4VERIWILD(path+'/train_test_split/test_3000_id_query.txt', path+'/images/', transform=test_transform, with_view=False)
        data_g = trs.CustomDataSet4VERIWILD(path+'/train_test_split/test_3000_id.txt', path+'/images/', transform=test_transform, with_view=False)
        data_train = trs.CustomDataSet4VERIWILD(path+'/train_test_split/train_list_start0.txt', path+'/images/', transform=train_transform, with_view=False)
        data_train = td.DataLoader(data_train, sampler=trs.RandomIdentitySampler(data_train, data['BATCH_SIZE'], data['NUM_INSTANCES']), num_workers=data['num_workers_train'], batch_size = data['BATCH_SIZE'], collate_fn=trs.train_collate_fn, pin_memory=True)
        data_q = td.DataLoader(data_q, batch_size=data['BATCH_SIZE'], shuffle=False, num_workers=data['num_workers_teste'])
        data_g = td.DataLoader(data_g, batch_size=data['BATCH_SIZE'], shuffle=False, num_workers=data['num_workers_teste'])

    if dataset == 'Veri776':
        data_q = trs.CustomDataSet4Veri776_withviewpont(data['query_list_file'], data['query_dir'], data['train_keypoint'], data['test_keypoint'], is_train=False, transform=test_transform)
        data_g = trs.CustomDataSet4Veri776_withviewpont(data['gallery_list_file'], data['teste_dir'], data['train_keypoint'], data['test_keypoint'], is_train=False, transform=test_transform)
        if data["LAI"]:
            data_train = trs.CustomDataSet4Veri776_withviewpont(data['train_list_file'], data['train_dir'], data['train_keypoint'], data['test_keypoint'], is_train=True, transform=train_transform)
        else:
            data_train = trs.CustomDataSet4Veri776(data['train_list_file'], data['train_dir'], is_train=True, transform=train_transform)
        data_train = td.DataLoader(data_train, sampler=trs.RandomIdentitySampler(data_train, data['BATCH_SIZE'], data['NUM_INSTANCES']), num_workers=data['num_workers_train'], batch_size = data['BATCH_SIZE'], collate_fn=trs.train_collate_fn, pin_memory=True)
        data_g = td.DataLoader(data_g, batch_size=data['BATCH_SIZE'], shuffle=False, num_workers=data['num_workers_teste'])
        data_q = td.DataLoader(data_q, batch_size=data['BATCH_SIZE'], shuffle=False, num_workers=data['num_workers_teste'])

    return data_train, data_g, data_q

def do_training(m_student, m_teacher, dataloader, num_epochs):
    optimizer = torch.optim.Adam(params=m_student.parameters(), lr=0.0001)
    
    print('training starts ...')
    for epoch in range(num_epochs):
        losses = []
        for image_batch, label, cam, view in dataloader: #tqdm(dataloader, desc='Epoch ' + str(epoch+1) +' (%)' , bar_format='{l_bar}{bar:20}{r_bar}'): 
            image_batch = image_batch.to(device)
            
            preds, embs, ffs, activations = m_teacher(image_batch, cam, view)
            print('teacher done ...')
            stud_emb = m_student(image_batch)
            print('student done ...')
            # print()
            exit()
            # optimizer.zero_grad()
            
          
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Selected device is: {device}')

    # setup
    data = {
        'model_arch':'MBR_4G',
        'n_classes': 20,
        
        'y_length':256,
        'x_length':256,
        
        'n_mean': [0.5, 0.5, 0.5],
        'n_std': [0.5, 0.5, 0.5],
        
        'BATCH_SIZE' : 32,
        'NUM_INSTANCES' : 4,
        'num_workers_teste': 16,
        'num_workers_train': 8
    }
    with open('baseline/cfg/' + "config.yaml", "r") as stream:
        data = yaml.safe_load(stream)
    # ======
    
    # create transforms, that will be applied to the test and train 
    test_transform = transforms.Compose([
                transforms.Resize((data['y_length'],data['x_length']), antialias=True),
                transforms.Normalize(data['n_mean'], data['n_std']),
    ])                  
    
    train_transform = transforms.Compose([
                transforms.Resize((data['y_length'],data['x_length']), antialias=True),
                transforms.Pad(10),
                transforms.RandomCrop((data['y_length'], data['x_length'])),
                # transforms.RandomHorizontalFlip(p=data['p_hflip']),
                transforms.Normalize(data['n_mean'], data['n_std']),
                # transforms.RandomErasing(p=data['p_rerase'], value=0),
    ])       
    
    m_teacher = getTeacherModel(weight_path='baseline/cfg/best_mAP.pt',data=data)
    m_student = getLeightweightModel(data).train() # set the lightweight model for training
    data_train, data_g, data_q = getDatasetInParts(path="C:/Users/holan/Downloads/VeriWild", data=data)
    
    # start training
    do_training(m_student=m_student, m_teacher=m_teacher, dataloader=data_train, num_epochs=1)
    


