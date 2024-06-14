from datetime import datetime
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import time
# from torch.cuda.amp import autocast
import numpy as np
from CVUSA_dataset import CVUSA_dataset_cropped, CVUSA_Dataset_Eval
# from CVUSA_dataset import CVUSA_Dataset_Eval
from custom_models import ResNet, VIT, CLIP_model
from losses import Contrastive_loss, SoftTripletBiLoss, InfoNCE
from train import train
from eval import predict, accuracy, calculate_scores
import torch.nn.functional as F
import copy
import math
from pytorch_metric_learning import losses as LS
from helper_func import get_rand_id, hyparam_info, save_exp, write_to_file, write_to_rank_file
from transformers import CLIPProcessor
from attributes import Configuration as hypm




# data_path = '/media/fahimul/2B721C03261BDC8D/Research/datasets/CVUSA' #don't include the / at the end
# data_path = '/home/fa947945/datasets/CVUSA_Cropped/CVUSA' #don't include the / at the end
data_path = '/data/Research/Dataset/CVUSA_Cropped/CVUSA' #don't include the / at the end

# train_data= pd.read_csv(f'{data_path}/splits/train-19zl.csv', header=None)
train_data= pd.read_csv(f'{data_path}/splits/train-19zl_5.csv', header=None)
# train_data= pd.read_csv(f'{data_path}/splits/train-19zl_30.csv', header=None)

val_data= pd.read_csv(f'{data_path}/splits/val-19zl.csv', header=None)

# df_loss = pd.DataFrame(columns=['Loss'])

transform = transforms.Compose([
    # transforms.Resize((224, 224)),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225]),
])


train_ds = CVUSA_dataset_cropped(df = train_data, path=data_path, transform=transform, train=True, lang=hypm.lang)
val_ds = CVUSA_dataset_cropped(df = val_data, path=data_path, transform=transform, train=False, lang=hypm.lang)

# val_que = CVUSA_Dataset_Eval(data_folder=data_path, split='val', img_type='query', transforms=transform)
# val_ref = CVUSA_Dataset_Eval(data_folder=data_path, split='val', img_type='reference', transforms=transform)





def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embed_dim = 512
    lr = 0.000001
    batch_size = 64
    epochs = 100
    expID = get_rand_id()
    loss_margin = 1

    hypm.expID = expID






    # print(f"Device: {device}")


    train_loader = DataLoader(train_ds, batch_size=hypm.batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=hypm.batch_size, shuffle=False)
    # val_loader_ref = DataLoader(val_ref, batch_size=hypm.batch_size, shuffle=False)

    if hypm.save_weights:
        os.mkdir(f'model_weights/{hypm.expID}')

    # model = ResNet(emb_dim=embed_dim).to(device)
    # model_r = ResNet(emb_dim=embed_dim).to(device)
    # model_q = ResNet(emb_dim=embed_dim).to(device)

    # model = ResNet().to(device)
    # model = VIT().to(device)
    model = CLIP_model(embed_dim=hypm.embed_dim)

    # model = torch.load(f'model_weights/{7355080}/model_tr.pth')

    # torch.save(model, f'model_weights/{expID}/model_st.pth')

    # criterion = TripletLoss(margin=loss_margin)
    # criterion = nn.TripletMarginLoss(margin=0.5)
  
    # criterion = SoftTripletBiLoss()

    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=hypm.label_smoothing)
    criterion = InfoNCE(loss_function=loss_fn,
                            device=hypm.device,
                            )



    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)
    optimizer = optim.Adam(parameters, lr=hypm.lr)
    # optimizer = optim.AdamW(parameters, lr=lr)
    # optimizer = optim.SGD(parameters, lr=lr)

    
    
    hyparam_info(emb_dim = hypm.embed_dim, 
                 loss_id = hypm.expID, 
                 ln_rate = hypm.lr, 
                 batch = hypm.batch_size, 
                 epc = hypm.epochs, 
                 ls_mrgn = hypm.loss_margin, 
                 trn_sz = train_data.shape[0],
                 val_sz= val_data.shape[0],
                 mdl_nm = model.modelName)
    
    save_exp(emb_dim=hypm.embed_dim, 
                loss_id=hypm.expID, 
                ln_rate=hypm.lr, 
                batch=hypm.batch_size, 
                epc=hypm.epochs, 
                ls_mrgn=hypm.loss_margin, 
                trn_sz=train_data.shape[0],
                val_sz= val_data.shape[0],
                mdl_nm=model.modelName,
                msg=f'Text with {hypm.lang_with} {hypm.lang}')

    print("Training Start")
    all_loses = train(model, criterion, optimizer, train_loader, num_epochs=hypm.epochs, dev=hypm.device)
    df_loss = pd.DataFrame({'Loss': all_loses})
    df_loss.to_csv(f'losses/losses_{hypm.expID}.csv')

    write_to_file(expID=hypm.expID, msg=f'End of training: ', content=datetime.now())


    print("\nExtract Features:")
    query_features, reference_features, labels = predict(model=model, dataloader=val_loader, dev=hypm.device, isQuery=True)
    # reference_features, reference_labels = predict(model = model, dataloader=val_loader_ref, dev=hypm.device, isQuery=False) 
    


    print("Compute Scores:")
    # r1 =  calculate_scores(query_features, reference_features, query_labels, reference_labels, step_size=1000, ranks=[1, 5, 10])
    r1 =  accuracy(query_features=query_features, reference_features=reference_features, query_labels=labels, topk=[1, 5, 10])
    print(f'{r1}\n') 

    write_to_file(expID=hypm.expID, msg=f'Final eval: ', content=r1)
    write_to_rank_file(expID=hypm.expID, step=hypm.epochs, row=r1)



    if hypm.save_weights:
        torch.save(model, f'model_weights/{hypm.expID}/model_tr.pth')
    





    torch.cuda.empty_cache()
        






if __name__ == '__main__':
    main()
