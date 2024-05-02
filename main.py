from datetime import datetime
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
from losses import SoftTripletBiLoss
from train import train
from eval import accuracy, predict
from eval import calculate_scores
import torch.nn.functional as F
import copy
import math
from pytorch_metric_learning import losses as LS




# data_path = '/media/fahimul/2B721C03261BDC8D/Research/datasets/CVUSA' #don't include the / at the end
# data_path = '/home/fa947945/datasets/CVUSA_Cropped/CVUSA' #don't include the / at the end
data_path = '/data/Research/Dataset/CVUSA_Cropped/CVUSA' #don't include the / at the end

train_data= pd.read_csv(f'{data_path}/splits/train-19zl.csv')
val_data= pd.read_csv(f'{data_path}/splits/val-19zl.csv')

# df_loss = pd.DataFrame(columns=['Loss'])

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_ds = CVUSA_dataset_cropped(df = train_data, path=data_path, transform=transform)
val_que = CVUSA_Dataset_Eval(data_folder=data_path, split='train', img_type='query', transforms=transform)
val_ref = CVUSA_Dataset_Eval(data_folder=data_path, split='train', img_type='reference', transforms=transform)


def hyparam_info(emb_dim, loss_id, ln_rate, batch, epc, ls_mrgn, mdl_nm):
    print('\nHyperparameter info:')
    print(f'Loss ID: {loss_id}')
    print(f'Embedded dimension: {emb_dim}')
    print(f'Learning rate: {ln_rate}')
    print(f'Batch Size: {batch}')
    print(f'Loss Margin: {ls_mrgn}')
    print(f'Epoch: {epc}')
    print(f'Training Size: {train_data.shape[0]}')
    print(f'Model Name: {mdl_nm}')
    print('\n')

def get_rand_id():
    dt = datetime.now()
    return f"{math.floor(dt.timestamp())}"[3:]


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    embed_dim = 512
    lr = 0.01
    batch_size = 64
    epochs = 10
    loss_id = get_rand_id()
    loss_margin = 1


    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader_que = DataLoader(val_que, batch_size=batch_size, shuffle=False)
    val_loader_ref = DataLoader(val_ref, batch_size=batch_size, shuffle=False)

    # model = ResNet(emb_dim=embed_dim).to(device)
    # model_r = ResNet(emb_dim=embed_dim).to(device)
    # model_q = ResNet(emb_dim=embed_dim).to(device)

    # model = VIT().to(device)
    model = CLIP_model()
    
    # criterion = TripletLoss(margin=loss_margin)
    # criterion = nn.TripletMarginLoss(margin=0.5)
    criterion = SoftTripletBiLoss(alpha=0.5)

    


    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)
    # optimizer = optim.Adam(parameters, lr=lr)
    # optimizer = optim.AdamW(parameters, lr=lr)
    optimizer = optim.SGD(parameters, lr=lr)

    
    
    hyparam_info(emb_dim=embed_dim, loss_id=loss_id, ln_rate=lr, batch=batch_size, epc=epochs, ls_mrgn=loss_margin, mdl_nm=model.modelName)

    print("Training Start")
    all_loses = train(model, criterion, optimizer, train_loader, num_epochs=epochs, dev=device)
    df_loss = pd.DataFrame({'Loss': all_loses})
    df_loss.to_csv(f'losses/losses_{loss_id}.csv')

    print("\nExtract Features:")
    query_features, query_labels = predict(model=model, dataloader=val_loader_que, dev=device, isQuery=True)
    reference_features, reference_labels = predict(model = model, dataloader=val_loader_ref, dev=device, isQuery=False) 


    print("Compute Scores:")
    r1 =  calculate_scores(query_features, reference_features, query_labels, reference_labels, step_size=1000, ranks=[1, 5, 10])
    # r1 =  accuracy(query_features=query_features, reference_features=reference_features, query_labels=query_labels, topk=[1, 5, 10])


    #print(r1) 
        







if __name__ == '__main__':
    main()
