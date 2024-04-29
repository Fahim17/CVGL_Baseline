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
from CVUSA_dataset import CVUSA_dataset_cropped
from CVUSA_dataset import CVUSA_Dataset_Eval
from resnet_model import ResNet

import torch.nn.functional as F
import copy


# data_path = '/media/fahimul/2B721C03261BDC8D/Research/datasets/CVUSA' #don't include the / at the end
data_path = '/home/fa947945/datasets/CVUSA_Cropped/CVUSA' #don't include the / at the end
train_data= pd.read_csv(f'{data_path}/splits/train-19zl.csv')
val_data= pd.read_csv(f'{data_path}/splits/val-19zl.csv')

# df_loss = pd.DataFrame(columns=['Loss'])

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_ds = CVUSA_dataset_cropped(df = train_data, path=data_path, transform=transform)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

val_ref = CVUSA_Dataset_Eval(data_folder=data_path, split='train', img_type='reference', transforms=transform)
val_loader_ref = DataLoader(val_ref, batch_size=64, shuffle=True)
val_que = CVUSA_Dataset_Eval(data_folder=data_path, split='train', img_type='query', transforms=transform)
val_loader_que = DataLoader(val_que, batch_size=64, shuffle=True)




# Define Triplet Loss
class TripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = torch.sum(torch.pow(anchor - positive, 2), dim=1)
        distance_negative = torch.sum(torch.pow(anchor - negative, 2), dim=1)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return torch.mean(losses)


def time_stamp():
    now = datetime.now()
    print(f'\nDate: {now}\n')


def train(model, criterion, optimizer, train_loader, num_epochs=10, dev='cpu'):
    model.train()

    time_stamp()
    for epoch in range(num_epochs):
        # total_loss = 0.0
        running_loss = []
        for i, (anchor, positive, negative) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            anchor, positive, negative = anchor.to(dev), positive.to(dev), negative.to(dev)
            anchor_embedding = model(anchor)
            positive_embedding = model(positive)
            negative_embedding = model(negative)
            loss = criterion(anchor_embedding, positive_embedding, negative_embedding)
            loss.backward()
            optimizer.step()
            # total_loss += loss.item()
            running_loss.append(loss.cpu().detach().numpy())
            # print(f"Epoch {epoch+1}, Loss: {loss.item()}")
        print(f"Epoch: {epoch+1}/{num_epochs} Loss: {np.mean(running_loss)}")
        # df_loss[epoch, "Loss"] = loss.cpu().detach().numpy()
  

    time_stamp()
    return running_loss

def predict(model, dataloader, verbose=True, dev=torch.device('cpu'), normalize_features=True):
    
    model.eval()
    
    # wait before starting progress bar
    time.sleep(0.1)
    
    if verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader
        
    img_features_list = []
    
    ids_list = []
    with torch.no_grad():
        
        for img, ids in bar:
        
            ids_list.append(ids)
            
            # with autocast():
         
            img = img.to(dev)
            img_feature = model(img)
        
            # # normalize is calculated in fp32
            # if normalize_features:
            #     img_feature = F.normalize(img_feature, dim=-1)
            
            # save features in fp32 for sim calculation
            img_features_list.append(img_feature.to(torch.float32))
      
        # keep Features on GPU
        img_features = torch.cat(img_features_list, dim=0) 
        ids_list = torch.cat(ids_list, dim=0).to(dev)
        
    if verbose:
        bar.close()
        
    return img_features, ids_list


def calculate_scores(query_features, reference_features, query_labels, reference_labels, step_size=1000, ranks=[1,5,10]):

    topk = copy.deepcopy(ranks)
    Q = len(query_features)
    R = len(reference_features)
    
    steps = Q // step_size + 1
    
    
    query_labels_np = query_labels.cpu().numpy()
    reference_labels_np = reference_labels.cpu().numpy()
    
    ref2index = dict()
    for i, idx in enumerate(reference_labels_np):
        ref2index[idx] = i
    
    
    similarity = []
    
    for i in range(steps):
        
        start = step_size * i
        
        end = start + step_size
          
        sim_tmp = query_features[start:end] @ reference_features.T
        
        similarity.append(sim_tmp.cpu())
     
    # matrix Q x R
    similarity = torch.cat(similarity, dim=0)
    

    topk.append(R//100)
    
    results = np.zeros([len(topk)])
    
    
    bar = tqdm(range(Q))
    
    for i in bar:
        
        # similiarity value of gt reference
        gt_sim = similarity[i, ref2index[query_labels_np[i]]]
        
        # number of references with higher similiarity as gt
        higher_sim = similarity[i,:] > gt_sim
        
         
        ranking = higher_sim.sum()
        for j, k in enumerate(topk):
            if ranking < k:
                results[j] += 1.
                        
        
    results = results/ Q * 100.
 
    
    bar.close()
    
    # wait to close pbar
    time.sleep(0.1)
    
    string = []
    for i in range(len(topk)-1):
        
        string.append('Recall@{}: {:.4f}'.format(topk[i], results[i]))
        
    string.append('Recall@top1: {:.4f}'.format(results[-1]))            
        
    print(' - '.join(string)) 

    return results[0]

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    model = ResNet(emb_dim=512).to(device)
    criterion = TripletLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Training Start")
    all_loses = train(model, criterion, optimizer, train_loader, num_epochs=10, dev=device)
    df_loss = pd.DataFrame({'Loss': all_loses})
    df_loss.to_csv('losses.csv')

    print("\nExtract Features:")
    reference_features, reference_labels = predict(model = model, dataloader=val_loader_ref, dev=device) 
    query_features, query_labels = predict(model=model, dataloader=val_loader_que, dev=device)


    print("Compute Scores:")
    r1 =  calculate_scores(query_features, reference_features, query_labels, reference_labels, step_size=1000, ranks=[1, 5, 10])

    #print(r1) 
        







if __name__ == '__main__':
    main()
