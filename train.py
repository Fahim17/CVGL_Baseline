import pandas as pd
import torch
from tqdm import tqdm
import numpy as np
import time
from datetime import datetime
from torch.utils.data import DataLoader
from eval import predict, accuracy

from CVUSA_dataset import CVUSA_Dataset_Eval
from eval import accuracy, predict


def time_stamp():
    now = datetime.now()
    print(f'\nDate: {now}\n')



def train_step_eval(step=-1, mdl=None, dev='cpu' ):
    print(f'Train Step Eval: {step}')

    data_path = '/data/Research/Dataset/CVUSA_Cropped/CVUSA' #don't include the / at the end

    val_que = CVUSA_Dataset_Eval(data_folder=data_path, split='train', img_type='query', transforms='Do Transform')
    val_ref = CVUSA_Dataset_Eval(data_folder=data_path, split='train', img_type='reference', transforms='Do Transform')
    val_loader_que = DataLoader(val_que, batch_size=64, shuffle=False)
    val_loader_ref = DataLoader(val_ref, batch_size=64, shuffle=False)

    print("\nExtract Features:")
    query_features, query_labels = predict(model=mdl, dataloader=val_loader_que, dev=dev, isQuery=True)
    reference_features, reference_labels = predict(model = mdl, dataloader=val_loader_ref, dev=dev, isQuery=False) 

    print("Compute Scores:")
    r1 =  accuracy(query_features=query_features, reference_features=reference_features, query_labels=query_labels, topk=[1, 5, 10])

        



def train(model, criterion, optimizer, train_loader, num_epochs=10, dev='cpu'):
    model.train()

    epoch_loss = []

    time_stamp()
    for epoch in range(num_epochs):
        # total_loss = 0.0
        print(f'Epoch#{epoch}')
        running_loss = []

        # if(epoch%10==0 and epoch != 0):
        #     train_step_eval(step=epoch, mdl=model, dev=dev)

        for i, (anchor, positive, negative) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            anchor, positive, negative = anchor.to(dev), positive.to(dev), negative.to(dev)
            anchor_embedding, positive_embedding = model(q = anchor, r = positive, isTrain = True, isQuery = True)


            # print(anchor_embedding)
            # loss, mean_p, mean_n  = criterion(anchor_embedding, positive_embedding)
            loss  = criterion(anchor_embedding, positive_embedding)


            # anchor_embedding, positive_embedding = model(q = anchor, r = positive, isTrain = True, isQuery = True)
            # _, negative_embedding = model(q = anchor, r = negative, isTrain = True, isQuery = True)
            # loss  = criterion(anchor_embedding, positive_embedding, negative_embedding)

            loss.backward()
            optimizer.step()
            # total_loss += loss.item()
            running_loss.append(loss.cpu().detach().numpy())
            # print(f"Epoch {epoch+1}, Loss: {loss.item()}")
        print(f"Epoch: {epoch+1}/{num_epochs} Loss: {np.mean(running_loss)}")
        epoch_loss.append(np.mean(running_loss))
        # df_loss[epoch, "Loss"] = loss.cpu().detach().numpy()
    
    # torch.save(model, f'model_weights/model_tr.pth')
    
    time_stamp()
    
    return epoch_loss






# def train(model, criterion, optimizer, train_loader, num_epochs=10, dev='cpu'):
#     model.train()
#     epoch_loss = []

#     time_stamp()
#     for epoch in range(num_epochs):
#         # total_loss = 0.0
#         print(f'Epoch#{epoch}')
#         running_loss = []
#         for i, (anchor, positive, negative) in enumerate(tqdm(train_loader)):
#             optimizer.zero_grad()
#             anchor, positive, negative = anchor.to(dev), positive.to(dev), negative.to(dev)
#             anchor_embedding = model(anchor, isQuery = True)
#             positive_embedding = model(positive, isQuery = False)
#             negative_embedding = model(negative, isQuery = False)
#             loss = criterion(anchor_embedding, positive_embedding, negative_embedding)
#             loss.backward()
#             optimizer.step()
#             # total_loss += loss.item()
#             running_loss.append(loss.cpu().detach().numpy())
#             # print(f"Epoch {epoch+1}, Loss: {loss.item()}")
#         print(f"Epoch: {epoch+1}/{num_epochs} Loss: {np.mean(running_loss)}")
#         epoch_loss.append(np.mean(running_loss))
#         # df_loss[epoch, "Loss"] = loss.cpu().detach().numpy()
    
#     time_stamp()
    
#     return epoch_loss




# only for HuggingFace CLIP

# def train_step_eval(step=-1, query_features=None, reference_features=None, topk=[1,5,10] ):
#     print(f'Train Step Eval: {step}')
#     print("Compute Scores:")
#     if(query_features is not None and reference_features is not None):
#         N = query_features.shape[0]
#         M = reference_features.shape[0]
#         topk.append(M//100)
#         results = np.zeros([len(topk)])
#         # for CVUSA, CVACT
#         query_features = query_features.cpu()
#         reference_features = reference_features.cpu()
        
#         query_features = query_features.detach().numpy()
#         reference_features = reference_features.detach().numpy()



#         if N < 80000:
#             query_features_norm = np.sqrt(np.sum((query_features**2), axis=1, keepdims=True))
#             reference_features_norm = np.sqrt(np.sum((reference_features ** 2), axis=1, keepdims=True))
#             similarity = np.matmul(query_features/query_features_norm, (reference_features/reference_features_norm).T)
            
#             # print(similarity)
#             # save_tensor(var_name='similarity', var=similarity)
#             for i in range(N):
#                 # ranking = np.sum((similarity[i,:]>similarity[i,query_labels[i]])*1.)
#                 ranking = np.sum((similarity[i,:]>similarity[i,i])*1.)


#                 for j, k in enumerate(topk):
#                     if ranking < k:
#                         results[j] += 1.

#         results = results/ query_features.shape[0] * 100.
#         print('Percentage-top1:{}, top5:{}, top10:{}, top1%:{}'.format(results[0], results[1], results[2], results[-1]))
#     else:
#         print('problem with embedding')

