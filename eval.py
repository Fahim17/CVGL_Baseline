import torch
from tqdm import tqdm
import time
import copy
import numpy as np
from torch.cuda.amp import autocast
import torch.nn.functional as F

from helper_func import save_tensor


def predict(model, dataloader, verbose=True, dev=torch.device('cpu'), normalize_features=True, isQuery=True):
    

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
            img = img.to(dev)
            if(isQuery):
                img_feature = model(q = img, r = img, isTrain = False, isQuery = True)
            else:
                img_feature = model(q = img, r = img, isTrain = False, isQuery = False)

            # if(dev !='cpu'):    
            #     with autocast():
            #         img = img.to(dev)
            #         if(isQuery):
            #             img_feature = model(q = img, r = img, isTrain = False, isQuery = True)
            #         else:
            #             img_feature = model(q = img, r = img, isTrain = False, isQuery = False)
            # else:
            #     img = img.to(dev)
            #     if(isQuery):
            #         img_feature = model(q = img, r = img, isTrain = False, isQuery = True)
            #     else:
            #         img_feature = model(q = img, r = img, isTrain = False, isQuery = False)

            
        
            # normalize is calculated in fp32
            # if normalize_features:
            #     img_feature = F.normalize(img_feature, dim=-1)
            
            # save features in fp32 for sim calculation
            img_features_list.append(img_feature.to(torch.float32))
      
        # keep Features on GPU
        img_features = torch.cat(img_features_list, dim=0) 
        ids_list = torch.cat(ids_list, dim=0).to(dev)

        # print(f'img_feature: {img_features.shape}')
        # print(f'ids: {ids_list.shape}')
        
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
        # print(f'i={i}, query_labels_np={query_labels_np[i]}, ref2index={ref2index[query_labels_np[i]]}')
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

    return results


def accuracy(query_features, reference_features, query_labels, topk=[1,5,10]):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    # print(f'query labels {query_labels}')
    ts = time.time()
    N = query_features.shape[0]
    M = reference_features.shape[0]
    topk.append(M//100)
    results = np.zeros([len(topk)])
    # for CVUSA, CVACT
    query_features = query_features.cpu()
    reference_features = reference_features.cpu()
    query_labels = query_labels.cpu()


    if N < 80000:
        query_features_norm = np.sqrt(np.sum((query_features**2).numpy(), axis=1, keepdims=True))
        reference_features_norm = np.sqrt(np.sum((reference_features ** 2).numpy(), axis=1, keepdims=True))
        similarity = np.matmul(query_features/query_features_norm, (reference_features/reference_features_norm).T)
        similarity = similarity.numpy()
        # print(similarity)
        # save_tensor(var_name='similarity', var=similarity)
        for i in range(N):
            # ranking = np.sum((similarity[i,:]>similarity[i,query_labels[i]])*1.)
            ranking = np.sum((similarity[i,:]>similarity[i,i])*1.)


            for j, k in enumerate(topk):
                if ranking < k:
                    results[j] += 1.
    else:
        # split the queries if the matrix is too large, e.g. VIGOR
        assert N % 4 == 0
        N_4 = N // 4
        for split in range(4):
            query_features_i = query_features[(split*N_4):((split+1)*N_4), :]
            query_labels_i = query_labels[(split*N_4):((split+1)*N_4)]
            query_features_norm = np.sqrt(np.sum(query_features_i ** 2, axis=1, keepdims=True))
            reference_features_norm = np.sqrt(np.sum(reference_features ** 2, axis=1, keepdims=True))
            similarity = np.matmul(query_features_i / query_features_norm,
                                   (reference_features / reference_features_norm).transpose())
            for i in range(query_features_i.shape[0]):
                ranking = np.sum((similarity[i, :] > similarity[i, query_labels_i[i]])*1.)
                for j, k in enumerate(topk):
                    if ranking < k:
                        results[j] += 1.

    results = results/ query_features.shape[0] * 100.
    print('Percentage-top1:{}, top5:{}, top10:{}, top1%:{}, time:{}'.format(results[0], results[1], results[2], results[-1], time.time() - ts))
    return results