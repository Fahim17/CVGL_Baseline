import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from CVUSA_dataset import CVUSA_dataset_cropped

train_data_path = '/media/fahimul/2B721C03261BDC8D/Research/datasets/CVUSA' #don't include the / at the end

train_data= pd.read_csv(f'{train_data_path}/splits/train-19zl.csv')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

custom_dataset = CVUSA_dataset_cropped(df = train_data, path=train_data_path, transform=transform)

train_loader = DataLoader(custom_dataset, batch_size=64, shuffle=True)

bar = tqdm(train_loader, total=len(train_loader))
# bar = train_loader

ids_list = []

for ids in bar:
    ids = np.array(ids)
    print(ids.shape)
    
#     with autocast():
    
#         img = img.to(train_config.device)
#         img_feature = model(img)
    
#         # normalize is calculated in fp32
#         if train_config.normalize_features:
#             img_feature = F.normalize(img_feature, dim=-1)
    
#     # save features in fp32 for sim calculation
#     img_features_list.append(img_feature.to(torch.float32))

# # keep Features on GPU
# img_features = torch.cat(img_features_list, dim=0) 
# ids_list = torch.cat(ids_list, dim=0).to(train_config.device)



# print(ids_list)





















# ang_img, pos_img, neg_img = custom_dataset[5241]
# ang_img, pos_img, neg_img = ang_img.T, pos_img.T, neg_img.T

# plt.figure(figsize=(10, 5))

# # Subplot 1
# plt.subplot(1, 3, 1)
# plt.imshow(ang_img)
# plt.title('Image 1')
# plt.axis('off')

# # Subplot 2
# plt.subplot(1, 3, 2)
# plt.imshow(pos_img)
# plt.title('Image 2')
# plt.axis('off')

# # Subplot 3
# plt.subplot(1, 3, 3)
# plt.imshow(neg_img)
# plt.title('Image 3')
# plt.axis('off')

# plt.show()