import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights, vit_b_16, ViT_B_16_Weights





# # Define the ResNet model
# class ResNet(nn.Module):
#     def __init__(self, emb_dim):
#         super(ResNet, self).__init__()
#         self.modelName = 'ResNet50'
#         # self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
#         # for param in self.resnet.parameters():
#         #     param.requires_grad = False
#         self.resnet = resnet50()
#         num_features = self.resnet.fc.in_features
#         self.resnet.fc = nn.Linear(num_features, emb_dim)



#     def forward(self, x):
#         return self.resnet(x)


# Define the ResNet model
class ResNet(nn.Module):
    def __init__(self, emb_dim):
        super(ResNet, self).__init__()
        self.modelName = 'ResNet18'
        self.q_net = resnet18(ResNet18_Weights.IMAGENET1K_V1)
        self.ref_net = resnet18(ResNet18_Weights.IMAGENET1K_V1)




    def forward(self, img, isQuery = True):
        if isQuery:
            return self.q_net(img)
        else:
            return self.ref_net(img)
    



# Define the ResNet model
class VIT(nn.Module):
    def __init__(self):
        super(VIT, self).__init__()
        self.modelName = 'VIT_B_16'
        self.q_vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.r_vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        
        # for param in self.vit.parameters():
        #     param.requires_grad = False
        # num_features = self.vit.heads
        # self.vit.fc = nn.Linear(num_features, emb_dim)



    def forward(self, img, isQuery = True):
        if isQuery:
            return self.q_vit(img)
        else:
            return self.r_vit(img)























# # Define the ResNet model
# class ResNet(nn.Module):
#     def __init__(self, emb_dim):
#         super(ResNet, self).__init__()
#         self.resnet = torchvision.models.resnet50(pretrained=True)
#         num_features = self.resnet.fc.in_features
#         self.resnet.fc = nn.Linear(num_features, emb_dim)

#     def forward(self, x):
#         return self.resnet(x)