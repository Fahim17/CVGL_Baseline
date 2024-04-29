import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights, vit_b_16, ViT_B_16_Weights
from vit_pytorch import ViT






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
    def __init__(self, emb_dim = 512):
        super(ResNet, self).__init__()
        self.modelName = 'ResNet18'
        self.q_net = resnet18(ResNet18_Weights.IMAGENET1K_V1)
        self.ref_net = resnet18(ResNet18_Weights.IMAGENET1K_V1)
        for param in self.q_net.parameters():
            param.requires_grad = False
        for param in self.ref_net.parameters():
            param.requires_grad = False
        self.resnet_output = self.q_net.fc.out_features
        self.fc_q = nn.Linear(self.resnet_output, emb_dim)
        self.fc_r = nn.Linear(self.resnet_output, emb_dim)
        self.sigmoid = nn.Sigmoid()





    def forward(self, img_q, img_r1=None, img_r2=None, isTrain = True, isQuery = True):
        if isTrain and img_r2 is not None:
            q = self.q_net(img_q)
            q = self.fc_q(q)
            q = self.sigmoid(q)

            r1 = self.ref_net(img_r1)
            r1 = self.fc_r(r1)
            r1 = self.sigmoid(r1)

            r2 = self.ref_net(img_r2)
            r2 = self.fc_r(r2)
            r2 = self.sigmoid(r2)

            return q, r1, r2

            # return self.q_net(img_q), self.ref_net(img_r1), self.ref_net(img_r2) 
        else:
            if isQuery:
                q = self.q_net(img_q)
                q = self.fc_q(q)
                q = self.sigmoid(q)
                return q
            else:
                r1 = self.ref_net(img_q)
                r1 = self.fc_r(r1)
                r1 = self.sigmoid(r1)
                return r1

    



class ResNet2(nn.Module):
    def __init__(self, emb_dim):
        super(ResNet2, self).__init__()
        self.modelName = 'ResNet18'
        self.net = resnet18()



    def forward(self, img):
        return self.net(img)


# Define the ResNet model
class VIT(nn.Module):
    def __init__(self):
        super(VIT, self).__init__()
        self.modelName = 'VIT'
        self.vit_q = ViT(
            image_size = 256,
            patch_size = 32,
            num_classes = 512,
            dim = 1024,
            depth = 6,
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1
            )
        self.vit_r = ViT(
            image_size = 256,
            patch_size = 32,
            num_classes = 512,
            dim = 1024,
            depth = 6,
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1
            )



    def forward(self, q, r, isTrain = True, isQuery = True):
        if isTrain:
            return self.vit_q(q), self.vit_r(r)
        else:
            if isQuery:
                return self.vit_q(q)
            else:
                return self.vit_r(r)








# Define the ResNet model
# class VIT(nn.Module):
#     def __init__(self):
#         super(VIT, self).__init__()
#         self.modelName = 'VIT_B_16'
#         self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
#         # for param in self.vit.parameters():
#         #     param.requires_grad = False
#         # num_features = self.vit.heads
#         # self.vit.fc = nn.Linear(num_features, emb_dim)



#     def forward(self, x):
#         return self.vit(x)


# # Define the ResNet model
# class ResNet(nn.Module):
#     def __init__(self, emb_dim):
#         super(ResNet, self).__init__()
#         self.resnet = torchvision.models.resnet50(pretrained=True)
#         num_features = self.resnet.fc.in_features
#         self.resnet.fc = nn.Linear(num_features, emb_dim)

#     def forward(self, x):
#         return self.resnet(x)







