import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights, vit_b_16, ViT_B_16_Weights
from vit_pytorch import ViT
from models.clip_b32 import getClipModel






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
        # for param in self.q_net.parameters():
        #     param.requires_grad = False
        # for param in self.ref_net.parameters():
        #     param.requires_grad = False
        self.resnet_output = self.q_net.fc.out_features
        # self.fc_q = nn.Linear(self.resnet_output, emb_dim)
        # self.fc_r = nn.Linear(self.resnet_output, emb_dim)
        # self.sigmoid = nn.Sigmoid()





    def forward(self, q, r, isTrain = True, isQuery = True):
        xq = self.q_net(q)
        # xq = self.fc_q(xq)
        # xq = torch.sigmoid(xq)

        xr = self.ref_net(r)
        # xr = self.fc_r(xr)
        # xr = torch.sigmoid(xr)
        
        if isTrain:
            # print(f'dukse train')
            return xq, xr
            # return self.query.encode_image(q), self.ref.encode_image(r)
        else:
            if isQuery:
                # print(f'dukse query')
                return xq
                # return self.query.encode_image(q)
            else:
                # print(f'dukse ref')
                return xr
                # return self.ref.encode_image(r)

    



class ResNet2(nn.Module):
    def __init__(self, emb_dim):
        super(ResNet2, self).__init__()
        self.modelName = 'ResNet18'
        self.net = resnet18()



    def forward(self, img):
        return self.net(img)


# Define the VIT model
class VIT(nn.Module):
    def __init__(self):
        super(VIT, self).__init__()
        self.modelName = 'VIT'
        self.query = ViT(
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
        self.ref = ViT(
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
            return self.query(q), self.ref(r)
        else:
            if isQuery:
                return self.query(q)
            else:
                return self.ref(r)






# # Define the CLIP model
# class CLIP_model(nn.Module):
#     def __init__(self):
#         super(CLIP_model, self).__init__()
#         self.modelName = 'CLIP'
#         self.query = getClipModel()
#         # for param in self.query.parameters():
#         #     param.requires_grad = False
#         self.ref = getClipModel()
#         # for param in self.ref.parameters():
#         #     param.requires_grad = False

#         self.query_fc1 = nn.Linear(512, 256) 
#         self.ref_fc1 = nn.Linear(512, 256) 

        



#     def forward(self, q, r, isTrain = True, isQuery = True):
#         xq = self.query.encode_image(q)
#         xq = self.query_fc1(xq)
#         xq = torch.sigmoid(xq)


#         xr = self.ref.encode_image(r)
#         xr = self.ref_fc1(xr)
#         xr = torch.sigmoid(xr)
        
#         if isTrain:
#             return xq, xr
#             # return self.query.encode_image(q), self.ref.encode_image(r)
#         else:
#             if isQuery:
#                 return xq
#                 # return self.query.encode_image(q)
#             else:
#                 return xr
#                 # return self.ref.encode_image(r)


# Define the Hugging face CLIP model
class CLIP_model(nn.Module):
    def __init__(self, embed_dim):
        super(CLIP_model, self).__init__()
        self.modelName = 'CLIP'
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.query = getClipModel()
        # for param in self.query.parameters():
        #     param.requires_grad = False
        self.ref = getClipModel()
        # for param in self.ref.parameters():
        #     param.requires_grad = False

        # self.norm_shape = self.query.vision_model.post_layernorm.normalized_shape[0]
        self.norm_shape = self.query.visual_projection.out_features

        self.query_fc1 = nn.Linear(self.norm_shape, embed_dim).to(device=self.device)
        self.query_fc2 = nn.Linear(embed_dim, 512).to(device=self.device)

        self.ref_fc1 = nn.Linear(self.norm_shape, embed_dim).to(device=self.device)
        self.ref_fc2 = nn.Linear(embed_dim, 512).to(device=self.device)


    def get_vision_embeddings(self, imgs, isQ):
        # Preprocess the images
        temp_dic = {'pixel_values':imgs}
        # Use the CLIP model to get vision embeddings
        
        # with torch.no_grad():
        if isQ:
            outputs = self.query(**temp_dic)
        else:
            outputs = self.ref(**temp_dic)
        # last_hidden_state = outputs.last_hidden_state
        # pooled_output = outputs.pooler_output  # pooled CLS states
        image_embeds = outputs.image_embeds

        return image_embeds


    def forward(self, q, r, isTrain = True, isQuery = True):
        xq = self.get_vision_embeddings(imgs = q, isQ = True )
        # xq = self.query_fc1(xq)
        # xq = torch.relu(xq)
        # xq = self.query_fc2(xq)
        # xq = torch.sigmoid(xq)

        xr = self.get_vision_embeddings(imgs = r, isQ = False )
        # xr = self.ref_fc1(xr)
        # xr = torch.relu(xr)
        # xr = self.ref_fc2(xr)
        # xr = torch.sigmoid(xr)
        
        if isTrain:
            # print(f'dukse train')
            return xq, xr
            # return self.query.encode_image(q), self.ref.encode_image(r)
        else:
            if isQuery:
                # print(f'dukse query')
                return xq
                # return self.query.encode_image(q)
            else:
                # print(f'dukse ref')
                return xr
                # return self.ref.encode_image(r)








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







