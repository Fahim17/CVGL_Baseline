import torch
import clip
# from transformers import CLIPModel
from transformers import CLIPVisionModel, CLIPVisionModelWithProjection


# OG CLIP
# def getClipModel():
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model, preprocess = clip.load("ViT-B/32", device=device)
#     for param in model.parameters():
#         param.requires_grad = False

#     return model


# HuggingFace CLIP
def getClipModel():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").to(device)
    # print(f'Model Device:{model.device}')
    # for param in model.parameters():
    #     param.requires_grad = False

    return model


# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device)

# image = preprocess(Image.open("CLIP.jpg")).unsqueeze(0).to(device)
# text = clip.tokenize(["a diagram", "a dog", "a stallion"]).to(device)

# with torch.no_grad():
#     image_features = model.encode_image(image)
#     text_features = model.encode_text(text)
    
#     logits_per_image, logits_per_text = model(image, text)
#     probs = logits_per_image.softmax(dim=-1).cpu().numpy()

# print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]