import torch
import clip
# from transformers import CLIPModel
from transformers import CLIPVisionModel, CLIPVisionModelWithProjection, CLIPTextModelWithProjection, CLIPModel
from attributes import Configuration as hypm

# OG CLIP
# def getClipModel():
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model, preprocess = clip.load("ViT-B/32", device=device)
#     for param in model.parameters():
#         param.requires_grad = False

#     return model


# HuggingFace CLIP
def getClipVisionModel():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    # model_v = CLIPModel.from_pretrained(hypm.v_pretrain_weight).to(hypm.device)
    model_v = CLIPVisionModelWithProjection.from_pretrained(hypm.v_pretrain_weight).to(hypm.device)
    # print(f'Model Device:{model_v.device}')
    for param in model_v.parameters():
        param.requires_grad = False

    return model_v


def getClipTextModel():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_t = CLIPTextModelWithProjection.from_pretrained(hypm.t_pretrain_weight).to(hypm.device)
    # print(f'Model Device:{model_t.device}')
    for param in model_t.parameters():
        param.requires_grad = False

    return model_t



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