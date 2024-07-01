import torch
from helper_func import get_rand_id

# openai/clip-vit-base-patch32
# openai/clip-vit-large-patch14
class Configuration:
    # Model
    model_name: str = '--'
    v_pretrain_weight: str = 'openai/clip-vit-large-patch14'
    t_pretrain_weight: str = 'openai/clip-vit-large-patch14'
    expID = -1
    embed_dim: int = 768
    save_weights = True

    # Training
    epochs: int = 50
    lr = 0.00001
    batch_size: int = 64
    lang_with: str = 'sat' # 'sat' or 'gnd'
    train_eval_per_epoch = 2

    # Data
    lang = 'T1' # T1, T2 or T3


    # Loss
    loss_margin = 1 # TripletMarginLoss
    label_smoothing=0.5 # Contrastive Loss

    # Device
    torch.cuda.set_device(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #others
    msg: str = f'{lang} Text with {lang_with}'







