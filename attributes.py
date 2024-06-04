import torch
from helper_func import get_rand_id

class Configuration:
    # Model
    model_name: str = '--'
    expID = get_rand_id()
    embed_dim: int = 512
    save_weights = True

    # Training
    epochs: int = 100
    lr = 0.00001
    batch_size: int = 64

    # Data
    lang = 'T2' # T1, T2 or T3


    # Loss
    loss_margin = 1 # TripletMarginLoss
    label_smoothing=0.5 # Contrastive Loss

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"







