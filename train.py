from tqdm import tqdm
import numpy as np
import time
from datetime import datetime


def time_stamp():
    now = datetime.now()
    print(f'\nDate: {now}\n')


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




# only for vit_pytorch

def train(model, criterion, optimizer, train_loader, num_epochs=10, dev='cpu'):
    model.train()


    epoch_loss = []

    time_stamp()
    for epoch in range(num_epochs):
        # total_loss = 0.0
        print(f'Epoch#{epoch}')
        running_loss = []
        for i, (anchor, positive, negative) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            anchor, positive, negative = anchor.to(dev), positive.to(dev), negative.to(dev)
            anchor_embedding, positive_embedding = model(q = anchor, r = positive, isTrain = True, isQuery = True)
            loss, mean_p, mean_n  = criterion(anchor_embedding, positive_embedding)
            loss.backward()
            optimizer.step()
            # total_loss += loss.item()
            running_loss.append(loss.cpu().detach().numpy())
            # print(f"Epoch {epoch+1}, Loss: {loss.item()}")
        print(f"Epoch: {epoch+1}/{num_epochs} Loss: {np.mean(running_loss)}")
        epoch_loss.append(np.mean(running_loss))
        # df_loss[epoch, "Loss"] = loss.cpu().detach().numpy()
    
    time_stamp()
    
    return epoch_loss