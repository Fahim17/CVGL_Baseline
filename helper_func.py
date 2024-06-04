from datetime import datetime
import math
import pandas as pd
import torch

def save_losses(df, emb_dim, loss_id, ln_rate, batch, epc, ls_mrgn, trn_sz, val_sz, mdl_nm, rslt, msg):

    filepath = f'logs/losses_{loss_id}.txt'
    with open(filepath, 'w') as file:
        file.write('\nHyperparameter info:' + "\n\n")
        file.write(f'Message: {msg}\n')
        file.write(f'Exp ID: {loss_id}\n')
        file.write(f'Embedded dimension: {emb_dim}\n')
        file.write(f'Learning rate: {ln_rate}\n')
        file.write(f'Batch Size: {batch}\n')
        file.write(f'Loss Margin: {ls_mrgn}\n')
        file.write(f'Epoch: {epc}\n')
        file.write(f'Training Size: {trn_sz}\n')
        file.write(f'Validation Size: {val_sz}\n')
        file.write(f'Model Name: {mdl_nm}\n')
        file.write('\n\n')
        file.write(f'Result: {rslt}\n')
    
        df.to_string(file, index=False)

# Example usage:
# Assuming you have a DataFrame named 'df' and you want to save it to 'data.txt'
# save_dataframe_to_txt(df, 'data.txt')


def hyparam_info(emb_dim, loss_id, ln_rate, batch, epc, ls_mrgn, trn_sz, val_sz, mdl_nm):
    print('\nHyperparameter info:')
    print(f'Exp ID: {loss_id}')
    print(f'Embedded dimension: {emb_dim}')
    print(f'Learning rate: {ln_rate}')
    print(f'Batch Size: {batch}')
    print(f'Loss Margin: {ls_mrgn}')
    print(f'Epoch: {epc}')
    print(f'Training Size: {trn_sz}')
    print(f'Validation Size: {val_sz}')
    print(f'Model Name: {mdl_nm}')
    print('\n')

def get_rand_id():
    dt = datetime.now()
    return f"{math.floor(dt.timestamp())}"[3:]



def save_tensor(var_name,  var):
    torch.save(var, f'logs/save_in/{var_name}.pt')



# def save_weights(mdl, pth = 'model_weights/'):
#     print("Model's state_dict:")
#     for param_tensor in mdl.state_dict():
#         print(param_tensor, "\t", mdl.state_dict()[param_tensor].size())

#     # Save the model's state_dict to a text file
#     state_dict = mdl.state_dict()

#     # Convert the state_dict to a human-readable format
#     formatted_state_dict = {k: v.tolist() for k, v in state_dict.items()}

#     # Write the formatted state_dict to a text file
#     with open(f"{pth}model_weights.txt", "w") as f:
#         for key, value in formatted_state_dict.items():
#             f.write(f"{key}: {value}\n")

#     print("Model weights have been saved to model_weights.txt")
    
        