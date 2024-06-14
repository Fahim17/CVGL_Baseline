from datetime import datetime
import math
import pandas as pd
import torch
import csv
import os

def save_exp(emb_dim, loss_id, ln_rate, batch, epc, ls_mrgn, trn_sz, val_sz, mdl_nm, msg):

    filepath = f'logs/log_{loss_id}.txt'
    with open(filepath, 'w') as file:
        file.write(f'\nHyperparameter info: {datetime.now()}' + "\n\n")
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
    
        # df.to_string(file, index=True)


def write_to_file(expID, msg, content):

    filepath = f'logs/log_{expID}.txt'
    with open(filepath, 'a') as file:
        file.write(f'\n{msg}')
        file.write(f'{content}\n')
    



def write_to_rank_file(expID, step, row):
    # Check if the file exists
    file_path = f'rank/rank_{expID}.csv'
    file_exists = os.path.isfile(file_path)

    row = row.tolist()
    row.insert(0, step)
    
    # Open the file in append mode ('a'), if the file doesn't exist, it will be created
    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # If the file doesn't exist, you might want to write the header
        if not file_exists:
            # Assuming the first row of the data to be added is the header
            header = ["epoch", "top1", "top5", "top10", "top1%"]  # Modify this according to your header
            writer.writerow(header)
        
        # Write the row to the CSV file
        writer.writerow(row)




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
    
        