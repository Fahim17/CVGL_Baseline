from datetime import datetime
import math
import pandas as pd

def save_losses(df, emb_dim, loss_id, ln_rate, batch, epc, ls_mrgn, trn_sz, mdl_nm):

    filepath = f'logs/losses_{loss_id}.txt'
    with open(filepath, 'w') as file:
        file.write('\nHyperparameter info:' + "\n\n")
        file.write(f'Exp ID: {loss_id}\n')
        file.write(f'Embedded dimension: {emb_dim}\n')
        file.write(f'Learning rate: {ln_rate}\n')
        file.write(f'Batch Size: {batch}\n')
        file.write(f'Loss Margin: {ls_mrgn}\n')
        file.write(f'Epoch: {epc}\n')
        file.write(f'Training Size: {trn_sz}\n')
        file.write(f'Model Name: {mdl_nm}\n')
        file.write('\n\n')
    
        df.to_string(file, index=False)

# Example usage:
# Assuming you have a DataFrame named 'df' and you want to save it to 'data.txt'
# save_dataframe_to_txt(df, 'data.txt')


def hyparam_info(emb_dim, loss_id, ln_rate, batch, epc, ls_mrgn, trn_sz, mdl_nm):
    print('\nHyperparameter info:')
    print(f'Exp ID: {loss_id}')
    print(f'Embedded dimension: {emb_dim}')
    print(f'Learning rate: {ln_rate}')
    print(f'Batch Size: {batch}')
    print(f'Loss Margin: {ls_mrgn}')
    print(f'Epoch: {epc}')
    print(f'Training Size: {trn_sz}')
    print(f'Model Name: {mdl_nm}')
    print('\n')

def get_rand_id():
    dt = datetime.now()
    return f"{math.floor(dt.timestamp())}"[3:]