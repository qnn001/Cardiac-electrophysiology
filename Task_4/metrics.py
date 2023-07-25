import utils
from torch.utils.data import DataLoader
import torch
import datetime
import numpy as np
import utils
import data_dssi as ds
import matplotlib.pyplot as plt
from model_t_cnn import TransformerModel 
from torch import nn, Tensor
from typing import Optional, Any, Union, Callable, Tuple
import normal_data_dssi as nds
import joblib
import torch

# import best_model_params_path
model_path = './save_model/model.pt'
scaler_path = 'save_model/act_scaler_mp.gz'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 64
ntokens = 1 # len(vocab)  # size of vocabulary
inp_s = 75  # embedding dimension
d_hid = 75  # dimension of the feedforward network model in ``nn.TransformerEncoder``
nlayers = 2  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
nhead = 2  # number of heads in ``nn.MultiheadAttention``
dropout = 0.2  # dropout probability

def evaluate_test(model: nn.Module, training_loader) -> float:
    
    model.eval()  # turn on evaluation mode
    err_list = []

    for i, v in training_loader: # replace with test_loader.
        
        x_dat, tar = i,v

        num_sub = len(tar)
        tar = np.vstack(tar)
        scaler = joblib.load(scaler_path)
        normalized_data = scaler.inverse_transform(tar)
        tar = np.vsplit(normalized_data, num_sub)
        tar = [i.reshape(75,-1) for i in tar]
 
        output = model(x_dat)

        num_sub = len(output)
        output = output.detach().numpy()
        output = np.vstack(output)
        scaler = joblib.load(scaler_path)
        normalized_data = scaler.inverse_transform(output)
        output = np.vsplit(normalized_data, num_sub)
        output = [i.reshape(75,-1) for i in output]

        for i in range(len(output)):
            err = output[i]-tar[i]
            err_list.append(torch.tensor(err))
    
    print(len(err_list))
    print(err_list)
    
    return err_list

class metrics(object):
    def __init__(self,err_list):
        self.err_l = err_list
        self.mean_l = []
        self.std_l = []

    def process(self):
        
        for i in self.err_l:
            self.mean_l.append(torch.mean(i))
            self.std_l.append(torch.std(i))
        
        mean_final = torch.mean(torch.tensor(self.mean_l))
        std_final = torch.mean(torch.tensor(self.std_l))

        return mean_final,std_final
    
if __name__ == "__main__":

    # Hyperparams
    test_size = 0.1
    batch_size = 64
    

    #data = ds.read_data()
    data = nds.read_normal_data()
    ds_data = ds.Custom_dataset(data)

    generator = torch.Generator().manual_seed(42)
    train_ds,val_ds,test_ds = torch.utils.data.random_split(ds_data, [0.8, 0.15, 0.05], generator=generator)

    #/Users/anshumansinha/DSSI/metrics.py
    # Create data loaders for our datasets; shuffle for training, not for validation
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=64, shuffle=False)

    # change according to the model
    model = TransformerModel(input_size = 12, batch_first = True, dim_val = 500, n_heads = 4, n_encoder_layers = 4, n_token = ntokens).to(device)
    
    model.load_state_dict(torch.load(model_path)) # load best model states
    
    test_loss = evaluate_test(model, test_loader)

    criterion = torch.nn.MSELoss()
    met = metrics(test_loss)
    m1,s1 = met.process()

    print('mean error time : ', m1)
    print('std error time : ',s1)


