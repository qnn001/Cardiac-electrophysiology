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
import torch

# import best_model_params_path
model_path = './save_model/model.pt'
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
 
        output = model(x_dat)
        output = output[0]
        tar = tar.reshape(-1, 75)
        target = tar[0]

        err = output-target
        err_list.append(err)
    
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

    data = ds.read_data()
    ds_data = ds.Custom_dataset(data)

    generator = torch.Generator().manual_seed(42)
    train_ds,val_ds,test_ds = torch.utils.data.random_split(ds_data, [0.8, 0.15, 0.05], generator=generator)

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


