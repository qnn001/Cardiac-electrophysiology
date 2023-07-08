# /Users/anshumansinha/Desktop/Project_victor/StructRepGen_Dev/Victor_code/StructRepGen_Dev

import utils
from torch.utils.data import DataLoader
import torch
import datetime
import numpy as np
import utils
import data_dssi as ds
import matplotlib.pyplot as plt

# /Users/anshumansinha/DSSI/model.py

# Hyperparams
test_size = 0.1
batch_size = 64

data = ds.read_data()
ds_data = ds.Custom_dataset(data)

generator = torch.Generator().manual_seed(42)
train_ds,val_ds,test_ds = torch.utils.data.random_split(ds_data, [0.8, 0.15, 0.05], generator=generator)

# Create data loaders for our datasets; shuffle for training, not for validation
training_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
validation_loader = torch.utils.data.DataLoader(val_ds, batch_size=64, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=64, shuffle=False)

import math
import os
from tempfile import TemporaryDirectory
from typing import Tuple

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
import positional_encoder as pe
import time

class TransformerModel(nn.Module):

    def __init__(self,
        input_size: int,
        batch_first: bool,  
        dim_val: int, 
        n_heads: int, 
        n_encoder_layers: int,
        n_token: int, 
        dropout_pos_enc: float = 0.5, 
        dropout_encoder: float=0.2, 
        dropout_decoder: float=0.2,
        dim_feedforward_encoder: int=200,
        num_predicted_features: int=12):
        super().__init__()
        
        self.model_type = 'Transformer'
        self.linear = nn.Linear(dim_val, n_token)

        self.final = nn.Sequential(

            # Define the convolutional layer
            nn.Conv1d(1, 16, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool1d(3),

            nn.Conv1d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool1d(3),

            #nn.Linear(500, 256),
            #nn.ReLU(),
            #nn.Linear(256, 128),
            #nn.ReLU(),
            #nn.Linear(128, 75),
            #nn.ReLU()
        )

        self.final_lin = nn.Linear(1760, 75)

        # Creating the three linear layers needed for the model
        # [batch_size, src length, dim_val]
        self.encoder_input_layer = nn.Linear(
            in_features=input_size, 
            out_features=dim_val 
            )
        
        self.linear_mapping = nn.Linear(
            in_features=dim_val, 
            out_features=num_predicted_features
            )
        
        # Create positional encoder
        self.positional_encoding_layer = pe.PositionalEncoder(
            d_model=dim_val,
            dropout=dropout_pos_enc
            )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_val, 
            nhead=n_heads,
            dim_feedforward=dim_feedforward_encoder,
            dropout=dropout_encoder,
            batch_first=batch_first
            )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_encoder_layers, 
            norm=None
            )

    def forward(self, src: Tensor) -> Tensor:
        
        #print('src', src.shape) # torch.Size([2, 500, 12])
        #print(self.encoder_input_layer.weight.dtype) # torch.float32
        
        src = self.encoder_input_layer(src) # src shape: [batch_size, src length, dim_val] regardless of number of input features
        src = self.positional_encoding_layer(src) # src shape: [batch_size, src length, dim_val] regardless of number of input features
        src = self.encoder( # src shape: [batch_size, enc_seq_len, dim_val]
            src=src
            )
        output = self.linear(src) # torch.Size([2, 500, 1])
        output = output.reshape(-1, 500) # torch.Size([2, 500])

        #print('output_before', output.shape) #output_before torch.Size([2, 500])
        output = output.unsqueeze(1)
        #print('output_aft', output.shape)
        final = self.final(output) # torch.Size([2, 75])
        #print('output_shape',final.shape)
        final = final.reshape(-1,1760)
        final = self.final_lin(final)
        #print(final.shape)
        return final

def train(model: nn.Module) -> None:
    
    model.train()  # turn on train mode
    total_loss = 0.
    log_interval = 200
    start_time = time.time()
    print('epochs : ', epoch)

    for batch, i in training_loader:

        data, targets = batch, i # torch.Size([2, 500, 12])
        
        output = model(data) # torch.Size([2, 500, 10])
        print(targets.shape)

        targets = targets.reshape(-1, 75)
        #print('train_output shape',output.shape)

        loss = criterion(output, targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
    
    return total_loss/64 # batch_size

def evaluate(model: nn.Module, eval_data: Tensor) -> float:
    
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    batch_l = len(eval_data)
    with torch.no_grad():
        for batch,i in validation_loader:
            data, targets = batch, i
            #print(data.shape)
            #exit()
            output = model(data)

            #print('val_output shape',output.shape)

            targets = targets.reshape(-1, 75)
            total_loss += criterion(output, targets).item()
    return total_loss/batch_l

def evaluate_test(model: nn.Module) -> float:
    
    model.eval()  # turn on evaluation mode
    total_loss = 0.

    for i, v in training_loader: # replace with test_loader.
        x_dat, tar = i,v
        break
    
    #print('x_data',x_dat.shape) # ([500, 12])
    output = model(x_dat)
    
    #print('output',output.shape)
    #print('v_data',tar.shape) # ([75, 1])

    output = output[0]
    tar = tar.reshape(-1, 75)
    target = tar[0]

    #print('1',output.shape)
    #print('2',target.shape)
    #targets = v_dat.reshape(-1, 75)

    total_loss = criterion(output, target).item()

    print('-' * 89)
    print('targets.shape',target.shape)
    print('output.shape',output.shape)

    print('targets',target)
    print('output',output)

    print('loss', total_loss)
    print('-' * 89)

    plt.figure(figsize=(1, 10))

    #print(output.shape)
    output = output.view(75,-1)
    #print(output.shape)

    ActTime = output.detach().numpy()

    # plot the Activation Time array
    plt.imshow(ActTime, cmap='jet', interpolation='nearest', aspect='auto')
    plt.title('Activation Time')
    plt.colorbar()
    plt.grid(visible=True, which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    # not xticks
    plt.xticks([])
    plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 64
    train_data = train_ds #batchify(, batch_size)  # shape ``[seq_len, batch_size]``
    val_data = val_ds #batchify(, eval_batch_size)
    test_data = test_ds #batchify(, eval_batch_size)

    ntokens = 1 # len(vocab)  # size of vocabulary
    inp_s = 75  # embedding dimension
    d_hid = 75  # dimension of the feedforward network model in ``nn.TransformerEncoder``
    nlayers = 2  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
    nhead = 2  # number of heads in ``nn.MultiheadAttention``
    dropout = 0.2  # dropout probability

    model = TransformerModel(input_size = 12, batch_first = True, dim_val = 500, n_heads = 4, n_encoder_layers = 4, n_token = ntokens).to(device)

    criterion = torch.nn.MSELoss()
    lr = 5.0  # learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    best_val_loss = float('inf')
    epochs = 100
    t_loss = []
    v_loss = []
    #model = TransformerModel()

    with TemporaryDirectory() as tempdir:
        
        best_model_params_path = os.path.join(tempdir, "best_model_params.pt")

        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()
            #print('trainer is running wild')

            tr_loss = train(model)
            val_loss = evaluate(model, val_data)

            t_loss.append(tr_loss)
            v_loss.append(val_loss)

            elapsed = time.time() - epoch_start_time
            print('-' * 89)
            print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
                f'valid loss {val_loss:5.2f} | ' f'train loss {tr_loss:5.2f}' )
            print('-' * 89)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_params_path)

            scheduler.step()


        model.load_state_dict(torch.load(best_model_params_path)) # load best model states
        
        #test_1 = train_data[0]
        #print('test_1',test_1.shape)
        test_loss = evaluate_test(model)

        print('=' * 89)
        print(f'| End of training | test loss {test_loss:5.2f}')
        print('=' * 89)