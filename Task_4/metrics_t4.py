import utils
from torch.utils.data import DataLoader
import torch
import datetime
import numpy as np
import utils

import data_dssi_t4 as ds
import matplotlib.pyplot as plt
from task_4 import Model_CNN

from torch import nn, Tensor
from typing import Optional, Any, Union, Callable, Tuple
import normal_data_t4 as ndt
import joblib
import torch

scaler_path = './save_model_t_4/'

path1 = scaler_path + 'leads_scaler_'
path2 = scaler_path + 'act_scaler_'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate_test(model: nn.Module, training_loader) -> float:
    
    model.eval()  # turn on evaluation mode
    err_list = []
    for i, v in training_loader: # replace with test_loader.
        
        x_dat, tar = i,v
        print(tar.shape) # torch.Size([5, 500, 75])
        num_sub = (tar.shape[0])
        tar= tar.reshape(num_sub,1,-1)
        print(tar.shape)
        tar = np.vstack(tar)
        print(tar.shape) # (5, 37500)
        scaler = joblib.load(path2)
        normalized_data = scaler.inverse_transform(tar)
        tar = np.vsplit(normalized_data, num_sub)
        print(tar[0].shape)
        tar = [i.reshape(75,-1) for i in tar]
        print(tar[0].shape) # (75, 500)

 
        output = model(x_dat)
        print('1', output.shape)
        output= output.reshape(num_sub,1,-1)
        #num_sub = len(output)
        output = output.detach().numpy()
        output = np.vstack(output)
        print('2', output.shape) # 3 (5, 37500)
        scaler = joblib.load(path2)
        normalized_data = scaler.inverse_transform(output)
        print('3', normalized_data.shape) # 3 (5, 37500)

        output = np.vsplit(normalized_data, num_sub)
        output = [i.reshape(75,-1) for i in output] # print(tar[0].shape) # (75, 500)

        for i in range(len(output)):
            err = np.abs(output[i]-tar[i])
            print('err',err.shape)
            err_list.append(torch.tensor(err))
    
    #print(len(err_list))
    #print(err_list)
    
    return err_list

class metrics(object):
    def __init__(self,err_list):
        self.err_l = err_list
        self.mean_l = []
        self.std_l = []

    def process(self):
        
        for i in self.err_l:
            print('i',i.shape) # i torch.Size([75, 500])
            print('torch.mean(i)',torch.mean(i).shape)
            self.mean_l.append(torch.mean(i))
            exit()
            self.std_l.append(torch.std(i))
        
        mean_final = torch.mean(torch.tensor(self.mean_l))
        std_final = torch.mean(torch.tensor(self.std_l))

        return mean_final,std_final

# /Users/anshumansinha/DSSI/metrics_t4.py

def plot_input(df, case):

    # ECG plot
    row = 3 
    column = 4
    num_timesteps = 500
    plt.figure(figsize=(10, 7))
    titles = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    reorder = {1:1,2:5,3:9,4:2,5:6,6:10,7:3,8:7,9:11,10:4,11:8,12:12} # reorder the leads to standard 12-lead ECG display

    #print('Case {} : {}'.format(case, file_pairs[case][0]))
    pECGData = df.iloc[case][0]
    #pECGData = np.load(file_pairs[case][0]) # 500 x 10
    #pECGData = get_standard_leads(pECGData)

    # create a figure with 12 subplots
    for i in range(pECGData.shape[1]):
        plt.subplot(row, column, reorder[i + 1])
        plt.plot(pECGData[0:num_timesteps,i],'r')
        plt.title(titles[i])
        plt.grid(visible=True, which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.xlabel('msec')
        plt.ylabel('mV')
    plt.tight_layout()
    plt.savefig('Input.pdf')
    
    return None

    
def plot_output(VmData, output):

    # Vm plot
    row = 7
    column = 10
    num_timesteps = 500
    plt.figure(figsize=(18, 9))


    for count, i in enumerate(range(VmData.shape[1])):
        plt.subplot(8, 10, count + 1)
        plt.plot(VmData[0:num_timesteps,i])
        plt.title(f'i = {i}')
        plt.grid(visible=True, which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        # plt.xlabel('msec')
        # plt.ylabel('mV')
    plt.tight_layout()
    plt.savefig(output+'.pdf')
    
    return None
    

if __name__ == "__main__":

    print('*******************')

    data = ndt.read_normal_data()
    data = ndt.Custom_dataset(data)

    print(len(data))
    generator = torch.Generator().manual_seed(42)

    print('alpha')
    train_ds,val_ds,test_ds = torch.utils.data.random_split(data, [0.8, 0.15, 0.05], generator=generator)
    
    batch_size = 16
    training_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    # Define the model
    model_path = './save_model_t_4/trained_model.pt'
    model = Model_CNN()
    model.load_state_dict(torch.load(model_path)) # load best model states
    
    test_loss = evaluate_test(model, test_loader)

    criterion = torch.nn.MSELoss()
    met = metrics(test_loss)
    m1,s1 = met.process()

    print('mean error time : ', m1)
    print('std error time : ',s1)


