import data_dssi as ds 
import numpy as np
import normal_data_t4 as ndt

import torch
import torch.nn as nn

# Define the convolutional layer
in_channels = 1  # Number of input channels
out_channels = 16  # Number of output channels
kernel_size = 3  # Size of the convolutional kernel
stride = 1  # Stride value for the convolution
padding = 1  # Padding value for the convolution

class Model_CNN(nn.Module):

    def __init__(self,time_dim):
        super().__init__()

        self.dim_time = time_dim

        self.module_up = nn.Sequential(
            nn.Conv1d(12,32,5,1,2),
            nn.ReLU(),

            nn.Conv1d(32,64,5,1,2),
            nn.ReLU(),

            nn.Conv1d(64,128,5,1,2),
            nn.ReLU(),

            nn.Conv1d(128,256,5,1,2),
            nn.ReLU(),

            nn.Conv1d(256,512,5,1,2),
            nn.ReLU()
        )

        self.module_middle = nn.Sequential(
            nn.Linear(512*self.dim_time,256),
            nn.ReLU(),

            nn.Linear(256,256),
            nn.ReLU(),

            nn.Linear(256,512*self.dim_time),
            nn.ReLU()
        )

        self.module_down = nn.Sequential(
            nn.Conv1d(512,256,5,1,2),
            nn.ReLU(),

            nn.Conv1d(256,128,5,1,2),
            nn.ReLU(),

            nn.Conv1d(128,75,5,1,2),
            nn.ReLU()
        )
    
    def forward(self, x,y):
        print(x.shape)
        x = x.permute(0,2,1) # add to device
        print(x.shape)
        z = self.module_up(x)
        return 0



if __name__ == '__main__':

    print('*******************')

    data = ndt.read_normal_data()
    print(data.head())
    data = ndt.Custom_dataset(data)

    #li_1 = data['leads'].tolist()
    #li_1 = data['act'].tolist()

    generator = torch.Generator().manual_seed(42)
    train_ds,val_ds,test_ds = torch.utils.data.random_split(data, [0.8, 0.15, 0.05], generator=generator)

    # Create data loaders for our datasets; shuffle for training, not for validation
    training_loader = torch.utils.data.DataLoader(train_ds, batch_size=2, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(val_ds, batch_size=2, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=2, shuffle=False)

    print(training_loader)

    for i,v in (training_loader):
        print(i[0].shape) # torch.Size([2, 500, 12])
        print(i[1].shape) # torch.Size([2, 500, 75])
        model = Model_CNN(500)
        ret = model(i,v)
        exit()




