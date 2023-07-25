import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the input tensor
#output_before torch.Size([2, 500]) [2,1,1,500]
input_dim = (500, 1, 1) # batch, rows (layers), col
input_tensor = torch.randn(input_dim)

# Define the convolutional layer
in_channels = 1  # Number of input channels
out_channels = 16  # Number of output channels
kernel_size = 3  # Size of the convolutional kernel
stride = 1  # Stride value for the convolution
padding = 1  # Padding value for the convolution

conv1d_1 = nn.Conv1d(1, 16, 3, 1, 1)
conv1d_2 = nn.Conv1d(16, 32, 3, 1, 1)
lin_1 = nn.Linear(1760, 75)
maxpool = nn.MaxPool1d(3)

a = input_tensor.permute(1, 2, 0) # torch.Size([1, 1, 500])


# Apply the convolutional layer to the input tensor
output_tensor = conv1d_1(input_tensor.permute(1, 2, 0))  # Permute the dimensions for Conv1d input
x = F.relu(output_tensor)
x = maxpool(x)
output_tensor = conv1d_2(x)
x = F.relu(output_tensor)
x = maxpool(x)
x = x.flatten()
x = lin_1(x)
