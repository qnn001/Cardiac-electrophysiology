import numpy as np
import matplotlib.pyplot as plt
import data_dssi_t4 as ds
import normal_data_t4 as ndt
import metrics as mt

import torch
import torch.nn as nn


class Model_CNN(nn.Module):

    def __init__(self, time_dim = 500):
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

#         self.module_middle = nn.Sequential(
#             nn.Linear(512*self.dim_time,256),
#             nn.ReLU(),

#             nn.Linear(256,256),
#             nn.ReLU(),

#             nn.Linear(256,512*self.dim_time),
#             nn.ReLU()
#         )

        self.module_down = nn.Sequential(
            nn.Conv1d(512,256,5,1,2),
            nn.ReLU(),

            nn.Conv1d(256,128,5,1,2),
            nn.ReLU(),

            nn.Conv1d(128,75,5,1,2),
            nn.ReLU()
        )
    
    def forward(self, x):
#         print('Raw input:', x.shape)
        x = x.permute(0,2,1) # add to device
#         print('Permuted input:', x.shape)
        
        z = self.module_up(x)
#         print('Output 1:', z.shape)
        _, w, h = z.shape
#         z = z.reshape(-1, w*h)
#         print('Output 1 reshaped:', z.shape)
#         z = self.module_middle(z)
#         print('Output 2:', z.shape)
        
#         z = z.reshape(-1, w, h)
#         print('Output 2 reshaped:', z.shape)
        out = self.module_down(z)
        out = out.permute(0, 2, 1) # torch.Size([2, 500, 75])
#         print('Output 3:', out.shape)
        
        return out


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Trainer():
    
    def __init__(self, config):
        self.config = config
        pass
    
    def train(self, data_loader, model):
        total_loss = []
        model.train()
        for i, v in data_loader:
            target = v
            result = model(i)
            loss = criterion(result, target)
            
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
            total_loss.append(loss.item())
            
        return total_loss
            
    
    def evaluate(self, data_loader, model):
        total_loss = []
        model.eval()
        with torch.no_grad():
            for i, v in data_loader:
                target = v
                result = model(i)

                total_loss.append(criterion(result, target).item())
            
        return total_loss



if __name__ == '__main__':

    print('*******************')

#     orig_data = ds.read_data()
#     mt.plot_input(orig_data, 21)
    data = ndt.read_normal_data()
#     print(data.head())
    data = ndt.Custom_dataset(data)

    generator = torch.Generator().manual_seed(42)
    train_ds,val_ds,test_ds = torch.utils.data.random_split(data, [0.8, 0.15, 0.05], generator=generator)

    # Create data loaders for our datasets; shuffle for training, not for validation
    batch_size = 16
    training_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    # Define the model
    model_path = './save_model_t_4/trained_model.pt'
    model = Model_CNN()
    print('Model parameters:', count_params(model)) # 1 mil
    criterion = nn.MSELoss()
    lr = 0.05
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    # Train & evaluate the model
    epochs = 3
    train_loss = []
    eval_loss = []
    
    config = []
    trainer = Trainer(config)
    for ep in range(1, epochs+1):
        print('Epoch:', ep)
        train_loss_0 = trainer.train(training_loader, model)
        train_loss.append(np.mean(train_loss_0))
    
        # evaluate
        eval_loss_0 = trainer.evaluate(validation_loader, model)
        eval_loss.append(np.mean(eval_loss_0))
    torch.save(model.state_dict(), model_path)
    
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.plot(eval_loss)
    plt.savefig('Loss.pdf')
    
    # Plot the output
    model.eval()
    with torch.no_grad():
        for i, v in test_loader:
            target = v
            print('Target:', target.shape)
            result = model(i)
            print('Result:', result.shape)
            break
    case = 3
    mt.plot_output(result[case], 'Result')
    mt.plot_output(target[case], 'Target')




