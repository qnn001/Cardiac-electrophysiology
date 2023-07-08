import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from torch import nn, Tensor
from typing import Optional, Any, Union, Callable, Tuple
import pandas as pd
from pathlib import Path
import glob, re, os
import cardiac_ml_tools as ct
import os
import pandas as pd

def read_data(data_dir: Union[str, Path] = "data"):

    data_dir = Path(data_dir)
    csv_files = list(data_dir.glob("*.csv"))

    data_dirs = []
    regex = r'data_hearts_dd_0p2*'
    DIR= './intracardiac_dataset/' # This should be the path to the intracardiac_dataset, it can be downloaded using data_science_challenge_2023/download_intracardiac_dataset.sh
    
    for x in os.listdir(DIR):
        if re.match(regex, x):
            data_dirs.append(DIR + x)
    
    file_pairs = ct.read_data_dirs(data_dirs)
    
    lf1 = np.load(file_pairs[0][0])
    lf1 = ct.get_standard_leads(lf1)

    lf2 = np.load(file_pairs[0][1])
    lf2 = ct.get_activation_time(lf2)

    data_book= []
    count = 0

    for lf1,lf2 in file_pairs:
        
        x1 = (np.load(lf1)) # (500,10)
        x1 = ct.get_standard_leads(x1) # (500,12)
        
        x2 = (np.load(lf2)) # (500, 75)
        x2 = ct.get_activation_time(x2) # (75,1)
        
        data_book.append((x1, x2))
        count+=1

        #if(count>5): # sanity check
        #    break
    
    df = pd.DataFrame(data_book, columns=['leads', 'act'])
    return df

class Custom_dataset(Dataset):
    
    def __init__(self, annotations_file):
        self.img_labels = annotations_file

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):

        x = self.img_labels.iloc[idx, 0]
        x = torch.tensor(x, dtype =torch.float32)
        y = self.img_labels.iloc[idx, 1]
        y = torch.tensor(y, dtype =torch.float32)
        return x, y

'''
if __name__ == "__main__":

    data = read_data()
    ds_data = Custom_dataset(data)
    generator = torch.Generator().manual_seed(42)
    train_ds,val_ds,test_ds = torch.utils.data.random_split(ds_data, [0.8, 0.15, 0.05], generator=generator)
    
    # Create data loaders for our datasets; shuffle for training, not for validation
    training_loader = torch.utils.data.DataLoader(train_ds, batch_size=4, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(val_ds, batch_size=4, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=4, shuffle=False)

    # Report split sizes
    print('Training set has {} instances'.format(len(train_ds)))
    print('Validation set has {} instances'.format(len(val_ds)))
    print('Test set has {} instances'.format(len(test_ds)))

    for i,j in training_loader:
        print('i : ',i)
        print('i : ',i.shape)
        print('j : ',j)
        print('j : ',j.shape)
        break
'''