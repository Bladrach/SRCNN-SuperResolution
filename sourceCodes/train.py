#import glob
#import os
from timeit import default_timer as timer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from custom_dataset import MyDataset
import model
import utilityFunc


# Learning parameters
batch_size = 64
max_epoch = 200
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Dataset variables
data_path = "C:\\Users\\Mehmet\\Desktop\\HighResPyTorch\\dataset64_51200_4\\data"
label_path = "C:\\Users\\Mehmet\\Desktop\\HighResPyTorch\\dataset64_51200_4\\label"

# Output files
netname = "net"

# Initialize the dataset and dataloader
traindataset = MyDataset(data_path = data_path, label_path = label_path, train = True)
trainloader = DataLoader(traindataset, batch_size = batch_size, shuffle = True, pin_memory = True, num_workers = 0)

testdataset = MyDataset(data_path = data_path, label_path = label_path, train = False)
testloader = DataLoader(testdataset, batch_size = 1, shuffle = True, pin_memory = True, num_workers = 0)

# Initialize the NN model
net = model.Net()
net = net.float()
net = net.to(device)

# Optimizer and Loss function
criterion = nn.MSELoss()
optimizer = optim.Adam(
    [
        {"params": net.conv1.parameters(), "lr": 1e-4},  
        {"params": net.conv2.parameters(), "lr": 1e-4},
        {"params": net.conv3.parameters(), "lr": 1e-5},
    ]
)

start = timer()  # start the timer
# Training
if __name__ == "__main__":
    for epoch in range(max_epoch + 1):
        batchiter = 0

        for batch in trainloader:
        
            batchiter += 1
            input_ = batch[0].unsqueeze(1).to(device)
            input_ = input_.permute(0, 1, 3, 2)
            label = batch[1].unsqueeze(1).to(device)
            label = label.permute(0, 1, 3, 2)
            y_pred = net(input_)   
            optimizer.zero_grad()    
            loss = criterion(y_pred.float(), label.float())
            loss.backward()
            optimizer.step()
            
            print("TRAIN","Epoch:",epoch+1, "Data-Num:",batchiter, "Loss:",loss.item())

        if epoch % 1 == 0:
            torch.save(net.state_dict(), "./saved_models_CROP64_51200_4/" + netname + "_epoch_%d"%(epoch) + ".pth")
        

    end = timer()  # end the timer
    elapsed_time = (end - start)/60  # elapsed time is calculated

    print('Elapsed time for training: {:.3f} minutes!'.format(float(elapsed_time)))
