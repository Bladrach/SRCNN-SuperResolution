import torch
from torch.utils.data import Dataset
import glob
import numpy as np
import utilityFunc


class MyDataset(Dataset):
    def __init__(self, data_path, label_path, train = True):
        if(train == True):
            self.data_path = glob.glob(data_path + "\\train_data\\*.png")
            self.label_path = glob.glob(label_path + "\\train_label\\*.png")
        else:
            self.data_path = glob.glob(data_path + "\\test_data\\*.png")
            self.label_path = glob.glob(label_path + "\\test_label\\*.png")
    

    def __getitem__(self, index):
        imgName = self.data_path[index]
        labelName = self.label_path[index]
        img_y, label_y = utilityFunc.prepareData(imgName, labelName)
        return img_y, label_y

    def __len__(self):
        return len(self.data_path)
