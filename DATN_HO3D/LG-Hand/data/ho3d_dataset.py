import torch
import os
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
import random
seed = 1 
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

connections = [(0,1,"red"), (1,6, "red"), (6,11, "red"), (11,16,"red"),
           (0,2,"yellow"), (2,7,"yellow"), (7,12,"yellow"), (12,17,"yellow"),
           (0,3,"green"), (3,8,"green"), (8,13,"green"), (13,18,"green"),
           (0,4, "blue"), (4,9, "blue"), (9,14, "blue"), (14,19, "blue"),
           (0,5,"purple"), (5,10,"purple"), (10,15,"purple"), (15,20,"purple")]

class HO3dDataset(data.Dataset):
    def __init__(self, phase="train", pad=1):
        self.phase = phase
        self.path_2d = "/home/lexuantu/DATN_HO3D/HO3Ddataset/points2d-train.npy"
        self.path_3d = "/home/lexuantu/DATN_HO3D/HO3Ddataset/points3d-train.npy"

        # self.path_2d = "points2d-train.npy"
        # self.path_3d = "points3d-train.npy"

        
        self.data_2d_all = torch.from_numpy(np.load(self.path_2d)[:,0:21,:])            #shape=(N, 29, 2)
        self.data_3d_all = torch.from_numpy(np.load(self.path_3d)[:,0:21,:])            #shape=(N, 29, 3)

        if self.phase == "train":
            self.data_2d = self.data_2d_all[0:int(len(self.data_2d_all)/100)*80,:,:]
            self.data_3d = self.data_3d_all[0:int(len(self.data_3d_all)/100)*80,:,:]
        else:
            self.data_2d = self.data_2d_all[int(len(self.data_2d_all)/100)*80:,:,:]
            self.data_3d = self.data_3d_all[int(len(self.data_3d_all)/100)*80:,:,:]

        self.pad = pad
        self.len_seq = pad*2 + 1

    def __len__(self):
        if self.phase == "train":
            return len(self.data_2d)        #65278
        elif self.phase == "val":
            return len(self.data_2d)        #756
        else:
            return len(self.data_2d)        #11524
    
    def __getitem__(self, index):

        sample2d = self.data_2d[index]
        sample3d = self.data_3d[index]

        start_frame = index - self.pad
        end_frame = index + self.pad

        
        if start_frame < 0:
            tensor_pad_2d = sample2d.view(1, 21, -1)
            tensor_pad_3d = sample3d.view(1, 21, -1)

            seq_sample2d = torch.cat((tensor_pad_2d, self.data_2d[index:end_frame + 1]), dim=0)
            seq_sample3d = torch.cat((tensor_pad_3d, self.data_3d[index:end_frame + 1]), dim=0)
        elif end_frame >= len(self.data_2d):
            tensor_pad_2d = sample2d.view(1, 21, -1)
            tensor_pad_3d = sample3d.view(1, 21, -1)

            seq_sample2d = torch.cat((self.data_2d[start_frame:index+1], tensor_pad_2d), dim=0)
            seq_sample3d = torch.cat((self.data_3d[start_frame:index+1], tensor_pad_3d), dim=0)
        else:
            seq_sample2d = self.data_2d[start_frame:end_frame + 1]
            seq_sample3d = self.data_3d[start_frame:end_frame + 1]
        return seq_sample2d, seq_sample3d        



train_dataset = HO3dDataset("train", pad=1)
test_dataset = HO3dDataset("val", pad=1)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=256,
                                                shuffle=False)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=256,
                                                shuffle=False)
