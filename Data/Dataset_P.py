from torch.utils.data import Dataset, DataLoader
import os
import torchvision.transforms as transforms
import torch
import h5py
import numpy as np
import pickle



class pickle_dataset(Dataset):
    def __init__(self, datapath='', ID=0, over_lap=0, layback=0, seq_len=80, increase_L=0,increase_R=0,Y_X=True):
        self.datapath = datapath
        self.ID = ID
        self.over_lap = over_lap
        self.layback = layback
        self.seq_len = seq_len
        self.increase_L = increase_L
        self.increase_R = increase_R
        self.clips_x = []
        self.clips_y = []
        self.Y_X = Y_X
        f1 = open(os.path.join(datapath,str(self.ID)), 'rb')
        self.data = pickle.load(f1)
        f1.close()
        assert len(self.data[0])==len(self.data[1])
        self.time_clips(self.seq_len,self.over_lap,self.layback)
        f1 = open('./trainMeanFace', 'rb')
        self.meanFace = pickle.load(f1)[self.ID]
        f1.close()

    def time_clips(self, seq_len, over_lap, layback):
        LEN = len(self.data[0])
        for i in range(0+self.increase_L,LEN,seq_len - over_lap):
            if i + seq_len+layback + self.increase_R >= LEN:
                break
            if self.Y_X:
                self.clips_y.append((i+layback-self.increase_L, i + seq_len+layback + self.increase_R))
                self.clips_x.append((i, i + seq_len))
            else:
                self.clips_x.append((i+layback-self.increase_L, i + seq_len+layback + self.increase_R))
                self.clips_y.append((i, i + seq_len))

    def __len__(self):
        return len(self.clips_x)

    def extractModal(self,frames,modal):
        if modal=='FL':
            return np.array([frame.FL for frame in frames])
        if modal=='MFCC':
            return np.array([frame.MFCC for frame in frames])
        if modal=='MEAN':
            return np.array(frames)
        if modal=='CLASS':
            return np.array([frame.CLASS for frame in frames])
        if modal=='IDENTITY':
            return np.array([frame.IDENTITY for frame in frames])

    def __getitem__(self, item):
        start = self.clips_x[item][0]
        end = self.clips_x[item][1]
        clips_x_FL = self.extractModal([self.data[0][i] for i in range(start,end)],'FL').reshape([end - start, -1])
        clips_x_FL = torch.tensor(clips_x_FL,dtype=torch.float32)
        clips_x_MFCC = self.extractModal([self.data[0][i] for i in range(start,end)],'MFCC').reshape([end - start, -1])
        clips_x_MFCC = torch.tensor(clips_x_MFCC, dtype=torch.float32)
        XmeanFace = self.extractModal([self.meanFace[0] for i in range(start, end)], 'MEAN').reshape([end - start, -1])
        XmeanFace = torch.tensor(XmeanFace, dtype=torch.float32)
        start = self.clips_y[item][0]
        end = self.clips_y[item][1]
        clips_y_FL = self.extractModal([self.data[1][i] for i in range(start, end)], 'FL').reshape([end - start, -1])
        clips_y_FL = torch.tensor(clips_y_FL,dtype=torch.float32)
        clips_y_MFCC = self.extractModal([self.data[1][i] for i in range(start, end)], 'MFCC').reshape([end - start, -1])
        clips_y_MFCC = torch.tensor(clips_y_MFCC,dtype=torch.float32)
        YmeanFace = self.extractModal([self.meanFace[1] for i in range(start, end)], 'MEAN').reshape([end - start, -1])
        YmeanFace = torch.tensor(YmeanFace, dtype=torch.float32)

        return clips_x_FL.transpose(1, 0),clips_y_FL.transpose(1, 0),clips_x_MFCC.transpose(1, 0),clips_y_MFCC.transpose(1, 0),XmeanFace.transpose(1, 0),YmeanFace.transpose(1, 0)
