
import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
import random
from PIL import Image


class TrainSetLoader(Dataset):
    def __init__(self, dataset_dir):
        super(TrainSetLoader, self).__init__()
        self.dataset_dir = dataset_dir
        self.train_input1_dirs = os.listdir(self.dataset_dir+"\\traindata\\0710")
        self.train_input2_dirs = os.listdir(self.dataset_dir+"\\traindata\\1185")
        self.train_input3_dirs = os.listdir(self.dataset_dir+"\\traindata\\1685")
        self.train_input4_dirs = os.listdir(self.dataset_dir+"\\traindata\\2178")
        self.train_input5_dirs = os.listdir(self.dataset_dir+"\\traindata\\2682")
        self.train_input6_dirs = os.listdir(self.dataset_dir+"\\traindata\\3180")



    def __getitem__(self, index):
        input1 = Image.open(self.dataset_dir + '\\traindata\\0710\\' + self.train_input1_dirs[index]);input1 = np.array(input1);input1 = torch.from_numpy(input1).float()
        input2 = Image.open(self.dataset_dir + '\\traindata\\1185\\' + self.train_input2_dirs[index]);input2 = np.array(input2);input2 = torch.from_numpy(input2).float()
        input3 = Image.open(self.dataset_dir + '\\traindata\\1685\\' + self.train_input3_dirs[index]);input3 = np.array(input3);input3 = torch.from_numpy(input3).float()
        input4 = Image.open(self.dataset_dir + '\\traindata\\2178\\' + self.train_input4_dirs[index]);input4 = np.array(input4);input4 = torch.from_numpy(input4).float()
        input5 = Image.open(self.dataset_dir + '\\traindata\\2682\\' + self.train_input5_dirs[index]);input5 = np.array(input5);input5 = torch.from_numpy(input5).float()
        input6 = Image.open(self.dataset_dir + '\\traindata\\3180\\' + self.train_input6_dirs[index]);input6 = np.array(input6);input6 = torch.from_numpy(input6).float()
        input1 = input1.unsqueeze(0);input2 = input2.unsqueeze(0);input3 = input3.unsqueeze(0)
        input4 = input4.unsqueeze(0);input5 = input5.unsqueeze(0);input6 = input6.unsqueeze(0)


        return input1, \
                    input2, \
                        input3, \
                            input4, \
                                input5, \
                                    input6

    def __len__(self):
        return len(self.train_input1_dirs)


class TestSetLoader(Dataset):
    def __init__(self, dataset_dir):
        super(TestSetLoader, self).__init__()
        self.dataset_dir = dataset_dir
        self.test_input1_dirs = os.listdir(self.dataset_dir+"\\testdata\\0710")
        self.test_input2_dirs = os.listdir(self.dataset_dir+"\\testdata\\1185")
        self.test_input3_dirs = os.listdir(self.dataset_dir+"\\testdata\\1685")
        self.test_input4_dirs = os.listdir(self.dataset_dir+"\\testdata\\2178")
        self.test_input5_dirs = os.listdir(self.dataset_dir+"\\testdata\\2682")
        self.test_input6_dirs = os.listdir(self.dataset_dir+"\\testdata\\3180")


    def __getitem__(self, index):
        input1 = Image.open(self.dataset_dir + '\\testdata\\0710\\' + self.test_input1_dirs[index]);input1 = np.array(input1);input1 = torch.from_numpy(input1).float()
        input2 = Image.open(self.dataset_dir + '\\testdata\\1185\\' + self.test_input2_dirs[index]);input2 = np.array(input2);input2 = torch.from_numpy(input2).float()
        input3 = Image.open(self.dataset_dir + '\\testdata\\1685\\' + self.test_input3_dirs[index]);input3 = np.array(input3);input3 = torch.from_numpy(input3).float()
        input4 = Image.open(self.dataset_dir + '\\testdata\\2178\\' + self.test_input4_dirs[index]);input4 = np.array(input4);input4 = torch.from_numpy(input4).float()
        input5 = Image.open(self.dataset_dir + '\\testdata\\2682\\' + self.test_input5_dirs[index]);input5 = np.array(input5);input5 = torch.from_numpy(input5).float()
        input6 = Image.open(self.dataset_dir + '\\testdata\\3180\\' + self.test_input6_dirs[index]);input6 = np.array(input6);input6 = torch.from_numpy(input6).float()
        input1 = input1.unsqueeze(0);input2 = input2.unsqueeze(0);input3 = input3.unsqueeze(0)
        input4 = input4.unsqueeze(0);input5 = input5.unsqueeze(0);input6 = input6.unsqueeze(0)


        return input1, \
                    input2, \
                        input3, \
                            input4, \
                                input5, \
                                    input6

    def __len__(self):
        return len(self.test_input1_dirs)