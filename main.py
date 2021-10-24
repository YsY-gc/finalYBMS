import torch
import torch.optim as optim
from train_support import * 

from models.YBMS import YBMS, build_model
from dataset import Data
import time

if __name__ == "__main__":

    print(time.asctime( time.localtime(time.time())))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # net = build_model().to(device)
    net = YBMS().to(device) 

 
    lr = 1e-4
    optimizer = optim.Adam(net.parameters(), lr=lr)
    num_epochs = 2000
    train_iter = Data().loader_train
    test_iter = Data().loader_test

    train_mu(net, train_iter, test_iter, optimizer, device, num_epochs)
