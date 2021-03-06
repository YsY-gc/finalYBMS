
import re
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from models.YBMS import *
from models.YBMS import AS
from models.YBMS5_2 import YBMS5_2
from PIL import Image
import os

def cal_psnr(y_hat,y):
    y_hat = y_hat.squeeze(0).squeeze(0)
    p1 = y_hat
    y = y.squeeze(0).squeeze(0)
    p2 = y

    diff = p1-p2
    diff *= diff
    mse = torch.mean(diff)
    psnr = 10*torch.log10(65535.0/mse)
    return psnr

def np_cal_psnr(y_hat,y):

    p1 = y_hat

    p2 = y

    diff = p1-p2
    diff *= diff
    mse = np.mean(diff)
    psnr = 10*np.log10(65535.0/mse)
    return psnr


# net = build_model(size=(768,768)).to('cuda')
# net = YBMS(size=(768,768)).to('cuda')
net = YBMS5_2(size=(768,768)).to('cuda')
net.load_state_dict(torch.load('I:\\finalYBMS\\params\\params23_2\\bestparams.pth'))
net.eval()


def test_param(file_name):

    input1 = Image.open("E:\\finallYBMS\\DATASETSU\\testdata\\0710\\1.bmp").convert('L');input1 = np.array(input1);input1 = torch.from_numpy(input1).float()
    input2 = Image.open("E:\\finallYBMS\\DATASETSU\\testdata\\1185\\1.bmp").convert('L');input2 = np.array(input2);input2 = torch.from_numpy(input2).float()
    input3 = Image.open("E:\\finallYBMS\\DATASETSU\\testdata\\1685\\1.bmp").convert('L');input3 = np.array(input3);input3 = torch.from_numpy(input3).float()
    input4 = Image.open("E:\\finallYBMS\\DATASETSU\\testdata\\2178\\1.bmp").convert('L');input4 = np.array(input4);input4 = torch.from_numpy(input4).float()
    input5 = Image.open("E:\\finallYBMS\\DATASETSU\\testdata\\2682\\1.bmp").convert('L');input5 = np.array(input5);input5 = torch.from_numpy(input5).float()
    input6 = Image.open("E:\\finallYBMS\\DATASETSU\\testdata\\3180\\1.bmp").convert('L');input6 = np.array(input6);input6 = torch.from_numpy(input6).float()
    input1 = input1.unsqueeze(0).unsqueeze(0);input2 = input2.unsqueeze(0).unsqueeze(0);input3 = input3.unsqueeze(0).unsqueeze(0)
    input4 = input4.unsqueeze(0).unsqueeze(0);input5 = input5.unsqueeze(0).unsqueeze(0);input6 = input6.unsqueeze(0).unsqueeze(0)
    print(input6.shape)

    input1 = input1[:,:,0:768,0:768]
    input1 = input1.to('cuda')

    input2 = input2[:,:,0:768,0:768]
    input2 = input2.to('cuda')

    input3 = input3[:,:,0:768,0:768]
    input3 = input3.to('cuda')

    input4 = input4[:,:,0:768,0:768]
    input4 = input4.to('cuda')

    input5 = input5[:,:,0:768,0:768]
    input5 = input5.to('cuda')

    input6 = input6[:,:,0:768,0:768]
    input6 = input6.to('cuda')
   

    x1p, x3p, U1, a1= net(input1, input3)


    print("x1 x1pp psnr: ", cal_psnr(x1p,input1))
    # print("x2 x2pp psnr: ", cal_psnr(x2p,input2))
    print("x3 x3pp psnr: ", cal_psnr(x3p,input3))
    # print("x4 x4pp psnr: ", cal_psnr(x4p,input4))
    # print("x5 x5pp psnr: ", cal_psnr(x5p,input5))


# z=[0.710mm 1.185mm 1.685mm 2.178mm 2.682mm 3.180mm]

    testnet = AS(-0.710e-3,size=(768,768))


    y = testnet(U1)

    y_real = y[0,0,:,:]
    y_imag = y[0,1,:,:]

    y_real_np = y_real.cpu().detach().numpy()
    y_imag_np = y_imag.cpu().detach().numpy()


    gt = torch.load("E:\\finallYBMS\\peoplePic\\DATASETS_X\\testdata\\U0\\" + file_name.split('.')[-2] + ".pt");
    # gt = torch.load("E:\\finallYBMS\\DATASETS4\\testdata\\U0\\" + file_name.split('.')[-2] + ".pt");
    # gt = torch.load("I:\\YBMS\\DATASETS\\testdata\\U0\\" + file_name.split('.')[-2] + ".pt");
    gt_real = gt[0,0,0:768,0:768]
    gt_imag = gt[0,1,0:768,0:768]
    gt_real = gt_real.numpy()
    gt_imag = gt_imag.numpy()

    gt_np = gt_real + gt_imag*1j
    y_np = y_real_np + y_imag_np*1j

    # print("gt_ang:\n", np.angle(gt_np,deg=True)) #?????? phi

    y_ang = (np.angle(y_np) - np.min(np.angle(y_np)))*200
    gt_ang = (np.angle(gt_np) - np.min(np.angle(gt_np)))*200

    # y_ang = np.angle(y_np)
    # gt_ang = np.angle(gt_np)

    # print("gt_ang:\n", np.max(gt_ang)) #?????? phi
    

    ang_res = np.abs(y_ang - gt_ang) * 100

    # print("y_ang:\n",y_ang )


    # print("gt_ang:\n",gt_ang )
    # print("ang_res:\n",ang_res)

    y_amp = pow(np.abs(y_np),2)
    gt_amp = pow(np.abs(gt_np),2)

    # print("y_amp:\n",y_amp )


    # print("gt_amp:\n",gt_amp )

    amp_psnr = np_cal_psnr(y_amp,gt_amp)
    ang_psnr = np_cal_psnr(y_ang,gt_ang)


    print("amp_psnr: ",amp_psnr)
    print("ang_psnr:", ang_psnr)

    

    y_amp = Image.fromarray(y_amp).convert('L')
    y_amp.show("net_amp")

    gt_amp = Image.fromarray(gt_amp).convert('L')
    gt_amp.show("gt_amp")

    # ang_res = Image.fromarray(ang_res).convert('L')
    # ang_res.show()

    y_ang = Image.fromarray(y_ang).convert('L')
    y_ang.show()

    gt_ang = Image.fromarray(gt_ang).convert('L')
    gt_ang.show()

    # Ureal = U_init[0,0,0:768,0:768]
    # Uimag = U_init[0,1,0:768,0:768]
    # Ureal = Ureal.cpu().detach().numpy()
    # Uimag = Uimag.cpu().detach().numpy()
    # U = Ureal + Uimag*1j
    # print(U)
    # U = Image.fromarray(np.abs(U)).convert('L')
    # U.show()


    return amp_psnr,ang_psnr





path = "I:\\YBMS\\DATASETS\\testdata\\07"
dirs = os.listdir(path)
avg_psnr = 0

# for image in dirs:
#     print(image,"====>")
#     amp_psnr, ang_psnr = test_param(image)
#     if ang_psnr == 0:
#         print(image)
#     avg_psnr += amp_psnr
# print("\033[1;35mavg_psnr: \033[0m",avg_psnr/30)

test_param("1.bmp")